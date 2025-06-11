# -*- coding: utf-8 -*-
"""
Data Preprocessing Module for SPM Prediction Pipeline, specfic to Delft3D-FM nc file structure/UGRID structure.

This module contains classes for preprocessing data, including UGRID rasterization,
dataset cleaning, shapefile creation, and optional CMEMS data integration.

Author: Beau van Koert, edits by L.Beyaard
Date: June 2025
"""

# Import libraries
import os
import gc
from datetime import datetime
import numpy as np
import geopandas as gpd
import dfm_tools as dfmt
import xugrid as xu
import xarray as xr
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import pandas as pd
import copernicusmarine  # to access the Copernicus Marine Toolbox API (included in DFM-Tools)
from matplotlib.ticker import AutoMinorLocator
from matplotlib.animation import FuncAnimation
from scipy.ndimage import generic_filter

#%% Processing DCSM data
class DCSMPreprocessor:
    """Preprocesses Delft3D DCSM hydrodynamic U-grid data for SPM prediction."""
    
    def __init__(self, config):
        """Initialize the preprocessor with configuration parameters."""
        # Base directories
        self.base_dir = config["base_dir"]
        self.data_dir = os.path.join(self.base_dir, "3.Data")
        self.results_dir = os.path.join(self.base_dir, "5.Results")
        self.figures_dir = os.path.join(self.results_dir, "Figures", "DCSM_Rasterized")
        self.map_dir = os.path.join(self.data_dir, "Map_Study_Area")
        self.dfm_dir = os.path.join(self.data_dir, "DFM_data")
        self.raster_dir = os.path.join(self.dfm_dir, "DCSM_rasterized_clip")

        # Input and output file paths (dynamically generated)
        self.input_ugrid = config["input_ugrid"]
        self.intermediate_ugrid = os.path.join(self.dfm_dir, "DCSM_ugrid_hydro_spm_intermediate.nc")
        self.input_raster = os.path.join(self.dfm_dir, "DCSM_log10_ly19_raster_nocl_32_timesort_2007.nc")
        self.merged_raster = os.path.join(self.raster_dir, "DCSM_raster_hydro_spm.nc")
        self.cleaned_raster = os.path.join(self.raster_dir, "DCSM_raster_hydro_spm_CleanedUp.nc")
        self.shapefile = os.path.join(self.map_dir, "bbox_DutchCoast_SPM_Plume_WaterOnly.shp")

        # Parameters
        self.start_time = config["start_time"]
        self.end_time = config["end_time"]
        self.time_range = [self.start_time, self.end_time]
        self.bbox = config.get("bbox", (2.4, 4.4, 51.0, 52.0))  # (x_min, x_max, y_min, y_max)
        self.buffer_size = config.get("buffer_size", 1.0)
        self.chunk_size = config.get("chunk_size", 10)
        self.plot_variables_flag = config.get("plot_variables", True)

        # Plotting labels
        self.subscript_labels = {
            'mesh2d_ucx': '(a) Eastward velocity (m/s)',
            'mesh2d_ucy': '(b) Northward velocity (m/s)',
            'mesh2d_ucz': '(c) Upward velocity (m/s)',
            'mesh2d_rho': '(d) Sea water density (kg/m³)',
            'mesh2d_sa1': '(e) Salinity (psu)',
            'mesh2d_tem1': '(f) Sea water temperature (°C)',
            'mesh2d_s1': '(g) Water Level (m)',
            'SPM': '(h) Suspended Particulate Matter (mg/L)',
            'mesh2d_s0': '(i) Water Level a day before (m)'
        }

    def ensure_directory(self, path):
        """Create directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

    def process_ugrid_data(self):
        """Process UGRID data and save as intermediate file."""
        if os.path.exists(self.intermediate_ugrid):
            print(f"File already exists: {self.intermediate_ugrid}. Skipping UGRID processing.")
            return xu.open_dataset(self.intermediate_ugrid, engine='netcdf4')
        
        ds = xu.open_dataset(self.input_ugrid, engine='netcdf4')
        ds = ds.sel(time=slice(self.time_range[0], self.time_range[1]))
        ds['mesh2d_s0'] = ds['mesh2d_s1'].shift(time=1)
        selected_vars = [var for var, data in ds.data_vars.items() if set(data.dims) == {'time', 'mesh2d_nFaces'}]
        ds = ds[selected_vars].rename({"mesh2d_water_quality_output_24": "SPM"})
        
        self.ensure_directory(os.path.dirname(self.intermediate_ugrid))
        print(f"Writing {os.path.basename(self.intermediate_ugrid)} to drive...")
        ds.ugrid.to_netcdf(self.intermediate_ugrid, mode='w')
        print(f"File saved: {self.intermediate_ugrid}")
        return ds

    def clip_datasets(self, ugrid_ds, raster_ds):
        """Clip UGRID and raster datasets to the specified bounding box with buffer."""
        x_min, x_max, y_min, y_max = self.bbox
        minx, maxx, miny, maxy = x_min - self.buffer_size, x_max + self.buffer_size, y_min - self.buffer_size, y_max + self.buffer_size
        
        print(f"\nBuffered Bounding Box: \nLongitude: {minx} to {maxx} (Range: {maxx - minx})"
              f"\nLatitude: {miny} to {maxy} (Range: {maxy - miny})")
        
        ugrid_clip = ugrid_ds.ugrid.sel(x=slice(minx, maxx), y=slice(miny, maxy))
        raster_clip = raster_ds.sel(x=slice(minx, maxx), y=slice(miny, maxy))
        return ugrid_clip, raster_clip

    def rasterize_dataset(self, ugrid_clip, raster_ref):
        """Rasterize the UGRID dataset in chunks and merge results."""
        if os.path.exists(self.merged_raster):
            print(f"Merged dataset already exists at {self.merged_raster}. Skipping rasterization...")
            return
        
        self.ensure_directory(os.path.dirname(self.merged_raster))
        start_day, end_day = 0, 365
        temp_files = []
        
        for lower_lim in range(start_day, end_day, self.chunk_size):
            upper_lim = min(lower_lim + self.chunk_size, end_day)
            limits = np.arange(lower_lim, upper_lim)
            print(f"\nProcessing days {lower_lim} to {upper_lim}...")
            
            base_raster = None
            for i in limits:
                print(f"Rasterizing time step {i}...")
                rast_temp = dfmt.rasterize_ugrid(ugrid_clip.isel(time=i), ds_like=raster_ref['SPM'][1, :, :])
                base_raster = rast_temp if base_raster is None else xr.concat([base_raster, rast_temp], dim='time')
            
            output_name = f'DCSM_raster_temp_{lower_lim}to{upper_lim}.nc'
            output_path_full = os.path.join(os.path.dirname(self.merged_raster), output_name)
            base_raster.to_netcdf(output_path_full, mode='w')
            temp_files.append(output_path_full)
            print(f"Raster data saved to {output_path_full}")
        
        print("Merging rasterized chunks...")
        ds_merged = xr.open_mfdataset(temp_files, concat_dim='time', combine='nested', engine='netcdf4')
        ds_merged.to_netcdf(self.merged_raster, mode='w', engine='netcdf4')
        ds_merged.close()
        print(f"Merged dataset saved as: {self.merged_raster}")

    def clean_dataset(self, ds):
        """Clean the dataset by adjusting coordinates."""
        ds = ds.sortby("time")
        
        ds = ds.transpose("time", "y", "x", ...)
        keep_coords = ['x', 'y', 'time', 'mesh2d_layer_sigma', 'mesh2d_nNodes']
        drop_coords = [coord for coord in ds.coords.keys() if coord not in keep_coords]
        for coord in drop_coords:
            ds = ds.reset_coords(coord).drop_vars(coord)
        
        ds.attrs['Conventions'] = 'CF-1.9, Custom Processing (U-GRID-1.0 to rasterized)'
        ds.attrs['Institution'] = 'Deltares'
        ds.attrs['Processed_by'] = f'Beau van Koert, {datetime.now().strftime("%B %d, %Y")}'
        return ds

    def create_shapefile(self, raster_ds):
        """Create a shapefile of water-only areas from the raster dataset."""
        if os.path.exists(self.shapefile):
            print(f"Shapefile already exists at {self.shapefile}. Skipping creation...")
            return gpd.read_file(self.shapefile)
        
        x_min, x_max, y_min, y_max = self.bbox
        rect = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)])
        gdf_rect = gpd.GeoDataFrame({"geometry": [rect]}, crs="EPSG:4326")
        
        spm_clipped = raster_ds['mesh2d_SPM'].sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
        water_mask = ~spm_clipped.isel(time=0).isnull()
        water_mask = np.flipud(water_mask)
        
        transform = rasterio.transform.from_bounds(x_min, y_min, x_max, y_max, len(spm_clipped.x), len(spm_clipped.y))
        water_shapes = shapes(water_mask.astype(np.uint8), mask=water_mask, transform=transform)
        water_polygons = [{'geometry': Polygon(shape['coordinates'][0]), 'value': value} 
                         for shape, value in water_shapes if value == 1]
        gdf_water = gpd.GeoDataFrame(water_polygons, crs="EPSG:4326")
        
        refined_gdf = gpd.overlay(gdf_rect, gdf_water, how='intersection')
        self.ensure_directory(os.path.dirname(self.shapefile))
        refined_gdf.to_file(self.shapefile)
        print(f"Refined shapefile saved as {self.shapefile}")
        
        # Visualize the shapefile
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_rect.boundary.plot(ax=ax, color='blue', linewidth=2, label='Original BBox')
        refined_gdf.boundary.plot(ax=ax, color='red', linewidth=1, label='Water-Only Shapefile')
        plt.title('Refined Study Area (Water Cells Only)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.show()
        return refined_gdf

    def plot_variables(self, raster_ds, shapefile_gdf):
        """Plot individual and combined figures for all variables."""
        if not self.plot_variables_flag:
            print("Skipping variables plotting as per config.")
            return
        
        self.ensure_directory(self.figures_dir)
        yearly_means = raster_ds.mean(dim='time')
        variables = [var for var in raster_ds.data_vars if 'mesh2d' in var or var in ['SPM', 'SPM_log']]
        
        # Individual plots
        for var in variables:
            plt.figure(figsize=(10, 6))
            im = plt.pcolormesh(yearly_means['x'], yearly_means['y'], yearly_means[var], cmap='jet', shading='auto')
            plt.colorbar(im, label=var.replace('mesh2d_', ''))
            plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.title(self.subscript_labels.get(var, f'{var.replace("mesh2d_", "")} - 2007 Mean Value'))
            shapefile_gdf.boundary.plot(ax=plt.gca(), color='black', linestyle='--', linewidth=1.5, label='Study Area')
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, f"DCSM-DCL-2007-{var.replace('mesh2d_', '')}-mean.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Combined subplot
        n_vars, n_cols = len(variables), 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
        
        for idx, var in enumerate(variables):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            im = ax.pcolormesh(yearly_means['x'], yearly_means['y'], yearly_means[var], cmap='jet', shading='auto')
            ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
            fig.colorbar(im, ax=ax)
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('Latitude (deg)')
            ax.text(0.5, -0.15, self.subscript_labels.get(var, f'({chr(97+idx)}) {var.replace("mesh2d_", "")}'), 
                    ha='center', va='center', transform=ax.transAxes)
            shapefile_gdf.boundary.plot(ax=ax, color='black', linestyle='--', linewidth=1.5, label='Study Area')
            ax.legend()
        
        for idx in range(len(variables), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "DCSM-DCL-2007-all-variables-mean.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"All plots saved in '{self.figures_dir}'.")

    def process(self):
        """Execute the full preprocessing pipeline."""
        # Step 1: Process UGRID data
        print("Step 1: Processing UGRID data...")
        ugrid_ds = self.process_ugrid_data()

        # Step 2: Clip datasets
        print("Step 2: Clipping datasets...")
        raster_ds = xr.open_dataset(self.input_raster)
        ugrid_clip, raster_clip = self.clip_datasets(ugrid_ds, raster_ds)

        # Step 3: Rasterize dataset
        print("Step 3: Rasterizing dataset...")
        self.rasterize_dataset(ugrid_clip, raster_clip)

        # Step 4: Clean dataset
        print("Step 4: Cleaning dataset...")
        if os.path.exists(self.cleaned_raster):
            print(f"Cleaned dataset already exists at {self.cleaned_raster}. Skipping cleaning...")
        else:
            ds_merged = xr.open_dataset(self.merged_raster)
            ds_cleaned = self.clean_dataset(ds_merged)
            ds_cleaned.to_netcdf(self.cleaned_raster, mode='w', engine='netcdf4')
            ds_cleaned.close()
            print(f"Cleaned dataset saved as: {self.cleaned_raster}")

        # Step 5: Create shapefile
        print("Step 5: Creating shapefile...")
        raster_ds = xr.open_dataset(self.cleaned_raster)
        shapefile_gdf = self.create_shapefile(raster_ds)

        # Step 6: Plot all variables
        print("Step 6: Plotting all variables...")
        self.plot_variables(raster_ds, shapefile_gdf)

        return self.cleaned_raster, self.shapefile

#%% Processing CMEMS satellite data
class CMEMS_processing:
    """Class to preprocess CMEMS satellite SPM data, regrid it to the DCSM dataset grid, and append it as a static input for ConvLSTM modeling."""
    
    def __init__(self, config, dcsm_path, output_dir_plots, shapefile_path):
        """
        Initialize the CMEMS_processing class with paths and configuration.

        Args:
            config (dict): Configuration dictionary containing temporal and spatial bounds.
            dcsm_path (str): Path to the input DCSM NetCDF file.
            output_dir_plots (str): Directory to save visualization plots.
            shapefile_path (str): Path to the shapefile defining the study area for plotting.

        Returns:
            None.
        """
        # Store config for access in methods
        self.config = config
        self.dcsm_path = dcsm_path
        # Use the directory of dcsm_path as the output directory
        self.output_dir = os.path.dirname(dcsm_path)
        self.output_dir_plots = output_dir_plots
        self.shapefile_path = shapefile_path
        
        # Validate required config parameters
        required_params = ["start_time", "end_time", "bbox", "buffer_size"]
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise KeyError(f"Config missing required parameters: {missing_params}")
        
        # Extract CMEMS username and password
        self.CMEMS_username = config["CMEMS_username"]
        self.CMEMS_password = config["CMEMS_password"]
        
        # Extract temporal and spatial bounds from config
        self.start_datetime = config["start_time"]
        self.end_datetime = config["end_time"]
        self.bbox = config["bbox"]              # (x_min, x_max, y_min, y_max)
        self.buffer_size = config["buffer_size"]
        self.minimum_longitude = self.bbox[0] - self.buffer_size
        self.maximum_longitude = self.bbox[1] + self.buffer_size
        self.minimum_latitude = self.bbox[2] - self.buffer_size
        self.maximum_latitude = self.bbox[3] + self.buffer_size

        # Derived attributes (to be set by methods)
        self.cmems_path = None
        self.cmems_ds = None
        self.dcsm_ds = None
        self.gdf = None
        self.land_mask = None
        self.nan_mask = None
        self.out_of_domain_mask = None
        self.x = None
        self.y = None
        self.nx = None
        self.ny = None
        self.time_steps = None
        self.cmems_spm_daily = None
        self.cmems_spm_values = None

        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir_plots, exist_ok=True)

    def __del__(self):
        """
        Destructor to ensure datasets are closed, large attributes are cleared, and memory is released.
        """
        # Close datasets
        if hasattr(self, 'dcsm_ds') and self.dcsm_ds is not None:
            try:
                self.dcsm_ds.close()
                print("Closed DCSM dataset.")
            except Exception as e:
                print(f"Error closing DCSM dataset: {str(e)}")
            self.dcsm_ds = None

        if hasattr(self, 'cmems_ds') and self.cmems_ds is not None:
            try:
                self.cmems_ds.close()
                print("Closed CMEMS dataset.")
            except Exception as e:
                print(f"Error closing CMEMS dataset: {str(e)}")
            self.cmems_ds = None

        # Clear large attributes to free memory
        attributes_to_clear = [
            'cmems_spm_daily', 'cmems_spm_values', 'cmems_spm_monthly',
            'land_mask', 'nan_mask', 'out_of_domain_mask', 'gdf'
        ]
        for attr in attributes_to_clear:
            if hasattr(self, attr):
                setattr(self, attr, None)

        # Force garbage collection
        gc.collect()
        print("Garbage collection completed for CMEMS_processing instance.")

    def process(self, create_animation=False):
        """
        Execute the full CMEMS preprocessing pipeline: download, process, regrid, append to DCSM dataset, and visualize.

        Args:
            create_animation (bool): If True, create an animation of the daily SPM grid (default: False).

        Returns:
            dict: Updated config dictionary.
        """
        # CMEMS pre-processing pipeline for 'monthly-to-daily CMEMS data'
        if self.config.get("add_cmems_data", False):
            self.config = self.download_CMEMS_data()
            self.calc_monthly_avg_CMEMS()
            self.interpolate_CMEMS_to_daily()
            self.load_DCSM_dataset()
            self.create_DCSM_masks()
            self.load_shapefile_AOI()
            self.rasterize_CMEMS_to_DCSM()
            self.append_CMEMS_to_DCSM_ds()
            self.save_updated_ds()
            self.plot_CMEMS_yearly_mean_SPM()
            self.plot_CMEMS_std_dev_SPM()
            if create_animation:
                self.create_SPM_animation()
            # return updated config
            return self.config
        
        # CMEMS pre-processing pipeline for 4DVarNet daily CMEMS data
        if self.config.get("add_cmems_4dvarnet_data", False):
            # Skip download, averaging, and interpolation step
            self.initialize_CMEMS_4DVarNet_data()
            self.load_DCSM_dataset()
            self.create_DCSM_masks()
            self.load_shapefile_AOI()
            self.rasterize_CMEMS_to_DCSM()
            self.append_CMEMS_to_DCSM_ds()
            self.save_updated_ds()
            self.plot_CMEMS_yearly_mean_SPM()
            self.plot_CMEMS_std_dev_SPM()
            if create_animation:
                self.create_SPM_animation()
            # return updated config
            return self.config

        raise ValueError("Neither 'add_cmems_data' nor 'add_cmems_4dvarnet_data' is set to True in config.")

    def download_CMEMS_data(self):
        """
        Download the CMEMS satellite SPM dataset for the specified time period
        and area using the Copernicus Marine Toolbox API.

        Returns:
            dict: Updated config dictionary.
        """
        try:
            # Define the CMEMS data directory (go up two levels from DCSM_rasterized_clip to 3.Data, then into CMEMS_data)
            cmems_data_dir = os.path.abspath(os.path.join('..', '..', "3.Data", "CMEMS_data"))
            os.makedirs(cmems_data_dir, exist_ok=True)

            # Define the output filename with dynamic lon/lat ranges
            output_filename = f'CMEMS_{self.start_datetime}_{self.end_datetime}_lon{self.minimum_longitude}to{self.maximum_longitude}_lat{self.minimum_latitude}to{self.maximum_latitude}.nc'
            self.cmems_path = os.path.join(cmems_data_dir, output_filename)

            # Store the raw CMEMS filepath in the config
            self.config['raw_cmems_path'] = self.cmems_path
            print(f"\nRaw CMEMS path: {self.config['raw_cmems_path']}")

            # Check if the file already exists
            if os.path.exists(self.cmems_path):
                print(f"CMEMS dataset already exists at {self.cmems_path}. Skipping download.")
                return self.config

            # Download the CMEMS dataset using copernicusmarine
            print(f"Downloading CMEMS dataset to {self.cmems_path}...")
            copernicusmarine.subset(
                username=self.CMEMS_username,
                password=self.CMEMS_password,
                dataset_id="cmems_obs-oc_atl_bgc-transp_my_l3-multi-1km_P1D",
                dataset_version="202311",
                variables=["SPM"],
                minimum_longitude=self.minimum_longitude,
                maximum_longitude=self.maximum_longitude,
                minimum_latitude=self.minimum_latitude,
                maximum_latitude=self.maximum_latitude,
                start_datetime=f"{self.start_datetime}T00:00:00",
                end_datetime=f"{self.end_datetime}T00:00:00",
                coordinates_selection_method="strict-inside",
                disable_progress_bar=False,
                output_directory=cmems_data_dir,
                output_filename=output_filename,
            )
            print(f"CMEMS dataset downloaded to {self.cmems_path}")
            return self.config
        
        except Exception as e:
            raise RuntimeError(f"Failed to download CMEMS dataset: {str(e)}")

    def calc_monthly_avg_CMEMS(self):
        """
        Calculate the monthly average SPM from the daily CMEMS dataset to fill
        gaps caused by cloud cover.

        Returns:
            None.
        """
        try:
            # Load the CMEMS dataset if not already loaded
            if self.cmems_ds is None:
                self.cmems_ds = xr.open_dataset(self.cmems_path)

            # Select data for the specified time period
            cmems_spm = self.cmems_ds['SPM'].sel(time=slice(self.start_datetime, self.end_datetime))

            # Group by month and compute the mean to fill gaps
            cmems_spm_monthly = cmems_spm.groupby('time.month').mean(dim='time')

            # Store the monthly averages
            self.cmems_spm_monthly = cmems_spm_monthly

            # Debug: Check the monthly averages
            print("CMEMS SPM Monthly Averages:")
            for month in range(1, 13):
                spm_month = cmems_spm_monthly.sel(month=month).values
                print(f"  Month {month}: Mean SPM = {np.nanmean(spm_month):.2f} mg/L, Fraction non-NaN = {np.sum(~np.isnan(spm_month)) / spm_month.size:.4f}")
        except Exception as e:
            raise RuntimeError(f"Failed to calculate monthly average CMEMS SPM: {str(e)}")

    def interpolate_CMEMS_to_daily(self):
        """
        Interpolate the monthly averaged CMEMS SPM data to daily resolution to match the temporal resolution of the DCSM dataset.
        Fill remaining NaN values using a neighborhood filter (7x7, then 49x49 if needed), and finally fill any remaining NaNs with 0.0.

        Returns:
            None.
        """
        try:
            # Create a daily time index for the target period
            daily_times = pd.date_range(start=self.start_datetime, end=self.end_datetime, freq='D')
            self.time_steps = len(daily_times)

            # Create a time array for the months (midpoint of each month)
            month_midpoints = pd.date_range(start=self.start_datetime, end=self.end_datetime, freq='MS') + pd.Timedelta(days=14)
            month_midpoints = month_midpoints[:12]  # Ensure exactly 12 months

            # Interpolate monthly averages to daily resolution
            month_indices = np.linspace(1, 12, self.time_steps)
            cmems_spm_daily = self.cmems_spm_monthly.interp(
                month=month_indices,
                method='linear'
            )

            # Create a new DataArray with the daily time index
            cmems_spm_daily = xr.DataArray(
                data=cmems_spm_daily.values,
                dims=['time', 'latitude', 'longitude'],
                coords={
                    'time': daily_times,
                    'latitude': cmems_spm_daily['latitude'],
                    'longitude': cmems_spm_daily['longitude']
                },
                name='SPM'
            )

            # Get the raw data array (shape: [time, latitude, longitude])
            spm_data = cmems_spm_daily.values

            # Define a function to compute the mean of non-NaN values in the neighborhood
            def nan_mean_filter(window):
                # Only compute if the center pixel is NaN
                center_idx = window.size // 2
                if not np.isnan(window[center_idx]):
                    return window[center_idx]  # Return the original value if not NaN
                # Compute the mean of non-NaN values in the window
                valid_values = window[~np.isnan(window)]
                return np.mean(valid_values) if valid_values.size > 0 else np.nan

            # Step 1: Apply a 7x7 neighborhood filter to NaN values
            print("Applying 7x7 neighborhood filter to NaN values...")
            footprint_7x7 = np.ones((7, 7))  # Define the 7x7 window
            for t in range(spm_data.shape[0]):  # Process each time step
                spm_data[t] = generic_filter(
                    spm_data[t],
                    function=nan_mean_filter,
                    footprint=footprint_7x7,
                    mode='nearest'  # Handle edges by extending the nearest value
                )

            # Step 2: Check for remaining NaNs and apply a 49x49 filter if needed
            remaining_nans = np.any(np.isnan(spm_data))
            if remaining_nans:
                print("Applying 49x49 neighborhood filter to remaining NaN values...")
                footprint_49x49 = np.ones((49, 49))  # Define the 49x49 window
                for t in range(spm_data.shape[0]):  # Process each time step
                    spm_data[t] = generic_filter(
                        spm_data[t],
                        function=nan_mean_filter,
                        footprint=footprint_49x49,
                        mode='nearest'
                    )

            # Step 3: Fill any remaining NaNs with 0.0
            remaining_nans_after_filters = np.any(np.isnan(spm_data))
            if remaining_nans_after_filters:
                print("Filling remaining NaNs with 0.0...")
                spm_data = np.nan_to_num(spm_data, nan=0.0)

            # Update the DataArray with the filtered data
            cmems_spm_daily = xr.DataArray(
                data=spm_data,
                dims=['time', 'latitude', 'longitude'],
                coords={
                    'time': daily_times,
                    'latitude': cmems_spm_daily['latitude'],
                    'longitude': cmems_spm_daily['longitude']
                },
                name='SPM'
            )

            # Store the daily interpolated data
            self.cmems_spm_daily = cmems_spm_daily

            # Debug: Check the daily interpolated data
            print("CMEMS SPM after interpolation to daily resolution and NaN handling:")
            print(f"  Min SPM: {np.nanmin(cmems_spm_daily.values):.2f} mg/L")
            print(f"  Max SPM: {np.nanmax(cmems_spm_daily.values):.2f} mg/L")
            print(f"  Mean SPM: {np.nanmean(cmems_spm_daily.values):.2f} mg/L")
            print(f"  Remaining NaNs: {np.any(np.isnan(cmems_spm_daily.values))}")
        except Exception as e:
            raise RuntimeError(f"Failed to interpolate CMEMS SPM to daily resolution: {str(e)}")

    def load_DCSM_dataset(self):
        """
        Load the DCSM dataset from the specified path and verify its structure.

        Returns:
            None.
        """
        try:
            self.dcsm_ds = xr.open_dataset(self.dcsm_path)
            print(f"DCSM dataset loaded from {self.dcsm_path}")

            # Verify required coordinates and variables
            required_coords = ['x', 'y', 'time']
            required_vars = ['mesh2d_s1']
            for coord in required_coords:
                if coord not in self.dcsm_ds.coords:
                    raise KeyError(f"DCSM dataset missing coordinate: {coord}")
            for var in required_vars:
                if var not in self.dcsm_ds.variables:
                    raise KeyError(f"DCSM dataset missing variable: {var}")

            # Extract grid dimensions and coordinates
            self.x = self.dcsm_ds['x'].values  # Longitudes
            self.y = self.dcsm_ds['y'].values  # Latitudes
            self.nx = len(self.x)
            self.ny = len(self.y)
            print(f"DCSM grid dimensions: nx={self.nx}, ny={self.ny}")
        except Exception as e:
            raise RuntimeError(f"Failed to load DCSM dataset: {str(e)}")

    def create_DCSM_masks(self):
        """
        Create land, NaN, and out-of-domain masks based on the DCSM dataset's water depth to distinguish water and land cells.

        Returns:
            None.
        """
        try:
            # Extract water depth at the first timestep
            waterdepth = self.dcsm_ds['mesh2d_s1'].isel(time=0).data  # Shape: (y, x)

            # Create a land mask (1 for water, 0 for land)
            self.land_mask = np.zeros((self.ny, self.nx), dtype=float)  # Shape: (y, x)
            self.land_mask[~np.isnan(waterdepth)] = 1  # Water cells (non-NaN) are 1
            self.land_mask = self.land_mask.T  # Transpose to (x, y)

            # Create a NaN mask (NaN for land, 1 for water)
            self.nan_mask = np.ones((self.ny, self.nx), dtype=float)  # Shape: (y, x)
            self.nan_mask[np.isnan(waterdepth)] = np.nan  # Land cells are NaN
            self.nan_mask = self.nan_mask.T  # Transpose to (x, y)

            # Create an out-of-domain mask (True for NaN cells)
            self.out_of_domain_mask = np.isnan(waterdepth)  # Shape: (y, x)
            self.out_of_domain_mask = self.out_of_domain_mask.T  # Transpose to (x, y)

            # Debug: Check the masks
            total_water_cells = np.sum(self.land_mask)
            print(f"Land mask created: Total water cells = {int(total_water_cells)}, Fraction of water cells = {total_water_cells / self.land_mask.size:.4f}")
        except Exception as e:
            raise RuntimeError(f"Failed to create DCSM masks: {str(e)}")

    def load_shapefile_AOI(self):
        """
        Load the shapefile defining the study area and project it to EPSG:4326 for plotting.

        Returns:
            None.
        """
        try:
            self.gdf = gpd.read_file(self.shapefile_path)
            self.gdf = self.gdf.to_crs(epsg=4326)
            print(f"Shapefile loaded from {self.shapefile_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load shapefile: {str(e)}")
    
    def initialize_CMEMS_4DVarNet_data(self):
        """
        Load the pre-downloaded, gap-filled CMEMS dataset (processed with 4DVarNet) and prepare it for processing.

        Returns:
            None.
        """
        try:
            # Define the CMEMS data directory (go up two levels from DCSM_rasterized_clip to 3.Data, then into CMEMS_data)
            cmems_data_dir = os.path.abspath(os.path.join('..', '..', "3.Data", "CMEMS_data"))
            os.makedirs(cmems_data_dir, exist_ok=True)
            
            # Define CMEMS input path
            self.cmems_path = os.path.join(cmems_data_dir, "CMEMS2007_gapfilled.nc")
            
            # Check if the file exists
            if not os.path.exists(self.cmems_path):
                raise FileNotFoundError(f"CMEMS 4DVarNet dataset not found at {self.cmems_path}")

            # Load the dataset
            self.cmems_ds = xr.open_dataset(self.cmems_path)
            print(f"CMEMS 4DVarNet dataset loaded from {self.cmems_path}")

            # Verify required variable and coordinates
            if 'SPM' not in self.cmems_ds.variables:
                raise KeyError("CMEMS dataset missing required variable: 'SPM'")
            
            # Check for coordinate names (support both 'lon'/'lat' and 'longitude'/'latitude')
            lon_coord = 'lon' if 'lon' in self.cmems_ds.coords else 'longitude'
            lat_coord = 'lat' if 'lat' in self.cmems_ds.coords else 'latitude'
            if lon_coord not in self.cmems_ds.coords or lat_coord not in self.cmems_ds.coords:
                raise KeyError(f"CMEMS dataset missing required coordinates: expected '{lon_coord}' and '{lat_coord}'")

            # Select data for the specified time period and variable
            self.cmems_spm_daily = self.cmems_ds['SPM'].sel(time=slice(self.start_datetime, self.end_datetime))

            # Create a daily time index for the target period
            daily_times = pd.date_range(start=self.start_datetime, end=self.end_datetime, freq='D')
            self.time_steps = len(daily_times)

            # Verify that the time dimension matches the expected daily frequency
            if len(self.cmems_spm_daily.time) != self.time_steps:
                raise ValueError(f"CMEMS 4DVarNet data time steps ({len(self.cmems_spm_daily.time)}) do not match expected daily frequency ({self.time_steps})")

            # Rename coordinates to 'longitude' and 'latitude' for consistency with the rest of the pipeline
            self.cmems_spm_daily = self.cmems_spm_daily.rename({lon_coord: 'longitude', lat_coord: 'latitude'})

            # Debug: Check the loaded data
            print("CMEMS 4DVarNet SPM Data:")
            print(f"  Time steps: {self.time_steps}")
            print(f"  Min SPM: {np.nanmin(self.cmems_spm_daily.values):.2f} mg/L")
            print(f"  Max SPM: {np.nanmax(self.cmems_spm_daily.values):.2f} mg/L")
            print(f"  Mean SPM: {np.nanmean(self.cmems_spm_daily.values):.2f} mg/L")
            print(f"  Remaining NaNs: {np.any(np.isnan(self.cmems_spm_daily.values))}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CMEMS 4DVarNet data: {str(e)}")
    
    def rasterize_CMEMS_to_DCSM(self):
        """
        Regrid the daily CMEMS SPM data to the spatial grid of the DCSM dataset using linear interpolation,
        and apply masks to handle land and out-of-domain cells.

        Returns:
            None.
        """
        try:
            # Plot 1: Raw CMEMS SPM data (mean of daily data, before interpolation)
            cmems_spm_mean = self.cmems_spm_daily.mean(dim='time')
            plt.figure(figsize=(10, 8))
            im = plt.pcolormesh(
                cmems_spm_mean['longitude'],
                cmems_spm_mean['latitude'],
                cmems_spm_mean.values,  # Use .values to pass a NumPy array to pcolormesh
                cmap='viridis',
                shading='nearest'
            )
            plt.colorbar(im, label='SPM (mg/L)')
            self.gdf.boundary.plot(ax=plt.gca(), color='black', linewidth=0.5, label='Study Area')
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.title('CMEMS SPM (Mean of 2007 Daily Data) Before Interpolation')
            plt.legend()
            # Add minor ticks and position them outside
            plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
            plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
            plt.tick_params(axis='both', which='minor', direction='out', length=3)
            plot_path_1 = os.path.join(self.output_dir_plots, 'CMEMS_SPM_Raw_Mean_2007_Before_Interpolation.png')
            plt.savefig(plot_path_1, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot 1 saved to: {plot_path_1}")

            # Interpolate CMEMS SPM data to the DCSM grid
            cmems_spm_interp = self.cmems_spm_daily.interp(
                latitude=self.dcsm_ds['y'],
                longitude=self.dcsm_ds['x'],
                method='linear'
            )

            # Plot 2: Interpolated CMEMS SPM data (mean, before masking)
            cmems_spm_interp_mean = cmems_spm_interp.mean(dim='time')
            plt.figure(figsize=(10, 8))
            im = plt.pcolormesh(
                self.x,
                self.y,
                cmems_spm_interp_mean.values,
                cmap='viridis',
                shading='nearest'
            )
            plt.colorbar(im, label='SPM (mg/L)')
            self.gdf.boundary.plot(ax=plt.gca(), color='black', linewidth=0.5, label='Study Area')
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.title('Interpolated CMEMS SPM (Mean of 2007) Before Masking')
            plt.legend()
            # Add minor ticks and position them outside
            plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
            plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
            plt.tick_params(axis='both', which='minor', direction='out', length=3)
            plot_path_2 = os.path.join(self.output_dir_plots, 'CMEMS_SPM_Interpolated_Mean_2007_Before_Masking.png')
            plt.savefig(plot_path_2, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot 2 saved to: {plot_path_2}")

            # Transpose to match (time, x, y) for masking
            cmems_spm_values = cmems_spm_interp.values.transpose(0, 2, 1)  # From (time, y, x) to (time, x, y)

            # Apply masks: set land and out-of-domain cells to NaN, and NaN water cells to 0
            cmems_spm_values = np.where(self.land_mask == 0, np.nan, cmems_spm_values)
            cmems_spm_values = np.where(self.out_of_domain_mask, np.nan, cmems_spm_values)
            cmems_spm_values = np.where((self.land_mask == 1) & np.isnan(cmems_spm_values), 0.0, cmems_spm_values)

            # Plot 3: Interpolated CMEMS SPM data (mean, after masking)
            cmems_spm_values_mean = np.nanmean(cmems_spm_values, axis=0)
            plt.figure(figsize=(10, 8))
            im = plt.pcolormesh(
                self.x,
                self.y,
                cmems_spm_values_mean.T,
                cmap='viridis',
                shading='nearest'
            )
            plt.colorbar(im, label='SPM (mg/L)')
            self.gdf.boundary.plot(ax=plt.gca(), color='black', linewidth=0.5, label='Study Area')
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.title('Interpolated CMEMS SPM (Mean of 2007) After Masking')
            plt.legend()
            # Add minor ticks and position them outside
            plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
            plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
            plt.tick_params(axis='both', which='minor', direction='out', length=3)
            plot_path_3 = os.path.join(self.output_dir_plots, 'CMEMS_SPM_Interpolated_Mean_2007_After_Masking.png')
            plt.savefig(plot_path_3, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot 3 saved to: {plot_path_3}")

            # Store the regridded data
            self.cmems_spm_values = cmems_spm_values
        except Exception as e:
            raise RuntimeError(f"Failed to rasterize CMEMS SPM to DCSM grid: {str(e)}")

    def append_CMEMS_to_DCSM_ds(self):
        """
        Create a DataArray for the regridded daily CMEMS SPM data and append it to the DCSM dataset.

        Returns:
            None.
        """
        # Define variable attributes
        if self.config.get("add_cmems_data", False):
            variable_attrs = {
                'unit': 'mg/L',
                'long_name': 'Daily CMEMS Suspended Particulate Matter for 2007',
                'description': 'Daily CMEMS SPM for 2007, derived from monthly averages and interpolated to the DCSM grid, used as a static input variable for ConvLSTM modeling.',
                'source': 'CMEMS satellite-derived SPM data (cmems_obs-oc_atl_bgc-transp_my_l3-multi-1km_P1D), monthly averaged and interpolated to daily',
                'interpolation_method': 'Linear interpolation from monthly to daily, then to DCSM grid',
                'masking': 'Land cells set to NaN, out-of-domain cells set to NaN, NaN water cells set to 0'
            }
        elif self.config.get("add_cmems_4dvarnet_data", False):
            variable_attrs = {
                'unit': 'mg/L',
                'long_name': 'Daily CMEMS Suspended Particulate Matter for 2007',
                'description': 'Daily CMEMS SPM for 2007, gapfilled using 4DVarNet method',
                'source': 'CMEMS satellite-derived SPM data (cmems_obs-oc_atl_bgc-transp_my_l3-multi-1km_P1D), gapfilled using 4DVarNet algorithm',
                'masking': 'Land cells set to NaN, out-of-domain cells set to NaN, NaN water cells set to 0'
            }
        else:
            raise ValueError("Neither 'add_cmems_data' nor 'add_cmems_4dvarnet_data' is set to True in config.")
        
        try:
            # Create a DataArray for the daily CMEMS SPM
            daily_times = pd.date_range(start=self.start_datetime, end=self.end_datetime, freq='D')
            cmems_spm_da = xr.DataArray(
                data=self.cmems_spm_values.transpose(0, 2, 1),  # Back to (time, y, x)
                dims=['time', 'y', 'x'],
                coords={
                    'time': self.dcsm_ds['time'].sel(time=slice(self.start_datetime, self.end_datetime)),
                    'y': self.dcsm_ds['y'],
                    'x': self.dcsm_ds['x']
                },
                name='CMEMS_SPM_daily_2007',
                attrs=variable_attrs
            )

            # Add the CMEMS_SPM_daily_2007 variable to the dataset
            self.dcsm_ds['CMEMS_SPM_daily_2007'] = cmems_spm_da
            print("CMEMS SPM data appended to DCSM dataset as 'CMEMS_SPM_daily_2007'")
        except Exception as e:
            raise RuntimeError(f"Failed to append CMEMS SPM to DCSM dataset: {str(e)}")

    def save_updated_ds(self):
        """
        Save the updated DCSM dataset with the CMEMS SPM data to the specified output directory.

        Returns:
            None.
        """
        try:
            output_filename = "DCSM_raster_hydro_spm_CMEMS_daily_2007.nc"
            if self.config.get("add_cmems_4dvarnet_data", False):
                output_filename = "DCSM_raster_hydro_spm_CMEMS_daily_4DVarNet_2007.nc"
            output_path = os.path.join(self.output_dir, output_filename)
            self.dcsm_ds.to_netcdf(output_path)
            print(f"Updated DCSM dataset saved to: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save updated DCSM dataset: {str(e)}")

    def plot_CMEMS_yearly_mean_SPM(self):
        """
        Plot the yearly mean of the regridded daily CMEMS SPM data and save the plot to the output directory.

        Returns:
            None.
        """
        try:
            cmems_spm_mean = np.nanmean(self.cmems_spm_values, axis=0)
            plt.figure(figsize=(10, 8))
            im = plt.pcolormesh(self.x, self.y, cmems_spm_mean.T, cmap='viridis', shading='nearest')
            plt.colorbar(im, label='SPM (mg/L)')
            self.gdf.boundary.plot(ax=plt.gca(), color='black', linewidth=0.5, label='Study Area')
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.title('Mean CMEMS SPM for 2007 (Static Input for ConvLSTM)')
            plt.legend()
            # Add minor ticks and position them outside
            plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
            plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
            plt.tick_params(axis='both', which='minor', direction='out', length=3)
            plot_path = os.path.join(self.output_dir_plots, 'CMEMS_SPM_Mean_2007_Static.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Mean SPM plot saved to: {plot_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to plot CMEMS yearly mean SPM: {str(e)}")

    def plot_CMEMS_std_dev_SPM(self):
        """
        Plot the standard deviation of the regridded daily CMEMS SPM data and save the plot to the output directory.

        Returns:
            None.
        """
        try:
            cmems_spm_std = np.nanstd(self.cmems_spm_values, axis=0)
            plt.figure(figsize=(10, 8))
            im = plt.pcolormesh(self.x, self.y, cmems_spm_std.T, cmap='plasma', shading='nearest')
            plt.colorbar(im, label='SPM (mg/L)')
            self.gdf.boundary.plot(ax=plt.gca(), color='black', linewidth=0.5, label='Study Area')
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.title('Standard Deviation of CMEMS SPM for 2007')
            plt.legend()
            # Add minor ticks and position them outside
            plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
            plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
            plt.tick_params(axis='both', which='minor', direction='out', length=3)
            plot_path = os.path.join(self.output_dir_plots, 'CMEMS_SPM_StdDev_2007.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Standard deviation SPM plot saved to: {plot_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to plot CMEMS SPM standard deviation: {str(e)}")

    def create_SPM_animation(self):
        """
        Create a GIF animation of the daily SPM grid over the entire time range to visualize the interpolation from monthly to daily.

        Returns:
            None.
        """
        try:
            print("Creating animation...")
            fig, ax = plt.subplots(figsize=(10, 8))
            # Initial frame (first day)
            spm_initial = self.cmems_spm_values[0].T  # Transpose to (y, x) for plotting
            im = ax.pcolormesh(self.x, self.y, spm_initial, cmap='viridis', shading='nearest')
            plt.colorbar(im, label='SPM (mg/L)')
            self.gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, label='Study Area')
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('Latitude (deg)')
            ax.set_title('CMEMS SPM on 2007-01-01')
            ax.legend()
            # Add minor ticks and position them outside
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', which='minor', direction='out', length=3)

            # Create daily time index for titles
            daily_times = pd.date_range(start=self.start_datetime, end=self.end_datetime, freq='D')

            def update(frame):
                spm_frame = self.cmems_spm_values[frame].T  # Transpose to (y, x) for plotting
                im.set_array(spm_frame.ravel())  # Update the data
                ax.set_title(f'CMEMS SPM on {daily_times[frame].strftime("%Y-%m-%d")}')
                return im,

            # Create the animation
            ani = FuncAnimation(fig, update, frames=range(self.time_steps), interval=100, blit=True)
            ani_path = os.path.join(self.output_dir_plots, 'CMEMS_SPM_Animation_2007.gif')
            ani.save(ani_path, writer='pillow', dpi=300)
            print(f"Animation saved to: {ani_path}")
            plt.close()
        except Exception as e:
            raise RuntimeError(f"Failed to create SPM animation: {str(e)}")