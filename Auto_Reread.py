# -*- coding: utf-8 -*-
"""
Module for defining model acrhitecture and warmup, for SPM Prediction Pipeline

Author: Beau van Koert for Deltares, built upon previous work by Senyang Li (unpublished). Edited by L.Beyaard. 
Date: June 2025
"""
#TODO: move utility from init to helper functions

# Import libraries
import os
import pandas as pd
import re
import csv
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

#%% SPMAnalyzer
class SPMAnalyzer:
    def __init__(self, name, config, data_vars):
        """
        Initializes the SPMAnalyzer class and performs initial data loading
        and processing.

        Args:
            name (str): Name of the model version
            config (dict): Configuration dictionary containing model and results paths.
            data_vars (dict): Dictionary containing data variables.
        """
        # Define model name
        self.name = name
        
        # Define config
        self.config = config
        
        # Define paths
        self.result_directory_main = self.config["results_file_path"]
        self.result_directory = os.path.join(self.result_directory_main, f'{self.name}')
        os.makedirs(self.result_directory, exist_ok=True)
        self.river_file_path = self.config["river_file_path"]
        self.path_in_DCSM_rast = self.config["input_file_path"]
        self.spm_validation_file_path = self.config["spm_validation_file_path"]
        
        # Load model results using path from config
        if 'results_npz_path' not in self.config:
            raise KeyError("self.config must contain 'results_npz_path' for loading .npz file")
        self.my_file = np.load(self.config['results_npz_path'])
        self.pred = self.my_file['pred']
        self.y_test = self.my_file['y_test']
        self.nan_indices = self.my_file['ni']
        self.losses = self.my_file['losses']
        self.val_losses = self.my_file['val_losses']
        self.epoches = self.my_file['epoches']
        
        # Load data variables
        self.feature_means = data_vars["feature_means"]
        self.feature_stds = data_vars["feature_stds"]
        self.X_test = data_vars["X_test"]
        self.dates_test = data_vars["dates_test"]
        
        # Sort dates_test chronologically and reorder corresponding arrays
        print("Sorting dates_test chronologically...")
        # Ensure dates_test is a list of datetime objects
        dates_test = [pd.to_datetime(date) for date in self.dates_test]
        # Get the sorted indices
        sorted_indices = np.argsort(dates_test)
        # Sort dates_test
        self.dates_test = [self.dates_test[i] for i in sorted_indices]
        # Reorder pred, y_test, and X_test to match the sorted dates
        self.pred = self.pred[sorted_indices]
        self.y_test = self.y_test[sorted_indices]
        self.X_test = self.X_test[sorted_indices]
        print("Dates sorted. First few dates:", self.dates_test[:5])

        # Analyze pred
        self.n_samples = self.pred.shape[0]
        self.days = self.pred.shape[1]
        self.vertical = self.pred.shape[2]
        self.horizontal = self.pred.shape[3]
        self.grid_shape = (self.vertical, self.horizontal)
        print(f"Grid shape from pred: {self.grid_shape}")
        
        # Load and process the DCSM dataset, storing it as self.DCSM_ds
        dcsm_data = self._load_and_process_dcsm_rast(required_vars=['y', 'x'])
        self.DCSM_ds = dcsm_data['dataset']
        self.y_coords = dcsm_data['y_coords']
        self.x_coords = dcsm_data['x_coords']
        self.min_lat, self.max_lat = self.y_coords.min(), self.y_coords.max()
        self.min_lon, self.max_lon = self.x_coords.min(), self.x_coords.max()
        print(f"Grid bounds: longitude ({self.min_lon}, {self.max_lon}), latitude ({self.min_lat}, {self.max_lat})")
        
        # Load raw CMEMS dataset if add_cmems_data is True
        self.raw_cmems_ds = None
        if self.config.get('add_cmems_data', False):
            if 'raw_cmems_path' not in self.config:
                raise KeyError("self.config must contain 'raw_cmems_path' when add_cmems_data is True.")
            raw_cmems_path = self.config['raw_cmems_path']
            if not os.path.exists(raw_cmems_path):
                raise FileNotFoundError(f"Raw CMEMS dataset not found at {raw_cmems_path}.")
            self.raw_cmems_ds = xr.open_dataset(raw_cmems_path)
            print(f"Loaded raw CMEMS dataset from {raw_cmems_path}")
        
        # Load plot_date_number
        self.plot_day_number = self.config.get("plot_day_number")
        if self.plot_day_number < 0 or self.plot_day_number >= self.n_samples:
            raise ValueError(f"plot_day_number {self.plot_day_number} is out of range. Must be between 0 and {self.n_samples-1}.")
        # Load time variables
        self.start_time = self.config.get("start_time")
        self.start_date = datetime.strptime(self.start_time, '%Y-%m-%d')
        self.end_time = self.config.get("end_time")
        self.end_date = datetime.strptime(self.end_time, '%Y-%m-%d')
        self.plot_date = self.start_date + timedelta(days=self.plot_day_number + self.days)
        
        # Convert SPM units
        self.pred_mgL = np.expm1(self.pred)
        self.y_test_mgL = np.expm1(self.y_test)

        if len(self.dates_test) != self.n_samples:
            raise ValueError(f"Length of dates_test ({len(self.dates_test)}) does not match n_samples ({self.n_samples}).")

        self.masked_p, self.masked_t, self.masked_p2D, self.masked_t2D = self.extract_nonnan_data()

        self.n_samples, self.days, self.n_features, self.vertical, self.horizontal = self.X_test.shape
        print(f"Grid shape from X_test: {(self.vertical, self.horizontal)}")
    
        # Initialize station data
        self.station_data = self.import_station_data()
        self.station_data = self.clean_station_data(self.station_data)
        self.locations, self.station_grid_mapping = self.map_stations_to_grid(self.station_data)

        # Existing plotting methods
        self.plot_loss_progression()
        self.mae_overall, self.mse_overall, self.rmse_overall, self.r2_overall, self.mad_overall, self.mae, self.mse, self.rmse, self.r2, self.mad = self.plot_metrics()
        self.plot_comparison_map()
        self.plot_probability_density_curve()
        self.plot_summary_map()
        # SPM validation map
        self.plot_validation_stations()
        # Longterm time-series plots/maps
        self.plot_spatial_stat_timeseries()
        self.plot_location_timeseries()
    
    # =========================================================================
    # Internal helper functions
    # =========================================================================
    def __del__(self):
        """
        Destructor to ensure datasets are closed when the SPMAnalyzer instance is destroyed.
        """
        if hasattr(self, 'DCSM_ds') and self.DCSM_ds is not None:
            try:
                self.DCSM_ds.close()
                print("Closed DCSM dataset.")
            except Exception as e:
                print(f"Error closing DCSM dataset: {str(e)}")
        if hasattr(self, 'raw_cmems_ds') and self.raw_cmems_ds is not None:
            try:
                self.raw_cmems_ds.close()
                print("Closed raw CMEMS dataset.")
            except Exception as e:
                print(f"Error closing raw CMEMS dataset: {str(e)}")
    
    def _load_and_process_dcsm_rast(self, required_vars=None, compute_spm_mean=False):
        """
        Loads and processes the DCSM_rast dataset, including clipping and extracting coordinates.

        Args:
            required_vars (list, optional): List of required variables to check in the dataset.
            compute_spm_mean (bool): If True, computes the mean SPM field over time.

        Returns:
            dict: Dictionary containing processed data (e.g., 'dataset', 'y_coords', 'x_coords', 'spm_mean').
        """
        if not self.path_in_DCSM_rast:
            raise ValueError("path_in_DCSM_rast must be set in self.config['input_file_path']")

        # Load the dataset
        ds = xr.open_dataset(self.path_in_DCSM_rast)
        
        # Check for required variables
        if required_vars:
            missing_vars = [var for var in required_vars if var not in ds]
            if missing_vars:
                ds.close()
                raise KeyError(f"Dataset must contain the following variables: {missing_vars}")

        # Clip the dataset using the trim_file_path from config
        trim_file_path = self.config.get('trim_file_path')
        if trim_file_path:
            print(f"Clipping dataset with shapefile: {trim_file_path}")
            shape = gpd.read_file(trim_file_path, crs="epsg:4326")
            ds.rio.write_crs("epsg:4326", inplace=True)
            ds = ds.rio.clip(
                shape.geometry.apply(mapping), shape.crs, drop=True
            )
        else:
            print("Warning: No trim_file_path provided in self.config. Using unclipped dataset bounds.")

        # Extract coordinates
        y_coords = ds['y'].values  # 1D array
        x_coords = ds['x'].values  # 1D array

        # Prepare the result dictionary
        result = {
            'dataset': ds,
            'y_coords': y_coords,
            'x_coords': x_coords
        }

        # Compute mean SPM field if requested
        if compute_spm_mean:
            if 'mesh2d_SPM' not in ds:
                ds.close()
                raise KeyError("Dataset must contain 'mesh2d_SPM' variable to compute mean SPM field.")
            result['spm_mean'] = ds['mesh2d_SPM'].mean(dim='time').values  # Shape: (vertical, horizontal)

        # Note: Dataset is not closed here; it will be stored as self.DCSM_ds and closed in __del__
        return result
    
    
    # =========================================================================
    # General model plots
    # =========================================================================
    def plot_loss_progression(self):
        """
        Plots the loss progression during training and validation over the epochs.
        """
        x_values = np.arange(0, self.epoches)
        plt.figure(figsize=(12, 6), constrained_layout=True)
        plt.plot(x_values, self.losses, label='Training Loss')
        plt.plot(x_values, self.val_losses, label='Validation Loss')
        plt.title(f"Loss Progression over {self.epoches} Epochs")
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.legend()
        plt.xlim(0, self.epoches)

        # Set x-axis ticks based on number of epochs
        if self.epoches <= 50:
            step = 1
        elif 51 <= self.epoches < 200:
            step = 5
        else:
            step = 10
        plt.xticks(np.arange(0, self.epoches + 1, step=step))
        plt.grid(which='both', color='lightgrey', linestyle='-', linewidth=0.5)
        plt.savefig(os.path.join(self.result_directory, 'loss_progression.png'), dpi=300)
        plt.close()
    
    
    def plot_probability_density_curve(self):
        """
        Plots the probability density curves for predictions and test data in mg/L.
        Uses a user defined percentile for plotted SPM range.
        """
        kde_pred = gaussian_kde(self.masked_p)
        kde_test = gaussian_kde(self.masked_t)
        
        percentile = 99
        percentile = int(percentile)
        
        pred_percentile = np.percentile(self.masked_p, percentile)
        y_test_percentile = np.percentile(self.masked_t, percentile)
        vmax = max(pred_percentile, y_test_percentile)
        vmin = 0

        x = np.linspace(vmin, vmax, 1000)
        plt.figure(figsize=(8, 6))
        plt.plot(x, kde_pred(x), color='blue', label='Prediction (ConvLSTM)')
        plt.plot(x, kde_test(x), color='red', label='Ground Truth (Delft3D-FM)')
        plt.legend()
        plt.xlabel('SPM (mg/L)')
        plt.ylabel('Probability Density')
        plt.title(f'Probability Density Curve - Predicted vs Ground Truth SPM ({percentile}th perc)')
        plt.grid(True)
        plt.savefig(os.path.join(self.result_directory, f'SPM_density_curve_{percentile}th_perc.png'), dpi=300)
        plt.close()
    
    
    def extract_nonnan_data(self):
        """
        Extracts non-NaN data from the predictions and test data in mg/L units.

        Returns:
            tuple:  (masked_p, masked_t, masked_p2D, masked_t2D) containing the
                    masked predictions and test data.
        """
        masked_p = np.array([])  # Flattened non-NaN predictions
        masked_t = np.array([])  # Flattened non-NaN ground truth
        masked_p2D = []  # Per-sample, per-day non-NaN predictions
        masked_t2D = []  # Per-sample, per-day non-NaN ground truth

        for i in range(self.n_samples):
            masked_p2D.append([])
            masked_t2D.append([])
            for j in range(self.days):
                p = self.pred_mgL[i, j].flatten()[~self.nan_indices]
                t = self.y_test_mgL[i, j].flatten()[~self.nan_indices]
                masked_p = np.concatenate((masked_p, p))
                masked_t = np.concatenate((masked_t, t))
                masked_p2D[i].append(p)
                masked_t2D[i].append(t)

        masked_p2D = np.array(masked_p2D)  # Shape: (n_samples, days, n_non_nan_pixels)
        masked_t2D = np.array(masked_t2D)
        print(f"Shape of masked_p2D: {masked_p2D.shape}")
        return masked_p, masked_t, masked_p2D, masked_t2D
    
    # =========================================================================
    # x-day window SPM prediction analysis
    # =========================================================================
    def plot_comparison_map(self):
        """
        Plots three maps in a 3x1 layout for a specified sample:
        - Top: Predicted SPM (mg/L)
        - Middle: Ground Truth SPM (mg/L)
        - Bottom: Difference (Predicted - Ground Truth) using RdBu colormap
        """
        # Reshape nan_indices to 2D for masking
        nan_indices_2d = self.nan_indices.reshape(self.vertical, self.horizontal)

        # Apply NaN mask to predictions and true values (in mg/L)
        pred_masked = np.where(nan_indices_2d, np.nan, self.pred_mgL[:, :, :self.vertical, :self.horizontal])
        y_test_masked = np.where(nan_indices_2d, np.nan, self.y_test_mgL[:, :, :self.vertical, :self.horizontal])

        # Calculate the difference (Predicted - Ground Truth)
        difference = pred_masked - y_test_masked

        # # Determine color scale for SPM plots (using 99th percentile of non-NaN values)
        pred_99th_percentile = np.percentile(pred_masked[~np.isnan(pred_masked)], 99)
        y_test_99th_percentile = np.percentile(y_test_masked[~np.isnan(y_test_masked)], 99)
        vmax_spm = max(pred_99th_percentile, y_test_99th_percentile)
        vmin_spm = 0

        # Determine color scale for the difference plot (symmetric around 0)
        diff_abs_max = np.percentile(np.abs(difference[~np.isnan(difference)]), 99)
        vmin_diff = -diff_abs_max
        vmax_diff = diff_abs_max

        # Loop over prediction days to create a plot for each
        for i in range(self.days):
            # Calculate the date for the current prediction day
            current_date = self.plot_date + timedelta(days=i)
            date_str = current_date.strftime('%Y-%m-%d')

            # Create a 3x1 subplot layout with shared x-axis
            fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

            # Plot 1: Predicted SPM (Top)
            im0 = axs[0].imshow(pred_masked[self.plot_day_number, i], cmap='viridis', vmin=vmin_spm, vmax=vmax_spm)
            axs[0].set_title('Prediction (ConvLSTM)')
            axs[0].set_ylabel('Pixel Y')

            # Plot 2: Ground Truth SPM (Middle)
            im1 = axs[1].imshow(y_test_masked[self.plot_day_number, i], cmap='viridis', vmin=vmin_spm, vmax=vmax_spm)
            axs[1].set_title('Ground Truth (Delft3D-FM)')
            axs[1].set_ylabel('Pixel Y')

            # Plot 3: Difference (Predicted - Ground Truth) (Bottom)
            im2 = axs[2].imshow(difference[self.plot_day_number, i], cmap='coolwarm', vmin=vmin_diff, vmax=vmax_diff)
            axs[2].set_title('Difference (ConvLSTM - Delft3D-FM)')
            axs[2].set_xlabel('Pixel X')
            axs[2].set_ylabel('Pixel Y')

            # Invert y-axis for all subplots
            for ax in axs:
                ax.invert_yaxis()

            # Add vertical colorbars on the right
            cbar0 = fig.colorbar(im0, ax=axs[0], orientation='vertical', fraction=0.06, pad=0.04)
            cbar0.set_label('SPM (mg/L)', fontsize=12)

            cbar1 = fig.colorbar(im1, ax=axs[1], orientation='vertical', fraction=0.06, pad=0.04)
            cbar1.set_label('SPM (mg/L)', fontsize=12)

            cbar2 = fig.colorbar(im2, ax=axs[2], orientation='vertical', fraction=0.06, pad=0.04)
            cbar2.set_label('Difference (mg/L)', fontsize=12)

            # Add a supertitle with the date and prediction day
            plt.suptitle(f'{date_str} (Day {i+1})', fontsize=16, y=0.95)

            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])

            # Save the figure
            plt.savefig(os.path.join(self.result_directory, f'map_comparison_day_index_{self.plot_day_number}_pred_day_{i+1}.png'), dpi=300)
            plt.close()


    def plot_metrics(self):
        """
        Computes and plots evaluation metrics (MAE, MSE, RMSE, R2, MAD) for
        each day in mg/L units.

        Returns:
            tuple: (mae_overall, mse_overall, rmse_overall, r2_overall, mad_overall, mae, mse, rmse, r2, mad)
        """
        mae = []
        mse = []
        rmse = []
        r2 = []
        mad = []

        # Compute metrics for each day
        for i in range(self.days):
            v = self.custom_mae(self.masked_t2D[:, i].flatten(), self.masked_p2D[:, i].flatten())
            w = self.custom_mse(self.masked_t2D[:, i].flatten(), self.masked_p2D[:, i].flatten())
            x = self.custom_rmse(self.masked_t2D[:, i].flatten(), self.masked_p2D[:, i].flatten())
            y = r2_score(self.masked_t2D[:, i].flatten(), self.masked_p2D[:, i].flatten())
            z = self.custom_mad(self.masked_t2D[:, i].flatten(), self.masked_p2D[:, i].flatten())
            mae.append(v)
            mse.append(w)
            rmse.append(x)
            r2.append(y)
            mad.append(z)

        # Plot MSE over days
        dates = [f'Day {i+1}' for i in range(len(mse))]
        plt.figure(figsize=(10, 6))
        plt.plot(dates, mse, marker='o', linestyle='-', color='purple')
        plt.title('Mean Squared Error (MSE) of Prediction for Each Day')
        plt.xlabel('Day in the Future')
        plt.ylabel('MSE (mg/L²)')
        plt.grid(True)
        plt.savefig(os.path.join(self.result_directory, 'evaluation_metrics.png'), dpi=300)
        plt.close()

        # Compute overall metrics
        mae_overall = self.custom_mae(self.masked_t2D.flatten(), self.masked_p2D.flatten())
        mse_overall = self.custom_mse(self.masked_t2D.flatten(), self.masked_p2D.flatten())
        rmse_overall = self.custom_rmse(self.masked_t2D.flatten(), self.masked_p2D.flatten())
        r2_overall = r2_score(self.masked_t2D.flatten(), self.masked_p2D.flatten())
        mad_overall = self.custom_mad(self.masked_t2D.flatten(), self.masked_p2D.flatten())

        # Save metrics to CSV
        with open(os.path.join(self.result_directory, 'evaluation_metrics.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Day", "MAE", "MSE", "RMSE", "R2", "MAD"])
            for i in range(len(dates)):
                writer.writerow([dates[i], mae[i], mse[i], rmse[i], r2[i], mad[i]])
            writer.writerow(["Overall", mae_overall, mse_overall, rmse_overall, r2_overall, mad_overall])

        return mae_overall, mse_overall, rmse_overall, r2_overall, mad_overall, mae, mse, rmse, r2, mad


    def plot_summary_map(self):
        """
        Plots spatial maps of evaluation metrics for each day and 
        overall in mg/L units.
        """
        metrics = ['MAE', 'MSE', 'RMSE', 'R2', 'MAD']
        summary = np.zeros((len(metrics), self.days + 1, self.vertical, self.horizontal))
        nan_indices_2d = self.nan_indices.reshape(self.vertical, self.horizontal)

        # Compute metrics for each pixel
        for i in range(self.vertical):
            for j in range(self.horizontal):
                if nan_indices_2d[i, j]:
                    summary[:, :, i, j] = np.nan
                else:
                    for k in range(self.days + 1):
                        if k == self.days:  # Overall metrics
                            p = self.pred_mgL[:, :, i, j].flatten()
                            t = self.y_test_mgL[:, :, i, j].flatten()
                        else:  # Daily metrics
                            p = self.pred_mgL[:, k, i, j].flatten()
                            t = self.y_test_mgL[:, k, i, j].flatten()
                        summary[0, k, i, j] = self.custom_mae(t, p)
                        summary[1, k, i, j] = self.custom_mse(t, p)
                        summary[2, k, i, j] = self.custom_rmse(t, p)
                        summary[3, k, i, j] = r2_score(t, p)
                        summary[4, k, i, j] = self.custom_mad(t, p)

        # Plot maps for each metric
        for i, metric in enumerate(metrics):
            value = summary[i, :-1, :, :]  # Daily values
            percentile_95 = np.percentile(value[~np.isnan(value)], 95)
            percentile_05 = np.percentile(value[~np.isnan(value)], 5)
            vmax = percentile_95 if metric != 'R2' else 1.0
            vmin = percentile_05 if metric != 'R2' else 0.0
            if metric == 'R2':
                value = np.clip(value, 0.0, 1)

            value_overall = summary[i, -1, :, :]
            overall_vmax = np.percentile(value_overall[~np.isnan(value_overall)], 95) if metric != 'R2' else 1.0
            overall_vmin = np.percentile(value_overall[~np.isnan(value_overall)], 5) if metric != 'R2' else 0.0

            cmap = plt.cm.plasma if metric != 'R2' else plt.cm.coolwarm

            for j in range(self.days + 1):
                plt.figure(figsize=(8, 6))
                if j == self.days:  # Overall map
                    im = plt.imshow(summary[i, j], cmap=cmap, vmin=overall_vmin, vmax=overall_vmax)
                    plt.title(f'{metric} Overall (5-day prediction window)')
                    filename = f'summary_{metric}_overall.png'
                else:  # Daily map
                    im = plt.imshow(summary[i, j], cmap=cmap, vmin=vmin, vmax=vmax)
                    plt.title(f'{metric} Day {j+1}')
                    filename = f'summary_{metric}_Day_{j+1}.png'
                plt.gca().invert_yaxis()
                cbar = plt.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label(metric + (' (mg/L)' if metric in ['MAE', 'MSE', 'RMSE', 'MAD'] else ''), fontsize=12)
                plt.xlabel('Pixel X')
                plt.ylabel('Pixel Y')
                plt.savefig(os.path.join(self.result_directory, filename), dpi=300)
                plt.close()

        # Save R2 spectral statistics
        r2 = summary[3, :, :, :]
        with open(os.path.join(self.result_directory, 'R2_spectral.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Day", "Min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "Max"])
            for i in range(self.days + 1):
                value = r2[i, :, :]
                value = value[~np.isnan(value)]
                day_label = "Overall" if i == self.days else f"Day {i+1}"
                writer.writerow([
                    day_label,
                    np.nanmin(value),
                    np.percentile(value, 1),
                    np.percentile(value, 5),
                    np.percentile(value, 25),
                    np.percentile(value, 50),
                    np.percentile(value, 75),
                    np.percentile(value, 95),
                    np.percentile(value, 99),
                    np.nanmax(value)
                ])
    
    # =========================================================================
    # SPM validation
    # =========================================================================
    def import_station_data(self):
        """
        Import buoy and measuring station data from the SPM validation CSV.
    
        Returns:
            pandas.DataFrame: Raw DataFrame containing station data.
        """
        df_stations = pd.read_csv(self.spm_validation_file_path, sep=';')
        print(f"Imported SPM validation data with {len(df_stations)} records")
        return df_stations
    
    
    def clean_station_data(self, df_stations):
        """
        Clean the station data by parsing coordinates, ensuring valid entries, filtering for the year 2007,
        ensuring stations have at least 5 measurements in 2007, and filtering stations within the spatial
        bounds of the DCSM dataset grid.
    
        Args:
            df_stations (pandas.DataFrame): Raw DataFrame from import_station_data.
    
        Returns:
            pandas.DataFrame: Cleaned DataFrame with parsed coordinates, valid entries, 2007 data only,
            stations with 5+ measurements, and stations within grid bounds.
        """
        # Create a copy to avoid modifying the input
        df_stations_clean = df_stations.copy()
    
        # Parse geom column: POINT (lon lat)
        def parse_geom(geom):
            if isinstance(geom, str):
                match = re.match(r'POINT \(([-\d.\s]+)\)', geom)
                if match:
                    lon, lat = map(float, match.group(1).split())
                    return lon, lat
            return None, None
    
        # Apply parsing to geom column
        df_stations_clean[['longitude', 'latitude']] = df_stations_clean['geom'].apply(parse_geom).apply(pd.Series)
    
        # Parse datetime using pandas
        df_stations_clean['datetime'] = pd.to_datetime(df_stations_clean['datetime'], errors='coerce')
    
        # Filter out rows with invalid entries (missing station, coordinates, datetime, or value)
        df_stations_clean = df_stations_clean.dropna(subset=['station', 'longitude', 'latitude', 'datetime', 'value'])
        df_stations_clean = df_stations_clean[df_stations_clean['value'].apply(lambda x: isinstance(x, (int, float)))]
    
        print(f"After initial cleaning, {len(df_stations_clean)} valid records remain")
    
        # Filter for records in the year 2007
        df_stations_clean = df_stations_clean[df_stations_clean['datetime'].dt.year == 2007]
        print(f"After filtering for year 2007, {len(df_stations_clean)} records remain")
    
        # Filter stations with at least 5 measurements in 2007
        station_counts = df_stations_clean['station'].value_counts()
        stations_with_enough_measurements = station_counts[station_counts >= 5].index
        df_stations_clean = df_stations_clean[df_stations_clean['station'].isin(stations_with_enough_measurements)]
        print(f"After filtering for stations with 5+ measurements in 2007, {len(df_stations_clean)} records remain")
    
        # Filter stations within the grid bounds (computed in __init__)
        df_stations_clean = df_stations_clean[
            (df_stations_clean['longitude'].between(self.min_lon, self.max_lon)) &
            (df_stations_clean['latitude'].between(self.min_lat, self.max_lat))
        ]
    
        print(f"After filtering stations within grid bounds, {len(df_stations_clean)} records remain")
        return df_stations_clean
    
    
    def map_stations_to_grid(self, df_stations):
        """
        Map station coordinates to the nearest grid cell in the DCSM dataset.

        Args:
            df_stations (pandas.DataFrame): Cleaned DataFrame with station data, including longitude and latitude.

        Returns:
            list of tuples: List of (y, x) pixel coordinates for station locations.
            dict: Dictionary mapping station names to their coordinates and grid indices.
        """
        # Get unique stations
        station_coords = df_stations[['station', 'longitude', 'latitude']].drop_duplicates()
        print(f"Mapping {len(station_coords)} unique stations to grid")

        # Use the pre-loaded DCSM dataset
        if not hasattr(self, 'DCSM_ds'):
            raise AttributeError("DCSM dataset not loaded. Ensure _load_and_process_dcsm_rast is called in __init__.")
        y_coords = self.y_coords
        x_coords = self.x_coords

        # Verify the lengths match the expected grid shape
        vertical, horizontal = self.grid_shape
        if len(y_coords) != vertical or len(x_coords) != horizontal:
            raise ValueError(f"Coordinate arrays do not match expected grid shape {self.grid_shape}: y has length {len(y_coords)}, x has length {len(x_coords)}")

        # Create 2D meshgrid of longitude and latitude
        lon_grid, lat_grid = np.meshgrid(x_coords, y_coords)  # Shapes: (vertical, horizontal)

        # Verify the resulting grid shape
        if lat_grid.shape != (vertical, horizontal) or lon_grid.shape != (vertical, horizontal):
            raise ValueError(f"Generated grid shape {lat_grid.shape} does not match expected shape {self.grid_shape}")

        # Map stations to grid
        locations = []
        station_grid_mapping = {}
        for _, row in station_coords.iterrows():
            station_name = row['station']
            lon, lat = row['longitude'], row['latitude']
            # Compute distance to grid points
            distance = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
            # Find nearest grid cell
            y, x = np.unravel_index(np.argmin(distance), distance.shape)
            locations.append((y, x))
            station_grid_mapping[station_name] = {
                'lon': lon,
                'lat': lat,
                'grid_y': y,
                'grid_x': x
            }
            print(f"Station {station_name}: (lon={lon}, lat={lat}) mapped to grid location (y={y}, x={x})")

        # Remove duplicates from locations while preserving order
        locations = list(dict.fromkeys(map(tuple, locations)))
        print(f"Found {len(locations)} unique grid locations")
        return locations, station_grid_mapping


    def plot_validation_stations(self):
        """
        Plots a map showing the locations of SPM validation stations within the study area.
    
        Uses the `locations` list and `station_grid_mapping` dictionary to plot station positions
        on a background map of the mean SPM field from the DCSM dataset.
    
        Returns
        -------
        None.
        """
        # Use the pre-loaded DCSM dataset
        if not hasattr(self, 'DCSM_ds'):
            raise AttributeError("DCSM dataset not loaded. Ensure _load_and_process_dcsm_rast is called in __init__.")
        ds = self.DCSM_ds
        if 'mesh2d_SPM' not in ds:
            raise KeyError("Dataset must contain 'mesh2d_SPM' variable to compute mean SPM field.")
        
        y_coords = self.y_coords
        x_coords = self.x_coords
        spm_mean = ds['mesh2d_SPM'].mean(dim='time').values  # Shape: (vertical, horizontal)

        # Create a 2D meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(x_coords, y_coords)
    
        # Create the plot
        plt.figure(figsize=(10, 6))
        # Plot the background SPM field
        plt.pcolormesh(lon_grid, lat_grid, spm_mean, cmap='viridis', shading='auto')
        plt.colorbar(label='Mean SPM (mg/L)')
    
        # Plot each station's location using station_grid_mapping
        for station_name, info in self.station_grid_mapping.items():
            lon = info['lon']
            lat = info['lat']
            # Plot the station as a red dot
            plt.scatter(lon, lat, marker='x', color='red', s=50, label='Stations' if station_name == list(self.station_grid_mapping.keys())[0] else "")
            # Add the station name as a label, slightly offset for readability
            text = plt.text(lon + 0.02, lat + 0.02, station_name, fontsize=8, color='black', ha='left', va='bottom')
            text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))    # bbbox around text
    
        # Add labels and title
        plt.xlabel('Longitude (deg)')
        plt.ylabel('Latitude (deg)')
        plt.title('SPM in-situ measurement stations with 5+ measurements')
        plt.grid(True)
    
        # Add a legend (only one entry for stations)
        plt.legend()
    
        # Save the plot
        plot_path = os.path.join(self.result_directory, 'validation_stations_map.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved validation stations map to {plot_path}")
        

    # =========================================================================
    # Full temporal range analysis
    # =========================================================================
    def plot_spatial_stat_timeseries(self):
        """
        Plots time series of spatially averaged, minimum, and maximum SPM (predicted vs. ground truth) over the entire test dataset.
    
        Focus: Full Temporal Range Analysis
    
        - X-axis: Time (months of the year 2007).
        - Y-axis: SPM in mg/L (average, minimum, or maximum).
        - Blue line: Predicted SPM (ConvLSTM).
        - Red line: Ground Truth (Delft3D-FM).
        Creates three plots: one for spatial average, one for spatial minimum, and one for spatial maximum.
        """
        # Reshape nan_indices to 2D for masking
        nan_indices_2d = self.nan_indices.reshape(self.vertical, self.horizontal)
    
        # Apply NaN mask to predictions and test data
        pred_masked = np.where(nan_indices_2d, np.nan, self.pred_mgL)
        y_test_masked = np.where(nan_indices_2d, np.nan, self.y_test_mgL)
    
        # Compute spatial statistics for each test sample and prediction day
        # Shape: (n_samples, days) -> compute over spatial dimensions (vertical, horizontal)
        # Average
        pred_spatial_avg = np.nanmean(pred_masked, axis=(2, 3))  # Shape: (60, 5)
        y_test_spatial_avg = np.nanmean(y_test_masked, axis=(2, 3))  # Shape: (60, 5)
        # Minimum
        pred_spatial_min = np.nanmin(pred_masked, axis=(2, 3))  # Shape: (60, 5)
        y_test_spatial_min = np.nanmin(y_test_masked, axis=(2, 3))  # Shape: (60, 5)
        # Maximum
        pred_spatial_max = np.nanmax(pred_masked, axis=(2, 3))  # Shape: (60, 5)
        y_test_spatial_max = np.nanmax(y_test_masked, axis=(2, 3))  # Shape: (60, 5)
        
        # Select the first prediction day (index 0) for each test sample
        pred_spatial_avg = pred_spatial_avg[:, 0]  # Shape: (60,)
        y_test_spatial_avg = y_test_spatial_avg[:, 0]  # Shape: (60,)
        pred_spatial_min = pred_spatial_min[:, 0]  # Shape: (60,)
        y_test_spatial_min = y_test_spatial_min[:, 0]  # Shape: (60,)
        pred_spatial_max = pred_spatial_max[:, 0]  # Shape: (60,)
        y_test_spatial_max = y_test_spatial_max[:, 0]  # Shape: (60,)
    
        # Debug: Print the number of NaN values in the averaged data
        print(f"Length of pred_spatial_avg: {len(pred_spatial_avg)} dates")
        print(f"Length of y_test_spatial_avg: {len(y_test_spatial_avg)} dates")
        print(f"Number of NaN in pred_spatial_avg: {np.isnan(pred_spatial_avg).sum()}")
        print(f"Number of NaN in y_test_spatial_avg: {np.isnan(y_test_spatial_avg).sum()}")
        print("Values of y_test_spatial_avg:")
        print(y_test_spatial_avg)
    
        # Convert dates_test to datetime objects for plotting
        dates = [pd.to_datetime(date) for date in self.dates_test]
        print("The dates of the test-dataset are:")
        print(self.dates_test)
    
        # Verify the date range
        print(f"Date range: {min(dates)} to {max(dates)}")
    
        # Function to create a plot for a given statistic
        def create_plot(data_pred, data_test, stat_name, filename):
            plt.figure(figsize=(10, 6))
            plt.plot(dates, data_pred, color='blue', label='Prediction (ConvLSTM)', marker='o', markersize=3)
            plt.plot(dates, data_test, color='red', label='Ground Truth (Delft3D-FM)', marker='d', markersize=3)
    
            # Customize the plot
            plt.xlabel('Time')
            plt.ylabel('SPM (mg/L)')
            plt.title(f'Spatially {stat_name} SPM, Prediction vs. Ground Truth (2007)')
            plt.legend()
            plt.grid(True)
    
            # Format x-axis to show months within 2007
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Show every month
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Format as month abbreviation (e.g., Jan, Feb)
    
            # Set x-axis limits to ensure only 2007 is shown
            start_date = pd.to_datetime('2007-01-01')
            end_date = pd.to_datetime('2007-12-31')
            ax.set_xlim(start_date, end_date)
    
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
    
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_directory, filename), dpi=300)
            plt.close()
            print(f"Saved spatially {stat_name.lower()} SPM time series to {self.result_directory}/{filename}")
    
        # Create plots for average, minimum, and maximum
        create_plot(pred_spatial_avg, y_test_spatial_avg, 'Averaged', 'spatial_average_timeseries_pred_vs_test.png')
        create_plot(pred_spatial_min, y_test_spatial_min, 'Minimum', 'spatial_minimum_timeseries_pred_vs_test.png')
        create_plot(pred_spatial_max, y_test_spatial_max, 'Maximum', 'spatial_maximum_timeseries_pred_vs_test.png')
        

    def plot_location_timeseries(self):
        """
        Plots time series of predicted SPM vs. test SPM at specific station locations over the entire test dataset,
        with in-situ SPM measurements overlaid as points. Optionally includes raw CMEMS SPM data if specified in config.
        """
        # Validate locations
        for y, x in self.locations:
            if not (0 <= y < self.vertical and 0 <= x < self.horizontal):
                raise ValueError(f"Location ({y}, {x}) is out of bounds for dimensions ({self.vertical}, {self.horizontal}).")
            if self.nan_indices.reshape(self.vertical, self.horizontal)[y, x]:
                print(f"Warning: Location ({y}, {x}) is masked as NaN (land area).")
    
        # Filter station data for the study period (2007)
        df = self.station_data[self.station_data['datetime'].dt.year == 2007]
        print(f"Filtered {len(df)} in-situ measurements for 2007")
    
        # Number of locations to plot
        n_locations = len(self.locations)
        if n_locations == 0:
            print("No valid station locations to plot.")
            return
    
        # Create subplots
        fig, axs = plt.subplots(n_locations, 1, figsize=(10, 4 * n_locations), sharex=True)
        dates = [pd.to_datetime(date) for date in self.dates_test]
    
        # Handle single location case
        if n_locations == 1:
            axs = [axs]
    
        # Check if CMEMS data should be included
        add_cmems = self.config.get('add_cmems_data', False)
        cmems_spm = None
        if add_cmems:
            if not hasattr(self, 'raw_cmems_ds'):
                raise AttributeError("Raw CMEMS dataset not loaded. Ensure it is loaded in __init__.")
            ds = self.raw_cmems_ds
            if 'SPM' not in ds:
                raise KeyError("Raw CMEMS dataset must contain 'SPM' variable to plot CMEMS SPM data.")

            # Select CMEMS SPM data for the dates in dates_test
            cmems_dates = pd.to_datetime(ds['time'].values)
            dates_test_np = np.array([np.datetime64(date) for date in dates])
            # Check which dates are available in CMEMS data
            date_mask = np.isin(dates_test_np, cmems_dates)
            if not date_mask.any():
                raise ValueError("None of the dates in dates_test are present in the raw CMEMS dataset.")
            if not date_mask.all():
                print(f"Warning: Only {date_mask.sum()} out of {len(dates_test_np)} dates in dates_test are present in the raw CMEMS dataset.")

            # Select the matching dates
            cmems_spm = ds['SPM'].sel(time=dates_test_np[date_mask])

            # Interpolate CMEMS SPM data to the DCSM grid
            cmems_spm = cmems_spm.interp(
                latitude=self.y_coords,
                longitude=self.x_coords,
                method='linear'
            )

        for idx, (y, x) in enumerate(self.locations):
            # Extract SPM values at this location
            pred_location = self.pred_mgL[:, 0, y, x]  # Shape: (n_samples,)
            y_test_location = self.y_test_mgL[:, 0, y, x]  # Shape: (n_samples,)
    
            # Plot predicted and ground truth
            axs[idx].plot(dates, pred_location, color='blue', label='Prediction (ConvLSTM)')
            axs[idx].plot(dates, y_test_location, color='red', label='Ground Truth (Delft3D-FM)')
    
            # Plot raw CMEMS SPM data if enabled
            if add_cmems and cmems_spm is not None:
                # Initialize an array for CMEMS SPM values, filled with NaN
                cmems_location = np.full(len(dates), np.nan)
                # Fill in the values for the dates we have
                cmems_values = cmems_spm[:, y, x].values  # Shape: (n_matching_dates,)
                cmems_location[date_mask] = cmems_values
                # Replace NaNs with 0 for plotting (to represent clouds)
                # cmems_location = np.nan_to_num(cmems_location, nan=0.0)
                axs[idx].scatter(dates, cmems_location, color='green', marker='d', s=50, label='CMEMS SPM (Raw)')
    
            # Find stations at this grid location
            stations_at_location = [
                station for station, info in self.station_grid_mapping.items()
                if info['grid_y'] == y and info['grid_x'] == x
            ]
            if stations_at_location:
                # Get in-situ measurements
                station_data = df[df['station'].isin(stations_at_location)]
                in_situ_dates = station_data['datetime'].tolist()
                in_situ_spm = station_data['value'].astype(float).tolist()
                if in_situ_dates and in_situ_spm:
                    axs[idx].scatter(in_situ_dates, in_situ_spm, color='black', label='In-situ Measurements', marker='o', s=50)
    
            # Customize subplot
            station_names = ', '.join(stations_at_location) if stations_at_location else f'Y={y}, X={x}'
            axs[idx].set_ylabel('SPM (mg/L)')
            axs[idx].set_title(f'Station: {station_names}')
            axs[idx].legend()
            axs[idx].grid(True)
    
        # Format x-axis
        axs[-1].set_xlabel('Time')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_directory, 'station_timeseries.png'), dpi=300)
        plt.close()
        print(f"Saved station-specific SPM time series to {self.result_directory}/station_timeseries.png")
            

    # =========================================================================
    # Custom error metrics
    # =========================================================================
    def custom_mae(self, y_true, y_pred):
        """
        Computes the Mean Absolute Error (MAE) between true and predicted values.

        Args:
            y_true (numpy.ndarray): Array of true values.
            y_pred (numpy.ndarray): Array of predicted values.

        Returns:
            float: MAE value.
        """
        return np.mean(np.abs(y_true - y_pred))


    def custom_mse(self, y_true, y_pred):
        """
        Computes the Mean Squared Error (MSE) between true and predicted values.

        Args:
            y_true (numpy.ndarray): Array of true values.
            y_pred (numpy.ndarray): Array of predicted values.

        Returns:
            float: MSE value.
        """
        return np.mean(np.square(y_true - y_pred))


    def custom_rmse(self, y_true, y_pred):
        """
        Computes the Root Mean Squared Error (RMSE) between true and predicted values.

        Args:
            y_true (numpy.ndarray): Array of true values.
            y_pred (numpy.ndarray): Array of predicted values.

        Returns:
            float: RMSE value.
        """
        return np.sqrt(self.custom_mse(y_true, y_pred))


    def custom_mad(self, y_true, y_pred):
        """
        Computes the Mean Absolute Deviation (MAD) between true and predicted values.

        Args:
            y_true (numpy.ndarray): Array of true values.
            y_pred (numpy.ndarray): Array of predicted values.

        Returns:
            float: MAD value.
        """
        e = y_true - y_pred
        return np.mean(np.abs(e - np.mean(e)))
    
    