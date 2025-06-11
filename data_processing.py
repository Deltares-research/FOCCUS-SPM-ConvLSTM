# -*- coding: utf-8 -*-
"""
Data Processing Module for SPM Prediction Pipeline

Author: Beau van Koert for Deltares, built upon previous work by Senyang Li (unpublished). Edited by L.Beyaard.
Date: June 2025
"""
# TODO: make the use of the variable "CMEMS_SPM_daily_2007" dynamic from data_preprocessing module

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import geopandas
from shapely.geometry import mapping
import datetime
import os
from sklearn.model_selection import train_test_split

#%% Data Processing
class Data_Processing(object):
    """Class that serves as a data container. Provide file path, date range, time step, and trim file for initialization."""

    def __init__(
        self,
        config,
        filename,
        start_time,
        end_time,
        time_step,
        trim_file_path,
    ):
        """
        Initialize the Data_Processing class.

        Args:
            filename (str): Path to the input NetCDF file.
            start_time (str): Start date in 'YYYY-MM-DD' format.
            end_time (str): End date in 'YYYY-MM-DD' format.
            time_step (int): Number of time steps to use for sequence generation.
            trim_file_path (str): Path to the shapefile for trimming the data.
        """
        self.config=config
        
        self.name = filename
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step
        [self.data, self.date] = self.loadRawData()
        print("Loaded raw data. Specs:")
        print(self)

        # Trim the file (key step, before getting lons and lats)
        if trim_file_path:
            self.trim(trim_file_path)
            print("Data trimmed.")

        # Extract longitude and latitude
        self.lons = self.data.x.values
        self.lats = self.data.y.values

        # Split the variables into mask, SPM (target), and features
        [self.mask, self.SPM, self.feature] = self.variable_split()
        
        # Save original SPM for reference (before transformation)
        self.SPM_original = self.SPM.copy()
        
        # Perform standardization and save statistics
        self.feature_means = {}
        self.feature_stds = {}
        self.statistic_standardisation()

        # Replace NaN with 0 (after standardization)
        self.nan_indices = self.fill_nan()

        # Generate Conv2D LSTM X (features) and y (target) structure
        self.X, self.y = self.genConv2DLSTMXy(SEQ_LEN=time_step)
        
        # Adjust dates to match X, y after sequence generation
        self.seq_dates = self.date[time_step:]  # Dates for X, y start after SEQ_LEN

    def loadRawData(self):
        """ 
        Loads the input dataset, slices on provided date range, and checks date format.

        Returns:
            tuple: (xarray.Dataset, numpy.ndarray) containing the dataset and time array.
        """
        ds = xr.open_dataset(self.name)
        ds = ds.sel(time=slice(self.start_time, self.end_time))
        time = ds.time.values.astype("datetime64[D]")
        time = np.array(
            [
                datetime.datetime.strptime(str_date, "%Y-%m-%d").date()
                for str_date in time.astype(str)
            ]
        )
        return [ds, time]

    def trim(self, trim_file_path):
        """
        Trims the input data file with the shapefile of the trim file.

        Args:
            trim_file_path (str): Path to the shapefile for trimming.
        """
        shape = geopandas.read_file(trim_file_path, crs="epsg:4326")
        self.data.rio.write_crs("epsg:4326", inplace=True)
        self.data = self.data.rio.clip(
            shape.geometry.apply(mapping), shape.crs, drop=True
        )
    
    def variable_split(self):
        """
        Splits the dataset into mask, SPM (target), and features (input variables).

        Returns:
            tuple: (mask, SPM, features) where:
                - mask: Boolean mask indicating non-NaN SPM values.
                - SPM: xarray.DataArray of SPM values.
                - features: xarray.Dataset of input features.
        """
        SPM_variable = "mesh2d_SPM"     # select target variable 
        water_quality_variables = [     # define feature variables
            "mesh2d_ucx",
            "mesh2d_ucy",
            "mesh2d_ucz",
            "mesh2d_rho",
            "mesh2d_sa1",
            "mesh2d_tem1",
            "mesh2d_s1",
            "mesh2d_s0"
        ]
        
        # Add more variables based on config settings
        if self.config.get("add_cmems_data", False):
            water_quality_variables.append("CMEMS_SPM_daily_2007")
            print("Added CMEMS_SPM_daily_2007 as the 9th variable (add_cmems_data=True)")
        else:
            print("Continuing with base 8 water quality variables (add_cmems_data=False)")
        
        SPM = self.data[SPM_variable]  # Target: SPM
        features = self.data[water_quality_variables]  # Input: 8 variables
        mask = ~np.isnan(SPM.values)  # Mask based on SPM
        return mask, SPM, features

    def statistic_standardisation(self):
        """
        Standardizes the SPM and feature variables.
        - SPM: Applies log transformation (ln(SPM+1)).
        - Features: Applies Z-score normalization and saves statistics.
        """
        # Standardize SPM (log transformation)
        self.SPM = self.SPM.clip(min=0)  # Clip SPM values to be non-negative
        self.SPM = np.log1p(self.SPM)  # Apply log transformation to SPM values

        # Standardize features with Z-score normalization and save statistics
        for var in self.feature.data_vars:
            self.feature_means[var] = float(self.feature[var].mean().values)  # Convert to float for serialization
            self.feature_stds[var] = float(self.feature[var].std().values)    # Convert to float for serialization
            self.feature[var] = (self.feature[var] - self.feature_means[var]) / self.feature_stds[var]

    def fill_nan(self):
        """
        Replaces NaN values with 0 in both SPM and features after standardization.

        Returns:
            numpy.ndarray: Flattened array of NaN indices from the first SPM time step.
        """
        nan_indices = np.isnan(self.SPM.values[0, :, :].flatten())  # NaN indices from the first time step
        self.feature = self.feature.fillna(0)  # Replace NaN with 0 in features
        self.SPM = self.SPM.fillna(0)  # Replace NaN with 0 in SPM
        return nan_indices

    def genConv2DLSTMXy(self, SEQ_LEN):
        """
        Generates the X (features) and y (target) arrays for Conv2D LSTM model training.

        Args:
            SEQ_LEN (int): Length of the time sequence (number of time steps).

        Returns:
            tuple: (X, y) where:
                - X: numpy.ndarray of shape (samples, SEQ_LEN, n_features, height, width)
                - y: numpy.ndarray of shape (samples, SEQ_LEN, height, width)
        """
        N_LONS = len(self.lons)
        N_LATS = len(self.lats)
        # n_points2D = N_LATS * N_LONS
        n_times = len(self.date)
        N_FEATURES = len(self.feature.data_vars)

        # Initialize arrays for features (X) and target (y)
        X = np.zeros((n_times, N_FEATURES, N_LATS, N_LONS))
        y = np.zeros((n_times, N_LATS, N_LONS))

        # Populate X with feature data
        for num, var in enumerate(list(self.feature.data_vars)):
            tmp = self.feature[var]
            X[:, num, :, :] = tmp

        # Populate y with SPM data
        y = np.array(self.SPM.values)

        # Convert to float16 to save memory
        X = X.astype(np.float16)
        y = y.astype(np.float16)

        # Create sequences of length SEQ_LEN
        X_final = np.zeros(
            (len(X) - SEQ_LEN, SEQ_LEN, N_FEATURES, N_LATS, N_LONS), dtype=np.float16
        )
        y_final = np.zeros(
            (len(X) - SEQ_LEN, SEQ_LEN, N_LATS, N_LONS), dtype=np.float16
        )

        for k in range(SEQ_LEN, len(X)):
            X_final[k - SEQ_LEN, :, :, :, :] = X[k - SEQ_LEN : k, :, :, :]
            y_final[k - SEQ_LEN, :, :, :] = y[k - SEQ_LEN : k, :, :]

        del X, y  # Free memory

        return X_final, y_final

    def __str__(self):
        """String representation of the Data_Processing object."""
        return f"Data_Processing object:\n  File: {self.name}\n  Time range: {self.start_time} to {self.end_time}\n  Time step: {self.time_step}\n  Data shape: {self.data.dims}"


def Train_Validation_Test_Split(X, y, dates, testPer, valPer, results_file_path, base_dir):
    """
    Splits the dataset into training, validation, and test sets by every month.
    Tracks dates and plots SPM values for each split.

    Args:
        X (numpy.ndarray): Feature data of shape (samples, SEQ_LEN, n_features, height, width).
        y (numpy.ndarray): Target data of shape (samples, SEQ_LEN, height, width).
        dates (numpy.ndarray): Array of dates corresponding to the samples.
        testPer (float): Proportion of data to use for testing.
        valPer (float): Proportion of data to use for validation.
        results_file_path (str): Directory for final results (not used for plots).
        base_dir (str): Base directory to derive model-specific results path.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test)
    """
    # Derive model-specific results directory
    os.makedirs(results_file_path, exist_ok=True)

    n_samples = len(X)
    n_splits = n_samples // 30  # Split by months (assuming 30 days per month)

    X_train_list, X_test_list, X_val_list = [], [], []
    y_train_list, y_test_list, y_val_list = [], [], []
    dates_train_list, dates_test_list, dates_val_list = [], [], []

    # Split data by month
    for i in range(n_splits):
        start = i * 30
        end = (i + 1) * 30
        X_temp, X_test, y_temp, y_test, dates_temp, dates_test = train_test_split(
            X[start:end], y[start:end], dates[start:end], test_size=testPer, random_state=42
        )
        X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(
            X_temp, y_temp, dates_temp, test_size=valPer / (1 - testPer), random_state=42
        )

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        X_val_list.append(X_val)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        y_val_list.append(y_val)
        dates_train_list.append(dates_train)
        dates_test_list.append(dates_test)
        dates_val_list.append(dates_val)

    # Handle remaining samples (if any)
    if n_samples % 30 > 7:  # Only include if there are enough samples for a meaningful split
        start = n_splits * 30
        X_temp, X_test, y_temp, y_test, dates_temp, dates_test = train_test_split(
            X[start:], y[start:], dates[start:], test_size=testPer, random_state=42
        )
        X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(
            X_temp, y_temp, dates_temp, test_size=valPer / (1 - testPer), random_state=42
        )

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        X_val_list.append(X_val)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        y_val_list.append(y_val)
        dates_train_list.append(dates_train)
        dates_test_list.append(dates_test)
        dates_val_list.append(dates_val)

    # Concatenate the splits
    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    X_val = np.concatenate(X_val_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)
    y_val = np.concatenate(y_val_list)
    dates_train = np.concatenate(dates_train_list)
    dates_test = np.concatenate(dates_test_list)
    dates_val = np.concatenate(dates_val_list)

    # Calculate mean SPM per date (across spatial dims) for plotting
    y_mean_train = np.expm1(np.mean(y_train[:, -1, :, :], axis=(1, 2)))  # Reverse log transformation
    y_mean_val = np.expm1(np.mean(y_val[:, -1, :, :], axis=(1, 2)))
    y_mean_test = np.expm1(np.mean(y_test[:, -1, :, :], axis=(1, 2)))

    # Plot distribution of datapoints over dates
    plt.figure(figsize=(12, 6))
    plt.scatter(dates_train, y_mean_train, c='blue', marker='o', label='Training data', alpha=0.5)
    plt.scatter(dates_val, y_mean_val, c='green', marker='^', label='Validation data', alpha=0.5)
    plt.scatter(dates_test, y_mean_test, c='red', marker='s', label='Test data', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('ln(SPM+1), mg/L')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(results_file_path, 'trainvaltest_dates_split.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved split plot to {plot_path}")

    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test
