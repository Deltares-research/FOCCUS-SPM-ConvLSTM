# -*- coding: utf-8 -*-
"""
SHAP analysis and best model logging for SPM prediction pipelin, adapted from work done by Senyang Li (unpublished).

Author: Beau van Koert, Edits by L.Beyaard
Date: June 2025 
"""
#TODO: Ploting SHAP values does not work yet. Debug and fix

# Import libraries
import os
import pandas as pd
import numpy as np
from datetime import datetime
import math
import itertools
from tqdm.auto import tqdm
from tensorflow import keras
import matplotlib.pyplot as plt

# Import modules
from model import ReflectionPadding2D  # Import the custom model architecture layer

#%% Helper functions
def convert_spm_to_mgL(spm_data):
    """
    Converts SPM data from ln(SPM+1) to mg/L.

    Args:
        spm_data (np.ndarray): SPM data in ln(SPM+1) format.

    Returns:
        np.ndarray: SPM data in mg/L.
    """
    return np.expm1(spm_data)

#%% KernelExplainer
class KernelExplainer:
    """
    A custom KernelExplainer for computing SHAP values for a deep learning model.
    This class calculates the contribution of each feature to the model's predictions
    using a kernel-based approach.
    """

    def __init__(self, model, data):
        """
        Initialize the KernelExplainer.

        Args:
            model: The trained Keras model (e.g., ConvLSTM).
            data (np.ndarray): Background dataset used for SHAP computation, shape (n_samples, timesteps, n_features, height, width).
        """
        self.model = model
        self.data = data
        # Extract dimensions of the data
        self.Ndatasample = self.data.shape[0]  # Number of background samples
        self.Ntimestep = self.data.shape[1]    # Number of timesteps
        self.Nfeature = self.data.shape[2]     # Number of features
        self.NY = self.data.shape[3]           # Height of the spatial grid
        self.NX = self.data.shape[4]           # Width of the spatial grid

    def shap_values(self, X, nsamples_per_feature=100):
        """
        Compute SHAP values for the input data X.

        Args:
            X (np.ndarray): Input data to explain, shape (n_samples, timesteps, n_features, height, width).
            nsamples_per_feature (int): Number of background samples to use per feature (default: 100).

        Returns:
            np.ndarray: SHAP values, shape (n_samples, n_features, timesteps, height, width).
        """
        explanations = []
        # Loop over each sample in X to compute SHAP values
        for i in tqdm(range(X.shape[0]), desc="Processing samples"):
            list_feature = []
            # Compute SHAP values for each feature
            for j in tqdm(range(X.shape[2]), desc=f"Sample {i+1} - Features"):
                data = X[i:i + 1, :]  # Select the i-th sample
                shap_for_feature = self.explain(data, j, nsamples_per_feature)
                list_feature.append(shap_for_feature)
            explanations.append(list_feature)
        explanations_array = np.array(explanations)  # Shape: (n_samples, n_features, timesteps, height, width)
        return explanations_array

    def explain(self, incoming_instance, j, nsamples_per_feature):
        """
        Compute the SHAP value for feature j for a single instance.

        Args:
            incoming_instance (np.ndarray): The instance to explain, shape (1, timesteps, n_features, height, width).
            j (int): Index of the feature to compute SHAP values for.
            nsamples_per_feature (int): Number of background samples to use.

        Returns:
            np.ndarray: SHAP values for feature j, shape (timesteps, height, width).
        """
        # Total number of subsets excluding the empty set and full set (2^(n_features-1) - 2)
        self.nsamples = 2 ** (self.Nfeature - 1) - 2
        # Compute weights for each subset size using the SHAP kernel
        self.weight_vector = np.array([
            math.factorial(i) * math.factorial(self.Nfeature - i - 1) / math.factorial(self.Nfeature)
            for i in range(0, self.Nfeature)
        ])
        result_shapley = np.zeros((self.Ntimestep, self.NY, self.NX))

        # Get indices of all features except feature j
        group_inds = np.arange(self.Nfeature)
        group_inds = np.delete(group_inds, np.where(group_inds == j))

        # Loop over subset sizes
        for i in tqdm(range(self.Nfeature), desc=f"Feature {j} - Subset sizes"):
            weight_vector = self.weight_vector[i]
            # Generate all combinations of features of size i (excluding feature j)
            indices = itertools.combinations(group_inds, i)
            result_average = np.zeros((self.Ntimestep, self.NY, self.NX))

            # Loop over all combinations of size i
            for ind in indices:
                # Compute the contribution for this subset across background samples
                result_sum = np.zeros((self.Ntimestep, self.NY, self.NX))
                # Use a subset of background samples to reduce computation
                sample_indices = np.random.choice(self.Ndatasample, nsamples_per_feature, replace=False)
                for k in sample_indices:
                    result = self.contribution(k, incoming_instance, ind, j)
                    result = result.squeeze()  # Remove singleton dimensions
                    result_sum += result
                result_average = result_sum / nsamples_per_feature
            # Weight the contribution by the SHAP kernel weight
            result_with_weight = result_average * weight_vector
            result_shapley += result_with_weight

        return result_shapley

    def contribution(self, k, fix_data, mask, j):
        """
        Compute the marginal contribution of feature j for a specific background sample.

        Args:
            k (int): Index of the background sample.
            fix_data (np.ndarray): The instance to explain, shape (1, timesteps, n_features, height, width).
            mask (tuple): Indices of features in the subset S (excluding j).
            j (int): Index of the feature to compute the contribution for.

        Returns:
            np.ndarray: Marginal contribution, shape (timesteps, height, width).
        """
        # Create two copies of the background sample
        data_withoutj = self.data[k:k+1, :].copy()  # Background sample k
        data_withoutj[:, :, mask] = fix_data[:, :, mask]  # Set features in S to values from fix_data
        data_withj = data_withoutj.copy()
        data_withj[:, :, j] = fix_data[:, :, j]  # Add feature j from fix_data

        # Compute predictions with and without feature j
        result_withoutj = self.model.predict(data_withoutj, verbose=0)
        result_withj = self.model.predict(data_withj, verbose=0)

        # Return the marginal contribution (f(S ∪ {j}) - f(S))
        return result_withj - result_withoutj

#%% SHAPAnalyzer
class SHAPAnalyzer:
    """
    A class to perform SHAP analysis on a trained model, create visualizations, and save the results.
    """
    
    def __init__(self, name, config, data_vars):
        
        # Define model name
        self.name = name
        
        self.model_path = config.get("model_path")
        self.model_name = config.get("specific_model_name")
        self.X_train = data_vars['X_train_original']
        self.X_test = data_vars['X_test']
        self.nan_indices = data_vars['nan_indices']
        self.feature_names = data_vars['feature_names']
        
        # Define paths (and ensure that they exist)
        self.output_dir_main = config.get("results_file_path")
        self.output_dir = os.path.join(self.output_dir_main, f'{self.name}')    # model specific folder
        os.makedirs(self.output_dir, exist_ok=True)
        self.intermediate_dir_main = config.get("intermediate_dir")
        self.intermediate_dir = os.path.join(self.intermediate_dir_main, f'{self.name}') # model specific folder
        os.makedirs(self.intermediate_dir, exist_ok=True)
        
        # Initialize SHAP variables
        self.shap_samples = config.get("shap_samples")
        self.background_samples = config.get("background_samples")
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.test_samples = None
        
        # Run the SHAP analysis pipeline
        self.load_model()
        self.initialize_explainer()
        self.compute_shap_values()
        self.plot_shap_beeswarm()
        self.plot_shap_feature_importance()
        self.plot_shap_spatial_maps()
        

    def load_model(self):
        """
        Load the trained Keras model.
        """
        print(f"Loading model from {self.model_path}...")
        # Use custom_object_scope to register the ReflectionPadding2D layer
        with keras.utils.custom_object_scope({'ReflectionPadding2D': ReflectionPadding2D}):
            self.model = keras.models.load_model(self.model_path)
        print("Model loaded successfully.")
        print(self.model.summary())


    def initialize_explainer(self):
        """
        Initialize the KernelExplainer with a subset of the training data as background samples.
        """
        # Select a subset of training data as background samples
        indices = np.linspace(0, len(self.X_train) - 1, self.background_samples, dtype=int)
        background_data = self.X_train[indices]
        print(f"Using {self.background_samples} background samples for SHAP explainer.")
        self.explainer = KernelExplainer(self.model, data=background_data)


    def compute_shap_values(self):
        """
        Compute or load SHAP values for a subset of the test data and store them.
        """
        # Select a subset of test samples to compute SHAP values for
        indices_test = np.linspace(0, len(self.X_test) - 1, self.shap_samples, dtype=int)
        print(f"Processing {self.shap_samples} test samples for SHAP values...")

        shap_values_list = []
        test_samples_list = []

        for i in range(self.shap_samples):
            index = indices_test[i]
            output_file = os.path.join(self.intermediate_dir, f'shap{self.background_samples}_S{i}.npz')

            # Check if the SHAP .npz file exists
            if os.path.exists(output_file):
                print(f"Loading existing SHAP values for test sample {i+1}/{self.shap_samples} (index {index}) from {output_file}...")
                data = np.load(output_file)
                shap_values = data['value']
                X_test_i = data['test']
                loaded_nan_indices = data['ni']
                # Verify that nan_indices match
                if not np.array_equal(loaded_nan_indices, self.nan_indices):
                    raise ValueError(f"NaN indices in {output_file} do not match the provided nan_indices.")
            else:
                print(f"Computing SHAP values for test sample {i+1}/{self.shap_samples} (index {index})...")
                X_test_i = self.X_test[index:index + 1]  # Select one test sample
                # Compute SHAP values for this sample
                shap_values = self.explainer.shap_values(X_test_i, nsamples_per_feature=self.background_samples)
                # Save the SHAP values along with the test data and nan_indices
                np.savez_compressed(output_file, value=shap_values, test=X_test_i, ni=self.nan_indices)
                print(f"Saved SHAP values to {output_file}")

            shap_values_list.append(shap_values)
            test_samples_list.append(X_test_i)

        # Store all SHAP values and test samples for plotting
        self.shap_values = np.concatenate(shap_values_list, axis=0)  # Shape: (shap_samples, n_features, timesteps, height, width)
        self.test_samples = np.concatenate(test_samples_list, axis=0)


    def plot_shap_beeswarm(self):
        """
        Create a beeswarm plot of temporal mean SHAP values for each feature at each pixel.
        """
        # Compute temporal mean SHAP values for each sample, feature, and pixel
        shap_mean = np.mean(self.shap_values, axis=2)  # Mean over timesteps: (shap_samples, n_features, height, width)
        # Convert SHAP values to mg/L units (since SHAP values are in ln(SPM+1) space)
        shap_mean_mgL = np.expm1(shap_mean)

        # Compute temporal mean of feature values
        feature_values = np.mean(self.test_samples, axis=2)  # (shap_samples, n_features, height, width)

        # Reshape nan_indices to 2D for masking
        nan_indices_2d = self.nan_indices.reshape(shap_mean_mgL.shape[2], shap_mean_mgL.shape[3])

        # Apply NaN mask to both shap_mean_mgL and feature_values
        shap_mean_masked = np.where(nan_indices_2d, np.nan, shap_mean_mgL)  # (shap_samples, n_features, height, width)
        feature_values_masked = np.where(nan_indices_2d, np.nan, feature_values)

        # Flatten the arrays, excluding NaN values
        n_features = shap_mean_mgL.shape[1]
        shap_flat = []
        feature_flat = []
        for i in range(n_features-1):
            # Extract non-NaN values for this feature
            shap_values_feature = shap_mean_masked[:, i, :, :].flatten()
            feature_values_feature = feature_values_masked[:, i, :, :].flatten()
            # Filter out NaN values
            valid_mask = ~np.isnan(shap_values_feature)
            shap_flat.append(shap_values_feature[valid_mask])
            feature_flat.append(feature_values_feature[valid_mask])

        # Create the beeswarm plot
        plt.figure(figsize=(10, 6))
        for i in range(n_features-1):
            y = np.ones(shap_flat[i].size) * (n_features - 1 - i)  # y-position for this feature
            # Normalize feature values to [0, 1] for the colormap
            feature_values_i = feature_flat[i]
            feature_min, feature_max = np.min(feature_values_i), np.max(feature_values_i)
            if feature_max == feature_min:  # Avoid division by zero
                feature_normalized = np.zeros_like(feature_values_i)
            else:
                feature_normalized = (feature_values_i - feature_min) / (feature_max - feature_min)
            plt.scatter(shap_flat[i], y, c=feature_normalized, cmap='viridis', alpha=0.5, s=10)

        # Customize the plot
        plt.yticks(range(n_features), self.feature_names[::-1])  # Reverse order for top-to-bottom
        plt.xlabel('SHAP Value (Impact on SPM Prediction, mg/L)')
        plt.title('Temporal Mean SHAP Values for Each Feature at Each Pixel')
        plt.colorbar(label='Normalized Feature Value', cmap='viridis')

        # Save the plot
        output_file = os.path.join(self.output_dir, f'shap_beeswarm_{self.shap_samples}Ss_{self.background_samples}BGs.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP beeswarm plot to {output_file}")


    def plot_shap_feature_importance(self):
        """
        Create a bar plot of mean absolute SHAP values to show overall feature importance.
        """
        # Compute mean absolute SHAP values for each feature
        shap_abs_mean = np.mean(np.abs(self.shap_values), axis=(0, 2, 3, 4))  # Mean over samples, timesteps, and spatial dims
        shap_abs_mean_mgL = np.expm1(shap_abs_mean)  # Convert to mg/L

        # Create the bar plot
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(self.feature_names)), shap_abs_mean_mgL, align='center', color='skyblue')
        plt.yticks(range(len(self.feature_names)), self.feature_names)
        plt.xlabel('Mean |SHAP Value| (Average Impact on SPM Prediction, mg/L)')
        plt.title('Feature Importance Based on SHAP Values')

        # Save the plot
        output_file = os.path.join(self.output_dir, 'shap_feature_importance.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP feature importance plot to {output_file}")


    def plot_shap_spatial_maps(self):
        """
        Create spatial maps of SHAP values for each feature, averaged over samples and timesteps.
        """
        # Compute mean SHAP values over samples and timesteps
        # shap_values shape: (shap_samples, n_features, timesteps, height, width)
        shap_mean = np.mean(self.shap_values, axis=(0, 2))  # Mean over samples and timesteps: (n_features, height, width)
        shap_mean = np.expm1(shap_mean)  # Convert to mg/L

        # Apply NaN mask
        nan_indices_2d = self.nan_indices.reshape(shap_mean.shape[1], shap_mean.shape[2])
        shap_mean_masked = np.where(nan_indices_2d, np.nan, shap_mean)

        # Determine colorbar limits
        abs_max = np.percentile(np.abs(shap_mean_masked[~np.isnan(shap_mean_masked)]), 95)
        vmin, vmax = -abs_max, abs_max

        # Plot a map for each feature
        for i, feature_name in enumerate(self.feature_names):
            plt.figure(figsize=(8, 6))
            im = plt.imshow(shap_mean_masked[i], cmap='RdBu', vmin=vmin, vmax=vmax)
            plt.gca().invert_yaxis()
            plt.title(f'Spatial Mean SHAP Values for {feature_name}')
            plt.xlabel('Pixel X')
            plt.ylabel('Pixel Y')
            plt.colorbar(im, label='SHAP Value (Impact on SPM Prediction)')

            # Save the plot
            output_file = os.path.join(self.output_dir, f'shap_spatial_map_{feature_name}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP spatial map for {feature_name} to {output_file}")

#%% Best model log
def find_best_model(results_dir, output_dir, print_best_models=True):
    """
    Analyze model results to find the best models based on various metrics using evaluation_metrics.csv.

    Args:
        results_dir (str): Directory containing model results (e.g., Results).
        output_dir (str): Directory to store output files (e.g., Results).
        print_best_models (bool): Whether to print best models to console.

    Returns:
        dict: Dictionary containing the best models for each metric.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths for results_allruns files
    results_allruns_csv = os.path.join(output_dir, 'results_allruns.csv')
    results_allruns_xlsx = os.path.join(output_dir, 'results_allruns.xlsx')
    
    # Check if results_allruns.csv exists, if not create an empty DataFrame
    if os.path.exists(results_allruns_csv):
        all_results_df = pd.read_csv(results_allruns_csv)
        print(f"Loaded existing results_allruns.csv with {len(all_results_df)} entries.")
    else:
        all_results_df = pd.DataFrame(columns=[
            'Name', 'mae_overall', 'mse_overall', 'rmse_overall', 
            'r2_overall', 'mad_overall'
        ])
        print("Created new empty DataFrame for results_allruns.")

    # Get list of model directories
    model_dirs = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d))]
    print(f"Found {len(model_dirs)} model directories in {results_dir}: {model_dirs}")
    
    if not model_dirs:
        print(f"No model directories found in {results_dir}. Exiting.")
        return {}

    # Process each model directory
    for model_dir in model_dirs:
        eval_metrics_file = os.path.join(results_dir, model_dir, 'evaluation_metrics.csv')
        
        if os.path.exists(eval_metrics_file):
            try:
                # Read the evaluation_metrics.csv file
                df = pd.read_csv(eval_metrics_file)
                
                if not df.empty:
                    # Extract the "Overall" row
                    overall_row = df[df['Day'] == 'Overall']
                    if overall_row.empty:
                        print(f"No 'Overall' row found in {eval_metrics_file}. Skipping.")
                        continue

                    # Verify that all required columns are present
                    required_columns = ['Day', 'MAE', 'MSE', 'RMSE', 'R2', 'MAD']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Missing columns in {eval_metrics_file}: {missing_columns}. Skipping this file.")
                        continue

                    # Extract relevant metrics from the "Overall" row
                    # Metrics are already in mg/L units since SPMAnalyzer converted them
                    model_data = {
                        'Name': model_dir,
                        'mae_overall': overall_row['MAE'].iloc[0],
                        'mse_overall': overall_row['MSE'].iloc[0],
                        'rmse_overall': overall_row['RMSE'].iloc[0],
                        'r2_overall': overall_row['R2'].iloc[0],
                        'mad_overall': overall_row['MAD'].iloc[0]
                    }
                    
                    # Check if model already exists in all_results_df
                    if model_data['Name'] in all_results_df['Name'].values:
                        # Update existing entry
                        model_idx = all_results_df[all_results_df['Name'] == model_data['Name']].index
                        all_results_df.loc[model_idx, model_data.keys()] = model_data.values()
                    else:
                        # Append new entry
                        all_results_df = pd.concat([all_results_df, pd.DataFrame([model_data])], 
                                                ignore_index=True)
                        print(f"Appended new entry for {model_data['Name']} to all_results_df.")
                else:
                    print(f"File {eval_metrics_file} is empty. Skipping.")
            except Exception as e:
                print(f"Error processing {eval_metrics_file}: {str(e)}")
                continue
        else:
            print(f"No evaluation_metrics.csv found in {model_dir}. Skipping.")
    
    print(f"Final all_results_df has {len(all_results_df)} entries:")
    print(all_results_df)

    if all_results_df.empty:
        print("No valid model data found to process. Exiting.")
        return {}

    # Save updated results to both CSV and Excel
    all_results_df.to_csv(results_allruns_csv, index=False)
    print(f"Saved results to {results_allruns_csv}")
    all_results_df.to_excel(results_allruns_xlsx, index=False)
    print(f"Saved results to {results_allruns_xlsx}")
    
    # Find the best models based on different metrics
    best_mae = all_results_df.loc[all_results_df['mae_overall'].idxmin()]
    best_mse = all_results_df.loc[all_results_df['mse_overall'].idxmin()]
    best_rmse = all_results_df.loc[all_results_df['rmse_overall'].idxmin()]
    best_r2 = all_results_df.loc[all_results_df['r2_overall'].idxmax()]
    best_mad = all_results_df.loc[all_results_df['mad_overall'].idxmin()]
    
    # Prepare the output log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"\nBest Models Report - {timestamp}\n"
    log_message += f"Number of models checked: {len(model_dirs)}\n"
    log_message += "=" * 50 + "\n"
    
    log_message += f"Best MAE Model: {best_mae['Name']}\n"
    log_message += f"MAE: {best_mae['mae_overall']} mg/L\n\n"
    
    log_message += f"Best MSE Model: {best_mse['Name']}\n"
    log_message += f"MSE: {best_mse['mse_overall']} mg/L^2\n\n"
    
    log_message += f"Best RMSE Model: {best_rmse['Name']}\n"
    log_message += f"RMSE: {best_rmse['rmse_overall']} mg/L\n\n"
    
    log_message += f"Best R2 Model: {best_r2['Name']}\n"
    log_message += f"R2: {best_r2['r2_overall']}\n\n"
    
    log_message += f"Best MAD Model: {best_mad['Name']}\n"
    log_message += f"MAD: {best_mad['mad_overall']} mg/L\n"
    log_message += "=" * 50 + "\n"
    
    # Write to output log file (prepend the new report)
    log_file = os.path.join(output_dir, 'best_models_log.txt')
    if os.path.exists(log_file):    # if file does exist
        with open(log_file, 'r') as f:
            existing_content = f.read()
        with open(log_file, 'w') as f:
            f.write(log_message + existing_content)
    else:                           # if file does not exist
        with open(log_file, 'w') as f:
            f.write(log_message)
    print(f"Wrote best models report to {log_file}")
    
    # Print to console if enabled
    if print_best_models:
        print(log_message)

    return {
        'best_mae': best_mae.to_dict(),
        'best_mse': best_mse.to_dict(),
        'best_rmse': best_rmse.to_dict(),
        'best_r2': best_r2.to_dict(),
        'best_mad': best_mad.to_dict()
    }