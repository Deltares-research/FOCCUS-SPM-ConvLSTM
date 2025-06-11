"""
ConvLSTM Model Pipeline for SPM Prediction, FOCCUS D6.2. https://foccus-project.eu/

Description:
This script orchestrates the preprocessing, training, prediction, and analysis
of a ConvLSTM model to predict Suspended Particulate Matter (SPM) using CMEMS
and DCSM output data. The pipeline processes unstructured and raster data, trains the 
model with specified hyperparameters, makes predictions, analyzes results, 
(and optionally performs SHAP analysis for interpretability.)

Step-by-Step Plan:
1. Initialize the pipeline by setting up data variables and handling path conversion.
2. Pre-process data, including UGRID rasterization and optional CMEMS data integration.
3. Process the raster data into ConvLSTM format and split into train/validation/test sets.
4. Loop over hyperparameter combinations to train the model, make predictions, and analyze results:
   - Adjust training data to fit batch size.
   - Compile and train the ConvLSTM model.
   - Make predictions on the test set.
   - Save the model and results.
   - Analyze the results using SPMAnalyzer.
5. Perform post-model analysis, including finding the best model and
   optionally running SHAP analysis.
6. Clear TensorFlow session to free memory.

Author: Beau van Koert, slight modifications by L.Beyaard. Works upon previous work of Senyang Li (unpublished).
Date: June 2025
"""

#TODO: Complete SHAP integration
#TODO: copy yaml file to intermediate results folder

# Standard Imports
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"       # disable more CuDNN installs
import tensorflow as tf
import pandas as pd
import yaml
import importlib.metadata

# Custom Module
from pipeline_utils import (initialize_pipeline, preprocess_data, process_data, train_model,
                           make_predictions, save_model_results, analyze_results,
                           post_model_analysis)

# --------------------------------------
# Helper Functions
# --------------------------------------
def print_package_versions():
    """Print versions of all installed packages in the environment."""
    print("\nAll installed packages in the environment:")
    print("-" * 50)
    installed_packages = sorted([(dist.name, dist.version) for dist in importlib.metadata.distributions()],
                               key=lambda x: x[0].lower())
    for pkg_name, pkg_version in installed_packages:
        print(f"{pkg_name}: {pkg_version}")
    print("-" * 50)
    print("\n")

def print_system_info():
    """Print system information about GPU availability and CUDA support."""
    print("-" * 25)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("cuDNN Enabled: ", tf.test.is_built_with_cuda())
    print("-" * 25)

# --------------------------------------
# Main Function
# --------------------------------------
def main(config):
    """Main function to run the ConvLSTM model pipeline."""
    # Initial Setup
    print_package_versions()
    print_system_info()
    print("\nRunning main function with config:", config)

    # Step 1: Initialize Pipeline
    data_vars = initialize_pipeline(config)

    # Step 2: Pre-process Data
    input_file_path, trim_file_path, data_vars, config = preprocess_data(config, data_vars)
    config["input_file_path"] = input_file_path     # save cleaned inpud data path to config for later steps
    config['trim_file_path'] = trim_file_path       # save trimfilepath to config for later steps

    # Step 3: Process Data for ConvLSTM
    data_vars = process_data(config, input_file_path, data_vars)

    # Step 4: Initialize Results DataFrame
    results = pd.DataFrame(columns=[
        "Name", "Losses", "Val_Losses", "mae_overall", "mse_overall",
        "rmse_overall", "r2_overall", "mad_overall", "mae", "mse",
        "rmse", "r2", "mad"
    ])

    # Step 5: Hyperparameter Loop
    print("Looping over combinations of hyper-parameters...")
    for batch_size in config.get("batch_size_list", [5]):
        for learning_rate_base in config.get("learning_rate_base_list", [0.002]):
            for T in config.get("T_list", [3]):
                for dropout in config.get("dropout_list", [0]):
                    for n_lstm_conv in config.get("n_lstm_conv_list", [48]):
                        # Construct model name
                        LRS = (f"cos-{config.get('num_conv_layers', 4)}L-{n_lstm_conv}C-"   # base
                               f"{dropout}D-{learning_rate_base}-{T}T"
                               f"{'-EarlyStop' if config.get('use_early_stopping', False) else '-NoEarlyStop'}")
                        if config.get("add_cmems_data", False):
                            dataset_suffix = "[CMEMS-intp-dly]" # suffix linear interp
                        elif config.get("add_cmems_4dvarnet_data", False):
                            dataset_suffix = "[CMEMS-4DVarNet-dly]" # suffix 4DVarNet interp
                        else:
                            dataset_suffix = ""
                        name = (f"{batch_size}-{config.get('time_step', 5)}-" # complete name
                                f"{config.get('epochs', 200)}-{LRS}-{dataset_suffix}")

                        # Train the model (call pipeline)
                        model, history, actual_epochs, data_vars['X_train'], data_vars['y_train'] = train_model(
                            config, data_vars, batch_size, learning_rate_base, T, dropout, n_lstm_conv)

                        # Make predictions (call pipeline)
                        model, pred, data_vars = make_predictions(
                            config, data_vars, model, batch_size, name, config.get("intermediate_dir", 'Intermediate'))

                        # Save model and results (call pipeline)
                        epoch_losses, val_losses = save_model_results(
                            config, model, history, actual_epochs, pred, data_vars, name,
                            config.get("intermediate_dir", 'Intermediate'))

                        # Analyze results (call pipeline)
                        results = analyze_results(
                            config, data_vars, epoch_losses, val_losses, name, results)

                        # Clear TensorFlow session
                        tf.keras.backend.clear_session()

    # Step 6: Post-Model Analysis (call pipeline)
    post_model_analysis(config, data_vars, name, results)

# =============================================================================
# Call main
# =============================================================================
if __name__ == "__main__":
    with open("config_single.yml", "r") as file:
        config = yaml.safe_load(file)
    main(config)