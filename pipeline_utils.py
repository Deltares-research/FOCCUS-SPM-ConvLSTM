"""
Utility functions for the ConvLSTM model pipeline.
Author: Beau van Koert, edits by L. Beyaard. 
Date: June 2025
"""

# Import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re
import math
from datetime import datetime
from tensorflow import keras

# Import modules
from data_preprocessing import DCSMPreprocessor, CMEMS_processing
from data_processing import Data_Processing, Train_Validation_Test_Split
from model import getModel, WarmUpCosineDecayScheduler, LossHistory
from post_model_analysis import find_best_model, SHAPAnalyzer
from Auto_Reread import SPMAnalyzer

#%% Helper functionns 
#Functions for helping with the conversion of 
def convert_path(path, to_linux=True):
    """
    Convert a file path between Windows and Linux formats.

    Args:
        path (str): The file path to convert.
        to_linux (bool): If True, convert to Linux format; if False, convert to Windows format.

    Returns:
        str: The converted file path.
    """
    if to_linux:
        # Convert Windows path to Linux path
        path = path.replace("\\", "/")
        if re.match(r"^[A-Z]:", path):
            drive = path[0].lower()
            path = f"/{drive}{path[2:]}"
    else:
        # Convert Linux path to Windows path
        path = path.replace("/", "\\")
        if path.startswith("/p/"):
            path = f"P:{path[2:]}"
    return path

def is_windows_path(path):
    """
    Check if a path is in Windows format.

    Args:
        path (str): The file path to check.

    Returns:
        bool: True if the path is in Windows format, False otherwise.
    """
    return re.match(r"^[A-Z]:", path) is not None or "\\" in path

def initialize_pipeline(config):
    """
    Initialize the pipeline by setting up data variables and handling path conversion.
    Checks if paths in config file are in correct format (Windows or Linux).
    This assumes that the cluster/computer runs on linux. 

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Initial data_vars dictionary.
    """
    # Check run_on_cluster flag (default to False if not specified)
    run_on_cluster = config.get("run_on_cluster", False)
    print(f"\nRunning on cluster: {run_on_cluster}\n")

    # Check if paths match the run_on_cluster setting
    path_keys = ["base_dir", "input_ugrid", "river_file_path", "spm_validation_file_path", "results_file_path", "intermediate_dir"]
    for key in path_keys:
        if key in config:
            print(f"{key}: {config[key]}")
            path = config[key]
            is_win = is_windows_path(path)
            if run_on_cluster and is_win:
                print(f"Warning: Path '{path}' ({key}) is in Windows format but run_on_cluster is True. Converting to Linux format.")
                config[key] = convert_path(path, to_linux=True)
            elif not run_on_cluster and not is_win:
                print(f"Warning: Path '{path}' ({key}) is in Linux format but run_on_cluster is False. Converting to Windows format.")
                config[key] = convert_path(path, to_linux=False)

    # Initialize data_vars
    data_vars = {
        'X_train': None, 'y_train': None, 'X_val': None, 'y_val': None,
        'X_test': None, 'y_test': None, 'dates_train': None, 'dates_val': None,
        'dates_test': None, 'nan_indices': None, 'pred': None,
        'feature_names': None, 'feature_means': None, 'feature_stds': None
    }
    return data_vars

#%% Data pre-processing
def preprocess_data(config, data_vars):
    """
    Preprocess the input data, including UGRID rasterization and optional CMEMS data integration.

    Args:
        config (dict): Configuration dictionary.
        data_vars (dict): Dictionary to store data variables.

    Returns:
        tuple: (updated input_file_path, updated trim_file_path, data_vars, config)
    """
    start_time = datetime.now()
    print(f"Starting preprocess_data at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Dynamically generate paths based on base_dir
    base_dir = config["base_dir"]
    if config.get("add_cmems_4dvarnet_data", False):
        output_dir_plots = os.path.join(base_dir, "5.Results", "Figures", "Post_Delft_3D", "CMEMS_4DVarNet_2007")
    else:
        output_dir_plots = os.path.join(base_dir, "5.Results", "Figures", "Post_Delft_3D", "CMEMS_Daily_2007")
    
    # Step 1: UGRID Preprocessing
    input_file_path = None
    trim_file_path = None
    # Check pipeline flag in config
    if config.get("run_ugrid_preprocessing", True):
        print(f"\nPreprocessing UGRID data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        dcsm_preprocessor = DCSMPreprocessor(config)
        input_file_path, trim_file_path = dcsm_preprocessor.process()
    else:
        print(f"\nSkipping UGRID preprocessing as per config at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        # Use default paths if preprocessing is skipped
        data_dir = os.path.join(base_dir, "3.Data")
        dfm_dir = os.path.join(data_dir, "DFM_data")
        raster_dir = os.path.join(dfm_dir, "DCSM_rasterized_clip")
        map_dir = os.path.join(data_dir, "Map_Study_Area")
        input_file_path = os.path.join(raster_dir, "DCSM_raster_hydro_spm_CleanedUp.nc")
        trim_file_path = os.path.join(map_dir, "bbox_DutchCoast_SPM_Plume_WaterOnly.shp")

    # Compute spatial bounds for CMEMS data paths
    def bbox_round_up(value):
        """ 
        Add 1 if number is an integer, otherwise round up.
        """
        if value == int(value):
            return int(value + 1)
        else:
            return math.ceil(value)
    
    bbox = config["bbox"]                      # (x_min, x_max, y_min, y_max)
    buffer_size = config.get("buffer_size", 1.0)    # degrees
    minimum_longitude = math.floor(bbox[0] - buffer_size)
    maximum_longitude = bbox_round_up(bbox[1] + buffer_size)
    minimum_latitude = math.floor(bbox[2] - buffer_size)
    maximum_latitude = bbox_round_up(bbox[3] + buffer_size)
    start_datetime = config["start_time"]
    end_datetime = config["end_time"]

    # Check for CMEMS data integration flags
    if config.get("add_cmems_data") and config.get("add_cmems_4dvarnet_data"):
        print("Cannot add CMEMS SPM data using both interpolation methods. Only one can be True.")
        raise
    
    # Step 2: CMEMS Data Integration (Optional)
    if config.get("add_cmems_data", False):
        # Define the CMEMS data directory (go up two levels from DCSM_rasterized_clip to 3.Data, then into CMEMS_data)
        cmems_data_dir = os.path.join(config['base_dir'], "3.Data", "CMEMS_data")
        # Define the raw CMEMS filename and path
        output_filename = f'CMEMS_{start_datetime}_{end_datetime}_lon{minimum_longitude}to{maximum_longitude}_lat{minimum_latitude}to{maximum_latitude}.nc'
        raw_cmems_path = os.path.join(cmems_data_dir, output_filename)
        # Store the raw CMEMS filepath in the config
        config['raw_cmems_path'] = raw_cmems_path
        print(f"\nRaw CMEMS path set in config: {config['raw_cmems_path']}")

        # Define the expected output file path after CMEMS processing
        cmems_output_path = os.path.join(os.path.dirname(input_file_path), "DCSM_raster_hydro_spm_CMEMS_daily_2007.nc")
        
        # Check if the CMEMS-processed file already exists
        if os.path.exists(cmems_output_path):
            print(f"\nCMEMS-processed file already exists at {cmems_output_path}. Skipping CMEMS data integration.")
            input_file_path = cmems_output_path
        else:
            print(f"\nAdding CMEMS satellite data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
            cmems_processor = CMEMS_processing(
                config=config,
                dcsm_path=input_file_path,
                output_dir_plots=output_dir_plots,
                shapefile_path=trim_file_path
            )
            config = cmems_processor.process(create_animation=True)
            input_file_path = cmems_output_path
    
    # Step 2b: Add the CMEMS data which is processed with 4DVarNet
    if config.get("add_cmems_4dvarnet_data", False):
        # Define the CMEMS data directory (go up two levels from DCSM_rasterized_clip to 3.Data, then into CMEMS_data)
        cmems_data_dir = os.path.join(config['base_dir'], "3.Data", "CMEMS_data")
        # Define the raw 4DVarNet CMEMS filepath
        raw_cmems_4dvarnet_path = os.path.join(cmems_data_dir, "CMEMS2007_gapfilled.nc")
        # Store the raw 4DVarNet CMEMS filepath in the config
        config['raw_cmems_4dvarnet_path'] = raw_cmems_4dvarnet_path
        print(f"\nRaw CMEMS 4DVarNet path set in config: {config['raw_cmems_4dvarnet_path']}")
        
        # Define the expected output file path after CMEMS 4DVarNet processing
        cmems_output_path = os.path.join(os.path.dirname(input_file_path), "DCSM_raster_hydro_spm_CMEMS_daily_4DVarNet_2007.nc")
        
        # Check if the CMEMS-processed file already exists
        if os.path.exists(cmems_output_path):
            print(f"\n4DVarNet CMEMS-processed file already exists at {cmems_output_path}. Skipping 4DVarNet CMEMS data integration.")
            input_file_path = cmems_output_path
        else:
            print(f"\nAdding 4DVarNet CMEMS satellite data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
            cmems_processor = CMEMS_processing(
                config=config,
                dcsm_path=input_file_path,
                output_dir_plots=output_dir_plots,
                shapefile_path=trim_file_path
            )
            config = cmems_processor.process(create_animation=False)
            input_file_path = cmems_output_path
    
    else:
        print(f"\nSkipping CMEMS data integration as per config at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Time duration preprocess_data: {str(duration)}")
    return input_file_path, trim_file_path, data_vars, config

#%% Data processing
def process_data(config, input_file_path, data_vars):
    """
    Process the raster data into ConvLSTM format and split into train/validation/test sets.

    Args:
        config (dict): Configuration dictionary.
        input_file_path (str): Path to the input dataset.
        data_vars (dict): Dictionary to store data variables.

    Returns:
        dict: Updated data_vars with processed data.
    """
    start_time = datetime.now()
    print(f"Starting process_data at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not config.get("run_data_processing", True):
        print(f"\nSkipping data processing as per config at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        return data_vars

    print(f"\nProcessing the raster data from Delft-3D into ConvLSTM at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    
    myFile = Data_Processing(config, input_file_path, config["start_time"], config["end_time"], config.get("time_step", 5), config['trim_file_path'])
    
    X = myFile.X
    y = myFile.y
    dates = myFile.seq_dates
    nan_indices = myFile.nan_indices
    feature_names = list(myFile.feature_means.keys())
    feature_means = myFile.feature_means
    feature_stds = myFile.feature_stds
    
    # Splitting the data
    X_train_original, X_val, X_test, y_train_original, y_val, y_test, dates_train, dates_val, dates_test = Train_Validation_Test_Split(
        X, y, dates, config.get("test_per", 0.15), config.get("val_per", 0.15), config.get("results_file_path", 'Results'), config["base_dir"]
    )
    print("\nTrain, Validation, Test splits:")
    print(X_train_original.shape, X_val.shape, X_test.shape)
    print(y_train_original.shape, y_val.shape, y_test.shape)

    data_vars.update({
        'X': X,
        'y': y,
        'dates': dates,
        'nan_indices': nan_indices,
        'feature_names': feature_names,
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'X_train_original': X_train_original,
        'X_val': X_val,
        'X_test': X_test,
        'y_train_original': y_train_original,
        'y_val': y_val,
        'y_test': y_test,
        'dates_train': dates_train,
        'dates_val': dates_val,
        'dates_test': dates_test
    })

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Time duration process_data: {str(duration)}")
    return data_vars

#%% Model
def train_model(config, data_vars, batch_size, learning_rate_base, T, dropout, n_lstm_conv):
    """
    Train the ConvLSTM model with the given hyperparameters.

    Args:
        config (dict): Configuration dictionary.
        data_vars (dict): Dictionary containing data variables.
        batch_size (int): Batch size for training.
        learning_rate_base (float): Base learning rate.
        T (int): Parameter T for the scheduler.
        dropout (float): Dropout rate.
        n_lstm_conv (int): Number of LSTM convolution filters.

    Returns:
        tuple: (model, history, actual_epochs, X_train, y_train, STEPS_PER_EPOCH)
    """
    start_time = datetime.now()
    print(f"Starting train_model at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not (config.get("run_model_training", True) and config.get("run_data_processing", True)):
        print(f"\nSkipping model training as per config at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        return None, None, None, None, None

    # Adjust training set to be a multiple of batch size
    X_train = data_vars['X_train_original'].copy()
    y_train = data_vars['y_train_original'].copy()
    N_TRAIN = X_train.shape[0] - X_train.shape[0] % batch_size
    X_train = X_train[len(X_train) - N_TRAIN :]
    y_train = y_train[len(y_train) - N_TRAIN :]
    STEPS_PER_EPOCH = len(X_train) // batch_size

    # Compute scheduler steps
    total_steps = STEPS_PER_EPOCH * config.get("epochs", 200)
    warmup_steps = STEPS_PER_EPOCH * config.get("warm_up_epochs", 10)

    # Compile the model
    print(f"\nCompiling the model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    model = getModel(
        X_train.shape[1:],
        config.get("activation", 'linear'),
        dropout,
        config.get("num_conv_layers", 4),
        n_lstm_conv
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=4e-06),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=["mse", "mae"]
    )
    print("\nModel summary:")
    model.summary()

    # Set up callbacks
    history = LossHistory()
    warm_up_lr = WarmUpCosineDecayScheduler(
        T=T,
        learning_rate_base=learning_rate_base,
        total_steps=total_steps,
        warmup_learning_rate=4e-06,
        warmup_steps=warmup_steps,
        hold_base_rate_steps=5
    )
    callbacks = [warm_up_lr, history]
    if config.get("use_early_stopping", False):
        earlystopper = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.get("early_stopping_patience", 50),
            verbose=1,
            mode="min"
        )
        callbacks.append(earlystopper)
        print(f"\nEarlyStopping enabled with patience={config.get('early_stopping_patience', 50)}")

    # Fit the model
    print(f"\nFitting the model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    model.fit(
        X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=config.get("epochs", 200),
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=(data_vars['X_val'], data_vars['y_val']),
        callbacks=callbacks,
        verbose=1
    )

    actual_epochs = len(history.losses) // STEPS_PER_EPOCH

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Time duration train_model: {str(duration)}")
    return model, history, actual_epochs, X_train, y_train

def make_predictions(config, data_vars, model, batch_size, name, intermediate_dir):
    """
    Make predictions using the trained model.

    Args:
        config (dict): Configuration dictionary.
        data_vars (dict): Dictionary containing data variables.
        model (tf.keras.Model): Trained model (or None if loading from disk).
        batch_size (int): Batch size for predictions.
        name (str): Model name.
        intermediate_dir (str): Directory for intermediate results.

    Returns:
        tuple: (model, pred, data_vars)
    """
    if not config.get("run_predictions", True):
        print("\nSkipping predictions as per config.")
        return model, None, data_vars

    # Load test data if not already loaded
    if data_vars['X_test'] is None:
        print("\nLoading test data since data processing was skipped...")
        myFile = Data_Processing(
            config["input_file_path"],
            config["start_time"],
            config["end_time"],
            config.get("time_step", 5),
            config['trim_file_path']
        )
        
        X = myFile.X
        y = myFile.y
        dates = myFile.seq_dates
        nan_indices = myFile.nan_indices
        feature_names = list(myFile.feature_means.keys())
        feature_means = myFile.feature_means
        feature_stds = myFile.feature_stds
        # Split data
        X_train_original, X_val, X_test, y_train_original, y_val, y_test, dates_train, dates_val, dates_test = Train_Validation_Test_Split(
            X, y, dates,
            testPer=config.get("test_per", 0.15),
            varPer=config.get("val_per", 0.15),
            results_file_path=config.get("results_file_path", 'Results'),
            base_dir=config["base_dir"]
        )
        
        data_vars.update({
            'X': X,
            'y': y,
            'dates': dates,
            'nan_indices': nan_indices,
            'feature_names': feature_names,
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'X_train_original': X_train_original,
            'X_val': X_val,
            'X_test': X_test,
            'y_train_original': y_train_original,
            'y_val': y_val,
            'y_test': y_test,
            'dates_train': dates_train,
            'dates_val': dates_val,
            'dates_test': dates_test
        })

    # Load model if not provided
    if model is None:
        print(f"\nLoading model from {intermediate_dir}/{name}.h5...")
        model = keras.models.load_model(f"{intermediate_dir}/{name}.h5")

    # Make predictions
    print("\nMaking SPM predictions...")
    pred = model.predict(data_vars['X_test'], batch_size=1, verbose=1)
    data_vars['pred'] = pred

    return model, pred, data_vars

def save_model_results(config, model, history, actual_epochs, pred, data_vars, name, intermediate_dir):
    """
    Save the model and its results, storing output paths in the config. If neither training nor predictions
    are run, load existing results from the .npz file.

    Args:
        config (dict): Configuration dictionary (will be modified to store paths).
        model (tf.keras.Model): Trained model.
        history (LossHistory): Training history.
        actual_epochs (int): Number of actual epochs.
        pred (np.ndarray): Predictions.
        data_vars (dict): Dictionary containing data variables.
        name (str): Model name.
        intermediate_dir (str): Base directory for intermediate results.

    Returns:
        tuple: (epoch_losses, val_losses) even if skipping training and predictions.
    """
    # Define file paths
    model_dir = os.path.join(intermediate_dir, name)
    model_path = os.path.join(model_dir, f"{name}.h5")
    results_npz_path = os.path.join(model_dir, f"{name}.npz")

    # Store paths in config (do this early so they're available regardless of the path)
    config['model_path'] = model_path
    config['results_npz_path'] = results_npz_path

    # Check if we need to skip training and prediction steps
    if not (config.get("run_model_training", True) or config.get("run_predictions", True)):
        print("\nNeither training nor predictions are set to run. Loading existing results.")
        # Check if the .npz file exists
        if not os.path.exists(results_npz_path):
            raise FileNotFoundError(f"No existing results found at {results_npz_path}. Cannot proceed without training or predictions.")
        
        # Load existing results
        npz_data = np.load(results_npz_path)
        epoch_losses = npz_data['losses']
        val_losses = npz_data['val_losses']
        actual_epochs = npz_data['epoches']
        
        # Update data_vars with loaded data if necessary
        if data_vars['y_test'] is None:
            data_vars['y_test'] = npz_data['y_test']
        if pred is None:
            pred = npz_data['pred']
        if data_vars['nan_indices'] is None:
            data_vars['nan_indices'] = npz_data['ni']
        
        print(f"\nLoaded existing results from {results_npz_path}")
        
        # Return losses from loaded files
        return epoch_losses, val_losses

    # Create named subdirectory for saving new results
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model.save(model_path)
    print(f"\nSaved the model to {model_path}")

    # Prepare losses
    if config.get("run_model_training", True):
        STEPS_PER_EPOCH = len(data_vars['X_train']) // config.get("batch_size_list", [5])[0]
        epoch_losses = history.losses[STEPS_PER_EPOCH - 1:actual_epochs * STEPS_PER_EPOCH:STEPS_PER_EPOCH]
        val_losses = history.val_losses[0:actual_epochs]
    else:
        # Load data from earlier training run
        npz_data = np.load(results_npz_path)
        epoch_losses = npz_data['losses']
        val_losses = npz_data['val_losses']
        actual_epochs = npz_data['epoches']
        if data_vars['y_test'] is None:
            data_vars['y_test'] = npz_data['y_test']
        if pred is None:
            pred = npz_data['pred']
        if data_vars['nan_indices'] is None:
            data_vars['nan_indices'] = npz_data['ni']
    
    # Save the original .npz file (without feature_means and feature_stds to avoid corruption)
    try:
        np.savez_compressed(
            results_npz_path,
            losses=epoch_losses,
            val_losses=val_losses,
            epoches=actual_epochs,
            y_test=data_vars['y_test'],
            pred=pred,
            ni=data_vars['nan_indices']
        )
        print(f"\nSaved the test/predict results to {results_npz_path}")
    except Exception as e:
        print(f"Error saving or verifying {results_npz_path}: {e}")
        raise
    
    # New .npz file path for normalization parameters
    norm_params_path = os.path.join(model_dir, f"{name}_norm_params.npz")
    
    # Save feature_means and feature_stds to a separate .npz file
    try:
        np.savez_compressed(
            norm_params_path,
            feature_names=data_vars.get('feature_names', {}),
            feature_means=data_vars.get('feature_means', {}),
            feature_stds=data_vars.get('feature_stds', {}),
            dates=data_vars.get('dates', {}),
            dates_train=data_vars.get('dates_train', {}),
            dates_val=data_vars.get('dates_val', {}),
            dates_test=data_vars.get('dates_test', {}),
        )
        print(f"Saved normalization parameters to {norm_params_path}")
    except Exception as e:
        print(f"Error saving or verifying {norm_params_path}: {e}")
        raise
    
    return epoch_losses, val_losses

def analyze_results(config, data_vars, epoch_losses, val_losses, name, results):
    """
    Analyze the results using SPMAnalyzer and save them.

    Args:
        config (dict): Configuration dictionary.
        data_vars (dict): Dictionary containing data variables.
        epoch_losses (list): Epoch losses.
        val_losses (list): Validation losses.
        name (str): Model name.
        results (pd.DataFrame): DataFrame to store results.

    Returns:
        pd.DataFrame: Updated results DataFrame.
    """
    # Check pipeline flag in config
    if not config.get("run_analysis", True):
        print("\nSkipping analysis as per config.")
        return results

    print("\nAnalyzing the results with SPMAnalyzer...")
    s = SPMAnalyzer(
        name=name,
        config=config,
        data_vars=data_vars
        )

    # Compile results
    result_dict = {
        "Name": name,
        "Losses": epoch_losses,
        "Val_Losses": val_losses,
        "mae_overall": s.mae_overall,
        "mse_overall": s.mse_overall,
        "rmse_overall": s.rmse_overall,
        "r2_overall": s.r2_overall,
        "mad_overall": s.mad_overall,
        "mae": s.mae,
        "mse": s.mse,
        "rmse": s.rmse,
        "r2": s.r2,
        "mad": s.mad
    }
    results = pd.concat([results, pd.DataFrame([result_dict])], ignore_index=True)

    # Save results
    results_dir = f"{config.get('results_file_path', 'Results')}/{name}"
    os.makedirs(results_dir, exist_ok=True)
    results.to_csv(f"{results_dir}/results.csv", index=False)
    print(f"\nResults saved to {results_dir}/results.csv")

    return results

#%% Post-model analysis
def post_model_analysis(config, data_vars, name, results):
    """
    Perform post-model analysis, including finding best models and SHAP
    analysis if enabled.

    Args:
        config (dict): Configuration dictionary.
        data_vars (dict): Dictionary containing data variables.
        name (str): Model name.
        results (pd.DataFrame): DataFrame containing results.

    Returns:
        dict: Best models (if applicable).
    """
    # Check pipeline flag in config
    if not config.get("run_analysis", True):
        print("\nSkipping post-model analysis as per config.")
        return None
    
    # Continue with post-model analysis
    print("\nPerforming post-model analysis...")
    
    # Find best models based on metrics in results_allruns.csv
    best_models = find_best_model(
        results_dir=config.get("results_file_path", "Results"),
        output_dir=config.get("results_file_path", "Results"),
        print_best_models=config.get("print_best_models", False)
    )
    
    # Perform SHAP analysis if enabled
    if config.get("run_shap", False):  
        print(f"\nPerforming SHAP analysis for {name}")
        shap_analyzer = SHAPAnalyzer(
            name=name,
            config=config,
            data_vars=data_vars,
        )
    
    return best_models