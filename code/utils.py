#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that contains re-usable functions for the modelling functions.

@author: christos, franz
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
from datetime import datetime
from re import sub
import os
import matplotlib.pyplot as plt
import warnings
import wandb
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import h2o
import numpy as np

warnings.filterwarnings("ignore")

def plot_psd_and_save(raw, subfolder=''):
    """Plot PSD of an MNE Raw object and save the plot as a PNG file."""
    # Get the file name(s) from the `filenames` attribute
    file_names = raw.filenames

    # If `filenames` is empty, return None
    if not file_names:
        return None

    # If `filenames` has one item, get the file name
    elif len(file_names) == 1:
        file_name = os.path.basename(file_names[0])

    # If `filenames` has multiple items, use the first file name
    else:
        file_name = os.path.basename(file_names[0])

    # Create the output directory if it does not exist
    output_dir = os.path.join(os.path.dirname(file_name), 'visualizations', 'psd', subfolder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot the PSD and save the figure
    fig = raw.plot_psd(fmin=1, fmax=50)
    plot_file = os.path.join(output_dir, f"{file_name}_psd.png")
    fig.savefig(plot_file)

    # Close the figure to free up memory
    plt.close(fig)

    # Return the path to the saved plot
    return plot_file,file_name

def plot_psd_and_save_rawarray(raw_array, file_name, subfolder=''):
    """Plot PSD of an MNE RawArray object and save the plot as a PNG file."""
    # Use the provided file name
    file_name = os.path.splitext(os.path.basename(file_name))[0]

    # Create the output directory if it does not exist
    output_dir = os.path.join(os.path.dirname(file_name), 'visualizations', 'psd', subfolder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot the PSD using the plot_psd() method
    fig = raw_array.plot_psd(fmin=1, fmax=50, show=False)

    # Save the figure
    plot_file = os.path.join(output_dir, f"{file_name}_psd.png")
    fig.savefig(plot_file)

    # Close the figure to free up memory
    plt.close(fig)

    # Return the path to the saved plot
    return plot_file
def get_datetime():
    """
    Returns the datetime in a snake case format that can be used as a
    suffix for the logger name.
    """

    def snake_case(s):
        return "_".join(
            sub(
                "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))
            ).split()
        ).lower()

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    date_time = snake_case(dt_string).replace(os.sep, "_").replace(":", "_")
    return date_time


def _get_number_of_sessions(path_to_mne_raw, subject):
    # get the number of sessions
    path2mne_raw = os.path.join(path_to_mne_raw, subject)
    n_sessions = [s for s in os.listdir(path2mne_raw) if not s.startswith(".")]
    return n_sessions

def extract_mne_signal_length_in_seconds(mne_file):
    # extract info in [samples]
    signal_length = mne_file.n_times # in samples
    sampling_rate = int(mne_file.info['sfreq'])
    # get the signal in [seconds]
    signal_length_in_seconds = int(signal_length/sampling_rate)
    return signal_length_in_seconds

def crop_signal(signal_length, window_size):
    '''
    Given a signal length and window size (in seconds),
    return equally sized intervals for that signal.

    '''
    intervals = []
    start = 0
    end = window_size
    while end <= signal_length:
        intervals.append((start, end))
        start += window_size
        end += window_size
    if end > signal_length:
        end = signal_length
        intervals.append((start, end))
    return intervals

def sort_list(given_list):
    return sorted(given_list, key=lambda x: int(x.split("-")[1]))

def get_best_non_stacked_model(aml):
    # Get the leaderboard
    lb = aml.leaderboard

    # Get model ids for all models in the AutoML Leaderboard
    model_ids = lb['model_id'].as_data_frame().iloc[:, 0]

    # Iterate and return the first non-stacked model
    for model_id in model_ids:
        model = h2o.get_model(model_id)
        if not isinstance(model, h2o.estimators.stackedensemble.H2OStackedEnsembleEstimator):
            return model
def analyze_cv_models(cv_models, condition_map, h2o_frame, conditions, n_top_features=10):
    confusion_matrices = []
    variable_importances = []
    # For each cross-validation model...
    for i, cv_model in enumerate(cv_models):
        print(f"\nCross-validation model {i + 1}:\n")

        # Get the variable importances
        var_imp = cv_model.varimp(use_pandas=True)
        variable_importances.append(var_imp)

        # Convert the H2OFrame to pandas DataFrame
        h2o_frame_pd = h2o_frame.as_data_frame()

        # Extract validation_data as pandas DataFrame
        validation_data = h2o_frame_pd[h2o_frame_pd["fold_column"] == i]

        # If you need to convert it back to H2OFrame
        validation_h2o_frame = h2o.H2OFrame(validation_data)
        # Get the confusion matrix
        train_performance = cv_model.model_performance(validation_h2o_frame)
        cm = train_performance._metric_json["cm"]['table'].cell_values
        # Convert to numpy array and ignore the last row and column as they're totals
        cm = np.array([row[0:-2] for row in cm[:-1]], dtype=float)
        confusion_matrices.append(cm)

        print("Confusion Matrix:")
        print(cm)

    # Sum the confusion matrices
    total_confusion_matrix = np.sum(confusion_matrices, axis=0)

    # Plot the total confusion matrix
    fig, ax = plt.subplots(figsize=(10, 7))
    # Get the classes
    classes = conditions

    # Map classes to your custom labels
    labels = [condition_map[int(c)] for c in classes]

    # Use labels for the xticklabels and yticklabels of your confusion matrix
    sns.heatmap(total_confusion_matrix, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Total Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Rotate x-axis labels for better readability
    #plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust the font size here
    plt.yticks(rotation=45, ha='right', fontsize=10)  # Adjust the font size here

    plt.tight_layout()

    # Log the confusion matrix plot to wandb
    wandb.log({"total_confusion_matrix": wandb.Image(fig)})

    # Concatenate all variable importances DataFrames
    all_var_imp = pd.concat(variable_importances)

    # Average the variable importances
    avg_var_imp = all_var_imp.groupby("variable").mean().reset_index()

    # Sort by average scaled_importance and take top n features
    top_n_var_imp = avg_var_imp.sort_values("relative_importance", ascending=False).head(n_top_features)

    # Plot the top n features
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x="relative_importance", y="variable", data=top_n_var_imp, ax=ax, palette="Blues_d")
    plt.title("Top {} Features by Relative Importance".format(n_top_features))
    plt.xlabel('Average Relative Importance')
    plt.ylabel('Feature')
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust the font size here
    plt.yticks(rotation=45, ha='right', fontsize=10)  # Adjust the font size here

    plt.tight_layout()

    # Log the feature importance plot to wandb
    wandb.log({"feature_importance": wandb.Image(fig)})

    # Initialize list for collecting SHAP values
    shap_values_list = []

    for i, cv_model in enumerate(cv_models):
        print(f"\nCross-validation model {i + 1}:\n")

        # Check if the model supports SHAP values calculation
        if not hasattr(cv_model, "shap_explain_frame"):
            print("Model does not support SHAP value calculation. Skipping...")
            continue

        # Get validation data for the current fold
        validation_data = h2o_frame[h2o_frame["fold_assignment"] == i]

        # Get SHAP values for the validation data of the current fold
        shap_values = cv_model.shap_explain_frame(validation_data)
        shap_values_list.append(shap_values)

    if not hasattr(cv_model, "shap_explain_frame"):
        print("Model does not support SHAP value calculation. Skipping...")
    else:
        # Concatenate all SHAP values DataFrames
        all_shap_values = pd.concat(shap_values_list)

        # Calculate the mean SHAP value for each feature
        mean_shap_values = all_shap_values.mean().sort_values(ascending=False)

        # Select top n features
        top_n_shap_values = mean_shap_values.head(n_top_features)

        # Plot mean SHAP values of top n features
        fig, ax = plt.subplots(figsize=(10, 7))
        top_n_shap_values.plot(kind='bar', ax=ax)
        plt.title("Mean SHAP Values (Top {} Features)".format(n_top_features))
        plt.xlabel('Feature')
        plt.ylabel('Mean SHAP Value')

        # Log the SHAP values plot to wandb
        wandb.log({"shap_values": wandb.Image(fig)})

def analyze_cv_models_multiclass(cv_models, condition_map, h2o_frame, n_top_features=10):
    confusion_matrices = []
    variable_importances = []
    # For each cross-validation model...
    for i, cv_model in enumerate(cv_models):
        print(f"\nCross-validation model {i + 1}:\n")

        # Get the variable importances
        var_imp = cv_model.varimp(use_pandas=True)
        variable_importances.append(var_imp)
        # Convert the H2OFrame to pandas DataFrame
        h2o_frame_pd = h2o_frame.as_data_frame()

        # Extract validation_data as pandas DataFrame
        validation_data = h2o_frame_pd[h2o_frame_pd["fold_column"] == i]

        # If you need to convert it back to H2OFrame
        validation_h2o_frame = h2o.H2OFrame(validation_data)
        # Get the confusion matrix
        train_performance = cv_model.model_performance(validation_h2o_frame)
        cm = train_performance._metric_json["cm"]['table'].cell_values
        # Convert to numpy array and ignore the last row and column as they're totals
        cm = np.array([row[0:-2] for row in cm[:-1]], dtype=float)
        confusion_matrices.append(cm)

        print("Confusion Matrix:")
        print(cm)

    # Sum the confusion matrices
    total_confusion_matrix = np.sum(confusion_matrices, axis=0)

    # Plot the total confusion matrix
    fig, ax = plt.subplots(figsize=(10, 7))
    # Get the classes

    # Map classes to your custom labels
    labels = list(condition_map.values())

    # Use labels for the xticklabels and yticklabels of your confusion matrix
    sns.heatmap(total_confusion_matrix, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Total Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust the font size here
    plt.yticks(rotation=45, ha='right', fontsize=10)  # Adjust the font size here

    plt.tight_layout()

    # Log the confusion matrix plot to wandb
    wandb.log({"total_confusion_matrix": wandb.Image(fig)})

    # Concatenate all variable importances DataFrames
    all_var_imp = pd.concat(variable_importances)

    # Average the variable importances
    avg_var_imp = all_var_imp.groupby("variable").mean().reset_index()

    # Sort by average scaled_importance and take top n features
    top_n_var_imp = avg_var_imp.sort_values("relative_importance", ascending=False).head(n_top_features)

    # Plot the top n features
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x="relative_importance", y="variable", data=top_n_var_imp, ax=ax, palette="Blues_d")
    plt.title("Top {} Features by Relative Importance".format(n_top_features))
    plt.xlabel('Average Relative Importance')
    plt.ylabel('Feature')
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust the font size here
    plt.yticks(rotation=45, ha='right', fontsize=10)  # Adjust the font size here

    plt.tight_layout()

    # Log the feature importance plot to wandb
    wandb.log({"feature_importance": wandb.Image(fig)})

    # Initialize list for collecting SHAP values
    shap_values_list = []

    for i, cv_model in enumerate(cv_models):
        print(f"\nCross-validation model {i + 1}:\n")

        # Check if the model supports SHAP values calculation
        if not hasattr(cv_model, "shap_explain_frame"):
            print("Model does not support SHAP value calculation. Skipping...")
            continue

        # Get validation data for the current fold
        h2o_frame_pd = h2o_frame.as_data_frame()

        # Extract validation_data as pandas DataFrame
        validation_data = h2o_frame_pd[h2o_frame_pd["fold_column"] == i]

        # If you need to convert it back to H2OFrame
        validation_h2o_frame = h2o.H2OFrame(validation_data)

        # Get SHAP values for the validation data of the current fold
        shap_values = cv_model.shap_explain_frame(validation_h2o_frame)
        shap_values_list.append(shap_values)

    if not hasattr(cv_model, "shap_explain_frame"):
        print("Model does not support SHAP value calculation. Skipping...")
    else:
        # Concatenate all SHAP values DataFrames
        all_shap_values = pd.concat(shap_values_list)

        # Calculate the mean SHAP value for each feature
        mean_shap_values = all_shap_values.mean().sort_values(ascending=False)

        # Select top n features
        top_n_shap_values = mean_shap_values.head(n_top_features)

        # Plot mean SHAP values of top n features
        fig, ax = plt.subplots(figsize=(10, 7))
        top_n_shap_values.plot(kind='bar', ax=ax)
        plt.title("Mean SHAP Values (Top {} Features)".format(n_top_features))
        plt.xlabel('Feature')
        plt.ylabel('Mean SHAP Value')

        # Log the SHAP values plot to wandb
        wandb.log({"shap_values": wandb.Image(fig)})