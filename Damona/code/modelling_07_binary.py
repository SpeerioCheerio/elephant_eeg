import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
import wandb
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from itertools import combinations
from utils import get_best_non_stacked_model, analyze_cv_models
h2o.init()


class ModelTrainer:
    def __init__(self, data, folder, condition_map, conditions = [1,2]):
        self.data = data
        self.folder = folder
        self._reformat_df(conditions)
        self.condition_map = condition_map

    def _reformat_df(self, conditions):
        unique_rats = self.data['rat_id'].unique()
        fold_assignments = np.arange(len(unique_rats)) % 4  # change 4 to the desired number of folds
        np.random.shuffle(fold_assignments)
        rat_to_fold = dict(zip(unique_rats, fold_assignments))
        self.data['fold_column'] = self.data['rat_id'].map(rat_to_fold)
        # only select the 2 conditions for binary classification
        self.data = df[df['condition'].isin(conditions)]

    def train(self):
        # Convert DataFrame to H2OFrame
        h2o_frame = h2o.H2OFrame(self.data)
        h2o_frame['condition'] = h2o_frame['condition'].asfactor()
        h2o_frame['fold_column'] = h2o_frame['fold_column'].asfactor()

        # Define predictors and response variable
        predictors = h2o_frame.columns
        response = 'condition'
        predictors.remove(response)
        predictors.remove('session')


        aml_params = {'seed': 1, 'nfolds': 5, 'max_models': 10, }
        self.aml = H2OAutoML(**aml_params, keep_cross_validation_predictions=True,
                 keep_cross_validation_models=True,
                 keep_cross_validation_fold_assignment=True)

        # Train the model
        self.aml.train(x=predictors, y=response, training_frame=h2o_frame, fold_column='fold_column')

        # View the AutoML Leaderboard
        lb = self.aml.leaderboard.as_data_frame()

        return lb, h2o_frame

    def predict(self, test_data):
        h2o_test_frame = h2o.H2OFrame(test_data)
        preds = self.aml.predict(h2o_test_frame)
        return preds.as_data_frame()

    def save_explain_plots(self, obj):
        for key in obj.keys():
            print(f"saving {key} plots")
            if not isinstance(obj[key], dict) or not "plots" in obj[key]:
                if key == 'confusion_matrix':
                    cell_values = obj[key]['subexplanations'][self.aml.leader.model_id]['plots'][self.aml.leader.model_id].table.cell_values
                    column_headers = obj[key]['subexplanations'][self.aml.leader.model_id]['plots'][self.aml.leader.model_id].table.col_header
                    # Create a DataFrame
                    df = pd.DataFrame(cell_values, columns=column_headers)

                    # Set the index to the first column
                    df.set_index('', inplace=True)

                    # Select only the numeric data for the confusion matrix
                    cm = df.iloc[:-1, :2]  # Exclude the 'Total' row and 'Error' and 'Rate' columns

                    # Convert to float
                    cm = cm.astype(float)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, cmap='YlGnBu', fmt='g', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')

                    # Define your path
                    fig_path = f"{self.folder}/{key}/{key}.png"

                    # Save the figure
                    fig.savefig(fig_path)
                    wandb.log({key: wandb.Image(fig_path)})
                    plots = None


                else:
                    continue
            else:
                plots = obj[key]["plots"]
            if plots is None or not isinstance(plots, dict):
                continue
            os.makedirs(f"{self.folder}/{key}", exist_ok=True)
            for plot in plots.keys():
                if hasattr(plots[plot], 'figure'):
                    fig = plots[plot].figure()
                    # Save the Matplotlib figure
                    fig_path = f"{self.folder}/{key}/{plot}.png"
                    fig.savefig(fig_path)
                    print(key,fig_path)

                    wandb.log({key: wandb.Image(fig_path)})

    def explain_plot(self, test_data):
        explanation = h2o.explain(self.aml.leader, test_data, render=False,
                                  include_explanations=
        ["confusion_matrix",
        "learning_curve",
        "varimp",
        "varimp_heatmap",
        "shap_summary",
        "ice"])

        self.save_explain_plots(explanation)

def train_and_evaluate_all_pairs(condition_map, df, save_dir, n_top_features=10):
    # Get all pairs of conditions
    all_pairs = list(combinations(condition_map.keys(), 2))

    for pair in all_pairs:
        conditions = list(pair)

        # Initialize wandb run
        wandb.init(
            # Set the project where this run will be logged
            project='DMNA_binary_CRL',
            # Track hyperparameters and run metadata
            config={
                "intervals": 'False',
                "cross_subject_validation": 'True',
            })

        selected_conditions = {key: condition_map[key] for key in conditions}
        trainer = ModelTrainer(df, save_dir, condition_map, conditions=conditions)
        leaderboard, h2o_frame = trainer.train()

        # Use the function
        best_non_stacked_model = get_best_non_stacked_model(trainer.aml)
        cv_models = best_non_stacked_model.cross_validation_models()
        analyze_cv_models(cv_models, condition_map, h2o_frame, conditions, n_top_features=10)

        # Convert leaderboard to pandas DataFrame
        leaderboard_df = pd.DataFrame(leaderboard)

        # Get the first and second items.
        first_item = selected_conditions[conditions[0]]
        second_item = selected_conditions[conditions[1]]

        wandb.log({"target_1": first_item, "target_2": second_item})
        wandb.log({"best_auc": leaderboard_df['auc'][0], 'best model':leaderboard_df['model_id'][0]})

        # Log leaderboard
        wandb.log({"leaderboard": leaderboard_df})
        wandb.finish()

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("/home/lucky/PycharmProjects/Damona/data/crl_results/result_crl_agg.csv")

    # Directory to save plots
    save_dir = "tmp/"

    cro = "crl"

    if cro == 'biotrial':
        condition_map = {
            0: 'Saline',
            1: 'Vehicle (5 mL/kg, ip)',
            2: 'DPX-101 (5 mg/kg, ip)',
            3: 'DPX-101 (15 mg/kg, ip)',
            4: 'Basmisanil (5 mg/kg, ip)',
            5: 'MRK016 (3 mg/kg, ip)',
            6: 'Diazepam (2 mg/kg, ip)'
        }
    elif cro == 'crl':
        condition_map = {
            1: 'Vehicle',
            2: 'DPX-101 1(mg / mL)',
            3: 'DPX-101 3(mg / mL)',
            4: 'DPX-101 10(mg / mL)',
            5: 'Diazepam 0.5(mg / mL)',
            6: 'Diazepam 1(mg / mL)'
        }

    train_and_evaluate_all_pairs(condition_map, df, save_dir)
