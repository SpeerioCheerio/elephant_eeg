import h2o
from h2o.automl import H2OAutoML
import numpy as np
import wandb
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_best_non_stacked_model, analyze_cv_models_multiclass

h2o.init()

class ModelTrainer:
    def __init__(self, data, folder):
        self.data = data
        self.folder = folder
        self._add_fold_column()

    def _add_fold_column(self):
        unique_rats = self.data['rat_id'].unique()
        fold_assignments = np.arange(len(unique_rats)) % 4  # change 4 to the desired number of folds
        np.random.shuffle(fold_assignments)
        rat_to_fold = dict(zip(unique_rats, fold_assignments))
        self.data['fold_column'] = self.data['rat_id'].map(rat_to_fold)

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


        aml_params = {'seed': 1, 'nfolds': 5, 'max_models': 20}
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
                    cell_values = obj[key]['subexplanations'][self.aml.leader.model_id]['plots'][self.aml.leader.model_id].cell_values
                    column_headers = obj[key]['subexplanations'][self.aml.leader.model_id]['plots'][self.aml.leader.model_id].col_header
                    # Create a DataFrame
                    df = pd.DataFrame(cell_values, columns=column_headers)

                    cm = df.iloc[:7, :7]  # Select the first 7 rows and 7 columns

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


if __name__ == "__main__":

    cro = "biotrial"

    if cro == 'biotrial':
        condition_map = {
            0: 'Saline',
            1: 'Vehicle (5 mL/kg, ip)',
            2: 'AC101 (5 mg/kg, ip)',
            3: 'AC101 (15 mg/kg, ip)',
            4: 'Basmisanil (5 mg/kg, ip)',
            5: 'MRK016 (3 mg/kg, ip)',
            6: 'Diazepam (2 mg/kg, ip)'
        }
    elif cro == 'crl':
        condition_map = {
            1: 'Vehicle',
            2: 'AC101 1(mg / mL)',
            3: 'AC101 3(mg / mL)',
            4: 'AC101 10(mg / mL)',
            5: 'Diazepam 0.5(mg / mL)',
            6: 'Diazepam 1(mg / mL)'
        }

    # Load your data
    df = pd.read_csv("/home/lucky/PycharmProjects/Damona/data/biotrial_results/neural_features.csv")
    # Initialize wandb run
    wandb.init(
            # Set the project where this run will be logged
            project='DMNA',
            # Track hyperparameters and run metadata
            config={
                "intervals": 'False',
                "cross_subject_validation": 'True',
            })
    # Directory to save plots
    save_dir = "tmp/"

    trainer = ModelTrainer(df, save_dir)
    leaderboard, h2o_frame = trainer.train()

    # Use the function
    best_non_stacked_model = get_best_non_stacked_model(trainer.aml)
    cv_models = best_non_stacked_model.cross_validation_models()
    analyze_cv_models_multiclass(cv_models, condition_map, h2o_frame, n_top_features=10)

    # Convert leaderboard to pandas DataFrame
    leaderboard_df = pd.DataFrame(leaderboard)



    # Log leaderboard
    wandb.log({"target": 'multiclass'})
    wandb.log({"best_mean_per_class_error": leaderboard_df['mean_per_class_error'][0], 'best model':leaderboard_df['model_id'][0]})

    wandb.log({"leaderboard": leaderboard_df})

    # Log explanations
    #trainer.explain_plot(h2o_frame)
