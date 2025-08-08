import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Read the csv file
x = pd.read_csv("/home/lucky/R_projects/neural_features_intervals.csv")
print(x.isnull().sum())
print(x.dtypes)


# Filtering conditions
x = x[x['condition'].isin([1, 2, 3, 6])]
condition_order = ["1. Vehicle", "2. AC101 (5 mg/kg)", "3. AC101 (15 mg/kg)", "6. Diazepam (2 mg/kg)"]
# Reassigning conditions
condition_dict = {0: "0. Saline", 1: "1. Vehicle", 2: "2. AC101 (5 mg/kg)", 3: "3. AC101 (15 mg/kg)", 4: "4. Basmisanil (5 mg/kg)", 5: "5. MRK016 (3 mg/kg)", 6: "6. Diazepam (2 mg/kg)"}

x['condition'] = x['condition'].replace(condition_dict)


x.reset_index(drop=True, inplace=True)
x = x.sort_values(by="rat_id").reset_index(drop=True)




import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages


def plot_model(data, y):
    plt.figure(figsize=(10, 8))

    # Define the colors for the stages
    colors = { 'manipulation': "blue"}
    markers = {'manipulation': "o"}

    # Plot confidence intervals for the predictions
    for stage, color in colors.items():
        stage_data = data[data['stage'] == stage]
        sns.pointplot(x='condition', y=y, data=stage_data, dodge=True, ci=95, join=False, scale=1, color=color,
                      capsize=0.3, order=condition_order)

    # Plot individual data points for each stage with different markers
    for stage, marker in markers.items():
        stage_data = data[data['stage'] == stage]
        palette = [colors[stage]] * 9  # Repeat the same color for all rats
        sns.stripplot(x='condition', y=y, hue='rat_id', data=stage_data, dodge=True, linewidth=0.5, palette=palette,
                      marker=marker, order=condition_order)

    #plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.title(f'{y}')

    # Custom legend
    legend_elements = [mpatches.Patch(color='blue', label='Manipulation')]
    plt.legend(handles=legend_elements, loc='upper right')

    #plt.show()

# Create a pdf file
pdf_pages = PdfPages('plots_sleep_only_manip.pdf')

# Iterate over each column (excluding 'condition', 'stage' and 'rat_id') in the dataframe
for column_name in x.drop(['condition', 'stage', 'rat_id', 'BAD', 'session'], axis=1).columns:
    plot_model(x, column_name)
    pdf_pages.savefig(plt.gcf())  # Save the current figure in the pdf file
    plt.close()  # Close the figure after it's saved

# Close the pdf file
pdf_pages.close()


def plot_data(data, y):
    plt.figure(figsize=(10,8))

    # Define the markers for the stages
    markers = {'baseline': "s", 'manipulation': "X"}

    # Plot confidence intervals for the predictions
    sns.pointplot(x='condition', y=y, hue='stage', data=data, dodge=True, ci=95, join=False, scale=0.6)

    # Plot individual data points for each stage with different markers
    for stage, marker in markers.items():
        stage_data = data[data['stage'] == stage]
        sns.stripplot(x='condition', y=y, hue='rat_id', data=stage_data, dodge=True, linewidth=0.5, palette='Set2', marker=marker)

    # Place legend outside of plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.title(f'Predicted {y}')

    #plt.show()

# Create a pdf file
pdf_pages = PdfPages('plotsorig_sleep.pdf')

# Iterate over each column (excluding 'condition', 'stage' and 'rat_id') in the dataframe
for column_name in x.drop(['condition', 'stage', 'rat_id', 'BAD', 'session', 'interval', 'interval_start'], axis=1).columns:
    plot_data(x, column_name)
    pdf_pages.savefig(plt.gcf())  # Save the current figure in the pdf file
    plt.close()  # Close the figure after it's saved

# Close the pdf file
pdf_pages.close()