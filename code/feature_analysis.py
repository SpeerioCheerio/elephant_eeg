import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

class DataAnalysis:
    def __init__(self, feature_file):
        self.feature_file = feature_file
        self.condition_map = {
            '0': 'Saline',
            '1': 'Vehicle (5 mL/kg, ip)',
            '2': 'AC101 (5 mg/kg, ip)',
            '3': 'AC101 (15 mg/kg, ip)',
            '4': 'Basmisanil (5 mg/kg, ip)',
            '5': 'MRK016 (3 mg/kg, ip)',
            '6': 'Diazepam (2 mg/kg, ip)'
        }

    def load_features(self):
        df = pd.read_csv(self.feature_file)
        df = df[df['stage'] == 'manipulation']
        df['condition'] = df['condition'].astype(str).map(self.condition_map)
        return df

    def do_stats_and_plot_violin(self, df, numerical_column):
        conditions = df['condition'].unique()
        p_values = []

        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                group1 = df[df['condition'] == conditions[i]][numerical_column]
                group2 = df[df['condition'] == conditions[j]][numerical_column]
                _, p_val = mannwhitneyu(group1, group2)
                p_values.append(p_val)

        reject_list, corrected_p_values = multipletests(p_values, method='bonferroni')[:2]

        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(x="condition", y=numerical_column, data=df)

        y_min, y_max = ax.get_ylim()
        range_y = y_max - y_min

        counter = 0
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                if reject_list[counter]:
                    ax.plot([i, j], [y_max + 0.05*range_y*(counter+1), y_max + 0.05*range_y*(counter+1)], color='black')
                    ax.text((i+j)/2, y_max + 0.05*range_y*(counter+1), f'p={corrected_p_values[counter]:.2f}', horizontalalignment='center')
                counter += 1

        ax.set_ylim(y_min - 0.1*range_y, y_max + 0.05*range_y*(counter+1))  # Adjust y-axis limits
        ax.set_title(f"Violin Plot of {numerical_column} for each condition")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'tmp/violin_plot_{numerical_column}.png')
        #plt.show()


if __name__ == "__main__":
    da = DataAnalysis("/home/lucky/PycharmProjects/Damona/data/biotrial_results/neural_features.csv")
    data = da.load_features()
    for feature in ['AlphaBandPAF', 'CenterofGravity', 'iaf', 'a3a2_ratio', 'hurst_exp_Delta', 'hurst_exp_Theta', 'relative_Theta', 'relative_Delta','relative_Alpha','relative_Beta1']:
        da.do_stats_and_plot_violin(data, feature)

