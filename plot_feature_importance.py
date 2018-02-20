import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Author : Jessica Lee
# This plots the feature importances given a CSV of the list.

curr_path = os.path.dirname(os.path.realpath(__file__))
png_path = os.path.join(curr_path, "images/")

df =pd.read_csv("Feature_Importance_List_50.csv")
y_pos = np.arange(len(df['importance'][0:20]))
plt.barh(y_pos, df['importance'].values[0:20], alpha=0.3)
plt.yticks(y_pos, df['feature'][0:20], rotation = (0), fontsize = 11, ha='right')
plt.ylabel('Feature Importance Scores')
plt.title('Feature Importance')
plt.tight_layout()

features_png = "{0}FeatureImportancePlot_XGBoost1.png".format(png_path)
plt.savefig(features_png, dpi=150)

plt.clf()
