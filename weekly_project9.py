import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

# Load the dataset
df = pd.read_csv('Placement_Data_Full_Class.csv')

# Inspect the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)  # Verify column names

# Encode categorical variables using one-hot encoding, but keep 'status' separately
df_encoded = pd.get_dummies(df.drop(columns=['status']), drop_first=True)

# Add the 'status' column back to the encoded dataframe
df_encoded['status'] = df['status']

# Calculate the correlation matrix
df_numeric = df_encoded.select_dtypes(include=[np.number])
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Analyze the influence of each factor using the original dataframe
for column in df.select_dtypes(include=[np.number]).columns:
    if column != 'status':
        sns.boxplot(x='status', y=column, data=df)
        plt.title(f'Boxplot of {column} by Placement Status')
        plt.show()

# Statistical tests
# Example for t-test
placed = df[df['status'] == 'Placed']
not_placed = df[df['status'] == 'Not Placed']
for column in df.select_dtypes(include=[np.number]).columns:
    if column != 'status':
        stat, p = ttest_ind(placed[column], not_placed[column])
        print(f'{column}: p-value={p}')

# Example for chi-square test
for column in df.select_dtypes(include=['object', 'category']):
    contingency_table = pd.crosstab(df[column], df['status'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f'{column}: p-value={p}')

# Visualize the demand for each specialization
sns.countplot(x='specialisation', hue='status', data=df)
plt.title('Demand for Each Specialisation by Placement Status')
plt.show()

# Analyze placement rate per specialization
placement_rate = df.groupby('specialisation')['status'].value_counts(normalize=True).unstack()
print(placement_rate)

# Statistical test for specialization
contingency_table = pd.crosstab(df['specialisation'], df['status'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f'Specialisation: p-value={p}')
