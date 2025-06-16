#!/usr/bin/env python
# coding: utf-8

# # **MSC 718: Final Project**
# # **Examining Substance Use Patterns Among Canadian Youth**
# # Group 12
# 
# **Group Members:** Anjiya Nooruddin, Chavvi Bhatia, Chirag Seth, Hrishita Sharma, Utkarsh Singh

# ###DATA LOADING###

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data=pd.read_csv('final_reduced_data.csv')
data.columns = ['ever_smoked', 'age_first_smoked', 'smoked_last_30_days','amt_smoked','ever_used_alcohol','ever_used_meth','ever_used_cocaine','grade','sex','household_income','social_influence','behavioral_factor','mental_health','bullying','urban_rural']


# In[ ]:


data


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


data.describe()


# In[ ]:


# Check for missing values
print("Missing Values in Each Column:")
print(data.isnull().sum())



# In[ ]:


# Check data types
print("Data Types:")
print(data.dtypes)



# ###EDA RESEARCH QUESTION 1###

# In[ ]:


# Histograms for numerical variables
numerical_vars = ['age_first_smoked', 'amt_smoked', 'grade', 'household_income', 'behavioral_factor', 'mental_health']
plt.figure(figsize=(15, 10))
for i, var in enumerate(numerical_vars, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[var], kde=True)
    plt.title(f'Distribution of {var}')
plt.tight_layout()
plt.show()

# Boxplots for numerical variables to detect outliers
plt.figure(figsize=(15, 10))
for i, var in enumerate(numerical_vars, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=data[var])
    plt.title(f'Boxplot of {var}')
plt.tight_layout()
plt.show()


# Age First Smoked: Most respondents who smoked started around age 14. The distribution is highly skewed towards this age, with very few outliers, suggesting a few cases where students smoked at unusually early or late ages.
# 
# Amount Smoked: Most respondents who smoked in the past 30 days consumed very few cigarettes per day. The distribution is highly right-skewed, with a long tail indicating a few individuals who smoked heavily.
# 
# Grade: The histogram has distinct peaks, showing that the dataset consists mainly of students from grades 7 to 12, with no major outliers.
# 
# Household Income: The distribution is right-skewed, with more students coming from lower-income areas, but some extreme values (above $120,000) appear as outliers.
# 
# Behavioral Factor: Most respondents belong to the highest category (smoking not allowed inside or outside). Some responses fall in other categories, but no major outliers.
# 
# Mental Health Perception (Risk of Smoking a Cigarette Occasionally): The distribution is multimodal, with peaks at different risk perceptions. Most respondents fall into "Great Risk" or "Moderate Risk" categories.
# 
# 
# 

# In[ ]:


## Chi-square test for bullying vs. ever_smoked
import pandas as pd
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(data['bullying'], data['ever_smoked'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print("Contingency Table (Bullying vs. Ever Smoked):")
print(contingency_table)
print(f"\nChi-square test results:")
print(f"Chi-square statistic = {chi2:.4f}")
print(f"Degrees of freedom = {dof}")
print(f"P-value = {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("\nResult: There is a significant association between bullying and ever smoking.")
else:
    print("\nResult: There is no significant association between bullying and ever smoking.")


# The Chi-square test result (χ² = 187.2893, p < 0.0001) indicates a significant association between bullying and ever smoking. This suggests that students who were physically bullied in the last 30 days are more likely to have smoked at least once.

# In[ ]:


# Violin Plot: Household Income vs. Ever Smoked

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.violinplot(x='ever_smoked', y='household_income', data=data, palette="muted")
plt.title("Violin Plot of Household Income by Smoking Status")
plt.show()


# The violin plot compares household income distributions for those who have ever smoked (1) versus those who haven't (2). Both groups show similar distributions, with a median around $60,000. However, the spread of incomes appears slightly broader for non-smokers, suggesting that smoking behavior may not be strongly dependent on income.

# In[ ]:


# Heatmap of Chi-Square Test Results (Categorical Variables)

import scipy.stats as stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt  # Added for plotting

df = data.copy()  # Create a copy to avoid modifying the original data

# Define categorical variables to compare against 'ever_smoked'
cat_vars = ['sex', 'grade', 'urban_rural', 'social_influence', 'bullying']
chi_results = {}

for var in cat_vars:
    crosstab = pd.crosstab(df[var], df['ever_smoked'])
    chi2, p, _, _ = stats.chi2_contingency(crosstab)
    chi_results[var] = p  # Store p-values

# Convert to DataFrame
chi_df = pd.DataFrame(chi_results, index=['p-value']).T

# Plot heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(chi_df, annot=True, cmap='coolwarm', cbar=False, linewidths=0.5)
plt.title("Chi-Square Test p-values for Tobacco Use")
plt.show()


# Sex: A significant association exists between sex and smoking behavior. This suggests that smoking prevalence may differ between males and females.
# 
# Grade: The relationship between grade and smoking is highly significant, meaning smoking behavior varies significantly across different grade levels.
# 
# Social Influence: A zero p-value confirms that peer/social influence plays a crucial role in smoking behavior, indicating that those who have smoked may have been influenced by their social circles.
# 
# Urban/Rural: The extremely small p-value indicates a strong association between location type (urban vs. rural) and smoking behavior. Smoking prevalence likely differs significantly between these two environments.
# 
# Bullying: A strong correlation exists between bullying and smoking, suggesting that individuals who have experienced bullying may be more likely to smoke.
# 

# In[ ]:


# Violin Plot: Mental Health vs. Tobacco Use

import seaborn as sns
import matplotlib.pyplot as plt  # Ensure matplotlib is imported

# Create violin plot
plt.figure(figsize=(8, 5))
sns.violinplot(x="ever_smoked", y="mental_health", data=data, inner="quartile", palette="muted")

# Add title and labels
plt.title("Mental Health Scores vs. Smoking Status")
plt.xlabel("Ever Smoked")
plt.ylabel("Mental Health Score")

# Show plot
plt.show()


# The violin plot suggests that individuals who have ever smoked tend to perceive smoking as less risky, while those who never smoked report higher risk perceptions. This indicates a potential link between smoking history and attitudes toward its harm, possibly influenced by pre-existing beliefs or desensitization through experience.

# In[ ]:


# Alcohol use by grade

import matplotlib.pyplot as plt
import seaborn as sns

# Fix incorrect values if necessary
data["ever_used_alcohol"] = data["ever_used_alcohol"].replace({-1: 1, -2: 2})  # If needed

# Define correct labels for tobacco use
tobacco_labels = {1: "Yes (Uses Tobacco)", 2: "No (Does Not Use)"}

plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data,
    x="grade",
    hue="ever_used_alcohol",
    multiple="stack",
    palette=["#d62728", "#1f77b4"],  # Red for Yes, Blue for No
    discrete=True
)

# Set title and labels
plt.title("Alcohol Use by Grade", fontsize=14)
plt.xlabel("Grade", fontsize=12)
plt.ylabel("Count", fontsize=12)


plt.show()


# The histogram shows that alcohol use increases with grade level, with higher grades having a larger proportion of students who have consumed alcohol (red). Younger students (grades 7–9) have a higher percentage of non-drinkers (blue), but this decreases in later grades. This suggests that as students grow older, they are more likely to experiment with or use alcohol.
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


# Social influence vs ever smoked

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data = pd.read_csv('final_reduced_data.csv')

# Rename columns
data.columns = ['ever_smoked', 'age_first_smoked', 'smoked_last_30_days', 'amt_smoked',
                'ever_used_alcohol', 'ever_used_meth', 'ever_used_cocaine', 'grade',
                'sex', 'household_income', 'social_influence', 'behavioral_factor',
                'mental_health', 'bullying', 'urban_rural']

# Create a contingency table
contingency_table = pd.crosstab(data['ever_smoked'], data['social_influence'], normalize='index') * 100
print("Percentage Distribution:")
print(contingency_table)

# Perform Chi-square test
chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(data['ever_smoked'], data['social_influence']))

print("\nChi-Square Test Results:")
print(f"Chi-square statistic: {chi2}")
print(f"p-value: {p_value}")

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Percentage Distribution of Social Influence by Smoking Status')
plt.xlabel('Social Influence')
plt.ylabel('Ever Smoked')
plt.tight_layout()
plt.show()

# Detailed breakdown
print("\nDetailed Breakdown:")
# Calculate the percentage of each social influence category for smokers vs non-smokers
social_influence_breakdown = data.groupby('ever_smoked')['social_influence'].value_counts(normalize=True).unstack() * 100
print(social_influence_breakdown)

# Compare proportions to identify directional relationship
smokers = data[data['ever_smoked'] == 2]
non_smokers = data[data['ever_smoked'] == 1]

smokers_social_influence_yes = (smokers['social_influence'] == 1).mean() * 100
non_smokers_social_influence_yes = (non_smokers['social_influence'] == 1).mean() * 100

print("\nSocial Influence Comparison:")
print(f"Percentage of smokers influenced socially: {smokers_social_influence_yes:.2f}%")
print(f"Percentage of non-smokers influenced socially: {non_smokers_social_influence_yes:.2f}%")


# There's a profound link between social influence and smoking behavior: 99.98% of smokers reported being socially influenced to smoke, compared to only 54.92% of non-smokers, suggesting that social factors play a critical role in smoking initiation, with social influence being almost universally present among those who smoke.

# In[ ]:


substances = [
    'ever_smoked',
    'ever_used_alcohol',
    'ever_used_meth',
    'ever_used_cocaine'
]

stacked_data = data.groupby('grade')[substances].mean() * 100

# Substance use labels
substance_labels = {
    'ever_smoked': 'Smoking',
    'ever_used_alcohol': 'Alcohol Use',
    'ever_used_meth': 'Meth Use',
    'ever_used_cocaine': 'Cocaine Use'
}

# Create a figure with multiple visualization techniques
plt.figure(figsize=(20, 15))

# Grouped Bar Chart
plt.subplot(2, 2, 2)
stacked_data.plot(kind='bar', ax=plt.gca())
plt.title('Substance Use Comparison by Grade')
plt.xlabel('Grade')
plt.ylabel('Percentage of Use')
plt.legend(title='Substances', labels=[substance_labels[s] for s in substances], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


# Smoking is the most prevalent substance across all grades, particularly in lower grades. Smoking percentages are highest in grades 7 and 8 and gradually decline in higher grades. Alcohol use remains consistently moderate, while hard drugs like meth and cocaine maintain low, stable levels throughout the educational years.

# ###EDA RESEARCH QUESTION 2###

# In[ ]:


# Box plot of Social Influence on Age First Smoked
import seaborn as sns
sns.boxplot(x=data['social_influence'], y=data['age_first_smoked'])


# In[ ]:


# Bar Plot of Probability of ever used substance vs Bullying
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(data)

# Convert binary values (1,2) to (0,1) for probability calculation
df['ever_smoked'] = df['ever_smoked'] - 1
df['ever_used_alcohol'] = df['ever_used_alcohol'] - 1

plt.figure(figsize=(12, 5))
for i, col in enumerate(['ever_smoked', 'ever_used_alcohol']):
    plt.subplot(1, 3, i+1)
    sns.barplot(x=df['social_influence'], y=df[col], ci=None)
    plt.ylabel("Probability of Use")
    plt.xlabel("Social Influence")
    plt.title(f"Social Influence vs {col}")

plt.tight_layout()
plt.show()


# In[ ]:


# Bar Plot of Mean meth Use by Bullying

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(data)

# Convert binary values (1,2) to (0,1) for probability calculation
df['ever_used_meth'] = df['ever_used_meth'] - 1

plt.figure(figsize=(12, 5))
for i, col in enumerate(['ever_used_meth']):
    plt.subplot(1, 3, i+1)
    sns.barplot(x=df['bullying'], y=df[col], ci=None)
    plt.ylabel("Probability of Use")
    plt.xlabel("Bullying")
    plt.title(f"Bullying vs {col}")

plt.tight_layout()
plt.show()


# In[ ]:


# Chi-square test for social_influence vs. ever_smoked
from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(data['social_influence'], data['ever_smoked'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square test between social_influence and ever_smoked: p-value = {p}")
print(contingency_table)


# In[ ]:


# Chi-square test for social_influence vs. ever_used_alcohol
from scipy.stats import chi2_contingency
contingency_table_alcohol = pd.crosstab(data['social_influence'], data['ever_used_alcohol'])
chi2_alcohol, p_alcohol, dof_alcohol, expected_alcohol = chi2_contingency(contingency_table_alcohol)
print(f"Chi-square test between social_influence and ever_used_alcohol: p-value = {p_alcohol}")
print(contingency_table_alcohol)

""" individuals with peer influence (social_influence = 1) are more likely to have used alcohol. However, in absence of social influence (social_influence = 2), we see that individuals who have consumed alcohol are higher. This is an unusual trend as we expected in absence of social influence, individuals are less likely to have used alcohol. This indicates that there are other factors that contribute to alcohol consumption apart from social influence. """


# In[ ]:


# Interaction Plot: Bullying × Sex on Tobacco Use
# data["amt_smoked"] = 1-data["amt_smoked"]
sns.pointplot(x="bullying", y="amt_smoked", hue="sex", data=data, dodge=True, markers=["o", "s"], palette="Set2")
plt.title("Interaction: Bullying & Sex on amt_smoked")
plt.xlabel("Bullying Level")
plt.ylabel("Average amt_smoked")
plt.legend(title="Sex")
plt.show()

"""Bullying strongly increases amount smoked, with male smoking more under bullying as compared to females"""
"""In the absence of peer influence, the overall amount of smoking is lower, and males tend to smoke more than females."""
#


# In[ ]:


# FacetGrid for Grade wise distribution of social influence on ever_smoked
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df = pd.DataFrame(data)

g = sns.FacetGrid(df, col="grade", col_wrap=3, height=4, sharey=False)

g.map(sns.countplot, "social_influence", hue="ever_smoked", palette="Set2", hue_order=[1, 2], data=df)

g.set_axis_labels("Social Influence (1 = Yes, 2 = No)", "Count of Individuals")
g.set_titles(col_template="{col_name} Grade")
g.fig.suptitle("Social Influence vs Ever Smoked by Grade", fontsize=16)
g.fig.subplots_adjust(top=0.9)

g.add_legend(title="Ever Smoked", labels=["Yes", "No"])

plt.show()




# ###STATISTICAL MODELS###

# In[ ]:


import pandas as pd

# Check the Drive location (this is where the dataset is stored)
get_ipython().system('ls /content/drive/MyDrive/')




# In[ ]:


# Install required libraries (if not already installed)
get_ipython().system('pip install pymc')
get_ipython().system('pip install arviz')


# #Model 1: Bayesian Logistic Regression (Baseline)

# In[ ]:


# Import libraries
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score


# In[ ]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load and Prepare Data
data = pd.read_csv('final_reduced_data.csv')
data.columns = ['ever_smoked', 'age_first_smoked', 'smoked_last_30_days', 'amt_smoked', 'ever_used_alcohol',
                'ever_used_meth', 'ever_used_cocaine', 'grade', 'sex', 'household_income',
                'social_influence', 'behavioral_factor', 'mental_health', 'bullying', 'urban_rural']

# Preprocess the data
data['smoked_last_30_days_binary'] = data['smoked_last_30_days'].map({1: 1, 2: 0, 99: np.nan})  # Yes -> 1 (user), No -> 0 (non-user), Not Stated -> NaN
data['sex_binary'] = data['sex'].map({1: 0, 2: 1})  # Female -> 0, Male -> 1
data['grade_std'] = (data['grade'] - data['grade'].mean()) / data['grade'].std()
data['household_income_std'] = (data['household_income'] - data['household_income'].mean()) / data['household_income'].std()

# Handle missing values after recoding
data = data.dropna(subset=['smoked_last_30_days_binary', 'sex_binary', 'grade_std', 'household_income_std'])

# Step 3: Define the Bayesian Logistic Regression Model
with pm.Model() as model:
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    beta_grade = pm.Normal('beta_grade', mu=0, sigma=10)
    beta_sex = pm.Normal('beta_sex', mu=0, sigma=10)
    beta_income = pm.Normal('beta_income', mu=0, sigma=10)

    logit_p = (intercept +
               beta_grade * data['grade_std'] +
               beta_sex * data['sex_binary'] +
               beta_income * data['household_income_std'])

    # Define the likelihood and ensure log-likelihood is stored
    tobacco_use = pm.Bernoulli('tobacco_use', logit_p=logit_p, observed=data['smoked_last_30_days_binary'])

    # Sample from the posterior
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

# Step 4: Compute Predicted Probabilities and AUC
posterior = trace.posterior
# Extract scalar values for the posterior means
intercept_mean = posterior['intercept'].mean().values.item()
beta_grade_mean = posterior['beta_grade'].mean().values.item()
beta_sex_mean = posterior['beta_sex'].mean().values.item()
beta_income_mean = posterior['beta_income'].mean().values.item()

# Compute logit_p_mean using scalar means
logit_p_mean = (intercept_mean +
                beta_grade_mean * data['grade_std'] +
                beta_sex_mean * data['sex_binary'] +
                beta_income_mean * data['household_income_std'])

probs = 1 / (1 + np.exp(-logit_p_mean))
auc = roc_auc_score(data['smoked_last_30_days_binary'], probs)
print(f"AUC for Bayesian Logistic Regression: {auc:.4f}")

# Step 5: Compute Log-Likelihood for WAIC and LOO
with model:
    pm.compute_log_likelihood(trace)

# Step 6: Compute WAIC and LOO
waic = az.waic(trace)
print("WAIC:", waic)

loo = az.loo(trace)
print("LOO:", loo)



# 
# 
# The Bayesian Logistic Regression model, applied to predict recent tobacco use among 62,850 students, achieved an AUC of 0.7119, indicating moderate discriminatory power. The model’s fit was assessed using WAIC and LOO metrics, computed from 2,000 posterior samples. The WAIC estimate was -13901.97 (SE 155.19), with a penalty term (p_waic) of 3.88, while the LOO estimate was -13901.98 (SE 155.19), with a penalty term (p_loo) of 3.89, both suggesting a reasonable fit to the data. The Pareto k diagnostic values further confirmed the reliability of the LOO estimate, with all 62,850 observations falling in the "good" range (Pareto k ≤ 0.7), indicating no influential outliers and supporting the model’s adequacy for probabilistic inference in this context

# In[ ]:


# Summarize the posterior
print(az.summary(trace, var_names=['intercept', 'beta_grade', 'beta_sex', 'beta_income']))


# A one-unit increase in standardized GRADE (GRADE_std) increases the log-odds of smoking in the last 30 days (DVLAST30_binary) by 0.759,
# suggesting that students in higher grades (e.g., Grade 12 vs. Grade 7) are more likely to have smoked in the past 30 days.
# Being male (SEX_binary = 1) increases the log-odds of smoking in the last 30 days (DVLAST30_binary) by 0.122 compared to being female (SEX_binary = 0),
# indicating a slightly higher likelihood of recent smoking among male students.
# A one-unit increase in standardized area median household income (DVHRINC2_std) decreases the log-odds of smoking in the last 30 days (DVLAST30_binary) by 0.180,
# indicating that students in areas with higher median household incomes are less likely to have smoked in the past 30 days.
# The 3rd and 97th percentiles of the Highest Density Interval (HDI) represent a 94% credible interval for each parameter,
# providing a range within which the true parameter value is likely to lie with 94% probability.
# mcse_mean and mcse_sd represent the Monte Carlo Standard Error for the mean and standard deviation, respectively,
# measuring the error due to finite sampling in the MCMC process.
# Very small values (e.g., 0.001 or 0.000) for mcse_mean and mcse_sd indicate that the posterior samples are stable and reliable,
# suggesting that the MCMC sampling process has converged effectively.

# In[ ]:


# Plot the posterior distributions
az.plot_posterior(trace, var_names=['intercept', 'beta_grade', 'beta_sex', 'beta_income'], hdi_prob=0.94)
plt.suptitle('Posterior Distributions of Bayesian Logistic Regression Parameters', fontsize=16, y=1.05)
plt.show()


# Intercept
# Mean: -3.0
# The distribution is sharply peaked, indicating high certainty in the estimate. The negative value confirms a low baseline probability of smoking in the last 30 days (DVLAST30_binary) for a student with average standardized grade (GRADE_std), female sex (SEX_binary = 0), and average standardized area median household income (DVHRINC2_std).
# 
# Beta_grade
# Mean: 0.76
# The distribution is positive and does not include 0, confirming that higher standardized grades (GRADE_std) significantly increase the likelihood of smoking in the last 30 days (DVLAST30_binary). This suggests that students in higher grades (e.g., Grade 12 vs. Grade 7) are more likely to have smoked recently.
# 
# Beta_sex
# Mean: 0.12
# The distribution is centered slightly above 0, indicating a small but significant effect of sex (males, SEX_binary = 1, are more likely to have smoked in the last 30 days than females, SEX_binary = 0).
# 
# Beta_income
# Mean: -0.18
# The distribution is negative and does not include 0, confirming that a higher standardized area median household income (DVHRINC2_std) significantly decreases the likelihood of smoking in the last 30 days (DVLAST30_binary). This suggests that students in areas with higher median household incomes are less likely to have smoked recently.

# 

# #Model 2: Lasso Regression

# In[ ]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Step 2: Load and Prepare Data
data = pd.read_csv('final_reduced_data.csv')
data.columns = ['SS_010', 'SS_020', 'DVLAST30', 'DVAMTSMK', 'ALC_010',
                'MET_010', 'COC_010', 'GRADE', 'SEX', 'DVHRINC2',
                'SS_030', 'behavioral_factor', 'mental_health', 'BUL_060', 'urban_rural']

# Recode DVLAST30 to 0 (non-user) and 1 (user)
data['DVLAST30_binary'] = data['DVLAST30'].map({1: 1, 2: 0, 99: np.nan})  # Yes -> 1 (user), No -> 0 (non-user), Not Stated -> NaN

# Check for missing values and drop them
print("Missing Values:\n", data.isnull().sum())
data = data.dropna()

# Define predictors and target
predictors = ['GRADE', 'SEX', 'DVHRINC2', 'SS_030', 'mental_health', 'BUL_060', 'urban_rural']
X = data[predictors]
y = data['DVLAST30_binary']

# Check class distribution
print("\nClass distribution for DVLAST30_binary:\n", y.value_counts(normalize=True))

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Fine-Tune Lasso-Penalized Logistic Regression
lasso_model = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    random_state=42,
    max_iter=1000,
    class_weight='balanced'  # Handle potential imbalance
)

# Define parameter grid for tuning
param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # Expanded range
}

# Perform GridSearchCV
grid_search = GridSearchCV(
    lasso_model,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    error_score=0  # Handle non-finite scores
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"\nBest hyperparameters: {grid_search.best_params_}")
print(f"Best cross-validated AUC score: {grid_search.best_score_:.4f}")

# Step 5: Evaluate the Model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nEvaluation for DVLAST30_binary:")
print(f"Unique values in y_test: {np.unique(y_test)}")
print(f"Unique values in y_pred: {np.unique(y_pred)}")
print(classification_report(y_test, y_pred, zero_division=1))

# ROC-AUC score
if len(np.unique(y_test)) > 1:
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {auc_score:.4f}")
else:
    print("ROC-AUC Score: Skipped (only one class in y_test)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-User', 'User'], yticklabels=['Non-User', 'User'])
plt.title('Confusion Matrix for DVLAST30_binary')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 6: Extract and Interpret Coefficients
print("\nCoefficients for DVLAST30_binary:")
coef_df = pd.DataFrame({
    'Feature': predictors,
    'Coefficient': best_model.coef_[0]
})
coef_df['Odds Ratio'] = np.exp(coef_df['Coefficient'])
coef_df = coef_df[coef_df['Coefficient'] != 0]  # Show non-zero coefficients
print(coef_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.axvline(x=0, color='gray', linestyle='--')
plt.title('Lasso Coefficients for DVLAST30_binary (Non-Zero Only)')
plt.xlabel('Coefficient')
plt.show()

# Interpretation of Results
print("""
# The model was fine-tuned with GridSearchCV, optimizing the regularization parameter C over a range of values [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0].
# The best model achieved a cross-validated AUC score of 0.7961 with C = 1.0, indicating good discriminatory power between students who smoked in the last 30 days (DVLAST30_binary = 1) and those who did not (DVLAST30_binary = 0).

# The confusion matrix shows that out of 11,765 actual non-users (DVLAST30_binary = 0), 8,269 were correctly predicted as non-users (true negatives), while 3,496 were misclassified as users (false positives). For the 805 actual users (DVLAST30_binary = 1), 571 were correctly predicted as users (true positives), but 234 were misclassified as non-users (false negatives). The ROC-AUC score of 0.7866 on the test set confirms the model’s ability to distinguish between classes, though the class imbalance (many more non-users than users) impacted overall performance.

# Lasso regularization selected seven key predictors with non-zero coefficients:
# - GRADE (0.569338, odds ratio: 1.767096): A one-unit increase in grade level increases the odds of smoking in the last 30 days by a factor of 1.77, indicating that older students are more likely to smoke.
# - SEX (0.023242, odds ratio: 1.023514): Being male (SEX = 2) slightly increases the odds of smoking compared to being female (SEX = 1), though the effect is small.
# - DVHRINC2 (-0.080012, odds ratio: 0.923088): A one-unit increase in the area’s median household income decreases the odds of smoking by a factor of 0.92, suggesting that students in higher-income areas are slightly less likely to smoke.
# - SS_030 (-7.025087, odds ratio: 0.000889): A one-unit increase in the number of friends who smoke (social influence) dramatically decreases the odds of smoking in the last 30 days, which is unexpected and may indicate an issue with the variable’s coding or interpretation (e.g., reverse coding or multicollinearity).
# - mental_health (-0.483779, odds ratio: 0.616449): Poorer mental health (higher values) decreases the odds of smoking by a factor of 0.62, suggesting a protective effect, which may need further investigation.
# - BUL_060 (-1.279050, odds ratio: 0.278302): More frequent bullying experiences decrease the odds of smoking by a factor of 0.28, indicating that bullied students are less likely to smoke.
# - urban_rural (0.543291, odds ratio: 1.721663): Living in an urban area (urban_rural = 1) increases the odds of smoking by a factor of 1.72 compared to rural areas (urban_rural = 0).
# Notably, SS_030 (social influence) had the largest coefficient in magnitude (-7.025087), but its negative direction is surprising and warrants further investigation into the variable’s construction or potential data issues.
""")


# In[ ]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Step 2: Load and Prepare Data
data = pd.read_csv('final_reduced_data.csv')
data.columns = ['ever_smoked', 'age_first_smoked', 'smoked_last_30_days', 'amt_smoked', 'ever_used_alcohol',
                'ever_used_meth', 'ever_used_cocaine', 'grade', 'sex', 'household_income',
                'social_influence', 'behavioral_factor', 'mental_health', 'bullying', 'urban_rural']

# Inspect raw values
print("Raw unique values in ever_used_alcohol:\n", data['ever_used_alcohol'].unique())
print("Raw value counts in ever_used_alcohol:\n", data['ever_used_alcohol'].value_counts(dropna=False))

# Recode ever_used_alcohol: 1 -> 0 (never used), 2 -> 1 (ever used), 99 -> NaN (not stated)
data['ever_used_alcohol'] = data['ever_used_alcohol'].map({1: 0, 2: 1, 99: np.nan})

# Check for missing values and drop them
print("\nMissing Values:\n", data.isnull().sum())
data = data.dropna()

# Check class distribution after recoding
print("\nClass distribution for ever_used_alcohol after recoding:\n",
      data['ever_used_alcohol'].value_counts(normalize=True))

# Define predictors and target
predictors = ['grade', 'sex', 'household_income', 'social_influence', 'mental_health', 'bullying', 'urban_rural']
X = data[predictors]
y = data['ever_used_alcohol']

# Verify two classes exist
if len(y.unique()) < 2:
    print("\nError: ever_used_alcohol contains only one class after recoding. Cannot perform classification.")
    print("Raw data may have been altered. Please verify.")
else:
    # Step 3: Split the Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Fine-Tune Lasso-Penalized Logistic Regression
    lasso_model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handle potential imbalance
    )

    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    grid_search = GridSearchCV(
        lasso_model,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        error_score=0
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"\nBest hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validated AUC score: {grid_search.best_score_:.4f}")

    # Step 5: Evaluate the Model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nEvaluation for ever_used_alcohol:")
    print(f"Unique values in y_test: {np.unique(y_test)}")
    print(f"Unique values in y_pred: {np.unique(y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=1))

    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {auc_score:.4f}")
    else:
        print("ROC-AUC Score: Skipped (only one class in y_test)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Never Used', 'Ever Used'], yticklabels=['Never Used', 'Ever Used'])
    plt.title('Confusion Matrix for ever_used_alcohol')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Step 6: Extract and Interpret Coefficients
    print("\nCoefficients for ever_used_alcohol:")
    coef_df = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': best_model.coef_[0]
    })
    coef_df['Odds Ratio'] = np.exp(coef_df['Coefficient'])
    coef_df = coef_df[coef_df['Coefficient'] != 0]  # Show non-zero coefficients
    print(coef_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title('Lasso Coefficients for ever_used_alcohol (Non-Zero Only)')
    plt.xlabel('Coefficient')
    plt.show()


# GridSearchCV selected C = 100.0, with a cross-validated AUC of 0.7776. The model achieved an accuracy of 0.71, with a macro-averaged F1-score of 0.71, indicating balanced performance across classes (precision: 0.75, recall: 0.69 for class 0; precision: 0.67, recall: 0.73 for class 1). The confusion matrix shows 4,659 true negatives and 4,213 true positives, with 2,109 false positives and 1,589 false negatives. The test ROC-AUC of 0.7779 demonstrates good discriminatory ability. Lasso identified six predictors: grade (-0.587434, odds ratio: 0.555752), sex (-0.037984, odds ratio: 0.962728), household_income (0.000007, odds ratio: 1.000007), social_influence (-1.747494, odds ratio: 0.174226), mental_health (0.242909, odds ratio: 1.274953), and urban_rural (-0.167378, odds ratio: 0.845879). Notably, mental_health (odds ratio: 1.274953) increases the likelihood of alcohol use, while social_influence (odds ratio: 0.174226), grade (odds ratio: 0.555752), and urban_rural (odds ratio: 0.845879) reduce it.

# In[ ]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Step 2: Load and Prepare Data
data = pd.read_csv('final_reduced_data.csv')
data.columns = ['ever_smoked', 'age_first_smoked', 'smoked_last_30_days', 'amt_smoked', 'ever_used_alcohol',
                'ever_used_meth', 'ever_used_cocaine', 'grade', 'sex', 'household_income',
                'social_influence', 'behavioral_factor', 'mental_health', 'bullying', 'urban_rural']

# Define predictors (updated to match your column names)
predictors = ['grade', 'sex', 'household_income', 'social_influence', 'mental_health', 'bullying', 'urban_rural']

# Targets to process
targets = ['ever_used_cocaine']

for target in targets:
    print(f"\n--- Processing {target} ---")

    # Inspect raw values
    print(f"Raw unique values in {target}:\n", data[target].unique())
    print(f"Raw value counts in {target}:\n", data[target].value_counts(dropna=False))

    # Recode target based on its values
    if target == 'ever_used_meth':
        # Meth: 1 -> 0 (never used), 2 -> 1 (ever used)
        data[target] = data[target].map({1: 0, 2: 1})
    elif target == 'ever_used_cocaine':
        # Cocaine: 1 -> 0 (never used), 2 and 3 -> 1 (ever used)
        data[target] = data[target].map({1: 0, 2: 1, 3: 1})

    # Check for missing values and drop them
    print("\nMissing Values:\n", data.isnull().sum())
    data = data.dropna()

    # Check class distribution after recoding
    print(f"\nClass distribution for {target} after recoding:\n",
          data[target].value_counts(normalize=True))

    # Define X and y
    X = data[predictors]
    y = data[target]

    # Verify two classes exist
    if len(y.unique()) < 2:
        print(f"\nError: {target} contains only one class after recoding. Cannot perform classification.")
        print("Raw data may have been altered. Please verify.")
        continue  # Skip to next target if only one class

    # Step 3: Split the Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Fine-Tune Lasso-Penalized Logistic Regression
    lasso_model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handle likely imbalance
    )

    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    grid_search = GridSearchCV(
        lasso_model,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        error_score=0
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"\nBest hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validated AUC score: {grid_search.best_score_:.4f}")

    # Step 5: Evaluate the Model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print(f"\nEvaluation for {target}:")
    print(f"Unique values in y_test: {np.unique(y_test)}")
    print(f"Unique values in y_pred: {np.unique(y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=1))

    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {auc_score:.4f}")
    else:
        print("ROC-AUC Score: Skipped (only one class in y_test)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Never Used', 'Ever Used'], yticklabels=['Never Used', 'Ever Used'])
    plt.title(f'Confusion Matrix for {target}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Step 6: Extract and Interpret Coefficients
    print(f"\nCoefficients for {target}:")
    coef_df = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': best_model.coef_[0]
    })
    coef_df['Odds Ratio'] = np.exp(coef_df['Coefficient'])
    coef_df = coef_df[coef_df['Coefficient'] != 0]  # Show non-zero coefficients
    print(coef_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title(f'Lasso Coefficients for {target} (Non-Zero Only)')
    plt.xlabel('Coefficient')
    plt.show()


# GridSearchCV selected C = 0.1, with a cross-validated AUC of 0.7313. The class distribution showed 96.8% never used and 3.2% ever used cocaine, indicating severe imbalance. The model achieved an accuracy of 0.67, with a macro-averaged F1-score of 0.46, reflecting challenges in predicting the minority class (precision: 0.06, recall: 0.67). The confusion matrix shows 8,136 true negatives and 272 true positives, with 4,030 false positives and 132 false negatives. The test ROC-AUC of 0.7262 indicates moderate discriminatory ability. Lasso identified seven predictors: grade (0.437771, odds ratio: 1.549366), sex (0.196974, odds ratio: 1.217713), household_income (-0.096009, odds ratio: 0.908459), social_influence (0.999391, odds ratio: 2.717113), mental_health (-0.244828, odds ratio: 0.782839), bullying (-1.725597, odds ratio: 0.178067), and urban_rural (0.622485, odds ratio: 1.863740). Notably, social_influence (odds ratio: 2.717113) and urban_rural (odds ratio: 1.863740) increase cocaine use likelihood, while bullying (odds ratio: 0.178067) and mental_health (odds ratio: 0.782839) reduce it.

# In[ ]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Step 2: Load and Prepare Data
data = pd.read_csv('final_reduced_data.csv')
data.columns = ['ever_smoked', 'age_first_smoked', 'smoked_last_30_days', 'amt_smoked', 'ever_used_alcohol',
                'ever_used_meth', 'ever_used_cocaine', 'grade', 'sex', 'household_income',
                'social_influence', 'behavioral_factor', 'mental_health', 'bullying', 'urban_rural']

# Target to process
target = 'ever_used_meth'
print(f"\n--- Processing {target} ---")

# Inspect raw values, including NaN
print(f"Raw unique values in {target} (including NaN):\n", data[target].unique())
print(f"Raw value counts in {target}:\n", data[target].value_counts(dropna=False))

# Check for missing values in the raw data
print("\nMissing Values in Raw Data:\n", data.isnull().sum())

# Handle NaN in ever_used_meth: assume NaN means "never used" (map to 1)
data[target] = data[target].fillna(1)

# Recode: 1 -> 0 (never used), 2 and 3 -> 1 (ever used)
data[target] = data[target].map({1: 0, 2: 1, 3: 1})

# Check for missing values after recoding
print("\nMissing Values After Recoding:\n", data.isnull().sum())

# Check class distribution after recoding
print(f"\nClass distribution for {target} after recoding:\n",
      data[target].value_counts(normalize=True))

# Define predictors and target
predictors = ['grade', 'sex', 'household_income', 'social_influence', 'mental_health', 'bullying', 'urban_rural']
X = data[predictors]
y = data[target]

# Verify two classes exist
if len(y.unique()) < 2:
    print(f"\nError: {target} contains only one class after recoding. Cannot perform classification.")
    print("Raw data may have been altered. Please verify.")
else:
    # Step 3: Split the Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Fine-Tune Lasso-Penalized Logistic Regression
    lasso_model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # Handle likely imbalance
    )

    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    grid_search = GridSearchCV(
        lasso_model,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        error_score=0
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"\nBest hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validated AUC score: {grid_search.best_score_:.4f}")

    # Step 5: Evaluate the Model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print(f"\nEvaluation for {target}:")
    print(f"Unique values in y_test: {np.unique(y_test)}")
    print(f"Unique values in y_pred: {np.unique(y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=1))

    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {auc_score:.4f}")
    else:
        print("ROC-AUC Score: Skipped (only one class in y_test)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Never Used', 'Ever Used'], yticklabels=['Never Used', 'Ever Used'])
    plt.title(f'Confusion Matrix for {target}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Step 6: Extract and Interpret Coefficients
    print(f"\nCoefficients for {target}:")
    coef_df = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': best_model.coef_[0]
    })
    coef_df['Odds Ratio'] = np.exp(coef_df['Coefficient'])
    coef_df = coef_df[coef_df['Coefficient'] != 0]  # Show non-zero coefficients
    print(coef_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title(f'Lasso Coefficients for {target} (Non-Zero Only)')
    plt.xlabel('Coefficient')
    plt.show()


# In[ ]:


get_ipython().system('pip install causalinference')


# In[ ]:


import causalinference


# GridSearchCV selected C = 100.0, with a cross-validated AUC of 0.7057. The model achieved an accuracy of 0.66, with a macro-averaged F1-score of 0.44, indicating challenges in predicting the minority class (precision: 0.05, recall: 0.64 for class 1). The confusion matrix shows 8,119 true negatives and 197 true positives, with 4,145 false positives and 109 false negatives. The test ROC-AUC of 0.7142 suggests moderate discriminatory ability despite the class imbalance. Lasso identified seven predictors: grade (0.339506, odds ratio: 1.404254), sex (0.269633, odds ratio: 1.309484), household_income (-0.000015, odds ratio: 0.999985), social_influence (0.216610, odds ratio: 1.241859), mental_health (-0.183416, odds ratio: 0.832422), bullying (-1.589269, odds ratio: 0.204075), and urban_rural (0.192106, odds ratio: 1.211798). Notably, social_influence (odds ratio: 1.241859), grade (odds ratio: 1.404254), and urban_rural (odds ratio: 1.211798) increase the likelihood of methamphetamine use, while bullying (odds ratio: 0.204075) and mental_health (odds ratio: 0.832422) reduce it.

# #Model 3: Multi Level Logistic Regression

# In[ ]:


pip install pymer4


# In[ ]:


# Install R in Colab
get_ipython().system('apt-get install -y r-base')

# Install rpy2 to interface Python with R
get_ipython().system('pip install rpy2')

# Install pymer4
get_ipython().system('pip install pymer4')

# Install the lme4 package in R
get_ipython().system('R -e "install.packages(\'lme4\', repos=\'http://cran.rstudio.com/\')"')


# In[ ]:


# Load rpy2 to enable R in Colab
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

# Import libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Import pymer4 for multilevel modeling
from pymer4.models import Lmer


# In[ ]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pymer4.models import Lmer
from sklearn.metrics import roc_auc_score  # Import for AUC calculation

# Step 2: Load and Prepare Data
data = pd.read_csv('final_reduced_data.csv')
data.columns = ['ever_smoked', 'age_first_smoked', 'smoked_last_30_days', 'amt_smoked', 'ever_used_alcohol',
                'ever_used_meth', 'ever_used_cocaine', 'grade', 'sex', 'household_income',
                'social_influence', 'behavioral_factor', 'mental_health', 'bullying', 'urban_rural']

# Create the binary outcome variable for any substance use
data['any_substance_use'] = ((data['smoked_last_30_days'] == 2) |
                             (data['ever_used_alcohol'] == 2) |
                             (data['ever_used_meth'] >= 2) |
                             (data['ever_used_cocaine'] >= 2)).astype(int)

# Scale continuous predictors to improve convergence
for col in ['grade', 'household_income', 'social_influence', 'mental_health', 'bullying']:
    data[col] = (data[col] - data[col].mean()) / data[col].std()

# Debug: Check the distribution of the outcome and clustering variable
print("Value counts of any_substance_use:\n", data['any_substance_use'].value_counts())
print("Value counts of urban_rural:\n", data['urban_rural'].value_counts())

# Step 3: Define the Multilevel Logistic Regression Model
# Fixed effects: grade, sex, household_income, social_influence, mental_health, bullying
# Random effect: (1 | urban_rural) to account for clustering by urban_rural
formula = 'any_substance_use ~ grade + sex + household_income + social_influence + mental_health + bullying + (1 | urban_rural)'

# Fit the model
model = Lmer(formula, data=data, family='binomial')
model.fit()

# Step 4: Display the Model Summary
print("\nMultilevel Logistic Regression Results:")
print(model)

# Step 5: Extract and Interpret Odds Ratios
fixed_effects = model.coefs
fixed_effects['Odds Ratio'] = np.exp(fixed_effects['Estimate'])
print("\nFixed Effects with Odds Ratios:")
print(fixed_effects[['Estimate', 'Odds Ratio', 'P-val']])

# Step 6: Extract Random Effects Variance
random_effects_variance = model.ranef_var
print("\nRandom Effects Variance (urban_rural clustering):")
print(random_effects_variance)

# Step 7: Compute AUC for Multilevel Logistic Regression
# Generate predicted probabilities
data['pred_prob'] = model.predict(data, verify_predictions=False)  # Added verify_predictions=False
auc = roc_auc_score(data['any_substance_use'], data['pred_prob'])
print(f"\nAUC for Multilevel Logistic Regression: {auc:.4f}")


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Data for Multilevel odds ratios
features = ['Grade', 'Sex', 'Household Income', 'Social Influence', 'Mental Health', 'Bullying']
odds_ratios = [0.45, 1.05, 1.18, 951.35, 1.68, 1.14]
ci_lower = [0.42, 0.97, 1.10, 50.0, 1.55, 1.05]  # Approximate 95% CI lower bounds
ci_upper = [0.48, 1.14, 1.27, 18100.0, 1.82, 1.24]  # Approximate 95% CI upper bounds

# For plotting, we'll log-transform the ORs and CIs due to the large social influence OR
log_odds_ratios = np.log10(odds_ratios)
log_ci_lower = np.log10(ci_lower)
log_ci_upper = np.log10(ci_upper)

# Create the forest plot
plt.figure(figsize=(8, 5))
y_pos = np.arange(len(features))

# Plot the error bars (CIs)
plt.errorbar(log_odds_ratios, y_pos, xerr=[log_odds_ratios - log_ci_lower, log_ci_upper - log_odds_ratios],
             fmt='o', color='black', ecolor='gray', capsize=5, label='95% CI')

# Highlight the social influence point
for i, feature in enumerate(features):
    if feature == 'Social Influence':
        plt.scatter(log_odds_ratios[i], y_pos[i], color='orange', s=100, label='Social Influence')

# Add a vertical line at OR = 1 (log(1) = 0)
plt.axvline(x=0, color='gray', linestyle='--')

# Customize the plot
plt.yticks(y_pos, features)
plt.xlabel('Odds Ratio (Log10 Scale)', fontsize=12)
plt.title('Multilevel Logistic Regression Odds Ratios for Any Substance Use', fontsize=14)
plt.legend()
plt.tight_layout()


plt.savefig('multilevel_forest_plot.png', dpi=300, bbox_inches='tight')
plt.show()


# Using Multilevel Logistic Regression to model the probability of substance use (smoking, cocaine, meth, or alcohol) while accounting for hierarchical clustering by urban/rural areas, we analyzed 62,850 students with demographic (grade, sex, household income) and social factors (social influence, mental health, bullying) as predictors. Social influence emerged as the strongest predictor (odds ratio 951.35, p = 0.001), increasing the odds of substance use by a factor of 951 per standard deviation, followed by mental health (odds ratio 1.68, p < 0.001) and bullying (odds ratio 1.14, p < 0.001). Household income (odds ratio 1.18, p < 0.001) and grade (odds ratio 0.45, p < 0.001) also had significant effects, with higher grades associated with a reduced likelihood of substance use, while sex was not significant (p = 0.258). The random intercept variance for urban/rural clustering (0.062) indicated minor geographic variation, demonstrating the model’s ability to control for confounders and clustering. The model achieved an AUC of 0.7896, reflecting good discriminatory power despite the class imbalance (60,186 users vs. 2,664 non-users), highlighting its effectiveness in assessing substance use risk across urban and rural contexts.

# #Model 4: Propensity Score Matching

# In[ ]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from causalinference import CausalModel
import xgboost as xgb  # For gradient boosting

# Step 2: Load and Prepare Data
data = pd.read_csv('final_reduced_data.csv')
data.columns = ['ever_smoked', 'age_first_smoked', 'smoked_last_30_days', 'amt_smoked', 'ever_used_alcohol',
                'ever_used_meth', 'ever_used_cocaine', 'grade', 'sex', 'household_income',
                'social_influence', 'behavioral_factor', 'mental_health', 'bullying', 'urban_rural']

# Debug: Inspect the social_influence column
print("Unique values in social_influence:", data['social_influence'].unique())
print("Number of NaN values in social_influence:", data['social_influence'].isna().sum())

# Check if social_influence contains only 1 and 2
if not set(data['social_influence'].dropna().unique()).issubset({1, 2}):
    raise ValueError("social_influence contains unexpected values. Expected only 1 and 2, got: {}".format(data['social_influence'].unique()))

# Define treatment: Map social_influence (1, 2) to treatment (0, 1)
# 1 -> 0 (low social influence, control), 2 -> 1 (high social influence, treated)
data['treatment'] = data['social_influence'].map({1: 0, 2: 1})

# Handle any NaN values in treatment (if social_influence had NaNs)
if data['treatment'].isna().sum() > 0:
    print("Found NaN values in treatment after mapping. Dropping rows with NaN in treatment.")
    data = data.dropna(subset=['treatment'])

# Debug: Inspect the treatment column
print("Unique values in treatment:", data['treatment'].unique())
print("Value counts of treatment:\n", data['treatment'].value_counts())

# Check if treatment has at least two classes
if len(data['treatment'].unique()) < 2:
    raise ValueError("The treatment variable contains only one class: {}. Cannot perform classification.".format(data['treatment'].unique()))

# Define covariates and outcome
covariates = ['grade', 'sex', 'household_income', 'ever_smoked', 'mental_health', 'bullying', 'urban_rural']
Y = data['smoked_last_30_days'].values  # Outcome variable
T = data['treatment'].values    # Treatment variable (high vs. low social influence)
X = data[covariates].values     # Covariates

# Step 3: Estimate Propensity Scores with Gradient Boosting (XGBoost)
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(data[covariates], data['treatment'])
data['propensity_xgb'] = xgb_model.predict_proba(data[covariates])[:, 1]

# Step 4: Visualize Propensity Score Distribution (Before Trimming)
plt.figure(figsize=(10, 6))
sns.histplot(data=data[data['treatment'] == 1], x='propensity_xgb', label='High Social Influence (XGB)', color='blue', alpha=0.5)
sns.histplot(data=data[data['treatment'] == 0], x='propensity_xgb', label='Low Social Influence (XGB)', color='orange', alpha=0.5)
plt.title('Propensity Score Distribution (Gradient Boosting - XGBoost)')
plt.legend()
plt.show()

# Step 5: Trim Data to Enforce Common Support
common_support = (data['propensity_xgb'] >= 0.05) & (data['propensity_xgb'] <= 0.95)
trimmed_data = data[common_support]

# Check for zero-variance covariates in the trimmed data
X_trimmed = trimmed_data[covariates].values
T_trimmed = trimmed_data['treatment'].values
Y_trimmed = trimmed_data['smoked_last_30_days'].values

variances = np.var(X_trimmed, axis=0)
zero_variance_cols = [covariates[i] for i in range(len(covariates)) if variances[i] == 0]
if zero_variance_cols:
    print(f"Warning: The following covariates have zero variance after trimming: {zero_variance_cols}")
    valid_cols = [cov for cov in covariates if cov not in zero_variance_cols]
    print(f"Proceeding with the following covariates: {valid_cols}")
    X_trimmed = trimmed_data[valid_cols].values
    covariates = valid_cols
else:
    print("No zero-variance covariates found after trimming.")

# Step 6: Run Causal Model with Trimmed Data
causal_trimmed = CausalModel(Y_trimmed, T_trimmed, X_trimmed)
causal_trimmed.est_propensity_s()

trimmed_data['propensity'] = causal_trimmed.propensity['fitted']

# Visualize the trimmed propensity score distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=trimmed_data[trimmed_data['treatment'] == 1], x='propensity', label='High Social Influence (Trimmed)', color='blue', alpha=0.5)
sns.histplot(data=trimmed_data[trimmed_data['treatment'] == 0], x='propensity', label='Low Social Influence (Trimmed)', color='orange', alpha=0.5)
plt.title('Propensity Score Distribution (After Trimming)')
plt.legend()
plt.show()

# Step 7: Perform Matching with a Relaxed Caliper
propensity_scores = causal_trimmed.propensity['fitted']
treated_indices = np.where(T_trimmed == 1)[0]
control_indices = np.where(T_trimmed == 0)[0]

matched_pairs = []
caliper = 0.2  # Relaxed caliper to increase matches

for treated_idx in treated_indices:
    treated_ps = propensity_scores[treated_idx]
    ps_diffs = np.abs(propensity_scores[control_indices] - treated_ps)
    within_caliper = control_indices[ps_diffs <= caliper]
    if len(within_caliper) > 0:
        closest_control_idx = within_caliper[np.argmin(ps_diffs[ps_diffs <= caliper])]
        matched_pairs.append((treated_idx, closest_control_idx))

# Step 8: Process Matched Pairs
if len(matched_pairs) == 0:
    print("No matches found within the caliper. Try increasing the caliper or adjusting the trimming thresholds.")
else:
    matched_pairs = np.array(matched_pairs)
    matched_data = pd.DataFrame({
        'treated_idx': matched_pairs[:, 0],
        'control_idx': matched_pairs[:, 1]
    })
    print("Number of treated units matched:", len(matched_data))
    print("Number of control units matched (with replacement):", len(np.unique(matched_data['control_idx'])))

    # Step 9: Create Matched Dataset
    matched_indices = np.concatenate([matched_data['treated_idx'].values, matched_data['control_idx'].values])
    matched_X = X_trimmed[matched_indices]
    matched_T = T_trimmed[matched_indices]
    matched_Y = Y_trimmed[matched_indices]

    # Check for zero variance in the matched dataset
    matched_df = pd.DataFrame(matched_X, columns=covariates)
    matched_df['treatment'] = matched_T
    matched_df['smoked_last_30_days'] = matched_Y

    variances_matched = np.var(matched_X, axis=0)
    zero_variance_cols_matched = [covariates[i] for i in range(len(covariates)) if variances_matched[i] == 0]
    if zero_variance_cols_matched:
        print(f"Warning: The following covariates have zero variance in the matched dataset: {zero_variance_cols_matched}")
        valid_cols_matched = [cov for cov in covariates if cov not in zero_variance_cols_matched]
        print(f"Proceeding with the following covariates for balance checking: {valid_cols_matched}")
        matched_X = matched_df[valid_cols_matched].values
        covariates = valid_cols_matched

    # Check variance of the outcome
    if np.var(matched_Y) == 0:
        print("Error: The outcome variable 'smoked_last_30_days' has zero variance in the matched dataset. Cannot estimate treatment effect.")
    else:
        # Step 10: Compute Covariate Balance Manually (Avoiding stratify_s)
        def compute_smd(data, covariates, treatment_col):
            treated = data[data[treatment_col] == 1]
            control = data[data[treatment_col] == 0]
            smd = {}
            for cov in covariates:
                mean_treated = treated[cov].mean()
                mean_control = control[cov].mean()
                sd_treated = treated[cov].std()
                sd_control = control[cov].std()
                pooled_sd = np.sqrt((sd_treated**2 + sd_control**2) / 2)
                smd[cov] = (mean_treated - mean_control) / pooled_sd
            return smd

        # Before matching (on trimmed data)
        trimmed_df = pd.DataFrame(X_trimmed, columns=covariates)
        trimmed_df['treatment'] = T_trimmed
        smd_before = compute_smd(trimmed_df, covariates, 'treatment')

        # After matching
        smd_after = compute_smd(matched_df, covariates, 'treatment')

        # Print SMDs
        print("\nStandardized Mean Differences (SMD) Before Matching:")
        for cov, smd in smd_before.items():
            print(f"{cov}: {smd:.4f}")
        print("\nStandardized Mean Differences (SMD) After Matching:")
        for cov, smd in smd_after.items():
            print(f"{cov}: {smd:.4f}")

        # Plot SMDs
        plt.figure(figsize=(10, 6))
        plt.plot(list(smd_before.keys()), list(smd_before.values()), 'o-', label='Before Matching', color='red')
        plt.plot(list(smd_after.keys()), list(smd_after.values()), 'o-', label='After Matching', color='green')
        plt.axhline(y=0.1, color='gray', linestyle='--', label='Threshold (0.1)')
        plt.axhline(y=-0.1, color='gray', linestyle='--')
        plt.xticks(rotation=45)
        plt.title('Standardized Mean Differences Before and After Matching')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Step 11: Estimate the Treatment Effect Manually
        treated_outcome = matched_df[matched_df['treatment'] == 1]['smoked_last_30_days'].mean()
        control_outcome = matched_df[matched_df['treatment'] == 0]['smoked_last_30_days'].mean()
        att = treated_outcome - control_outcome
        print("\nEstimated Treatment Effect (Average Treatment Effect on the Treated - ATT):", att)


# Using Propensity Score Matching to estimate the causal effect of social influence on tobacco use outcomes, students were matched on confounders including grade, sex, household income, prior smoking, mental health, bullying, and urban/rural status. The propensity score distribution before trimming showed poor overlap, with high social influence students concentrated at higher scores (0.3–0.8), while low social influence students were absent. After trimming (0.05–0.95), overlap improved, with both groups distributed between 0.2–0.8. Matching paired 6,012 treated (high social influence) with 1,070 control units (with replacement). Standardized Mean Differences (SMDs) reduced significantly post-matching (e.g., grade: -0.2906 to -0.0166, mental health: 0.1286 to 0.0289), all within ±0.1, indicating good covariate balance. The estimated Average Treatment Effect on the Treated (ATT) of 0.5158 suggests that high social influence increases recent tobacco use (smoked_last_30_days) by approximately 0.52 units among students with high social influence, highlighting its significant impact on smoking behavior.

# #Model 5: Random Forest

# In[ ]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load and Prepare Data
data = pd.read_csv('final_reduced_data.csv')
data.columns = ['ever_smoked', 'age_first_smoked', 'smoked_last_30_days', 'amt_smoked', 'ever_used_alcohol',
                'ever_used_meth', 'ever_used_cocaine', 'grade', 'sex', 'household_income',
                'social_influence', 'behavioral_factor', 'mental_health', 'bullying', 'urban_rural']

# Debug: Check the unique values of substance use columns
print("Unique values in smoked_last_30_days:", data['smoked_last_30_days'].unique())
print("Unique values in ever_used_alcohol:", data['ever_used_alcohol'].unique())
print("Unique values in ever_used_meth:", data['ever_used_meth'].unique())
print("Unique values in ever_used_cocaine:", data['ever_used_cocaine'].unique())

# Create a new binary outcome variable for any substance use
# 1 if the student has used any substance (smoking, cocaine, meth, or alcohol), 0 otherwise
# Define "use" based on the value ranges:
# - smoked_last_30_days: 2 = use
# - ever_used_alcohol: 2 = use
# - ever_used_meth: 2 or 3 = use
# - ever_used_cocaine: 2 or 3 = use
data['any_substance_use'] = ((data['smoked_last_30_days'] == 2) |
                             (data['ever_used_alcohol'] == 2) |
                             (data['ever_used_meth'] >= 2) |
                             (data['ever_used_cocaine'] >= 2)).astype(int)

# Debug: Check the distribution of the new outcome variable
print("Unique values in any_substance_use:", data['any_substance_use'].unique())
print("Value counts of any_substance_use:\n", data['any_substance_use'].value_counts())

# Define predictors and outcome
predictors = ['grade', 'sex', 'household_income', 'social_influence', 'mental_health', 'bullying', 'urban_rural']
X = data[predictors]
y = data['any_substance_use']

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest Classifier with Class Weighting
# Use class_weight='balanced' to handle class imbalance
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # Adjusts weights inversely proportional to class frequencies
)
rf_model.fit(X_train, y_train)

# Step 5: Make Predictions and Evaluate the Model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-User', 'User'], yticklabels=['Non-User', 'User'])
plt.title('Confusion Matrix (With Class Weighting)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 6: Identify Key Predictor Variables (Feature Importance)
feature_importance = pd.DataFrame({
    'Feature': predictors,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("\nFeature Importance Scores:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Predicting Any Substance Use (Classification)')
plt.show()

# Step 7: Classify High-Risk Groups
data['predicted_substance_use'] = rf_model.predict(X)
# High-risk group: Students predicted to use at least one substance (any_substance_use = 1)
data['high_risk'] = (data['predicted_substance_use'] == 1).astype(int)  # Class 1 is "User"

# Summarize high-risk group characteristics
high_risk_group = data[data['high_risk'] == 1]
print("\nSummary of High-Risk Group Characteristics (Predicted Substance Users):")
for feature in predictors:
    print(f"{feature}: Mean = {high_risk_group[feature].mean():.2f}, Std = {high_risk_group[feature].std():.2f}")

# Compare high-risk vs. low-risk groups
low_risk_group = data[data['high_risk'] == 0]
print("\nComparison of High-Risk vs. Low-Risk Groups:")
for feature in predictors:
    high_mean = high_risk_group[feature].mean()
    low_mean = low_risk_group[feature].mean()
    print(f"{feature}: High-Risk Mean = {high_mean:.2f}, Low-Risk Mean = {low_mean:.2f}, Difference = {high_mean - low_mean:.2f}")

from sklearn.metrics import roc_auc_score

y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC for Random Forest: {auc:.4f}")


# Using Random Forest, a machine learning method, we predicted substance use outcomes (smoking, cocaine, meth, or alcohol) and identified key predictors for classifying high-risk groups among students. The model achieved an accuracy of 0.7266, with a weighted F1-score of 0.81, though it struggled with the minority class (non-users, F1-score 0.19) due to class imbalance (558 non-users vs. 12,012 users). The confusion matrix revealed 8,734 true positives (users correctly predicted) but 3,278 false negatives (users misclassified as non-users). Feature importance scores highlighted grade (0.4063), mental health (0.1754), and social influence (0.1624) as the top predictors, indicating their critical role in identifying students at high risk of substance use, with urban/rural status (0.0522) and sex (0.0229) being less influential.

# -------------

# #Overall Analysis
# 
# **Bayesian Logistic Regression**: Using Bayesian Logistic Regression, a probabilistic approach, we estimated the probability of tobacco use (smoked_last_30_days) based on CSTADS variables such as grade, sex, and household income for 62,850 students. The model revealed that a one-unit increase in standardized grade increases the log-odds of recent smoking by 0.758, males have a slightly higher log-odds (0.123) compared to females, and higher household income decreases the log-odds by 0.179. The 94% Highest Density Interval provided credible ranges for these estimates, and the sharply peaked posterior distributions (e.g., beta_grade mean: 0.76) confirmed high certainty, offering a robust probabilistic framework for understanding tobacco use likelihood while accounting for uncertainty in parameter estimates.
# 
# **Lasso Regression**: Employing Lasso Regression, a feature selection model, we analyzed substance use trends across multiple outcomes (smoking, alcohol, cocaine, meth) by identifying the most influential factors affecting behavior among 62,850 students. For smoking (DVLAST30_binary), the model (AUC 0.7961) selected predictors like grade (odds ratio 1.77) and urban_rural (odds ratio 1.72) as key drivers, while for alcohol use (AUC 0.7776), social_influence (odds ratio 0.174) and mental_health (odds ratio 1.275) were significant. For cocaine (AUC 0.7313) and meth (AUC 0.7057), social_influence increased use likelihood (odds ratios 2.72 and 1.24, respectively). Lasso’s ability to shrink less important coefficients to zero ensured focus on the most impactful variables, enhancing interpretability despite class imbalances.
# 
# **Multilevel Logistic Regression**: Using Multilevel Logistic Regression to model the probability of substance use (smoking, cocaine, meth, or alcohol) while accounting for hierarchical clustering by urban/rural areas, we analyzed 62,850 students with demographic (grade, sex, household income) and social factors (social influence, mental health, bullying) as predictors. Social influence was the strongest predictor (odds ratio 951.35, p = 0.001), followed by mental health (odds ratio 1.68, p < 0.001) and bullying (odds ratio 1.14, p < 0.001). The random intercept variance for urban/rural clustering (0.062) indicated minor geographic variation, demonstrating the model’s ability to control for confounders and clustering, thus providing a nuanced understanding of substance use risk across different contexts.
# 
# **Propensity Score Matching**: Using Propensity Score Matching to estimate the causal effect of social influence on tobacco use outcomes, we matched 62,850 students on confounders like grade, sex, household income, prior smoking, mental health, bullying, and urban/rural status. After trimming for common support, matching paired 6,012 high social influence students with 1,070 controls, achieving good covariate balance (SMDs within ±0.1, e.g., grade: -0.2906 to -0.0166). The Average Treatment Effect on the Treated (ATT) of 0.5158 indicated that high social influence increases recent tobacco use by 0.52 units, offering a robust causal estimate by reducing confounding bias through effective matching.
# 
# **Random Forest**: Using Random Forest, a machine learning method, we predicted substance use outcomes (smoking, cocaine, meth, or alcohol) and identified key predictors for classifying high-risk groups among 62,850 students. The model achieved an accuracy of 0.7266 (weighted F1-score 0.81), despite challenges with the minority class (non-users, F1-score 0.19). Feature importance scores highlighted grade (0.4063), mental health (0.1754), and social influence (0.1624) as top predictors, enabling precise identification of high-risk students. The model’s ability to handle non-linear relationships and interactions made it effective for classifying at-risk groups, with high-risk students showing distinct characteristics (e.g., higher social influence).

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data for AUC comparison
data = {
    'Model': ['Bayesian', 'Lasso', 'Multilevel', 'Random Forest'],
    'AUC': [0.7119, 0.7866, 0.7896, 0.7266]
}
auc_df = pd.DataFrame(data)

# Sort the dataframe by AUC for better visualization (highest to lowest)
auc_df = auc_df.sort_values(by='AUC', ascending=True)

# Create the lollipop chart
plt.figure(figsize=(8, 5))

# Plot the horizontal lines (the "sticks")
plt.hlines(y=auc_df['Model'], xmin=0.65, xmax=auc_df['AUC'], color='gray', alpha=0.7)

# Plot the points (the "lollipops")
plt.scatter(auc_df['AUC'], auc_df['Model'], color='teal', s=100, label='AUC')

# Add annotations for the AUC values
for i, (auc, model) in enumerate(zip(auc_df['AUC'], auc_df['Model'])):
    plt.text(auc + 0.005, model, f'{auc:.4f}', ha='left', va='center', fontsize=10, color='black')

# Add a vertical line at the median AUC for reference
median_auc = np.median(auc_df['AUC'])
plt.axvline(x=median_auc, color='orange', linestyle='--', label=f'Median AUC ({median_auc:.4f})')

# Customize the plot
plt.title('AUC Comparison Across Models', fontsize=14)
plt.xlabel('AUC (Area Under the ROC Curve)', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.xlim(0.65, 0.82)  # Adjust x-axis limits to focus on the range of AUC values
plt.legend()
plt.tight_layout()

# Add a caption (optional, for presentation slide)
caption = "Note: Lasso AUC shown for smoking (range: 0.7057–0.7866); PSM excluded (not a classification model)."
plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=10)

# Save the plot for your presentation
plt.savefig('auc_comparison_lollipop.png', dpi=300, bbox_inches='tight')
plt.show()

