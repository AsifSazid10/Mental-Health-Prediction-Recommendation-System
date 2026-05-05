# -*- coding: utf-8 -*-
"""
Fully commented version of the mental health related machine learning workflow.
Original notebook content was reorganized and annotated so that every major part
of the work has a clear comment before it.
"""

# ============================================================
# PART A: IMPORT REQUIRED LIBRARIES
# This section imports all libraries used for data handling,
# visualization, machine learning, evaluation, and explainability.
# ============================================================
import sys
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import shap
import sklearn
import imblearn
import xgboost

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from scipy.stats import pearsonr
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ============================================================
# PART B: LOAD THE DATASET
# This section reads the CSV file into a pandas DataFrame.
# Replace the file path if your dataset is stored somewhere else.
# ============================================================
df = pd.read_csv('/content/Untitled form-3.csv')


# ============================================================
# PART C: INITIAL DATA PREVIEW
# This section displays the first few rows in transposed format
# so that all variables can be reviewed vertically.
# ============================================================
df.head(2).T


# ============================================================
# PART D: REMOVE UNNECESSARY COLUMNS
# This section removes columns that are not required for analysis,
# including timestamp, attention check, and optional free-text response.
# ============================================================
df.drop(['Timestamp'], inplace=True, axis=1)
df.drop(['23. Attention check: To ensure data quality, please select 3 for this question.'], inplace=True, axis=1)
df.drop(columns=['24. Optional: Anything else you\'d like to share about your social media use and work–life balance?'], inplace=True)


# ============================================================
# PART E: RENAME LONG SURVEY QUESTIONS INTO SHORT VARIABLE NAMES
# This makes the dataset easier to work with during analysis,
# visualization, scoring, and model building.
# ============================================================
col_names_dict = {
    '1. Age: _______ (years)': 'age',
    '2. Gender:': 'gender',
    '3. Relationship status:': 'relationship_status',
    '4. Current primary occupation:': 'occupation_status',
    '5. Primary work / activity setting:': 'org_affiliation',
    ' 6. Do you currently use social media?': 'uses_social_media',
    '7. Which social media platforms do you use regularly? (tick all that apply)': 'platforms_used',
    '8. Average time spent on social media per day:': 'daily_time_spent',
    '9. How often do you use social media without a specific purpose (mindless browsing)?': 'mindless_use_freq',
    '10. How often do social media notifications or content distract you during important tasks (work, study, chores etc.)?': 'distraction_when_busy_freq',
    '11. When you haven’t used social media for a while, how restless or uneasy do you feel?': 'restless_without_sm',
    '12. What aspects of social media distract you the most? (tick all that apply)': 'top_distraction_sources',
    '13. On a scale of 1–5, how strongly do these distractions affect your ability to focus?': 'distraction_impact',
    '14. How often do you find it difficult to concentrate on tasks?( Using Social Media)': 'concentration_difficulty_freq',
    '15. On a typical day, how productive do you feel?': 'daily_productivity',
    '16. How often do you feel that social media negatively affects your productivity or performance?': 'sm_negative_impact_freq',
    '17. How often do you compare yourself with others on social media?': 'social_comparison_freq',
    '18. Overall, how do these comparisons make you feel?': 'comparison_feelings',
    '19. How often do you seek validation on social media (likes, comments, reactions)?': 'validation_seeking_freq',
    '20. Over the past two weeks, how often have you felt depressed, upset, or unusually sad?': 'low_mood_freq',
    '21. How often does your motivation or interest in daily activities fluctuate?': 'interest_fluctuation_freq',
    '22. How often do you experience sleep-related issues (difficulty falling or staying asleep)?': 'sleep_issues_freq',
}

df.rename(columns=col_names_dict, inplace=True)


# ============================================================
# PART F: CHECK COLUMN NAMES AND DATA SHAPE
# This helps verify whether renaming and column removal worked correctly.
# ============================================================
df.columns
df.shape


# ============================================================
# PART G: CLEAN AND STANDARDIZE AGE COLUMN
# This section converts age to numeric, removes invalid entries,
# and stores age as an integer column.
# ============================================================
df['age_num'] = pd.to_numeric(df['age'], errors='coerce')
df = df.dropna(subset=['age_num']).copy()
df['age'] = df['age_num'].astype(int)
df.drop(columns=['age_num'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.shape


# ============================================================
# PART H: SAVE COLUMN TITLES FOR REFERENCE
# This creates a list of all current column names.
# ============================================================
titles = list(df.columns)
titles


# ============================================================
# PART I: CHECK FOR MISSING VALUES
# This section identifies null values before imputation.
# ============================================================
df.isnull().sum()


# ============================================================
# PART J: FILL MISSING VALUES IN SELECTED COLUMNS USING MODE
# Mode imputation is used for categorical variables with missing data.
# ============================================================
fill_cols = ['org_affiliation', 'platforms_used', 'daily_time_spent', 'sleep_issues_freq']
for col in fill_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)


# ============================================================
# PART K: REMOVE DUPLICATE ROWS AND RESET INDEX
# This ensures the dataset contains only unique responses.
# ============================================================
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print('Shape after cleaning:', df.shape)
print(df.isnull().sum())
print([
    f'Nan values in: {(element, value)}'
    for (element, value) in zip(df.isnull().sum().index, df.isnull().sum())
    if value > 0
])


# ============================================================
# PART L: OPTIONAL REPEATED NULL HANDLING CHECKS
# These lines further confirm percentages of missing values and
# refill mode values where needed.
# ============================================================
f"Percent NaN = {df.org_affiliation.isnull().sum()/len(df.org_affiliation)*100:0.2f} %"
df.org_affiliation.fillna(df['org_affiliation'].value_counts().index[0], inplace=True)

f"Percent NaN = {df.platforms_used.isnull().sum()/len(df.platforms_used)*100:0.2f} %"
df.platforms_used.fillna(df['platforms_used'].value_counts().index[0], inplace=True)

f"Percent NaN = {df.daily_time_spent.isnull().sum()/len(df.daily_time_spent)*100:0.2f} %"
df.daily_time_spent.fillna(df['daily_time_spent'].value_counts().index[0], inplace=True)

f"Percent NaN = {df.sleep_issues_freq.isnull().sum()/len(df.sleep_issues_freq)*100:0.2f} %"
df.sleep_issues_freq.fillna(df['sleep_issues_freq'].value_counts().index[0], inplace=True)

df.isnull().sum()
df.info()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape
titles
df.info()


# ============================================================
# PART M: DEMOGRAPHIC BREAKDOWN
# This section summarizes age, gender distribution, and occupation.
# It also creates a combined demographic table for reporting.
# ============================================================
df['age'] = pd.to_numeric(df['age'], errors='coerce')
demo_df = df.dropna(subset=['age']).copy()

age_min = demo_df['age'].min()
age_max = demo_df['age'].max()
age_mean = demo_df['age'].mean()
age_std = demo_df['age'].std()

print('Age Summary')
print(f'Age range: {age_min} - {age_max}')
print(f'Mean age: {age_mean:.2f}')
print(f'SD age: {age_std:.2f}')

gender_dist = demo_df['gender'].value_counts(dropna=False).reset_index()
gender_dist.columns = ['Gender', 'Count']
gender_dist['Percentage'] = (gender_dist['Count'] / len(demo_df) * 100).round(2)

print('\nGender Distribution')
print(gender_dist)

occupation_dist = demo_df['occupation_status'].value_counts(dropna=False).reset_index()
occupation_dist.columns = ['Occupation Status', 'Count']
occupation_dist['Percentage'] = (occupation_dist['Count'] / len(demo_df) * 100).round(2)

print('\nOccupation Status Distribution')
print(occupation_dist)

age_row = pd.DataFrame({
    'Variable': ['Age'],
    'Category': [f'Range: {age_min}-{age_max}, Mean: {age_mean:.2f}, SD: {age_std:.2f}'],
    'Count': [len(demo_df)],
    'Percentage': [100.00]
})

gender_table = gender_dist.copy()
gender_table.columns = ['Category', 'Count', 'Percentage']
gender_table.insert(0, 'Variable', 'Gender')

occupation_table = occupation_dist.copy()
occupation_table.columns = ['Category', 'Count', 'Percentage']
occupation_table.insert(0, 'Variable', 'Occupation Status')

demographic_table = pd.concat([age_row, gender_table, occupation_table], ignore_index=True)

print('\nDemographic Characteristics Table')
print(demographic_table)

demographic_table.to_csv('demographic_characteristics_table.csv', index=False)


# ============================================================
# PART N: INSPECT AND RECODE COMPARISON FEELINGS VARIABLE
# This section converts text-based emotional comparison responses
# into numeric values for analysis.
# ============================================================
df['comparison_feelings'].value_counts()

comparison_map = {
    'Much Better': 1,
    'Slightly Better': 2,
    'No Change': 3,
    'Slightly Worse': 4,
    'Much Worse': 5
}

df['comparison_feelings'] = (
    df['comparison_feelings']
    .astype(str)
    .str.strip()
    .str.title()
)

df['comparison_feelings'] = (
    df['comparison_feelings']
    .map(comparison_map)
    .fillna(0)
    .astype(int)
)

sex = set(df['gender'])
print(sex)


# ============================================================
# PART O: RECHECK AND REFORMAT AGE COLUMN
# This section handles any leftover formatting issues in age values.
# ============================================================
df['age'] = (
    df['age']
    .astype(str)
    .str.strip()
    .replace('', pd.NA)
)

df['age'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')

df.describe()
df.median(numeric_only=True)


# ============================================================
# PART P: REASSIGN COMPARISON FEELINGS SCORES FOR NEGATIVE IMPACT
# Positive responses are set to zero because the study focuses on
# negative mental health effects rather than positive effects.
# ============================================================
df.loc[df['comparison_feelings'] == 1, 'comparison_feelings'] = 0
df.loc[df['comparison_feelings'] == 2, 'comparison_feelings'] = 0
df.loc[df['comparison_feelings'] == 3, 'comparison_feelings'] = 0
df.loc[df['comparison_feelings'] == 4, 'comparison_feelings'] = 4
df.loc[df['comparison_feelings'] == 5, 'comparison_feelings'] = 2

df.head()


# ============================================================
# PART Q: REVIEW FREQUENCY-BASED COLUMNS BEFORE MAPPING
# This helps verify response categories before numeric conversion.
# ============================================================
df['mindless_use_freq'].value_counts()
df['distraction_when_busy_freq'].value_counts()
df['distraction_impact']
df['mindless_use_freq'].value_counts()


# ============================================================
# PART R: MAP FREQUENCY RESPONSES TO NUMERIC SCORES
# Text responses like Never, Rarely, Sometimes, Often, Always
# are converted into a 1 to 5 scale.
# ============================================================
freq_map = {
    'Never': 1,
    'Rarely': 2,
    'Sometimes': 3,
    'Often': 4,
    'Always': 5
}

df['mindless_use_freq'] = (
    df['mindless_use_freq']
    .astype(str).str.strip().str.title()
    .map(freq_map)
)

df['distraction_when_busy_freq'] = (
    df['distraction_when_busy_freq']
    .astype(str).str.strip().str.title()
    .map(freq_map)
)

df['concentration_difficulty_freq'] = (
    df['concentration_difficulty_freq']
    .astype(str).str.strip().str.title()
    .map(freq_map)
)

df['sm_negative_impact_freq']


# ============================================================
# PART S: CREATE COMPOSITE SCORES
# This section combines selected items into broader psychological
# or behavior-related score groups.
# ============================================================
purpose = ['mindless_use_freq', 'distraction_when_busy_freq', 'restless_without_sm', 'distraction_impact']
df['purpose'] = df[purpose].sum(axis=1)

Anxiety = ['sm_negative_impact_freq', 'concentration_difficulty_freq']
df['Anxiety Score'] = df[Anxiety].sum(axis=1)

SelfEsteem = ['social_comparison_freq', 'comparison_feelings', 'validation_seeking_freq']
df['Self Esteem Score'] = df[SelfEsteem].sum(axis=1)

Depression = ['low_mood_freq', 'interest_fluctuation_freq', 'sleep_issues_freq']
df['Depression Score'] = df[Depression].sum(axis=1)

Total = ['purpose', 'Anxiety Score', 'Self Esteem Score', 'Depression Score']
df['Total Score'] = df[Total].sum(axis=1)

df['Total Score']

TOT = df['Total Score'].max()
TOT

average = TOT / 12
print(average)

df.head()


# ============================================================
# PART T: MAP TOTAL SCORE TO OUTCOME CLASSES
# This section creates the target variable for classification.
# The original notebook uses three final classes.
# ============================================================
def map_score(score):
    if score < 35:
        return 0
    elif score < 40:
        return 1
    else:
        return 2


df['Outcome'] = df['Total Score'].apply(map_score).astype(int)

class_labels = {
    0: 'Normal',
    1: 'Moderate Risk',
    2: 'Severe Risk',
}

df['Outcome Label'] = df['Outcome'].map(class_labels)
df[['Total Score', 'Outcome', 'Outcome Label']].head()
df.head()


# ============================================================
# PART U: CLASS DISTRIBUTION REPORT
# This section counts how many samples belong to each outcome class.
# ============================================================
class_distribution = df['Outcome'].value_counts().reset_index()
class_distribution.columns = ['Class', 'Count']
class_distribution['Percentage'] = (class_distribution['Count'] / len(df) * 100).round(2)

print('Class Distribution')
print(class_distribution)

class_distribution.to_csv('class_distribution.csv', index=False)


# ============================================================
# PART V: CRONBACH'S ALPHA RELIABILITY ANALYSIS
# This evaluates internal consistency for each composite score.
# ============================================================
def cronbach_alpha(items_df):
    items_df = items_df.dropna().astype(float)
    k = items_df.shape[1]
    if k < 2:
        return np.nan
    item_variances = items_df.var(axis=0, ddof=1)
    total_score = items_df.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    if total_variance == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - (item_variances.sum() / total_variance))
    return alpha

purpose_items = ['mindless_use_freq', 'distraction_when_busy_freq', 'restless_without_sm', 'distraction_impact']
anxiety_items = ['sm_negative_impact_freq', 'concentration_difficulty_freq']
self_esteem_items = ['social_comparison_freq', 'comparison_feelings', 'validation_seeking_freq']
depression_items = ['low_mood_freq', 'interest_fluctuation_freq', 'sleep_issues_freq']

reliability_results = pd.DataFrame({
    'Composite Score': ['Purpose', 'Anxiety Score', 'Self Esteem Score', 'Depression Score'],
    "Cronbach's Alpha": [
        cronbach_alpha(df[purpose_items]),
        cronbach_alpha(df[anxiety_items]),
        cronbach_alpha(df[self_esteem_items]),
        cronbach_alpha(df[depression_items])
    ]
})

reliability_results["Cronbach's Alpha"] = reliability_results["Cronbach's Alpha"].round(4)

print("Reliability Analysis (Cronbach's Alpha)")
print(reliability_results)

reliability_results.to_csv('reliability_analysis_cronbach_alpha.csv', index=False)


# ============================================================
# PART W: INSTALL FACTOR ANALYZER PACKAGE
# This is needed for exploratory factor analysis.
# Run this in notebook environments when the package is missing.
# ============================================================
# !pip install factor_analyzer


# ============================================================
# PART X: EXPLORATORY FACTOR ANALYSIS (EFA)
# This section checks whether grouped questionnaire items form
# meaningful latent factors.
# ============================================================
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

factor_items = [
    'mindless_use_freq', 'distraction_when_busy_freq', 'restless_without_sm', 'distraction_impact',
    'sm_negative_impact_freq', 'concentration_difficulty_freq',
    'social_comparison_freq', 'comparison_feelings', 'validation_seeking_freq',
    'low_mood_freq', 'interest_fluctuation_freq', 'sleep_issues_freq'
]

efa_df = df[factor_items].copy()
efa_df = efa_df.apply(pd.to_numeric, errors='coerce')
efa_df = efa_df.dropna()

chi_square_value, bartlett_p = calculate_bartlett_sphericity(efa_df)
kmo_all, kmo_model = calculate_kmo(efa_df)

print("Bartlett's Test of Sphericity")
print(f'Chi-square value: {chi_square_value:.4f}')
print(f'p-value: {bartlett_p:.6f}')

print('\nKMO Test')
print(f'KMO Overall: {kmo_model:.4f}')

fa_test = FactorAnalyzer(rotation=None)
fa_test.fit(efa_df)

eigenvalues, vectors = fa_test.get_eigenvalues()

eigen_table = pd.DataFrame({
    'Factor': range(1, len(eigenvalues) + 1),
    'Eigenvalue': eigenvalues
})

print('\nEigenvalues')
print(eigen_table)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.axhline(y=1, linestyle='--')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

fa = FactorAnalyzer(n_factors=4, rotation='varimax')
fa.fit(efa_df)

loadings = pd.DataFrame(
    fa.loadings_,
    index=factor_items,
    columns=['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4']
)

print('\nFactor Loadings')
print(loadings.round(4))

communalities = pd.DataFrame({
    'Item': factor_items,
    'Communality': fa.get_communalities()
})

variance_df = pd.DataFrame({
    'Factor': ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4'],
    'SS Loadings': fa.get_factor_variance()[0],
    'Proportion Var': fa.get_factor_variance()[1],
    'Cumulative Var': fa.get_factor_variance()[2]
})

print('\nCommunalities')
print(communalities.round(4))

print('\nVariance Explained')
print(variance_df.round(4))

eigen_table.to_csv('efa_eigenvalues.csv', index=False)
loadings.to_csv('efa_factor_loadings.csv')
communalities.to_csv('efa_communalities.csv', index=False)
variance_df.to_csv('efa_variance_explained.csv', index=False)


# ============================================================
# PART Y: ENCODE GENDER INTO NUMERIC VALUES
# This converts categorical gender responses into integer codes.
# ============================================================
df['gender'].value_counts()
df.loc[df['gender'] == 'Male', 'gender'] = 0
df.loc[df['gender'] == 'Female', 'gender'] = 1
df.loc[df['gender'] == 'Prefer not to say', 'gender'] = 2
df['gender'] = df['gender'].astype('int64')

df


# ============================================================
# PART Z: ENCODE DAILY TIME SPENT ON SOCIAL MEDIA
# This converts time-range categories into ordinal numeric values.
# ============================================================
df['daily_time_spent'].value_counts()
df.loc[df['daily_time_spent'] == 'Less than 30 minutes', 'daily_time_spent'] = 1
df.loc[df['daily_time_spent'] == '30-60 minutes', 'daily_time_spent'] = 2
df.loc[df['daily_time_spent'] == '1-2 hours', 'daily_time_spent'] = 3
df.loc[df['daily_time_spent'] == '2-4 hours', 'daily_time_spent'] = 4
df.loc[df['daily_time_spent'] == 'More than 4 hours', 'daily_time_spent'] = 5
df['daily_time_spent'] = df['daily_time_spent'].astype('int64')

df


# ============================================================
# PART AA: PREPARE MODELING DATASET
# This keeps only the features and target used for machine learning.
# ============================================================
df_full = df.copy()

columns_to_keep = [
    'age', 'gender', 'daily_time_spent',
    'purpose', 'Anxiety Score', 'Self Esteem Score',
    'Depression Score', 'Outcome'
]

df_model = df[columns_to_keep].copy()

df
print(df_model['Outcome'].value_counts())
print(df_model['Outcome'].value_counts(normalize=True) * 100)
df['Outcome'].value_counts()


# ============================================================
# PART AB: CORRELATION ANALYSIS AND HEATMAP
# This explores relationships among features and the target variable.
# ============================================================
df_model.corr()

corr = df_model[['age', 'gender', 'daily_time_spent', 'purpose', 'Anxiety Score', 'Self Esteem Score', 'Depression Score', 'Outcome']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

df_model


# ============================================================
# PART AC: TRAIN-TEST SPLIT
# This separates the data into training and testing sets while
# preserving class balance using stratified splitting.
# ============================================================
RANDOM_STATE = 42

X = df_model.drop('Outcome', axis=1)
y = df_model['Outcome'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f'Random seed used: {RANDOM_STATE}')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('\nTraining set class distribution:')
print(y_train.value_counts())

print('\nTest set class distribution:')
print(y_test.value_counts())


# ============================================================
# PART AD: LOGISTIC REGRESSION WITH PROPER CV AND SMOTE
# SMOTE is applied inside each fold to avoid data leakage.
# ============================================================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
    ('clf', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring={
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'roc_auc_ovr': 'roc_auc_ovr'
    },
    return_train_score=True,
    n_jobs=-1
)

print('CV Accuracy Scores:', cv_results['test_accuracy'])
print(f"CV Mean Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
print(f"CV Std Dev: {np.std(cv_results['test_accuracy']):.4f}")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

print('\nTest Accuracy:', round(accuracy_score(y_test, y_pred), 4))
print('Test Precision Macro:', round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4))
print('Test Recall Macro:', round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4))
print('Test F1 Macro:', round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4))


# ============================================================
# PART AE: COMPARE MULTIPLE CLASSIFICATION MODELS
# This section evaluates several algorithms using the same
# train/test split, CV setup, and SMOTE strategy.
# ============================================================
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
    'Support Vector Machine': SVC(probability=True, random_state=RANDOM_STATE),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

class_labels = {
    0: 'Normal',
    1: 'Moderate Risk',
    2: 'Severe Risk'
}

results = []
all_reports = {}
all_conf_matrices = {}
all_predictions = {}
classes = np.array(sorted(y_train.unique()))

for model_name, model in classifiers.items():
    print('\n' + '=' * 80)
    print(f'MODEL: {model_name}')
    print('=' * 80)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
        ('clf', model)
    ])

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro',
            'roc_auc_ovr': 'roc_auc_ovr'
        },
        return_train_score=True,
        n_jobs=-1
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    y_test_bin = label_binarize(y_test, classes=classes)
    test_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')

    report = classification_report(
        y_test,
        y_pred,
        target_names=[class_labels[c] for c in sorted(class_labels.keys())],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    all_reports[model_name] = report_df
    all_conf_matrices[model_name] = confusion_matrix(y_test, y_pred)
    all_predictions[model_name] = {'y_pred': y_pred, 'y_prob': y_prob, 'pipeline': pipeline}

    print('CV Accuracy Scores:', cv_results['test_accuracy'])
    print(f"CV Mean Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"CV Std Dev: {np.std(cv_results['test_accuracy']):.4f}")

    print('\nTest Accuracy:', round(test_accuracy, 4))
    print('Test Precision Macro:', round(test_precision, 4))
    print('Test Recall Macro:', round(test_recall, 4))
    print('Test F1 Macro:', round(test_f1, 4))
    print('Test AUC OVR Macro:', round(test_auc, 4))

    print('\nClassification Report:')
    print(classification_report(
        y_test,
        y_pred,
        target_names=[class_labels[c] for c in sorted(class_labels.keys())],
        zero_division=0
    ))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    results.append({
        'Model': model_name,
        'CV_Train_Accuracy_Mean': round(np.mean(cv_results['train_accuracy']), 4),
        'CV_Test_Accuracy_Mean': round(np.mean(cv_results['test_accuracy']), 4),
        'CV_Test_Accuracy_SD': round(np.std(cv_results['test_accuracy']), 4),
        'CV_Test_Precision_Macro_Mean': round(np.mean(cv_results['test_precision_macro']), 4),
        'CV_Test_Recall_Macro_Mean': round(np.mean(cv_results['test_recall_macro']), 4),
        'CV_Test_F1_Macro_Mean': round(np.mean(cv_results['test_f1_macro']), 4),
        'CV_Test_AUC_OVR_Mean': round(np.mean(cv_results['test_roc_auc_ovr']), 4),
        'Test_Accuracy': round(test_accuracy, 4),
        'Test_Precision_Macro': round(test_precision, 4),
        'Test_Recall_Macro': round(test_recall, 4),
        'Test_F1_Macro': round(test_f1, 4),
        'Test_AUC_OVR_Macro': round(test_auc, 4)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by=['Test_F1_Macro', 'Test_AUC_OVR_Macro', 'Test_Accuracy', 'CV_Test_Accuracy_Mean'],
    ascending=False
).reset_index(drop=True)

print('\n' + '=' * 80)
print('FINAL MODEL COMPARISON TABLE')
print('=' * 80)
print(results_df)

results_df.to_csv('model_comparison_results.csv', index=False)

best_model_name = results_df.loc[0, 'Model']
best_model_info = all_predictions[best_model_name]
best_pipeline = best_model_info['pipeline']
best_y_pred = best_model_info['y_pred']
best_y_prob = best_model_info['y_prob']

print('\n' + '=' * 80)
print(f'BEST MODEL BASED ON OVERALL PERFORMANCE: {best_model_name}')
print('=' * 80)
print(results_df.loc[0])

print('\nBest Model Classification Report:')
print(classification_report(
    y_test,
    best_y_pred,
    target_names=[class_labels[c] for c in sorted(class_labels.keys())],
    zero_division=0
))

print('Best Model Confusion Matrix:')
print(confusion_matrix(y_test, best_y_pred))


# ============================================================
# PART AF: CLASS-WISE AUC FOR THE BEST MODEL
# This computes one-vs-rest AUC for each outcome class.
# ============================================================
y_test_bin = label_binarize(y_test, classes=classes)
auc_rows = []

for i, c in enumerate(classes):
    auc_value = roc_auc_score(y_test_bin[:, i], best_y_prob[:, i])
    auc_rows.append({
        'Class_Code': int(c),
        'Class_Label': class_labels[int(c)],
        'AUC_OVR': round(auc_value, 4)
    })

best_auc_table = pd.DataFrame(auc_rows)

print('\nClass-wise AUC for Best Model:')
print(best_auc_table)

best_auc_table.to_csv('best_model_classwise_auc.csv', index=False)


# ============================================================
# PART AG: BOOTSTRAP 95 PERCENT CONFIDENCE INTERVALS
# This section estimates confidence intervals for evaluation metrics.
# ============================================================
def bootstrap_metric_ci(y_true, y_pred, metric_func, n_bootstrap=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        indices = rng.choice(np.arange(n), size=n, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        try:
            score = metric_func(y_true_sample, y_pred_sample)
            scores.append(score)
        except Exception:
            continue

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return lower, upper


def bootstrap_auc_ci(y_true, y_prob, classes, n_bootstrap=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    auc_scores = []

    for _ in range(n_bootstrap):
        indices = rng.choice(np.arange(n), size=n, replace=True)
        y_true_sample = y_true[indices]
        y_prob_sample = y_prob[indices]

        if len(np.unique(y_true_sample)) < len(classes):
            continue

        y_true_bin = label_binarize(y_true_sample, classes=classes)

        try:
            auc_score = roc_auc_score(y_true_bin, y_prob_sample, multi_class='ovr', average='macro')
            auc_scores.append(auc_score)
        except Exception:
            continue

    lower = np.percentile(auc_scores, 2.5)
    upper = np.percentile(auc_scores, 97.5)
    return lower, upper

classes = np.array(sorted(np.unique(y_test)))
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

y_test_bin = label_binarize(y_test, classes=classes)
auc_macro = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')

acc_ci = bootstrap_metric_ci(y_test, y_pred, accuracy_score)
prec_ci = bootstrap_metric_ci(y_test, y_pred, lambda yt, yp: precision_score(yt, yp, average='macro', zero_division=0))
rec_ci = bootstrap_metric_ci(y_test, y_pred, lambda yt, yp: recall_score(yt, yp, average='macro', zero_division=0))
f1_ci = bootstrap_metric_ci(y_test, y_pred, lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0))
auc_ci = bootstrap_auc_ci(y_test, y_prob, classes=classes)

ci_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1 Macro', 'AUC OVR Macro'],
    'Estimate': [acc, prec, rec, f1, auc_macro],
    'CI Lower (95%)': [acc_ci[0], prec_ci[0], rec_ci[0], f1_ci[0], auc_ci[0]],
    'CI Upper (95%)': [acc_ci[1], prec_ci[1], rec_ci[1], f1_ci[1], auc_ci[1]]
})

ci_table = ci_table.round(4)

print('Performance Metrics with 95% Confidence Intervals')
print(ci_table)

ci_table.to_csv('performance_metrics_with_95CI.csv', index=False)


# ============================================================
# PART AH: SENSITIVITY ANALYSIS WITH DIFFERENT FEATURE SETS
# This checks how model performance changes when symptom-related
# variables are included or excluded.
# ============================================================
best_model_name = 'Logistic Regression'

model_lookup = {
    'Logistic Regression': LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
    'Support Vector Machine': SVC(probability=True, random_state=RANDOM_STATE),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')
}

best_model = model_lookup[best_model_name]
print('Best model selected:', best_model_name)
print('Estimator object:', best_model)

behavior_features = ['age', 'gender', 'daily_time_spent', 'purpose']
symptom_features = ['Anxiety Score', 'Self Esteem Score', 'Depression Score']

feature_sets = {
    'Behavior_Only': behavior_features,
    'Behavior_Plus_Symptoms': behavior_features + symptom_features
}

class_labels = {
    0: 'Normal',
    1: 'Moderate Risk',
    2: 'Severe Risk'
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
sensitivity_results = []
sensitivity_details = {}

for set_name, features in feature_sets.items():
    print('\n' + '=' * 80)
    print(f'SENSITIVITY ANALYSIS: {set_name}')
    print('=' * 80)
    print('Features used:', features)

    model_df_temp = df_model[features + ['Outcome']].copy()
    model_df_temp = model_df_temp.apply(pd.to_numeric, errors='coerce').dropna()

    X = model_df_temp[features]
    y = model_df_temp['Outcome'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
        ('clf', model_lookup[best_model_name])
    ])

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro',
            'roc_auc_ovr': 'roc_auc_ovr'
        },
        return_train_score=True,
        n_jobs=-1
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    classes = np.array(sorted(y.unique()))
    y_test_bin = label_binarize(y_test, classes=classes)

    result = {
        'Best_Model': best_model_name,
        'Feature_Set': set_name,
        'Features_Used': ', '.join(features),
        'CV_Train_Accuracy_Mean': round(np.mean(cv_results['train_accuracy']), 4),
        'CV_Test_Accuracy_Mean': round(np.mean(cv_results['test_accuracy']), 4),
        'CV_Test_Accuracy_SD': round(np.std(cv_results['test_accuracy']), 4),
        'CV_Test_Precision_Macro_Mean': round(np.mean(cv_results['test_precision_macro']), 4),
        'CV_Test_Recall_Macro_Mean': round(np.mean(cv_results['test_recall_macro']), 4),
        'CV_Test_F1_Macro_Mean': round(np.mean(cv_results['test_f1_macro']), 4),
        'CV_Test_AUC_OVR_Mean': round(np.mean(cv_results['test_roc_auc_ovr']), 4),
        'Test_Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Test_Precision_Macro': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'Test_Recall_Macro': round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'Test_F1_Macro': round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'Test_AUC_OVR_Macro': round(roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro'), 4)
    }

    sensitivity_results.append(result)
    sensitivity_details[set_name] = {
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'pipeline': pipeline
    }

    print(f"CV Mean Accuracy: {result['CV_Test_Accuracy_Mean']:.4f}")
    print(f"CV Std Dev: {result['CV_Test_Accuracy_SD']:.4f}")
    print(f"Test Accuracy: {result['Test_Accuracy']:.4f}")
    print(f"Test Precision Macro: {result['Test_Precision_Macro']:.4f}")
    print(f"Test Recall Macro: {result['Test_Recall_Macro']:.4f}")
    print(f"Test F1 Macro: {result['Test_F1_Macro']:.4f}")
    print(f"Test AUC OVR Macro: {result['Test_AUC_OVR_Macro']:.4f}")

    print('\nClassification Report:')
    print(classification_report(
        y_test,
        y_pred,
        target_names=[class_labels[c] for c in sorted(class_labels.keys())],
        zero_division=0
    ))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df = sensitivity_df.sort_values(by='Feature_Set').reset_index(drop=True)

print('\n' + '=' * 80)
print('SENSITIVITY ANALYSIS RESULTS')
print('=' * 80)
print(sensitivity_df)

sensitivity_df.to_csv('sensitivity_analysis_best_model.csv', index=False)


# ============================================================
# PART AI: SHAP EXPLAINABILITY FOR LOGISTIC REGRESSION
# This section generates SHAP summary and bar plots to explain
# the model's feature contributions.
# ============================================================
scaler = pipeline.named_steps['scaler']
clf = pipeline.named_steps['clf']

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

explainer = shap.LinearExplainer(clf, X_train_scaled_df)
shap_values = explainer.shap_values(X_test_scaled_df)

print('Type of shap_values:', type(shap_values))
print('Shape info:', np.array(shap_values).shape if not isinstance(shap_values, list) else [np.array(sv).shape for sv in shap_values])

class_labels = {
    0: 'Normal',
    1: 'Moderate Risk',
    2: 'Severe Risk'
}

if isinstance(shap_values, list):
    for class_idx, sv in enumerate(shap_values):
        print(f'\nSHAP Summary Plot for class {class_idx}: {class_labels.get(class_idx, class_idx)}')
        shap.summary_plot(sv, X_test_scaled_df, feature_names=X_test_scaled_df.columns.tolist(), show=True)
else:
    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        if shap_array.shape[0] == X_test_scaled_df.shape[0]:
            for class_idx in range(shap_array.shape[2]):
                print(f'\nSHAP Summary Plot for class {class_idx}: {class_labels.get(class_idx, class_idx)}')
                shap.summary_plot(shap_array[:, :, class_idx], X_test_scaled_df, feature_names=X_test_scaled_df.columns.tolist(), show=True)
        elif shap_array.shape[1] == X_test_scaled_df.shape[0]:
            for class_idx in range(shap_array.shape[0]):
                print(f'\nSHAP Summary Plot for class {class_idx}: {class_labels.get(class_idx, class_idx)}')
                shap.summary_plot(shap_array[class_idx, :, :], X_test_scaled_df, feature_names=X_test_scaled_df.columns.tolist(), show=True)
    else:
        shap.summary_plot(shap_values, X_test_scaled_df, feature_names=X_test_scaled_df.columns.tolist(), show=True)

if isinstance(shap_values, list):
    for class_idx, sv in enumerate(shap_values):
        print(f'\nSHAP Bar Plot for class {class_idx}: {class_labels.get(class_idx, class_idx)}')
        shap.summary_plot(sv, X_test_scaled_df, feature_names=X_test_scaled_df.columns.tolist(), plot_type='bar', show=True)
else:
    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        if shap_array.shape[0] == X_test_scaled_df.shape[0]:
            for class_idx in range(shap_array.shape[2]):
                print(f'\nSHAP Bar Plot for class {class_idx}: {class_labels.get(class_idx, class_idx)}')
                shap.summary_plot(shap_array[:, :, class_idx], X_test_scaled_df, feature_names=X_test_scaled_df.columns.tolist(), plot_type='bar', show=True)
        elif shap_array.shape[1] == X_test_scaled_df.shape[0]:
            for class_idx in range(shap_array.shape[0]):
                print(f'\nSHAP Bar Plot for class {class_idx}: {class_labels.get(class_idx, class_idx)}')
                shap.summary_plot(shap_array[class_idx, :, :], X_test_scaled_df, feature_names=X_test_scaled_df.columns.tolist(), plot_type='bar', show=True)


# ============================================================
# PART AJ: DETAILED LOGISTIC REGRESSION REPORTING AND PLOTS
# This produces the classification report, confusion matrix,
# ROC curves, and cross-validation accuracy plots.
# ============================================================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
    ('clf', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring={
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'roc_auc_ovr': 'roc_auc_ovr'
    },
    return_train_score=True,
    n_jobs=-1
)

print('CV Accuracy Scores:', cv_results['test_accuracy'])
print(f"CV Mean Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
print(f"CV Std Dev: {np.std(cv_results['test_accuracy']):.4f}")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

class_labels = {
    0: 'Normal',
    1: 'Moderate Risk',
    2: 'Severe Risk'
}

print('\nClassification Report:')
print(classification_report(
    y_test,
    y_pred,
    target_names=[class_labels[c] for c in sorted(class_labels.keys())],
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[class_labels[c] for c in sorted(class_labels.keys())]
)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f'Test Accuracy: {acc:.4f}')
print(f'Test Precision Macro: {prec:.4f}')
print(f'Test Recall Macro: {rec:.4f}')
print(f'Test F1 Macro: {f1:.4f}')

classes = np.array(sorted(y_test.unique()))
y_test_bin = label_binarize(y_test, classes=classes)

plt.figure(figsize=(8, 6))
auc_rows = []

for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    auc_rows.append({
        'Class_Code': int(c),
        'Class_Label': class_labels[int(c)],
        'AUC': round(roc_auc, 4)
    })
    plt.plot(fpr, tpr, linewidth=2, label=f"{class_labels[int(c)]} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curves - Logistic Regression')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

auc_table = pd.DataFrame(auc_rows)
print('\nClass-wise ROC-AUC:')
print(auc_table)

macro_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
print(f'\nMacro AUC OVR: {macro_auc:.4f}')

folds = np.arange(1, 6)

plt.figure(figsize=(8, 5))
plt.plot(folds, cv_results['test_accuracy'], marker='o', linewidth=2, label='Validation Accuracy')
plt.axhline(np.mean(cv_results['test_accuracy']), linestyle='--', linewidth=1.5, label='Mean CV Accuracy')
plt.xticks(folds)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('5-Fold Cross-Validation Accuracy - Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(folds, cv_results['train_accuracy'], marker='o', linewidth=2, label='Train Accuracy')
plt.plot(folds, cv_results['test_accuracy'], marker='s', linewidth=2, label='Validation Accuracy')
plt.xticks(folds)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy Across 5 Folds - Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()


test_acc = accuracy_score(y_test, y_pred)
folds = np.arange(1, 6)

plt.figure(figsize=(8, 5))
plt.plot(folds, cv_results['test_accuracy'], marker='o', linewidth=2, label='CV Accuracy (per fold)')
plt.axhline(np.mean(cv_results['test_accuracy']), linestyle='--', linewidth=1.5, label='Mean CV Accuracy')
plt.axhline(test_acc, linestyle='-.', linewidth=1.5, label=f'Final Test Accuracy = {test_acc:.4f}')
plt.xticks(folds)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('5-Fold CV Accuracy with Final Test Accuracy - Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(folds, cv_results['train_accuracy'], marker='o', linewidth=2, label='Train Accuracy')
plt.plot(folds, cv_results['test_accuracy'], marker='s', linewidth=2, label='CV Accuracy')
plt.axhline(test_acc, linestyle='-.', linewidth=1.5, label=f'Final Test Accuracy = {test_acc:.4f}')
plt.xticks(folds)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Train, CV, and Final Test Accuracy - Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# PART AK: SAVE AUC VALUES FOR EACH CLASS
# This creates a table of class-specific AUC scores.
# ============================================================
class_labels = {
    0: 'Normal',
    1: 'Moderate Risk',
    2: 'Severe Risk'
}

classes = np.array(sorted(y_test.unique()))
y_test_bin = label_binarize(y_test, classes=classes)

auc_rows = []
for i, c in enumerate(classes):
    auc_value = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
    auc_rows.append({
        'Class_Code': int(c),
        'Class_Label': class_labels[int(c)],
        'AUC': round(auc_value, 4)
    })

auc_table = pd.DataFrame(auc_rows)
print('AUC Values for Each Class')
print(auc_table)
auc_table.to_csv('auc_values_each_class.csv', index=False)


# ============================================================
# PART AL: REPORT HYPERPARAMETERS FOR ALL MODELS
# This section exports all hyperparameter settings for transparency.
# ============================================================
hyperparameter_rows = []

for model_name, model in classifiers.items():
    params = model.get_params()
    for param_name, param_value in params.items():
        hyperparameter_rows.append({
            'Model': model_name,
            'Hyperparameter': param_name,
            'Value': param_value
        })

hyperparameter_table = pd.DataFrame(hyperparameter_rows)
print('Hyperparameter Table for All Models')
print(hyperparameter_table)
hyperparameter_table.to_csv('all_model_hyperparameters.csv', index=False)


# ============================================================
# PART AM: REPORT HYPERPARAMETERS FOR THE BEST MODEL ONLY
# This focuses only on the selected best-performing classifier.
# ============================================================
best_model = classifiers[best_model_name]
best_params = best_model.get_params()

best_hyperparameter_table = pd.DataFrame({
    'Hyperparameter': list(best_params.keys()),
    'Value': list(best_params.values())
})

print(f'Hyperparameters for Best Model: {best_model_name}')
print(best_hyperparameter_table)
best_hyperparameter_table.to_csv('best_model_hyperparameters.csv', index=False)


# ============================================================
# PART AN: REPORT PIPELINE HYPERPARAMETERS
# This records all parameters in the preprocessing and model pipeline.
# ============================================================
pipeline_params = pipeline.get_params()

pipeline_hyperparameter_table = pd.DataFrame({
    'Parameter': list(pipeline_params.keys()),
    'Value': list(pipeline_params.values())
})

print('Pipeline Hyperparameters')
print(pipeline_hyperparameter_table)
pipeline_hyperparameter_table.to_csv('pipeline_hyperparameters.csv', index=False)


# ============================================================
# PART AO: RECORD SOFTWARE AND LIBRARY VERSIONS
# This improves reproducibility by documenting package versions.
# ============================================================
software_versions = pd.DataFrame({
    'Software / Library': [
        'Python', 'Platform', 'pandas', 'numpy', 'scikit-learn',
        'imbalanced-learn', 'matplotlib', 'seaborn', 'scipy', 'shap', 'xgboost'
    ],
    'Version': [
        sys.version.split()[0],
        platform.platform(),
        pd.__version__,
        np.__version__,
        sklearn.__version__,
        imblearn.__version__,
        plt.matplotlib.__version__,
        sns.__version__,
        scipy.__version__,
        shap.__version__,
        xgboost.__version__
    ]
})

print('Software and Library Versions')
print(software_versions)
software_versions.to_csv('software_library_versions.csv', index=False)


# ============================================================
# PART AP: SHAP INTERACTION ANALYSIS FOR TREE MODEL
# This section attempts to visualize pairwise feature interaction
# effects for a tree-based model.
# NOTE: rf_model must be defined before running this section.
# ============================================================
class_labels = {
    0: 'Normal',
    1: 'Moderate Risk',
    2: 'Severe Risk'
}

feature_names = X_test.columns.tolist()
n_samples, n_features = X_test.shape
n_classes = len(class_labels)

# IMPORTANT:
# The variable rf_model is not defined in the original notebook before use.
# Define and fit rf_model first if you want this section to run.
# Example:
# rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
# rf_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf_model)
shap_interaction_values = explainer.shap_interaction_values(X_test)

print('type:', type(shap_interaction_values))
print('shape:', np.array(shap_interaction_values).shape if not isinstance(shap_interaction_values, list)
      else [np.array(v).shape for v in shap_interaction_values])


def get_interaction_slice(shap_interaction_values, class_idx, X):
    n_samples, n_features = X.shape
    arr = np.array(shap_interaction_values)

    if isinstance(shap_interaction_values, list):
        out = np.array(shap_interaction_values[class_idx])
    elif arr.ndim == 3:
        out = arr
    elif arr.ndim == 4:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features and arr.shape[2] == n_features:
            out = arr[:, :, :, class_idx]
        elif arr.shape[1] == n_samples and arr.shape[2] == n_features and arr.shape[3] == n_features:
            out = arr[class_idx, :, :, :]
        else:
            raise ValueError(f'Unexpected 4D SHAP interaction shape: {arr.shape}')
    else:
        raise ValueError(f'Unsupported SHAP interaction shape: {arr.shape}')

    if out.shape != (n_samples, n_features, n_features):
        raise ValueError(f'Wrong interaction slice shape: {out.shape}, expected {(n_samples, n_features, n_features)}')

    return out

for class_idx in range(n_classes):
    class_siv = get_interaction_slice(shap_interaction_values, class_idx, X_test)

    print(f'\nInteraction summary for class {class_idx}: {class_labels[class_idx]}')
    print('class slice shape:', class_siv.shape)

    shap.summary_plot(class_siv, X_test, feature_names=feature_names, show=True)
    shap.dependence_plot(
        ('age', 'daily_time_spent'),
        class_siv,
        X_test,
        feature_names=feature_names,
        display_features=X_test,
        show=True
    )


# ============================================================
# PART AQ: CORRELATION MATRIX WITH P-VALUES
# This section calculates Pearson correlation coefficients and
# their significance values for key variables.
# ============================================================
corr_vars = [
    'age', 'gender', 'daily_time_spent', 'purpose',
    'Anxiety Score', 'Self Esteem Score', 'Depression Score', 'Outcome'
]

corr_df = df_model[corr_vars].copy()
corr_df = corr_df.apply(pd.to_numeric, errors='coerce')


def corr_with_pvalues(df):
    cols = df.columns
    n = len(cols)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
    p_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)

    for i in range(n):
        for j in range(n):
            x = df[cols[i]]
            y = df[cols[j]]
            valid = pd.concat([x, y], axis=1).dropna()
            if len(valid) > 1:
                r, p = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                corr_matrix.iloc[i, j] = r
                p_matrix.iloc[i, j] = p
            else:
                corr_matrix.iloc[i, j] = np.nan
                p_matrix.iloc[i, j] = np.nan
    return corr_matrix, p_matrix

corr_matrix, p_matrix = corr_with_pvalues(corr_df)

print('Pearson Correlation Coefficients')
print(corr_matrix.round(4))

print('\nP-values for Correlations')
print(p_matrix.round(6))

combined_table = corr_matrix.copy().astype(str)

for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        r_val = corr_matrix.iloc[i, j]
        p_val = p_matrix.iloc[i, j]
        combined_table.iloc[i, j] = f'r={r_val:.3f}\np={p_val:.3g}'

print('\nCorrelation Matrix with P-values')
print(combined_table)

corr_matrix.to_csv('correlation_coefficients.csv')
p_matrix.to_csv('correlation_pvalues.csv')
combined_table.to_csv('correlation_with_pvalues_combined.csv')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix (Pearson r)')
plt.tight_layout()
plt.show()


# ============================================================
# PART AR: HEATMAP WITH SIGNIFICANCE STARS
# This adds significance markers to the correlation heatmap.
# ============================================================
annot_matrix = corr_matrix.copy().astype(str)

for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        r = corr_matrix.iloc[i, j]
        p = p_matrix.iloc[i, j]
        if pd.isna(r) or pd.isna(p):
            annot_matrix.iloc[i, j] = ''
        else:
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = ''
            annot_matrix.iloc[i, j] = f'{r:.2f}{stars}'

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=annot_matrix,
    fmt='',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    linewidths=0.5
)
plt.title('Correlation Matrix with Significance Levels')
plt.tight_layout()
plt.show()
