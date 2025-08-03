# Credit-Card-Fraud-Detection---EDA
Training set for Credit Card Transactions
index - Unique Identifier for each row
trans_date_trans_time - Transaction DateTime
cc_num - Credit Card Number of Customer
merchant - Merchant Name
category - Category of Merchant
amt - Amount of Transaction
first - First Name of Credit Card Holder
last - Last Name of Credit Card Holder
gender - Gender of Credit Card Holder
street - Street Address of Credit Card Holder
city - City of Credit Card Holder
state - State of Credit Card Holder
zip - Zip of Credit Card Holder
lat - Latitude Location of Credit Card Holder
long - Longitude Location of Credit Card Holder
city_pop - Credit Card Holder's City Population
job - Job of Credit Card Holder
dob - Date of Birth of Credit Card Holder
trans_num - Transaction Number
unix_time - UNIX Time of transaction
merch_lat - Latitude Location of Merchant
merch_long - Longitude Location of Merchant
is_fraud - Fraud Flag <--- Target Class
# Importing all essential python libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# reading the data
data_train = pd.read_csv("/kaggle/input/fraud-detection/fraudTrain.csv")
data_test = pd.read_csv("/kaggle/input/fraud-detection/fraudTest.csv")

# combining all train and test data for detailed-EDA
full_data = pd.concat([data_train, data_test], axis=0)
full_data.head()

# Data Cleaning

full_data.info()

full_data.isnull().sum()

# check for duplicated records
full_data.duplicated().sum()

# drop irrelevent col 
full_data.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'first', 'last', 'unix_time', 'trans_num', 'street', 'city'], inplace=True)

# convert to correct datetime type
full_data['trans_date_trans_time'] = pd.to_datetime(full_data['trans_date_trans_time'])

# extract features from data - feature engg
full_data['day'] = full_data['trans_date_trans_time'].dt.day
full_data['month'] = full_data['trans_date_trans_time'].dt.month
full_data['year'] = full_data['trans_date_trans_time'].dt.year
full_data['hour'] = full_data['trans_date_trans_time'].dt.hour
full_data['minute'] = full_data['trans_date_trans_time'].dt.minute

full_data.drop(columns=['trans_date_trans_time'], inplace=True)

# extract age of person
full_data['dob'] = pd.to_datetime(full_data['dob'])
full_data['year_dob'] = full_data['dob'].dt.year
full_data['age'] = full_data['year'] - full_data['year_dob']


full_data['is_fraud'].unique()

# EDA - Detailed

**1. insights for numeric-type columns**

# statistical insights for numeric-datatypes features
full_data.describe()

**2. insights of non-numeric types columns:**

# statistical insights for non-numeric-datatypes features

full_data.describe(include='object')

**3. Fraud Rate by Category, State, Gender**

data = full_data

# List of categorical columns to compare
categorical_cols = ['category', 'gender', 'state']

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(16, 4 * len(categorical_cols)))

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(len(categorical_cols), 1, i)
    temp = data.groupby(col)['is_fraud'].mean().sort_values(ascending=False)
    sns.barplot(x=temp.index, y=temp.values, palette="viridis")
    plt.ylabel('Fraud Rate (%)')
    plt.title(f'Fraud Rate by {col}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

plt.suptitle("Fraud Rate by Categorical Features", fontsize=18, y=1.02)
plt.show()


**4. Fraud Rate byJob Titles**

# Calculate fraud rate (%) by job title
fraud_by_jobs = (
    data.groupby('job')['is_fraud']
    .mean()
    .multiply(100)
    .reset_index(name='fraud_rate_%')
    .sort_values(by='fraud_rate_%', ascending=False)
)

# Top 10 jobs with highest fraud rate
top_fraud_jobs = fraud_by_jobs.head(10)
print("Top 10 Jobs with Highest Fraud Rate:\n")
print(top_fraud_jobs.to_string(index=False))

print()

# Bottom 10 jobs with lowest fraud rate
bottom_fraud_jobs = fraud_by_jobs.tail(10)
print("\nBottom 10 Jobs with Lowest Fraud Rate:\n")
print(bottom_fraud_jobs.to_string(index=False))


# Set figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

# ----- Top 10 Jobs -----
ax1.bar(top_fraud_jobs['job'], top_fraud_jobs['fraud_rate_%'], color='crimson')
ax1.set_title('🔝 Top 10 Jobs with Highest Fraud Rate', fontsize=14)
ax1.set_ylabel('Fraud Rate (%)')
ax1.set_xticklabels(top_fraud_jobs['job'], rotation=45, ha='right')

# ----- Bottom 10 Jobs -----
ax2.bar(bottom_fraud_jobs['job'], bottom_fraud_jobs['fraud_rate_%'], color='seagreen')
ax2.set_title('🔻 Bottom 10 Jobs with Lowest Fraud Rate', fontsize=14)
ax2.set_ylabel('Fraud Rate (%)')
ax2.set_xticklabels(bottom_fraud_jobs['job'], rotation=45, ha='right')

# Overall title
plt.suptitle('Fraud Rate by Job Title (Top & Bottom)', fontsize=16)
plt.show()


***code check to compare our comparison methods for fraud_rate_%***

# code check to compare our comparison methods for fraud_rate_%

# Count fraud and non-fraud per job
job_counts = data.groupby(['job', 'is_fraud']).size().unstack(fill_value=0)

# Total transactions per job
job_counts['total'] = job_counts[0] + job_counts[1]

# Fraud rate calculation
job_counts['fraud_rate_%'] = (job_counts[1] / job_counts['total']) * 100

# Check if both are same
method1 = data.groupby('job')['is_fraud'].mean().sort_index()
method2 = (job_counts[1] / job_counts['total']).sort_index()

(method1.equals(method2))  # Should return True


**5. Fraud Rate by State**


# -------------------- Fraud Rate by State --------------------
state_grouped = (
    data.groupby('state')['is_fraud']
    .mean()
    .multiply(100)
    .reset_index(name='fraud_rate')
    .sort_values(by='fraud_rate', ascending=False)
)

top_states = state_grouped.head(10)
bottom_states = state_grouped.tail(10)

# -------------------- Plotting --------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)


# Top 10 States
axes[0].bar(top_states['state'], top_states['fraud_rate'], color='tomato')
axes[0].set_title('🔝 Top 10 States with Highest Fraud Rate')
axes[0].set_ylabel('Fraud Rate (%)')
axes[0].set_xticklabels(top_states['state'], rotation=45, ha='right')

# Bottom 10 States
axes[1].bar(bottom_states['state'], bottom_states['fraud_rate'], color='mediumseagreen')
axes[1].set_title('🔻 Bottom 10 States with Lowest Fraud Rate')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].set_xticklabels(bottom_states['state'], rotation=45, ha='right')

plt.suptitle('Fraud Rate by State (Top vs Bottom)', fontsize=18)
plt.show()


**6. Fraud Rate by Year, Month, Day, Hour, Minute**

def fraud_rate_by_time(feature):
    # Group by time feature and fraud status
    temp = data.groupby([feature, 'is_fraud'])['is_fraud'].count().unstack(fill_value=0)

    # Rename columns for clarity
    temp.columns = ['non_fraud', 'fraud']

    # Add total transactions and fraud rate
    temp['total_txn'] = temp['non_fraud'] + temp['fraud']
    temp['fraud_rate_%'] = (temp['fraud'] / temp['total_txn']) * 100

    return temp.sort_values(by=feature)

# Apply to all time-based features
fraud_by_year = fraud_rate_by_time('year')
fraud_by_month = fraud_rate_by_time('month')
fraud_by_day = fraud_rate_by_time('day')
fraud_by_hour = fraud_rate_by_time('hour')
fraud_by_minute = fraud_rate_by_time('minute')

# Display all results
print("📆 Fraud Rate by Year:\n", fraud_by_year)
print("\n📅 Fraud Rate by Month:\n", fraud_by_month)
print("\n📅 Fraud Rate by Day:\n", fraud_by_day)
print("\n⏰ Fraud Rate by Hour:\n", fraud_by_hour)
print("\n⏱️ Fraud Rate by Minute:\n", fraud_by_minute)


**7. Fraud Rate by Age**

import matplotlib.pyplot as plt
import pandas as pd

# 1. Define age bins and labels
bins = [10, 20, 30, 40, 50, 60, 70, 80, 90]
labels = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

# 2. Create a new age group column
full_data['age_group'] = pd.cut(full_data['age'], bins=bins, labels=labels, right=False)

# 3. Group by age group and calculate fraud rate
age_grouped = (
    full_data.groupby('age_group')['is_fraud']
    .mean()
    .multiply(100)
    .reset_index(name='fraud_rate_%')
)

# 4. Plot histogram-style bar plot
plt.figure(figsize=(10, 6))
plt.bar(age_grouped['age_group'], age_grouped['fraud_rate_%'], color='cornflowerblue')
plt.title('Fraud Rate by Age Group', fontsize=14)
plt.xlabel('Age Group')
plt.ylabel('Fraud Rate (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()




**8. Finding correlation of variables (numerics only) with target variable**

ans = data.corr(numeric_only=True)
ans['is_fraud']

data.columns

# Data extracting for Model Training

X = data.drop(columns=[
    'gender', 'state', 'zip',
    'dob', 'year_dob', 'is_fraud'  
])

y = data['is_fraud']

from sklearn.preprocessing import LabelEncoder

# Columns to encode
cat_cols = ['merchant', 'category', 'job', 'age_group']

# Initialize encoder
le = LabelEncoder()

# Encode each column
for col in cat_cols:
    X[col] = le.fit_transform(X[col])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


print(f"X_train shape: ", X_train.shape)
print(f"y_train shape: ", y_train.shape)
print()
print(f"X_test shape: ", X_test.shape)
print(f"y-test shape: ", y_test.shape)

from sklearn.preprocessing import StandardScaler

# Select only numeric columns to scale
cols_to_scale = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'day', 'month', 'year', 'hour', 'minute', 'age']

# Initialize scaler
scaler = StandardScaler()

# Fit the scaler on X_train only and transform
X_train_scaled = X_train.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])

# Use the same scaler (do not refit) to transform X_test
X_test_scaled = X_test.copy()
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])


# 1. Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt

# Base model with important hyperparameters
lr = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

# Fit the model
lr.fit(X_train_scaled, y_train)

# Predict
y_pred = lr.predict(X_test_scaled)

# Metrics
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
conf_mt = confusion_matrix(y_test, y_pred)

# Print results
print("Accuracy     :", acc)
print("Recall       :", rec)
print("Precision    :", prec)
print("Confusion Matrix:\n", conf_mt)

# Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mt, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


The model gives about 95% accuracy, which is considered as good number. But recall is 75%, which is quite low. It means, while predicting frauds, we missed the vast number of frauds (about of 25%). So, need to optimise it!
