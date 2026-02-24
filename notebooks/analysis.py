import pandas as pd


df = pd.read_csv("data/loan_data.csv")


print("First 5 rows of dataset:")
print(df.head())


print("\nDataset Info:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())
print("\nCleaning Missing Values...")

# Categorical columns
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Numerical columns
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nEncoding Loan_Status...")

df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

print("\nLoan_Status Value Counts:")
print(df['Loan_Status'].value_counts())


print("\nApproval Rate by Credit History:")

approval_by_credit = df.groupby('Credit_History')['Loan_Status'].mean()

print(approval_by_credit)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.barplot(x='Credit_History', y='Loan_Status', data=df)

plt.title("Approval Rate by Credit History")
plt.xlabel("Credit History (0 = Bad, 1 = Good)")
plt.ylabel("Approval Rate")

plt.show()

print("\nCreating Income Groups...")

# Create income bins
df['Income_Group'] = pd.qcut(df['ApplicantIncome'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

print("\nApproval Rate by Income Group:")
print(df.groupby('Income_Group')['Loan_Status'].mean())

print("\nApproval Rate by Income Group AND Credit History:")

combined_analysis = df.groupby(['Income_Group', 'Credit_History'])['Loan_Status'].mean()

print(combined_analysis)

plt.figure()
sns.barplot(x='Income_Group', y='Loan_Status', hue='Credit_History', data=df)

plt.title("Approval Rate by Income & Credit History")
plt.ylabel("Approval Rate")

plt.show()

print("\nApproval Rate by Property Area:")

property_analysis = df.groupby('Property_Area')['Loan_Status'].mean()

print(property_analysis)

plt.figure()
sns.barplot(x='Property_Area', y='Loan_Status', data=df)

plt.title("Approval Rate by Property Area")
plt.ylabel("Approval Rate")

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Convert categorical columns using one-hot encoding
df_model = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df_model.drop('Loan_Status', axis=1)
y = df_model['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))