import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title='Fraud Detection EDA', layout='wide')
st.title('📊 Fraud Detection - EDA Dashboard')

##_____EDA_____

import numpy as np
print ("Numpy version:", np.__version__)

import pandas as pd
df = pd.read_csv("/content/financial-fraud-detection-dataset/Synthetic_Financial_datasets_log.csv")
df

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

print(df.shape)

print(df.columns)

print(df.dtypes)

print(df.isnull().sum())

df.head()

#Count of fraudulent vs non-fraudulent transactions
fraud_counts = df['isFraud'].value_counts()
print(fraud_counts[0])
print(fraud_counts[1])

fraud_percent = (fraud_counts[1] / len(df)) * 100
print(f"\n⚠️ Fraudulent Transactions Percentage: {fraud_percent:.4f}%")

# Bar chart
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='isFraud', data=df, palette={'0': 'skyblue', '1': 'salmon'})
plt.xticks([0, 1], ['Not Fraud (0)', 'Fraud (1)'])
plt.title("Fraudulent vs Non-Fraudulent Transactions")
plt.xlabel("isFraud")
plt.ylabel("Count")
plt.show()

# Count of transaction types
print(df['type'].value_counts())

#Visualize fraud distribution by transaction type
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
sns.countplot(data=df, x='type', hue='isFraud', palette='Set2')
plt.title("Fraud Count by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title='isFraud', labels=['Not Fraud (0)', 'Fraud (1)'])
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

#Boxplot to show amount distribution by fraud status
plt.figure(figsize=(8,5))
sns.boxplot(x='isFraud', y='amount', data=df, palette={'0': 'skyblue', '1': 'salmon'})
plt.yscale('log')  # log scale helps handle extreme outliers
plt.title("Transaction Amount by Fraud Status (Log Scale)")
plt.xlabel("Fraud Status (0 = Not Fraud, 1 = Fraud)")
plt.ylabel("Transaction Amount (Log Scale)")
plt.grid(True)
plt.tight_layout()
plt.show()

#Histogram for visual comparison
plt.figure(figsize=(10,5))
sns.histplot(data=df, x='amount', bins=100, hue='isFraud', multiple='stack', palette={0: 'skyblue', 1: 'salmon'}, log_scale=(False, True))
plt.title("Distribution of Transaction Amounts (Stacked by Fraud Status)")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency (Log Scale)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

#Check time-based distribution using 'step'
plt.figure(figsize=(12,5))
sns.histplot(data=df[df['isFraud'] == 1], x='step', bins=100, color='red', label='Fraud', alpha=0.7)
sns.histplot(data=df[df['isFraud'] == 0], x='step', bins=100, color='blue', label='Not Fraud', alpha=0.5)
plt.title("Fraudulent vs Non-Fraudulent Transactions Over Time (step)")
plt.xlabel("Time Step")
plt.ylabel("Number of Transactions")
plt.legend()
plt.tight_layout()
plt.show()

#Top 10 sender accounts involved in fraud
fraud_origins = df[df['isFraud'] == 1]['nameOrig'].value_counts().head(10)
print(fraud_origins)

#Top 10 receiver accounts involved in fraud
fraud_destinations = df[df['isFraud'] == 1]['nameDest'].value_counts().head(10)
print(fraud_destinations)

#visualize as bar chart
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
fraud_origins.plot(kind='barh', color='crimson')
plt.title("Top 10 Origin Accounts Involved in Fraud")
plt.xlabel("Number of Fraudulent Transactions")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
fraud_destinations.plot(kind='barh', color='darkorange')
plt.title("Top 10 Destination Accounts Involved in Fraud")
plt.xlabel("Number of Fraudulent Transactions")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Select only numerical columns before computing correlation matrix
numerical_df = df.select_dtypes(include=[np.number])
corr = numerical_df.corr()

# Display the correlation matrix (optional, but helpful)
display(corr)

# You can then proceed to visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

#title step

from matplotlib import pyplot as plt
corr['step'].plot(kind='hist', bins=20, title='step')
plt.gca().spines[['top', 'right',]].set_visible(False)

#Transaction Amount by Type & Fraud
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[df['amount'] > 0], x='type', y='amount', hue='isFraud')
plt.yscale('log')
plt.title("Transaction Amount by Type and Fraud Status")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Hourly Pattern of Fraud
df['hour'] = df['step'] % 24

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='hour', hue='isFraud', multiple='stack', bins=24, palette='coolwarm')
plt.title("Fraud vs Non-Fraud Transactions by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Count")
plt.show()

#Balance Analysis Before & After Transaction
df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

sns.boxplot(x='isFraud', y='errorBalanceOrig', data=df)
plt.title("Error in Origin Balance by Fraud Status")
plt.show()

sns.boxplot(x='isFraud', y='errorBalanceDest', data=df)
plt.title("Error in Destination Balance by Fraud Status")
plt.show()

#High-Amount Frauds Detection
high_value_frauds = df[(df['isFraud'] == 1) & (df['amount'] > 1e6)]
print("High-value frauds:\n", high_value_frauds[['type', 'amount', 'nameOrig', 'nameDest']].head())

#Pairplot of Key Features
import seaborn as sns
sample_df = df.sample(1000, random_state=42)  # Limit for performance
sns.pairplot(sample_df, hue='isFraud', vars=['amount', 'oldbalanceOrg', 'newbalanceOrig'])
plt.show()





##______Feature Engineering + Model Training________

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Copy to avoid changing original
data = df.copy()

# --- Feature Engineering ---

# Convert 'step' to datetime (assuming step represents time in some unit, e.g., hours)
# Note: The dataset documentation would clarify the unit of 'step'.
# For now, we'll treat 'step' as hours and convert to a timedelta from a reference point
data['transaction_time'] = pd.to_timedelta(data['step'], unit='h')

# Extract time-based features
data['transaction_hour'] = data['transaction_time'].dt.components.hours
data['transaction_day'] = data['transaction_time'].dt.days

# High-value transaction flag (top 5% amounts)
data['is_large_amount'] = (data['amount'] > data['amount'].quantile(0.95)).astype(int)

# --- Handle Categorical Variables ---
categorical_cols = ['transaction_type', 'device_type', 'merchant']  # Replace with actual categorical columns
le = LabelEncoder()

# --- Handle Categorical Variables ---
# One-Hot Encode the 'type' column
X = pd.get_dummies(X, columns=['type'], drop_first=True)

# --- Drop Unnecessary Columns ---
# Drop columns that are not needed for training. We'll keep 'nameOrig' and 'nameDest' for now,
# as they might be useful for graph-based analysis later, but drop IDs and the original time column.
drop_cols = ['nameOrig', 'nameDest', 'transaction_time']
for col in drop_cols:
    if col in data.columns:
        # We should drop from the X dataframe after splitting
        if col in X.columns:
            X.drop(col, axis=1, inplace=True)

# --- Split into Features (X) and Target (y) ---
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print("Preprocessing Done!")
print(f"X_train shape: {X_train.shape}, y_train fraud count: {sum(y_train)}")

##______Model Training (XGBoost)

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib # Import joblib

# Initialize XGBoost Classifier
xgb_model = XGBClassifier(
    use_label_encoder=False,
    # eval_metric='logloss', # Removed eval_metric
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # handle class imbalance
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # For AUC

# Save the XGBoost model
joblib.dump(xgb_model, 'xgboost_fraud_model.pkl')
print("XGBoost model saved as 'xgboost_fraud_model.pkl'")

# Evaluation Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(xgb_model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

##______Model Evaluation & Visualization______

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    precision_recall_curve,
    auc
)

# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC-AUC Score
roc_score = roc_auc_score(y_test, y_pred_proba)
print(f"📈 ROC-AUC Score: {roc_score:.4f}")

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_pred_proba)
plt.title("ROC Curve")
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.', label=f'PR AUC = {pr_auc:.4f}')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.show()

##______LightGBM_______

pip install lightgbm

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Initialize LightGBM classifier
lgb_model = LGBMClassifier(
    class_weight='balanced',  # handle class imbalance
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7
)

# Train the model
lgb_model.fit(X_train, y_train)

# Predict
y_pred = lgb_model.predict(X_test)
y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]  # For AUC/PR curves

# Classification metrics
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title("Confusion Matrix - LightGBM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC AUC Score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"📈 ROC-AUC Score: {auc_score:.4f}")

import joblib
joblib.dump(lgb_model, 'lightgbm_fraud_model.pkl')
print("LightGBM model saved as 'lightgbm_fraud_model.pkl'")



pip install secure-smtplib

##_______Email Alert Function After Prediction ________

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert_email(transaction_info, triggered_by="Model", to_email=None):
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")
    subject = "Fraud Alert Detected!"
    body = f"""
    A potential fraudulent transaction has been detected by: {triggered_by}

    Transaction Details:
    ---------------------
    {transaction_info}
    """

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()
        print(f"Email sent for alert by {triggered_by}")
    except Exception as e:
        print(f"Failed to send email: {e}")

import numpy as np
import os # Import the os module

xgb_preds = xgb_model.predict(X_test)
lgb_preds = lgb_model.predict(X_test)
combined_preds = np.where((xgb_preds==1) | (lgb_preds==1), 1, 0)

# Map back to original DataFrame indices
df_test = df.iloc[y_test.index].reset_index(drop=True)
X_test_df = pd.DataFrame(X_test, columns=X.columns)  # for context

for idx, pred in enumerate(combined_preds):
    if pred == 1:
        triggered_by = []
        if xgb_preds[idx] == 1: triggered_by.append("XGBoost")
        if lgb_preds[idx] == 1: triggered_by.append("LightGBM")
        model_names = ", ".join(triggered_by)

        trans_info = df_test.iloc[idx].to_string()
        send_alert_email(transaction_info=trans_info, triggered_by=model_names, to_email=os.getenv("ALERT_EMAIL"))






