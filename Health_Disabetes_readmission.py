import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1. Setup
os.makedirs('outputs', exist_ok=True)
data_path = 'D:\\hospital_readmissions.csv'

# 2. Load Data
df = pd.read_csv(data_path)

# 3. Clean Data
# Check for '?' as missing value
if (df == '?').any().any():
    df.replace('?', pd.NA, inplace=True)
    df = df.dropna()

# 4. Target Encoding
df['readmitted_binary'] = df['readmitted'].map({'yes': 1, 'no': 0})

# 5. Encode Categorical Features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col != 'readmitted':
        df[col] = le.fit_transform(df[col])

# 6. EDA Visualizations
# 6.1 Class Balance
plt.figure(figsize=(5,3))
sns.countplot(x='readmitted_binary', data=df)
plt.title('Class Balance: Readmitted')
plt.xticks([0,1], ['No', 'Yes'])
plt.savefig('outputs/class_balance.png')
plt.close()

# 6.2 Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('outputs/correlation_heatmap.png')
plt.close()

# 7. Modeling
X = df.drop(['readmitted', 'readmitted_binary'], axis=1)
y = df['readmitted_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# 8. Model Evaluation
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

# 8.1 Confusion Matrix
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# 8.2 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('outputs/roc_curve.png')
plt.close()

# 8.3 Feature Importance
feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')
plt.close()

# 9. Save Summary Report
with open('outputs/summary_report.txt', 'w') as f:
    f.write("=== Hospital Readmission Prediction Report ===\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write(f"ROC AUC Score: {roc_auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Top 10 Feature Importances:\n")
    f.write(str(feat_imp.head(10)) + "\n")

print("\n✅ Project complete! All plots and report are saved in the 'outputs' folder.")

# 10. (Optional) Streamlit Dashboard
"""
# Save this as dashboard.py and run: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:\\hospital_readmissions.csv')
df['readmitted_binary'] = df['readmitted'].map({'yes': 1, 'no': 0})
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col != 'readmitted':
        df[col] = le.fit_transform(df[col])

st.title("Hospital Readmission Dashboard")
st.subheader("Class Balance")
st.bar_chart(df['readmitted_binary'].value_counts())
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm', ax=ax)
st.pyplot(fig)
st.subheader("Raw Data")
st.write(df.head())
"""
