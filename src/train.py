import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
from mlflow.models import infer_signature
from sklearn import metrics
import joblib 
import warnings
warnings.filterwarnings('ignore')
import os, sys

mlflow_flag = 0

try:
    mlflow.set_tracking_uri("http://34.68.129.98:8200")
    print(f"MLFLOW tracking uri: {mlflow.get_tracking_uri()}")
    mlflow.set_experiment("OPPE2 Pract Experiment")
    print("Experiment set: OPPE2 Pract Experiment")
    mlflow_flag = 1
except Exception as e:
    print("Could not connect to mlflow server")

# Define paths
pathname = os.path.dirname(os.path.dirname(sys.argv[0]))
path = os.path.abspath(pathname)
# data_dir = os.path.join('data','iris.csv')
# csv_path = os.path.join(path,data_dir)
model_path = os.path.join(path,'model','model.pkl')

# Load wine dataset as DataFrame
X, y = load_wine(return_X_y=True, as_frame=True)
features = X.columns

np.random.seed(42)  
location = np.random.choice([0, 1], size=X.shape[0])
X['location'] = location

# Split dataset into train and test (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

X_train = X_train[features]
X_test = X_test[features]
# Initialize and train classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
f1_macro = metrics.f1_score(y_test, y_pred, average='macro')

# # Print evaluation metrics
# print("Classification Report:\n", classification_report(y_test, y_pred, target_names=load_wine().target_names))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {f1_macro:.4f}")


# Log to MLFLOW
if mlflow_flag:
    with mlflow.start_run():
        # mlflow.log_params({"POISON_PERCENT": POISON_PERCENT})
        mlflow.log_metric("Accuracy", f"{accuracy:.4f}")
        mlflow.log_metric("Macro F1 Score", f" {f1_macro:.4f}")
        mlflow.set_tag("Training Info",f"Decision Tree on Wine Dataset")
        signature = infer_signature(X_train,y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model = clf,
            name = "Classifier",
            signature = signature,
            input_example = X_train,
            registered_model_name = f"DT_model"
        )
        print("Run tracked in mlflow")

# Save the model
with open(model_path,'wb') as file:
    joblib.dump(clf,file)
    print("Written to ", model_path)