import io
import base64

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

report_lines = []
# --- Load dataset ---
X, y = load_wine(return_X_y=True, as_frame=True)
features = X.columns

# Add synthetic sensitive attribute
np.random.seed(42)
X['location'] = np.random.choice([0, 1], size=X.shape[0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

sensitive_feature = X_test['location']

X_train = X_train[features]
X_test = X_test[features]

# Train classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# --- Fairness evaluation ---

# Example metrics
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Overall Macro F1: {f1_macro:.4f}")

# Fairness metrics using MetricFrame
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)

print("\nFairness Metrics by Location:")
print(metric_frame.by_group)
report_lines.append("## Fairness Metrics by Location\n")
# report_lines.append(metric_frame.by_group)

# --- SHAP explainability ---
# Train SHAP explainer on the trained model
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Focus on class "Wine cultivar 3" (index 2)
class_index = 2
for group in [0,1]:
    mask = (sensitive_feature == group)
    summ_plot_html = shap.summary_plot(
        shap_values[...,class_index][mask],
        X_test[mask],
        feature_names=X_test.columns,
    )
    print(f"Displayed SHAP plot for {class_index}, location={group}")

    plt.tight_layout()
    
    # Convert plot to PNG and encode as Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    
    # Embed in Markdown
    report_lines.append(f"### SHAP Summary Plot for Class 3, Location {group}")
    report_lines.append(f"![SHAP](data:image/png;base64,{img_base64})")

# --- Combine everything into Markdown ---
report_md = "\n\n".join(report_lines)

# --- Save Markdown report ---
with open("fairness_explainability_report.md", "w") as f:
    f.write(report_md)
    
sample_idx = 0
force_plot_html = shap.force_plot(
    explainer.expected_value[class_index],
    shap_values[...,class_index],
    X_test
)

# Save as HTML to view in browser
shap.save_html("shap_force_plot.html", force_plot_html)