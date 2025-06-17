import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib


X = np.load("embeddings.npy")
df_labels = pd.read_csv("labels.csv")


y = df_labels["category"]
filenames = df_labels["filename"]


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X, y_encoded, filenames, test_size=0.2, stratify=y_encoded, random_state=42
)


train_df = pd.DataFrame({
    "filename": filenames_train,
    "category": le.inverse_transform(y_train)
})
train_df.to_csv("train_files.csv", index=False)

test_df = pd.DataFrame({
    "filename": filenames_test,
    "category": le.inverse_transform(y_test)
})
test_df.to_csv("test_files.csv", index=False)

print("Saved train_files.csv and test_files.csv.")


clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=500, early_stopping=True, verbose=True)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))


joblib.dump(clf, "mlp_classifier.joblib")
joblib.dump(le, "label_encoder.joblib")
print("Model saved.")


# training loss
plt.figure(figsize=(8, 5))
plt.plot(clf.loss_curve_, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png")
print("Saved training_loss.png")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")