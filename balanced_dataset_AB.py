import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

def print_confusion_table(cm):
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix")
    print("--------------------------------")
    print(f"True Negative (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Positive (TP): {tp}")
    print("--------------------------------")


df_a = pd.read_csv("Dataset_A_clean.csv")
df_b = pd.read_csv("Dataset_B_clean.csv")

majority = df_a[df_a["Is Fake"] == 0]
minority = df_a[df_a["Is Fake"] == 1]

majority_down = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

balanced_a = pd.concat([majority_down, minority])
print("Balanced Dataset A:")
print(balanced_a["Is Fake"].value_counts())


balanced_a = balanced_a.rename(columns={
    "Tamil": "text",
    "Is Fake": "label"
})

df_b = df_b.rename(columns={
    "News": "text",
    "Authenticity": "label"
})


combined_df = pd.concat(
    [balanced_a, df_b],
    ignore_index=True
)

print("\nCombined Dataset:")
print(combined_df["label"].value_counts())

X = combined_df["text"]
y = combined_df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "SVM": LinearSVC(
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}


for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)
    print(f"\n=== {name} (Balanced A + Dataset B) ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    print_confusion_table(cm)
    