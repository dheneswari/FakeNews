import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Dataset_A_clean.csv")

print("Before balancing:")
print(df["Is Fake"].value_counts())

majority = df[df["Is Fake"] == 0]
minority = df[df["Is Fake"] == 1]

majority_down = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

balanced_df = pd.concat([majority_down, minority])

print("\nAfter balancing:")
print(balanced_df["Is Fake"].value_counts())

X = balanced_df["Tamil"]
y = balanced_df["Is Fake"]

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

    print(f"\n=== {name} (Balanced Dataset A) ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))