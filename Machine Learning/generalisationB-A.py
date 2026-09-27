import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df_a = pd.read_csv("ML Project/Dataset_A_clean.csv")
df_b = pd.read_csv("ML Project/Dataset_B_clean.csv")

X_train = df_b["News"]
y_train = df_b["Authenticity"]

X_test = df_a["Tamil"]
y_test = df_a["Is Fake"]

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "SVM": LinearSVC(class_weight="balanced"),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)

    print(f"\n=== {name} ( B → A) ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
   