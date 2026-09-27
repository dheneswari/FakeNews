import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df_a = pd.read_csv("ML Project/Dataset_A_clean.csv")
df_b = pd.read_csv("ML Project/Dataset_B_clean.csv")

X = pd.concat([df_a["Tamil"], df_b["News"]], ignore_index=True)
y = pd.concat([df_a["Is Fake"], df_b["Authenticity"]], ignore_index=True)

print("Dataset A size:")
print("Total:", len(df_a))
print(df_a["Is Fake"].value_counts())

print("\nDataset B size:")
print("Total:", len(df_b))
print(df_b["Authenticity"].value_counts())

print("\nCombined Dataset size:")
print("Total:", len(df_a) + len(df_b))

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
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)

    print(f"\n=== {name} (Combined Dataset) ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))