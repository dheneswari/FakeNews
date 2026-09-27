import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


df_a = pd.read_csv("DL Project/Dataset_A_clean.csv")

print("===================================")
print("DATASET A")
print("===================================")
print("\nDataset shape:")
print(df_a.shape)

print("\nDataset A class distribution:")
print(df_a["Is Fake"].value_counts())


df_b = pd.read_csv("DL Project/Dataset_B_clean.csv")

print("\n===================================")
print("DATASET B")
print("===================================")

print("\nDataset shape:")
print(df_b.shape)

print("\nDataset B class distribution:")
print(df_b["Authenticity"].value_counts())


df_a = df_a.dropna(
    subset=["Tamil", "Is Fake"]
)

df_a["Tamil"] = df_a["Tamil"].astype(str)
df_a["Is Fake"] = df_a["Is Fake"].astype(int)
df_a = df_a.rename(
    columns={
        "Tamil": "Text",
        "Is Fake": "Label"
    }
)


df_b = df_b.dropna(
    subset=["News", "Authenticity"]
)

df_b["News"] = df_b["News"].astype(str)
df_b["Authenticity"] = df_b["Authenticity"].astype(int)

df_b = df_b.rename(
    columns={
        "News": "Text",
        "Authenticity": "Label"
    }
)

df_combined = pd.concat(
    [
        df_a[["Text", "Label"]],
        df_b[["Text", "Label"]]
    ],
    ignore_index=True
)


print("\n===================================")
print("COMBINED DATASET")
print("===================================")
print("\nCombined dataset shape:")
print(df_combined.shape)

print("\nCombined class distribution:")
print(df_combined["Label"].value_counts())

real_news = df_combined[
    df_combined["Label"] == 0
]

fake_news = df_combined[
    df_combined["Label"] == 1
]


print("\nReal samples:")
print(len(real_news))

print("Fake samples:")
print(len(fake_news))

BALANCE_SIZE = min(
    len(real_news),
    len(fake_news)
)

real_balanced = real_news.sample(
    n=BALANCE_SIZE,
    random_state=42
)

fake_balanced = fake_news.sample(
    n=BALANCE_SIZE,
    random_state=42
)

df_balanced = pd.concat(
    [
        real_balanced,
        fake_balanced
    ],
    ignore_index=True
)
df_balanced = df_balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\n===================================")
print("BALANCED COMBINED DATASET")
print("===================================")
print("\nBalanced dataset shape:")
print(df_balanced.shape)

print("\nBalanced class distribution:")
print(df_balanced["Label"].value_counts())

X = df_balanced["Text"]
y = df_balanced["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n===================================")
print("TRAIN / TEST SPLIT")
print("===================================")
print("\nTraining samples:")
print(len(X_train))

print("Testing samples:")
print(len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


MAX_WORDS = 20000
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(
    X_train
)

X_test_sequences = tokenizer.texts_to_sequences(
    X_test
)


print("\n===================================")
print("TOKENIZATION")
print("===================================")
print("\nVocabulary size:")
print(len(tokenizer.word_index))

MAX_LENGTH = 200
X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("\n===================================")
print("PADDING")
print("===================================")
print("\nTraining data shape:")
print(X_train_padded.shape)

print("Testing data shape:")
print(X_test_padded.shape)

y_train = np.array(y_train)

y_test = np.array(y_test)

VOCAB_SIZE = min(
    MAX_WORDS,
    len(tokenizer.word_index) + 1
)

EMBEDDING_DIM = 128
model = Sequential([

    # Convert word IDs into dense vectors
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    # CNN layer
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    ),

    # Extract strongest feature
    GlobalMaxPooling1D(),

    # Prevent overfitting
    Dropout(0.5),

    # Fully connected layer
    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.5),

    # Binary classification
    Dense(
        1,
        activation="sigmoid"
    )
])

model.build(
    input_shape=(None, MAX_LENGTH)
)

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n===================================")
print("CNN MODEL SUMMARY")
print("===================================")
model.summary()

EPOCHS = 10
BATCH_SIZE = 32
history = model.fit(
    X_train_padded,
    y_train,
    validation_split=0.10,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose=0
)

print("\n===================================")
print("BALANCED COMBINED DATASET ")
print("===================================")
print( f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")

y_probability = model.predict(
    X_test_padded,
    verbose=0
)

y_pred = (
    y_probability >= 0.5
).astype(int).flatten()

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n===================================")
print("CNN RESULTS")
print("BALANCED COMBINED DATASET")
print("===================================")

print( f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Real",
            "Fake"
        ],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\n===================================")
print("CONFUSION MATRIX VALUES")
print("===================================")
print(f"True Negative  (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive  (TP): {tp}")
