import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
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

df_A = pd.read_csv("DL Project/Dataset_A_clean.csv")

print("DATASET A")
print("\nFirst 5 rows:")
print(df_A.head())
print("\nColumns:")
print(df_A.columns)
print("\nDataset A shape:")
print(df_A.shape)
print("\nDataset A class distribution:")
print(df_A["Is Fake"].value_counts())

df_A = df_A.dropna(
    subset=["Tamil", "Is Fake"]
)

df_A["Tamil"] = df_A["Tamil"].astype(str)
df_A["Is Fake"] = df_A["Is Fake"].astype(int)


print("\nDataset A shape after removing missing values:")
print(df_A.shape)

df_B = pd.read_csv("DL Project/Dataset_B_clean.csv")

print("\n===================================")
print("DATASET B")
print("===================================")

print("\nFirst 5 rows:")
print(df_B.head())

print("\nColumns:")
print(df_B.columns)

print("\nDataset B shape:")
print(df_B.shape)

print("\nDataset B class distribution:")
print(df_B["Authenticity"].value_counts())


df_B = df_B.dropna(
    subset=["News", "Authenticity"]
)

df_B["News"] = df_B["News"].astype(str)
df_B["Authenticity"] = df_B["Authenticity"].astype(int)


print("\nDataset B shape after removing missing values:")
print(df_B.shape)


X_A = df_A["Tamil"]
y_A = df_A["Is Fake"]


print("\n===================================")
print("DATASET A TRAINING DATA")
print("===================================")

print("Training samples:")
print(len(X_A))

print("\nTraining class distribution:")
print(y_A.value_counts())


X_B = df_B["News"]
y_B = df_B["Authenticity"]


print("\n===================================")
print("DATASET B TESTING DATA")
print("===================================")

print("Testing samples:")
print(len(X_B))

print("\nTesting class distribution:")
print(y_B.value_counts())

MAX_WORDS = 20000
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_A)

print("\n===================================")
print("TOKENIZATION")
print("===================================")

print("Vocabulary size:")
print(len(tokenizer.word_index))


X_A_sequences = tokenizer.texts_to_sequences(
    X_A
)

X_B_sequences = tokenizer.texts_to_sequences(
    X_B
)

MAX_LENGTH = 200
X_A_padded = pad_sequences(
    X_A_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_B_padded = pad_sequences(
    X_B_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)
print("PADDING")
print("Dataset A training data shape:")
print(X_A_padded.shape)

print("Dataset B testing data shape:")
print(X_B_padded.shape)


y_A = np.array(y_A)

y_B = np.array(y_B)

classes = np.unique(y_A)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_A
)

class_weights = dict(
    zip(classes, weights)
)


print("\n===================================")
print("CLASS WEIGHTS")
print("===================================")
print(class_weights)

VOCAB_SIZE = min(
    MAX_WORDS,
    len(tokenizer.word_index) + 1
)

EMBEDDING_DIM = 128


model = Sequential([

    # Word embedding
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    # CNN feature extraction
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    ),

    # Select strongest feature
    GlobalMaxPooling1D(),

    # Reduce overfitting
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

    X_A_padded,
    y_A,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    verbose=1
)

B_loss, B_accuracy = model.evaluate(
    X_B_padded,
    y_B,
    verbose=0
)

print("\n===================================")
print("EXPERIMENT 3")
print("DATASET A → DATASET B")
print("===================================")

print(f"Test Loss     : {B_loss:.4f}")
print(f"Test Accuracy : {B_accuracy:.4f}")


y_probability = model.predict(
    X_B_padded,
    verbose=0
)

y_pred = (
    y_probability >= 0.5
).astype(int).flatten()

accuracy = accuracy_score(
    y_B,
    y_pred
)

precision = precision_score(
    y_B,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_B,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_B,
    y_pred,
    zero_division=0
)

print("\n===================================")
print("CNN CROSS-DATASET RESULTS")
print("DATASET A → DATASET B")
print("===================================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_B,
        y_pred,
        target_names=["Real", "Fake"],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_B,
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
