import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

df = pd.read_csv("DL Project/Dataset_A_clean.csv")

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["Is Fake"].value_counts())

df = df.dropna(subset=["Tamil", "Is Fake"])
df["Tamil"] = df["Tamil"].astype(str)
df["Is Fake"] = df["Is Fake"].astype(int)

print("\nDataset shape after removing missing values:")
print(df.shape)

X = df["Tamil"]
y = df["Is Fake"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
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

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
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


print("\nTraining data shape after padding:")
print(X_train_padded.shape)
print("Testing data shape after padding:")
print(X_test_padded.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))
print("\nClass weights:")
print(class_weights)

VOCAB_SIZE = min(MAX_WORDS, len(tokenizer.word_index) + 1)
EMBEDDING_DIM = 128
model = Sequential([
    
    # Convert word IDs into dense vectors
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LENGTH
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

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary:")
model.summary()
EPOCHS = 10
BATCH_SIZE = 32

history = model.fit(
    X_train_padded,
    y_train,
    validation_split=0.10,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    verbose=1
)

loss, accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose=0
)

print("\n===================================")
print("CNN TEST RESULTS")
print("===================================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

y_probability = model.predict(
    X_test_padded,
    verbose=0
)

y_pred = (y_probability >= 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)

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
print("CLASSIFICATION METRICS")
print("===================================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Real", "Fake"],
        zero_division=0
    )
)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)