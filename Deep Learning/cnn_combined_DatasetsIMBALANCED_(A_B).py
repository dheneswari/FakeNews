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
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

df_A = pd.read_csv(
    "DL Project/Dataset_A_clean.csv"
)

print("===================================")
print("DATASET A")
print("===================================")
print("\nDataset A shape:")
print(df_A.shape)

print("\nDataset A class distribution:")
print(df_A["Is Fake"].value_counts())


df_B = pd.read_csv(
    "DL Project/Dataset_B_clean.csv"
)

print("\n===================================")
print("DATASET B")
print("===================================")
print("\nDataset B shape:")
print(df_B.shape)

print("\nDataset B class distribution:")
print(df_B["Authenticity"].value_counts())


df_A_combined = pd.DataFrame({
    "Text": df_A["Tamil"],
    "Label": df_A["Is Fake"]
})

df_B_combined = pd.DataFrame({
    "Text": df_B["News"],
    "Label": df_B["Authenticity"]
})

df_combined = pd.concat(
    [
        df_A_combined,
        df_B_combined
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

X = df_combined["Text"]
y = df_combined["Label"]

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

tokenizer.fit_on_texts(
    X_train
)

print("\n===================================")
print("TOKENIZATION")
print("===================================")
print("Vocabulary size:")
print(len(tokenizer.word_index))

X_train_sequences = tokenizer.texts_to_sequences(
    X_train
)

X_test_sequences = tokenizer.texts_to_sequences(
    X_test
)

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
print("Training data shape:")
print(X_train_padded.shape)

print("Testing data shape:")
print(X_test_padded.shape)

y_train = np.array(y_train)

y_test = np.array(y_test)

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
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

    # Strongest feature
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
    class_weight=class_weights,
    verbose=1
)

loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose=0
)

print("\n===================================")
print("EXPERIMENT 5")
print("COMBINED DATASET → COMBINED DATASET")
print("===================================")
print(f"Test Loss     : {loss:.4f}")
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
print("COMBINED DATASET")
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

