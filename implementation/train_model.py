

from pathlib import Path
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR /
    "data" /
    "twitter_processed.csv"
)

MODELS_DIR = (
    BASE_DIR /
    "models"
)

MODELS_DIR.mkdir(
    exist_ok=True
)

df = pd.read_csv(DATA_PATH)

df.dropna(inplace=True)

# For CPU-friendly testing, we sample a subset of the dataset by default.
# Set SUBSET_SIZE to None or len(df) to train on the complete dataset.
SUBSET_SIZE = 500
if SUBSET_SIZE and SUBSET_SIZE < len(df):
    df = df.sample(SUBSET_SIZE, random_state=42).reset_index(drop=True)

X = df["Tweet"].astype(str)

y = df["Sentiment"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

with open(
    MODELS_DIR /
    "label_encoder.pkl",
    "wb"
) as f:

    pickle.dump(
        encoder,
        f
    )

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

# Tokenize to lists rather than TensorFlow tensors first
# to prevent compatibility issues with train_test_split
encoded = tokenizer(
    X.tolist(),
    padding=True,
    truncation=True,
    max_length=128
)

X_train_ids, X_test_ids, y_train, y_test = train_test_split(
    encoded["input_ids"],
    y,
    test_size=0.2,
    random_state=42
)

# Convert lists to numpy arrays for compatibility with Keras/TF training
import numpy as np
X_train_ids = np.array(X_train_ids)
y_train = np.array(y_train)

model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(
        encoder.classes_
    )
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=2e-5
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True
)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=["accuracy"]
)

model.fit(
    X_train_ids,
    y_train,
    epochs=2,
    batch_size=8
)

model.save_pretrained(
    MODELS_DIR /
    "bert_model"
)

tokenizer.save_pretrained(
    MODELS_DIR /
    "tokenizer"
)

print(
    "BERT Model Saved"
)