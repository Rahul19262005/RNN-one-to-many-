# RNN Practical (One to Many)

# Text Generation using Encoder-Decoder Simple RNN

# Dataset : technology_sentences.csv

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SimpleRNN,
    Dense,
    RepeatVector,
    TimeDistributed
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

MODEL = "one_to_many_rnn.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000

# ----------------------------------------
# Load Dataset
# ----------------------------------------

df = pd.read_csv("technology_sentences.csv")

keywords = df["keyword"].astype(str).tolist()
sentences = df["sentence"].astype(str).tolist()

# ----------------------------------------
# Tokenizer
# ----------------------------------------

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(keywords + sentences)

vocab_size = len(tokenizer.word_index) + 1

# ----------------------------------------
# Prepare Input
# ----------------------------------------

X = tokenizer.texts_to_sequences(keywords)

max_input_len = max(len(i) for i in X)

X = pad_sequences(
    X,
    maxlen=max_input_len,
    padding="post"
)

# ----------------------------------------
# Prepare Output
# ----------------------------------------

Y = tokenizer.texts_to_sequences(sentences)

max_output_len = max(len(i) for i in Y)

Y = pad_sequences(
    Y,
    maxlen=max_output_len,
    padding="post"
)

Y = to_categorical(
    Y,
    num_classes=vocab_size
)

# ----------------------------------------
# Train-Test Split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------
# Train Model
# ----------------------------------------

def train_model():

    st.write("Training One-to-Many RNN...")

    encoder_inputs = Input(
        shape=(max_input_len,)
    )

    x = Embedding(
        vocab_size,
        128
    )(encoder_inputs)

    encoded = SimpleRNN(
        128
    )(x)

    repeated = RepeatVector(
        max_output_len
    )(encoded)

    decoded = SimpleRNN(
        128,
        return_sequences=True
    )(repeated)

    outputs = TimeDistributed(
        Dense(
            vocab_size,
            activation="softmax"
        )
    )(decoded)

    model = Model(
        encoder_inputs,
        outputs
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=4,
        validation_data=(X_test, y_test),
        verbose=1
    )

    model.save(MODEL)

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    st.success(f"Model Accuracy : {accuracy*100:.2f}%")

# ----------------------------------------
# Generate Sentence
# ----------------------------------------

def generate_sentence(keyword):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    seq = tokenizer.texts_to_sequences([keyword.lower()])

    seq = pad_sequences(
        seq,
        maxlen=max_input_len,
        padding="post"
    )

    prediction = model.predict(
        seq,
        verbose=0
    )

    prediction = np.argmax(
        prediction,
        axis=-1
    )[0]

    reverse_word_index = {
        value: key
        for key, value in tokenizer.word_index.items()
    }

    sentence = []

    for index in prediction:

        if index == 0:
            continue

        word = reverse_word_index.get(index)

        if word is not None:
            sentence.append(word)

    return " ".join(sentence)

# ----------------------------------------
# Train Only Once
# ----------------------------------------

if not os.path.exists(MODEL):
    train_model()
    
# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.title("One-to-Many RNN")

st.write("Text Generation using Encoder-Decoder Simple RNN")

keyword = st.text_input(
    "Enter Keyword",
    "Python"
)

if st.button("Generate Sentence"):

    if keyword.strip() == "":

        st.warning("Please enter a keyword.")

    else:

        sentence = generate_sentence(keyword)

        st.subheader("Generated Sentence")

        if sentence.strip() == "":
            st.error("Unable to generate a sentence.")
        else:
            st.success(sentence)