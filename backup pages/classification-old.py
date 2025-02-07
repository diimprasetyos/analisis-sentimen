import pandas as pd
from datetime import datetime

import requests
import json
import re
import string
import emoji
from sklearn.utils import resample

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.layers import LSTM, Dense, SimpleRNN, Embedding, Flatten, Dropout, Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.activations import softmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine
import psycopg2
import io
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname="postlit",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )

conn = get_connection()
@st.cache_data
def get_comments():
    global conn
    try:
        query = "SELECT full_text, label FROM tb_train;"
        df = pd.read_sql(query, conn)
        if df.empty:
            st.warning("Tidak ada data yang ditemukan.")
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return pd.DataFrame()

def model_to_binary(model):
    """Convert the Keras model to binary format."""
    model_bytes = io.BytesIO()
    model.save(model_bytes, save_format='h5')  # Save the model as HDF5 format to the BytesIO object
    model_bytes.seek(0)  # Rewind the file pointer to the beginning
    return model_bytes.read()  # Return the binary data

def save_model_and_tokenizer(model, tokenizer):
    conn = get_connection()
    cursor = conn.cursor()

    # Convert the model to binary format
    model_binary = model_to_binary(model)
    tokenizer_binary = pickle.dumps(tokenizer)  # Serialize tokenizer with pickle

    # Insert model and tokenizer into database
    cursor.execute("""
        INSERT INTO tb_model_storage (model_name, model_data, tokenizer_data)
        VALUES (%s, %s, %s)
        """, ('sentiment_model', model_binary, tokenizer_binary))

    conn.commit()
    cursor.close()
    conn.close()

    print("Model and Tokenizer have been saved to the database.")

# Simpan evaluasi ke database
def save_evaluation_to_db(accuracy, loss):
    try:
        # Create DataFrame for evaluation results
        eval_data = pd.DataFrame({
            'training_date': [datetime.now()],
            'accuracy': [accuracy],
            'loss': [loss]
        })

        # Save the evaluation results to the database
        engine = create_engine('postgresql+psycopg2://postgres:admin@localhost:5432/postlit')
        eval_data.to_sql('tb_model_stats', con=engine, if_exists='append', index=False)

        st.success("Hasil evaluasi berhasil disimpan di tb_result.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menyimpan hasil evaluasi: {e}")

def save_classification_results(results_df):
    try:
        # SQLAlchemy
        engine = create_engine('postgresql+psycopg2://postgres:admin@localhost:5432/postlit')
        results_df.to_sql('tb_result', con=engine, if_exists='append', index=False)
        st.success("Hasil klasifikasi berhasil disimpan di tb_result.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menyimpan hasil: {e}")

# Simpan evaluasi ke database
def save_evaluation_to_db(accuracy, loss):
    try:
        # Create DataFrame for evaluation results
        eval_data = pd.DataFrame({
            'training_date': [datetime.now()],
            'accuracy': [accuracy],
            'loss': [loss]
        })

        # Save the evaluation results to the database
        engine = create_engine('postgresql+psycopg2://postgres:admin@localhost:5432/postlit')
        eval_data.to_sql('tb_model_stats', con=engine, if_exists='append', index=False)

        st.success("Hasil evaluasi berhasil disimpan di tb_result.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menyimpan hasil evaluasi: {e}")
def classification_process():
    with st.spinner("Memproses data..."):

        df = get_comments()



        #Stemming
        stem_factory = StemmerFactory()
        stemmer = stem_factory.create_stemmer()

        def stemming_text(text):
            words = text.split()
            hasil = [stemmer.stem(word) for word in words]
            return ' '.join(hasil)

        df['full_text'] = df['full_text'].apply(stemming_text)

        #lemmatize
        lemmatizer = WordNetLemmatizer()

        def lemmatize_text(text):
            #pisah kalimat menjadi list kata
            words = text.split()
            # mengubah kata kerja menjadi bentuk sederhana pos='v' (kata kerja, verb)
            lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words]
            return ' '.join(lemmatized_words)

        df['full_text'] = df['full_text'].apply(lemmatize_text)

        #Tokenizing
        df.dropna(subset=['full_text'], inplace=True)
        max_features = 3000
        max_sequence_length = 200
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(df['full_text'].values)

        X = tokenizer.texts_to_sequences(df['full_text'].values)
        X = pad_sequences(X)
        Y = pd.get_dummies(df['label']).values

        #split dataset
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=128)

        #modelling
        embed_dim = 128
        lstm_out = 32

        #layer
        model = Sequential()
        model.add(Embedding(max_features, embed_dim, input_length=max_sequence_length))
        model.add(SpatialDropout1D(0.3))
        model.add(Bidirectional(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.3)))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='sigmoid'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            metrics=['accuracy']
        )
        model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True),
            ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', save_best_only=True)
        ]

        #train model
        batch_size = 128
        progress_bar = st.progress(0)

        for epoch in range(epochs):
            history = model.fit(
                X_train, Y_train,
                epochs=20,
                batch_size=batch_size,
                validation_data=(X_test, Y_test),
                callbacks=callbacks,
                verbose=0
            )

            # Update the progress bar
            progress_bar.progress((epoch + 1) / epochs)

            # Update the display with accuracy and loss after each epoch
            accuracy = history.history['accuracy'][-1]
            loss = history.history['loss'][-1]
            st.write(f"Epoch {epoch + 1}/{epochs}")
            st.write(f"Akurasi: {accuracy:.4f}, Loss: {loss:.4f}")

            # Final evaluation
        accuracy = history.history['accuracy'][-1]
        loss = history.history['loss'][-1]
        st.write(f"Akurasi Test: {accuracy}")
        st.write(f"Loss Test: {loss}")

        # Save model and tokenizer
        save_model_and_tokenizer(model, tokenizer)
        # Save model stats
        save_evaluation_to_db(accuracy, loss)


        return model, tokenizer
def classification_page():
    st.title("Klasifikasi")

    # Check if classification results are stored in session state
    if 'classification_results' in st.session_state:
        # Display saved classification results
        st.subheader("Hasil Klasifikasi Sebelumnya")
        st.write(st.session_state.classification_results)
    else:
        # Show the available comments if no results in session state
        st.write(get_comments())

    # epochs = st.text_input("Masukkan jumlah epoch pelatihan", "10")
    # if epochs.strip() == "":
    #     st.warning("Masukkan jumlah epoch yang valid.")

    # Only show the button to classify if there are no results in session state
    # if st.button("Klasifikasi Data") and 'classification_results' not in st.session_state:
    #     try:
    #         epochs_int = int(epochs)  # Ensure the epochs value is an integer
    #         classification_process(epochs_int)  # Pass epochs_int as an argument
    #     except ValueError:
    #         st.error("Masukkan jumlah epoch yang valid (angka).")
    if st.button("Klasifikasi Data") and 'classification_results' not in st.session_state:
        classification_process()

classification_page()
