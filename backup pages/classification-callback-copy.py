import streamlit as st
from datetime import datetime

# --- Manipulasi Data ---
import pandas as pd
import numpy as np

# --- PostgreSQL ---
import psycopg2
from sqlalchemy import create_engine

# --- Preprocessing Teks ---
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- TensorFlow dan Keras ---
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split

@st.cache_resource
def get_postgres_connection():
    return psycopg2.connect(
        dbname="postlit",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )

@st.cache_data
def get_comments():
    conn = get_postgres_connection()
    try:
        query = "SELECT full_text, label FROM tb_train;"
        df = pd.read_sql(query, conn)
        if df.empty:
            st.warning("Tidak ada data yang ditemukan.")
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def save_classification_results(results_df):
    """Save classification results to PostgreSQL."""
    conn = get_postgres_connection()
    try:
        engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postlit")
        results_df.to_sql('tb_callback', engine, if_exists='append', index=False)
    except Exception as e:
        st.error(f"Gagal menyimpan hasil klasifikasi: {e}")
    finally:
        conn.close()

def classification_process():
    with st.spinner("Memproses data..."):
        df = get_comments()

        # Prepare data
        df.dropna(subset=['full_text'], inplace=True)
        df = df[df['label'].isin(['positif', 'negatif'])]

        # Pre-Processing
        df['full_text'] = df['full_text'].apply(lambda x: x.lower())
        st.write('Mengubah teks menjadi huruf kecil:')
        st.write(df['full_text'])
        # Menghapus URL
        df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'http[s]?://\S+|www\.\S+', '', x))
        st.write('Menghapus url/tautan:')
        st.write(df['full_text'])
        # Menghapus username/mention
        df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'@@[A-Za-z0-9_-]+|@[A-Za-z0-9_-]+', '', x))
        st.write('Menghapus username/mention:')
        st.write(df['full_text'])
        # Menghapus angka
        df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'\d+', '', x))
        st.write('Menghapus karakter non huruf/angka:')
        st.write(df['full_text'])
        # Mengganti tanda hubung (-) dengan spasi
        df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'(?<=\w)-(?=\w)', ' ', x))
        st.write('menghapus kata dengan tanda hubung:')
        st.write(df['full_text'])
        # Menghapus semua simbol punctuation kecuali spasi
        df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        st.write('menghapus semua tanda baca:')
        st.write(df['full_text'])

        max_features = 5000
        max_sequence_length = 200
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(df['full_text'].values)

        X = tokenizer.texts_to_sequences(df['full_text'].values)
        X = pad_sequences(X, maxlen=max_sequence_length)
        Y = pd.get_dummies(df['label']).values

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=128)

        # Build model
        model = Sequential([
            Embedding(max_features, 128, input_length=max_sequence_length),
            SpatialDropout1D(0.3),
            Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
            Dense(2, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Streamlit output components
        progress_bar = st.progress(0)
        status_text = st.empty()
        epoch_logs_container = st.container()

        # Callback for logging progress
        def log_epoch(epoch, logs):
            progress = (epoch + 1) / 7  # Assume 15 epochs
            progress_bar.progress(progress)
            status_text.text(
                f"Epoch {epoch + 1}/7 - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
                f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}"
            )
            with epoch_logs_container:
                st.write(f"Epoch {epoch + 1}: Loss={logs['loss']:.4f}, Accuracy={logs['accuracy']:.4f}, Val Loss={logs['val_loss']:.4f}, Val Accuracy={logs['val_accuracy']:.4f}")

        logging_callback = LambdaCallback(on_epoch_end=log_epoch)

        # Train model
        model.fit(
            X_train, Y_train,
            epochs=7, batch_size=32,
            validation_data=(X_test, Y_test),
            callbacks=[
                # EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),
                logging_callback
            ],
            verbose=0
        )

        # Evaluate model
        score = model.evaluate(X_test, Y_test, verbose=0)
        predictions = model.predict(X_test)
        predicted_classes = predictions.argmax(axis=-1)

        # Results
        labels = df['label'].unique()
        predicted_labels = [labels[i] for i in predicted_classes]
        results_df = pd.DataFrame({
            'full_text': df['full_text'].iloc[Y_test.argmax(axis=1)],
            'predicted_label': predicted_labels,
            'actual_label': df['label'].iloc[Y_test.argmax(axis=1)]
        })

        save_classification_results(results_df)
        st.success("Pelatihan selesai!")

def classification_page():
    st.title("Klasifikasi")
    st.write(get_comments())
    if st.button("Klasifikasi Data"):
        classification_process()

classification_page()
