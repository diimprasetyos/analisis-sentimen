import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sqlalchemy import create_engine
import psycopg2

import tensorflow as tf

# PostgreSQL Connection
def get_postgres_connection():
    return psycopg2.connect(
        dbname="postgres-streamlit",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )

def get_comments():
    """Retrieve comments from PostgreSQL."""
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
        engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres-streamlit")
        results_df.to_sql('tb_result_train', engine, if_exists='append', index=False)
    except Exception as e:
        st.error(f"Gagal menyimpan hasil klasifikasi: {e}")
    finally:
        conn.close()

def update_progress(epoch, total_epochs, progress_bar, status_text, logs):
    """Update the progress bar during training."""
    progress = (epoch + 1) / total_epochs
    progress_bar.progress(progress)
    status_text.text(f"Epoch {epoch + 1}/{total_epochs} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")

def classification_process():
    with st.spinner("Memproses data..."):
        df = get_comments()

        # MODELLING
        # Prepare data
        df = df[df['label'] != 'Neutral']
        df.dropna(subset=['full_text'], inplace=True)
        max_features = 5000
        max_sequence_length = 200
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(df['full_text'].values)

        X = tokenizer.texts_to_sequences(df['full_text'].values)
        X = pad_sequences(X, maxlen=max_sequence_length)
        Y = pd.get_dummies(df['label']).values

        # Split data into training and testing sets
        X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, np.arange(len(X)),
                                                                                         test_size=0.33,
                                                                                         random_state=128)

        # Build LSTM model
        embed_dim = 128
        lstm_out = 32

        model = Sequential([
            Embedding(max_features, embed_dim, input_length=max_sequence_length),
            SpatialDropout1D(0.3),
            Bidirectional(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.3)),
            Dense(2, activation='softmax')
        ])
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        epochs = 15
        progress_bar = st.progress(0)
        status_text = st.empty()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True),
            ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', save_best_only=True),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: update_progress(epoch, epochs, progress_bar, status_text, logs))
        ]

        batch_size = 128
        history = model.fit(
            X_train, Y_train,
            epochs=15,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            callbacks=callbacks,
            verbose=1
        )

        score = model.evaluate(X_test, Y_test, verbose=0)

        # Classification on test data
        predictions = model.predict(X_test)
        predicted_classes = predictions.argmax(axis=-1)

        # Map predicted classes to labels
        labels = df['label'].unique()
        predicted_labels = [labels[i] for i in predicted_classes]

        # Create DataFrame for classification results
        results_df = pd.DataFrame({
            'full_text': df['full_text'].iloc[indices_test].values,
            'predicted_label': predicted_labels,
            'actual_label': df['label'].iloc[indices_test].values
        })

        # Save results and display evaluation metrics
        save_classification_results(results_df)
        st.write(f'Test Loss: {score[0]}')
        st.write(f'Test Accuracy: {score[1]}')

# Classification page content
def classification_page():
    st.title("Klasifikasi")
    st.write(get_comments())
    if st.button("Klasifikasi Data"):
        classification_process()

classification_page()
