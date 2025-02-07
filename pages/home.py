import pandas as pd
import streamlit as st
import psycopg2
from psycopg2 import sql

# Init koneksi database
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname="postgres-streamlit",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )

conn = get_connection()

# Fungsi untuk mendapatkan data
@st.cache_data
def get_comments():
    try:
        query = "SELECT full_text, label FROM tb_train;"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return pd.DataFrame()

# Fungsi untuk halaman beranda
def beranda():
    st.title("Dashboard Analisis Sentimen")

    st.write(""" 
    Aplikasi ini dibuat untuk melakukan analisis sentimen pada komentar terkait isu tertentu 
    menggunakan metode Long Short-Term Memory (LSTM).
    """)

    # Ambil data dari database
    df = get_comments()

    if not df.empty:
        # Hitung total data dan komposisi label
        total_data = len(df)
        positif_count = len(df[df['label'] == 'Positive'])
        negatif_count = len(df[df['label'] == 'Negative'])
        avg_length = df['full_text'].str.split().apply(len).mean()
        max_length = df['full_text'].str.split().apply(len).max()
        min_length = df['full_text'].str.split().apply(len).min()

        # baris 1
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", total_data)
        with col2:
            st.metric("Label Positif", f"{positif_count}")
        with col3:
            st.metric("Label Negatif", f"{negatif_count}")
        # baris 2
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Komentar Terpanjang", f"{max_length} kata")
        with col2:
            st.metric("Komentar Terpendek", f"{min_length} kata")
        with col3:
            st.metric("Rata - Rata Panjang Komentar", f"{avg_length} kata")

        import os
        st.write("Working Directory:", os.getcwd())
# Panggil fungsi beranda
beranda()
