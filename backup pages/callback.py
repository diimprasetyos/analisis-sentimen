import pandas as pd
import streamlit as st
import time
import psycopg2
from psycopg2 import sql

# Init koneksi database
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

# fungsi get data
@st.cache_data
def get_comments():
    try:
        query = "SELECT full_text, actual_label, predicted_label FROM tb_callback;"
        df = pd.read_sql(query, conn)
        if df.empty:
            st.warning("Tidak ada data yang ditemukan.")
        else:
            st.write(df)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return pd.DataFrame()


def main():
    # tampilkan table data
    get_comments()

main()