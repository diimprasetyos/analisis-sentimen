import pandas as pd
import streamlit as st
import time
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

# fungsi get data
@st.cache_data
def get_comments():
    try:
        query = "SELECT full_text, label FROM tb_train;"
        df = pd.read_sql(query, conn)
        if df.empty:
            st.warning("Tidak ada data yang ditemukan.")
        else:
            st.write(df)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return pd.DataFrame()

# fungsi insert data
def insert_comments(file):
    try:
        cursor = conn.cursor()

        if file.name.endswith('.csv'):
            df = pd.read_csv(file, delimiter=',')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Format file tidak didukung. Unggah file dengan ekstensi .csv atau .xlsx.")
            return

        # hanya mengambil 2 kolom
        if 'full_text' in df.columns and 'label' in df.columns:
            df = df[['full_text', 'label']]
        else:
            st.error("Tidak terdapat kolom 'full_text' dan 'label' di file.")
            return

        # insert ke database
        for _, row in df.iterrows():
            cursor.execute(
                sql.SQL("""
                    INSERT INTO tb_train (full_text, label)
                    VALUES (%s, %s)
                """),
                (row['full_text'], row['label'])
            )

        conn.commit()
        cursor.close()

        success_message = st.empty()
        success_message.success("Data berhasil ditambahkan!")
        time.sleep(3)
        success_message.empty()
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# fungsi hapus data
def delete_all_comments():
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tb_train;")
        conn.commit()
        cursor.close()
        st.success("Semua data berhasil dihapus!")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# fungsi main
def main():
    st.title("Upload Data Pelatihan")
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button:first-child {
            color: red;
            border: 2px solid red;
            transition: all 0.3s ease;
        }
        div[data-testid="stButton"] > button:first-child:hover {
            background-color: red;
            color: white;
            border: 2px solid red;
        }
        div[data-testid="stButton"] > button:first-child:active {
            background-color: red;
            color: white;
            border: 2px solid red;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # komponen file uploader
    uploaded_file = st.file_uploader("Upload CSV", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        insert_comments(uploaded_file)

    if st.button("Hapus Semua Data"):
        delete_all_comments()

    st.link_button("Klasifikasi Data", "classification")

    # tampilkan table data
    get_comments()

main()
