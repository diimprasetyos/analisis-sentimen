import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import psycopg2


# PostgreSQL Connection
def get_postgres_connection():
    return psycopg2.connect(
        dbname="postgres-streamlit",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )

# Fungsi untuk mengambil data hasil klasifikasi
@st.cache_data
def get_classified_comments():
    try:
        with get_postgres_connection() as conn:
            query = "SELECT full_text, predicted_label FROM tb_result;"
            df = pd.read_sql(query, conn)
            return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data: {e}")
        return pd.DataFrame()


# Fungsi untuk menghapus semua data hasil klasifikasi
def delete_all_data():
    try:
        with get_postgres_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tb_result;")
            conn.commit()
            cursor.close()
            st.success("Semua data berhasil dihapus dari tb_result.")
            # Clear cache setelah penghapusan
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghapus data: {e}")


# Halaman evaluasi
def evaluation_page():
    st.title("Evaluasi Hasil Pelatihan")

    # Ambil data dari PostgreSQL
    comments_df = get_classified_comments()

    # Validasi jika tidak ada data
    if comments_df.empty:
        st.warning("Tidak ada data hasil klasifikasi yang ditemukan.")
        return

    # Tampilkan data hasil klasifikasi
    st.write("Data:")
    st.dataframe(comments_df)

    # Tombol untuk menghapus semua data
    if st.button("Hapus Semua Data"):
        delete_all_data()

    # Distribusi label
    st.subheader("Distribusi Label")
    if 'predicted_label' in comments_df.columns:
        label_counts = comments_df['predicted_label'].value_counts()

        # Plot donut chart
        plt.figure(figsize=(4, 4))
        plt.pie(
            label_counts,
            labels=label_counts.index,
            autopct='%1.1f%%',  # Menampilkan persentase
            colors=['#1f77b4', '#ff7f0e'],  # Warna untuk tiap label
            startangle=90,  # Awal sudut dari pie chart
        )
        # Tambahkan lingkaran putih di tengah untuk efek donut
        center_circle = plt.Circle((0, 0), 0.70, color='white')
        plt.gca().add_artist(center_circle)
        st.pyplot(plt)
    else:
        st.warning("Kolom 'predicted_label' tidak ditemukan dalam data.")

    # Wordcloud dari teks komentar
    st.subheader("Word Cloud dari Komentar")
    if 'full_text' in comments_df.columns:
        text = " ".join(comments_df['full_text'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.warning("Kolom 'full_text' tidak ditemukan dalam data.")


# Tampilkan halaman evaluasi
evaluation_page()