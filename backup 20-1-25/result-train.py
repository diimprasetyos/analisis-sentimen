import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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


# Function to fetch classified data from PostgreSQL
@st.cache_data
def get_classified_comments():
    conn = get_postgres_connection()
    try:
        query = "SELECT full_text, predicted_label, actual_label FROM tb_result_train;"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


# Function to count positive and negative labels
def get_data_count():
    conn = get_postgres_connection()
    try:
        query = """
            SELECT
                SUM(CASE WHEN predicted_label = 'Positive' THEN 1 ELSE 0 END) AS positive_count,
                SUM(CASE WHEN predicted_label = 'Negative' THEN 1 ELSE 0 END) AS negative_count
            FROM tb_result_train;
        """
        df = pd.read_sql(query, conn)
        positive_count = int(df['positive_count'].iloc[0] or 0)
        negative_count = int(df['negative_count'].iloc[0] or 0)
        total_count = positive_count + negative_count
        return positive_count, negative_count, total_count
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghitung data: {e}")
        return 0, 0, 0
    finally:
        conn.close()


# Function to display WordCloud for a label
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)


# Function to display sentiment distribution as a pie chart
def display_sentiment_distribution(positive_count, negative_count):
    if positive_count == 0 and negative_count == 0:
        st.warning("Tidak ada data untuk ditampilkan dalam grafik pie.")
        return

    labels = ['Positive', 'Negative']
    sizes = [positive_count, negative_count]
    colors = ['#66b3ff', '#ff9999']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title("Distribusi Sentimen (Positive vs Negative)")
    plt.axis('equal')
    st.pyplot(plt)


# Function to display a confusion matrix
def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

pg_conn = get_postgres_connection()

# Function to delete all data from tb_result_train table
def delete_all_data():
    try:
        cursor = pg_conn.cursor()
        cursor.execute("DELETE FROM tb_result_train;")
        pg_conn.commit()
        cursor.close()
        st.success("Semua data berhasil dihapus dari tb_result_train.")

        # Clear cache immediately after data deletion
        st.cache_data.clear()
        st.cache_resource.clear()

        # Refresh the page after deletion
        st.rerun()

    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghapus data: {e}")

# Evaluation page with cache-clearing functionality
def evaluation_page():
    st.title("Evaluasi Hasil Pelatihan")

    # Fetch data from PostgreSQL
    comments_df = get_classified_comments()
    if not comments_df.empty:
        st.write(comments_df)

    if comments_df.empty:
        st.warning("Tidak ada data hasil klasifikasi yang ditemukan.")
        return

    # Separate comments by sentiment label
    positive_comments = comments_df[comments_df['predicted_label'] == 'Positive']
    negative_comments = comments_df[comments_df['predicted_label'] == 'Negative']
    if st.button("Hapus Semua Data"):
        delete_all_data()
    # Generate WordCloud for positive comments
    st.subheader("WordCloud untuk Komentar Positive")
    if not positive_comments.empty:
        positive_text = ' '.join(positive_comments['full_text'])
        generate_wordcloud(positive_text, "Komentar Positive")
    else:
        st.warning("Tidak ada komentar Positive yang ditemukan.")

    # Generate WordCloud for negative comments
    st.subheader("WordCloud untuk Komentar Negative")
    if not negative_comments.empty:
        negative_text = ' '.join(negative_comments['full_text'])
        generate_wordcloud(negative_text, "Komentar Negative")
    else:
        st.warning("Tidak ada komentar Negative yang ditemukan.")

    # Get counts of positive and negative data from PostgreSQL
    positive_count, negative_count, total_count = get_data_count()

    # Display sentiment distribution as a pie chart
    st.subheader("Distribusi Sentimen (Pie Chart)")
    display_sentiment_distribution(positive_count, negative_count)

    # Calculate and display accuracy, precision, recall, F1 score
    y_true = comments_df['actual_label']
    y_pred = comments_df['predicted_label']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Positive', average='binary')
    recall = recall_score(y_true, y_pred, pos_label='Positive', average='binary')
    f1 = f1_score(y_true, y_pred, pos_label='Positive', average='binary')

    st.subheader("Evaluasi Model")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    display_confusion_matrix(y_true, y_pred)

    # Clear cache
    st.cache_data.clear()


# Display evaluation page
evaluation_page()
