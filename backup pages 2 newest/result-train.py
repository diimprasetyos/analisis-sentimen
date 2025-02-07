import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import psycopg2


# Initialize PostgreSQL connection
def init_connection_postgres():
    return psycopg2.connect(
        dbname="postgres-streamlit",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )


pg_conn = init_connection_postgres()


# Function to fetch data from tb_result_train in PostgreSQL
def get_classified_comments_postgres():
    query = "SELECT full_text, predicted_label, actual_label FROM tb_result_train;"
    df = pd.read_sql(query, pg_conn)
    return df


# Function to count positive and negative labels
def get_data_count():
    query = "SELECT COUNT(*) FROM tb_result_train WHERE predicted_label = 'positif';"
    positif_count = pd.read_sql(query, pg_conn).iloc[0, 0]

    query = "SELECT COUNT(*) FROM tb_result_train WHERE predicted_label = 'negatif';"
    negatif_count = pd.read_sql(query, pg_conn).iloc[0, 0]

    total_count = positif_count + negatif_count
    return positif_count, negatif_count, total_count


# Function to display WordCloud for a label
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)

    st.pyplot(plt)
    plt.close()


# Function to display sentiment distribution as a pie chart
def display_sentiment_distribution(positif_count, negatif_count):
    if positif_count is None or positif_count == 0:
        positif_count = 1
    if negatif_count is None or negatif_count == 0:
        negatif_count = 1

    total = positif_count + negatif_count

    labels = ['Positif', 'Negatif']
    sizes = [positif_count / total * 100, negatif_count / total * 100]
    colors = ['#66b3ff', '#ff9999']

    plt.figure(figsize=(4, 4))
    plt.pie(
        sizes,
        labels=[f'{l} ({s:.1f}%)' for l, s in zip(labels, sizes)],
        startangle=90,
        colors=colors,
        wedgeprops={'width': 0.4}
    )
    plt.title("Distribusi Sentimen (Positif vs Negatif)")
    plt.axis('equal')
    st.pyplot(plt)


# Function to display a confusion matrix
def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=['positif', 'negatif'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Positif', 'Negatif'],
                yticklabels=['Positif', 'Negatif'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix")
    st.pyplot(plt)


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

    # Fetch data from 'tb_result_train' table (PostgreSQL)
    comments_df_postgres = get_classified_comments_postgres()

    # If data is empty, re-run the page to reflect the latest state
    if comments_df_postgres.empty:
        st.warning("Tidak ada data hasil klasifikasi yang ditemukan.")
        return

    comments_df_postgres = comments_df_postgres[['full_text', 'predicted_label', 'actual_label']]
    st.write("Data hasil klasifikasi:")
    st.write(comments_df_postgres)

    # Separate comments by sentiment label
    positive_comments = comments_df_postgres[comments_df_postgres['predicted_label'] == 'positif']
    negative_comments = comments_df_postgres[comments_df_postgres['predicted_label'] == 'negatif']

    if st.button("Hapus Semua Data"):
        delete_all_data()

    # Create columns for WordCloud (only one column for both)
    col1, col2 = st.columns(2)

    # WordCloud for positive comments in col1
    with col1:
        st.subheader("WordCloud untuk Komentar Positif")
        if not positive_comments.empty:
            positive_text = ' '.join(positive_comments['full_text'])
            generate_wordcloud(positive_text, "Komentar Positif")
        else:
            st.warning("Tidak ada komentar positif yang ditemukan.")

    # WordCloud for negative comments in col2
    with col2:
        st.subheader("WordCloud untuk Komentar Negatif")
        if not negative_comments.empty:
            negative_text = ' '.join(negative_comments['full_text'])
            generate_wordcloud(negative_text, "Komentar Negatif")
        else:
            st.warning("Tidak ada komentar negatif yang ditemukan.")

    # Get counts of positive and negative data from PostgreSQL
    positif_count, negatif_count, total_count = get_data_count()

    # Display sentiment distribution as a pie chart
    st.subheader("Distribusi Sentimen (Pie Chart)")
    display_sentiment_distribution(positif_count, negatif_count)

    # Calculate and display accuracy, precision, recall, F1 score
    y_true = comments_df_postgres['actual_label']
    y_pred = comments_df_postgres['predicted_label']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')  # Using 'weighted' for multiclass
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    st.subheader("Evaluasi Klasifikasi")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    display_confusion_matrix(y_true, y_pred)

    # Display classification report
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=['Positif', 'Negatif'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)


# Display evaluation page
evaluation_page()
