# --- Library Built-in ---
import re
import string
from datetime import datetime

# --- Library Eksternal ---
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sqlalchemy import create_engine
import psycopg2
import requests
import json
import emoji

# --- Library dari Proyek Lokal atau Khusus ---
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Keras/TensorFlow ---
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

# --- Custom Function ---
from sqlalchemy import create_engine  # Used for saving to PostgreSQL


@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname="postlit",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )

# Ambil koneksi global
conn = get_connection()

# Load pre-saved tokenizer and model from the 'model' directory
@st.cache_resource
def load_tokenizer():
    with open('model/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

@st.cache_resource
def load_trained_model():
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the tokenizer and model
tokenizer = load_tokenizer()
model = load_trained_model()

@st.cache_data
def get_comments():
    global conn
    try:
        # Replace with your actual database query code
        query = "SELECT full_text, label FROM tb_train;"
        df = pd.read_sql(query, conn)  # Replace 'conn' with your actual connection variable
        if df.empty:
            st.warning("Tidak ada data yang ditemukan.")
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return pd.DataFrame()

def save_classification_results(results_df):
    """Save classification results to the PostgreSQL table 'tb_result'."""
    try:
        # Create SQLAlchemy engine for bulk insert
        engine = create_engine('postgresql+psycopg2://postgres:admin@localhost:5432/postlit')
        results_df.to_sql('tb_result', con=engine, if_exists='append', index=False)
        st.success("Hasil klasifikasi berhasil disimpan di tb_result.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menyimpan hasil: {e}")

def classification_process():
    # Mengambil data dari database
    df = get_comments()

    #menghapus data kosong
    df.dropna(subset=['full_text'], inplace=True)

    # Filter hanya untuk label 'positif' dan 'negatif'
    df = df[df['label'].isin(['positif', 'negatif'])]

    #Pre-Processing
    # Mengubah semua teks menjadi huruf kecil
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
    #mengganti kata tidak baku
    custom_dict = {
        'gpp': 'nggak apa-apa',
        'wkwk': 'haha',
        'wkwkwk': 'haha',
        'mager': 'malas gerak',
        'ngedepak': 'menendang',
        'jur': 'jurusan',
        'gw': 'saya',
        'dstnya': 'dan seterusnya',
        'dgn': 'dengan',
        'keciduk': 'ketahuan',
        'klo': 'kalau',
        'kl': 'kalau',
        'g': 'tidak',
        'y': 'ya',
        'sy': 'saya',
        'bgt': 'banget',
        'gx': 'tidak',
        'diaa': 'dia',
        'yaa': 'ya',
        'yaaang': 'yang',
        'ttg': 'tentang',
        'knp': 'kenapa',
        'xixixixi': 'haha',
        'xixixi': 'haha',
        'xixi': 'haha',
        'bnyk': 'banyak',
        'bnyak': 'banyak',
        'bnyknya': 'banyaknya',
        'bnyknyaa': 'banyaknya',
        'gue': 'saya',
        'guee': 'saya',
        'krn': 'karena',
        'd*sen': 'dosen',
        'wkwkwkw': 'haha',
        'bangget': 'banget',
        'bangett': 'banget',
        'yg': 'yang',
        'emg': 'memang',
        'nyogok': 'menyogok',
        'konoha': 'indonesia',
        'sdm': 'sumber daya manusia',
        'univ': 'universitas',
        'indo': 'indonesia',
        'tak': 'tidak',
        'org': 'orang',
        'krg': 'kurang',
        'kaaaan': 'kan',
        'nii': 'ini',
        'kebanyaksn': 'kebanyakan',
        'mna': 'mana',
        'wkwkwkwk': 'haha',
        'kyk': 'seperti',
        'bgini': 'begini',
        'bgn': 'begini',
        'dtg': 'datang',
        'pr': 'pekerjaan rumah',
        'sd': '',
        'sma': '',
        'smk': 'sekolah menengah kejuruan',
        'pkl': 'praktek kerja lapangan',
        'd': 'di',
        'jokey': 'joki',
        'lu': 'kamu',
        'k': 'kakak',
        'kk': 'kakak',
        'loe': 'kamu',
        'ane': 'saya',
        'ente': 'kamu',
        'antum': 'kamu',
        'info': 'informasi',
        'bngt': 'banget',
        'slah': 'salah',
        'slh': 'salah',
        'bet': 'banget',
        'mk': 'mahkamah konstitusi',
        'medsos': 'media sosial',
        'ybs': 'yang bersangkutan',
        'trnyt': 'ternyata',
        'jd': 'jadi',
        'jdny': 'jadinya',
        'bkn': 'bukan',
        'tololll': 'bodoh',
        'tololl': 'bodoh',
        'tolol': 'bodoh',
        'wkwkwkwk': 'haha',
        'wkwkwk': 'haha',
        'wkwkwk': 'haha',
        'wkwkwkwkwk': 'haha',
        'wkwkwkwkwkwk': 'haha',
        'wkwkwkwkwkwkwk': 'haha',
        'wkwkw': 'haha',
        'wkwkwkkwk': 'haha',
        'wkwkwkwk': 'haha',
        'potokopi': 'fotocopy',
        'capai': 'mencapai',
        'reall': 'real',
        'wkwkwkw': 'haha',
        'anceman': 'ancaman',
        'angus': 'hangus',
        'sd': 'sampai dengan',
        'pd': 'pada',
        'krna': 'karena',
        'STM': 'sekolah teknik',
        'nan': 'dan',
        'bahlul': 'bodoh',
        'putuss': 'putus',
        'rs': 'rumah sakit',
        'bpnya': 'bapaknya',
        'tp': 'tapi',
        'dll': 'dan lain lain',
        'satir': 'sindiran',
        'gmau': 'tidak mau',
        'gblk': 'goblok',
        'wkkkwkkwwkwk': 'haha',
        'thn': 'tahun',
        'ckckckck': 'hadeh',
        'byk': 'banyak',
        'negri': 'negeri',
        'ni': 'ini',
        'vangke': 'bangsat',
        'vangkenya': 'bangsatnya',
        'kek': 'seperti',
        'sbg': 'sebagai',
        'gelo': 'gila',
        'diblg': 'dibilang',
        'ush': 'usah',
        'bljr': 'belajar',
        'elu': 'kamu',
        'gibah': 'omongan',
        'plongo': 'diam tercengang',
        'planga plongo': 'diam tercengang',
        'nntik': 'nanti',
        'nnti': 'nanti',
        'nti': 'nanti',
        'thn': 'tahun',
        'nantik': 'nanti',
        'bs': 'bisa',
        'aj': 'aja',
        'kurleb': 'kurang lebih',
        's2': 'sarjana',
        's1': 'sarjana',
        's3': 'sarjana',
        'dr': '',
        'ama': 'sama',
        'ane': 'saya',
        'ente': 'kamu',
        'mpot': 'susah',
        'x': 'media sosial',
        'sm': 'sama',
        'bgt': 'banget',
        'tmn': 'teman',
        'fk': 'fakultas kedokteran',
        'fkg': 'fakultas kedokteran gigi',
        'n': 'dan',
        'bodok': 'bodoh',
        'kalok': 'kalau',
        'yt': 'youtube',
        'ybs': 'yang bersangkutan',
        'uu': 'legislasi',
        'rp': 'rupiah',
        'edan': 'gila',
        'ampe': 'sampe',
        'gws': 'get well soon',
        'ortu': 'orang tua',
        'cemoohan': 'ejekan',
        'ak': 'aku',
        'd': 'di',
        'cogan': ' ganteng',
        'n': 'dan',
        'sidikitt': 'sedikit',
        'PTS': 'perguruan tinggi swasta',
        'diindo': 'di indonesia',
        'penganggurrrrrrrrrrrrrr': 'pengangguran',
        'botakkkkkkkkkkk': 'botak',
        'bangsad': 'bangsat',
        'hahahahaha': 'haha',
        'meng gebu gebu': 'menggebu',
        'gokss': 'gokil',
        'tahun': '',
        'a': '',
        'z': '',
        'ber amanah': 'amanah',
        'jt': '',
        'h': '',
        'd': '',
        'an': '',
        'th': '',
        'ordal': 'orang dalam',
        'ajaa': 'aja',
        'anjj': 'anjing',
        'osn': 'olimpiade sains nasional',
        'endonesah': 'indonesia',
        'tll': 'tolol',
        'tulalit': 'telat berpikir',
        'cpns': 'calon pegawai negara sipil',
        'kampus eno': 'universitas indonesia',
        'gen': '',
        'blookkk': 'goblok',
        'bae': 'aja',
        'sp': '',
        'ad': 'ada',
        'ir': '',
        'negarannya': 'negaranya',
        'b': '',
        'blepotan': 'belepotan',
        'skripshit': 'skripsi',
        'ilang': 'hilang',
        'Maturnuwun': 'terima kasih',
        'm': '',
        'mugi': 'semoga',
        'mberkah': 'berkah',
        'finis': 'finish',
        'jkt': 'jakarta',
        'tpp': '',
        'digit': '',
        'ehh': '',
        'eh': '',
        'ah': '',
        'ahh': '',
        'penjokiwah': '',
        'am': 'sama',
        'bimbel': 'bimbingan belajar',
        'sangarrrr': 'bencana',
        'reeek': '',
        'dospem': 'dosen pembimbing',
        'bloon': 'bodoh',
        'mnyikapi': 'menyikapi',
        'sbgai': 'sebagai',
        'spya': 'supaya',
        'sgla': 'segala',
        'k': '',
        'dll': '',
        'asdos': 'asisten dosen',
        'mhsswa': 'mahasiswa',
        'bro': '',
        'elu': 'kamu',
        'elo': 'kamu',
        'gwa': 'saya',
        'bngt': 'banget',
        'sahih': 'sah',
        'iq': 'nilai kecerdasan seseorang',
        'wa': '',
        'wow': '',
        'belaain': 'membela',
        'wahhhh': '',
        'anjirrr': '',
        'anjir': '',
        'anjirr': '',
        'DPR': 'pemerintahan',
        'c': '',
        'e': '',
        'f': '',
        'g': '',
        'p': '',
        'q': '',
        'aq': 'aku',
        'chat gpt': 'ai',
        'gpt': 'ai',
        'cv': 'surat lamaran kerja',
        'kelar': 'selesai',
        'gais': '',
        'guys': '',
        'ges': '',
        'inj': 'ini',
        'bhs': 'bahasa',
        'dkv': 'desain',
        'akwkwkwwk': 'haha',
        'hah': '',
        'ngeh': 'sadar',
        'waow': '',
        'bablas': 'kelewatan',
        'fak ': '',
        'bhsa': 'bahasa',
        'pnjang': 'panjang',
        'hey': '',
        'nnti': 'nanti',
        'krna': 'karena',
        'pke': 'pakai',
        'pr': 'pekerjaan rumah',
        'tsb': 'tersebut',
        'ttg': 'tentang',
        'lg': 'lagi',
        'lgi': 'lagi',
        'blum': 'belum',
        'anjg': 'anjing',
        'yutub': 'youtube',
        'cengoh': 'bodoh',
        'per': 'setiap',
        'do': 'dikeluarkan',
        'ni': 'ini',
        'k': '',
        'acc': 'disetujui',
        'bet': 'banget',
        'bab': '',
        'tu': 'itu',
        'jokersssss': 'joki',
        'baddddddddddd': 'buruk',
        'lbih': 'lebih',
        'drpd': 'daripada',
        'klw': 'kalau',
        'jokii': 'joki',
        'vcs': '',
        'bo': '',
        'dll': '',
        'vc': 'video call',
        'michat': '',
        'Skrip51': 'skripsi',
        'J0k1': 'joki',
        'aing': 'saya',
        'tong': '',
        'skrg ': 'sekarang',
        'bmw': '',
        'lc': '',
        'm4': '',
        'mmr': '',
        'mercy': '',
        'flexing': 'pamer',
        'bangeet': 'banget',
        'kalimt': 'kalimat',
        'ipk': 'nilai',
        'fyi': 'info',
        'wkwkwkwkkwkw': 'haha',
        'apakh': 'apakah',
        'Booo': '',
        'Doohh': 'bodoh',
        'jirr': '',
        'jir': '',
        'jirrr': '',
        'njir': '',
        'njirr': '',
        'njirrr': '',
        'bisadeh': 'bisa',
        'uu': 'undang undang',
        'un': 'ujian',
        'om': '',
        'univku': 'universitas',
        'w': '',
        'd3': 'diploma',
        's': '',
        'd': '',
        'blng': 'bilang',
        'epep': 'game',
        'jadiiiii': 'jadi',
        'msh': 'masih',
        'higher': 'tinggi',
        'think': 'pikir',
        'and': 'dan',
        'as': 'sebagai',
        'to': 'untuk',
        'covernya': 'sampul',
        'white': 'putih',
        'art': 'seni',
        'supply': 'menyediakan',
        'least': 'paling sedikit',
        'field': 'bidang',
        'them': 'mereka',
        'might': 'mungkin',
        'why': 'kenapa',
        'top': 'atas',
        'topic': 'topik',
        'loundrying': 'pencucian',
        'bpaknya': 'bapaknya',
        'mony': 'uang',
        'title': 'judul',
        'footnote': 'catatan kaki',
        'insight': 'wawasan',
        'other': 'lainnya',
        'together': 'bersama',
        'highlight': 'sorotan',
        'oversimplification': 'penyederhanaan yang berlebihan',
        'what': 'apa',
        'thegw': '',
        'responsibility': 'tanggungjawab',
        'cool': 'keren',
        'industry': 'industri',
        'high': 'tinggi',
        'demand': 'kebutuhan',
        'tell': 'katakan',
        'that': 'itu',
        'culture': 'budaya',
        'pal': '',
        'sesunghunya': 'sesungguhnya',
        'cmiiw': '',
        'pangilan': 'panggilan',
        'bad': 'buruk',
        'oversimplify': 'menggampangkan',
        'prosespdahal': 'proses padahal',
        'naek': 'naik',
        'wkawkakak': '',
        'enddengan': 'end dengan',
        'etc': 'lainnya',
        'profeson': 'professor',
        'siniworks': 'sini bekerja',
        'ttp': 'tetap',
        'lie': 'bohong',
        'diplekoto': 'dibodohin',
        'atooooo': 'atau',
        'literally': 'secara',
        'dimention': 'menyebutkan',
        'mention': 'menyebutkan',
        'fb': 'facebook',
        'publish': 'publikasi',
        'quality': 'berkualitas',
        'correct me if i wrong': 'koreksi saya',
        'pkm': 'pekan karya mahasiswa',
        'prof': 'professor',
        'ngising': 'goblok',
        'say': 'kata',
        'obsess': 'obsesi',
        'man': '',
        'somehow': 'bagaimanapun',
        'people': 'orang',
        'watch': 'menonton',
        'remind': 'mengingatkan',
        'hopeless': 'tanpa harapan',
        'shock': 'kaget',
        'ppl': 'orang',
        'hehehehe': 'haha',
        'with': 'dedngan',
        'average': 'rata rata',
        'you': 'kamu',
        'expect': 'harapkan',
        'from': 'dari',
        'die': 'mati',
        'dumb': 'bodoh',
        'stupid': 'bodoh',
        'emangnge': 'memang',
        'sbnernya': 'sebenarnya',
        'silly': 'konyol',
        'wrong': 'salah',
        'whats': 'apa',
        'pngen': 'ingin',
        'common':'umum',
        'gin': 'gini',
        'stunt be real hih': '',
        'simak lengkap thread ikut': '',
        'tolil': 'tolol',
        'goblog':'goblok',
        'gblg':'goblok',
        'gblk': 'goblok',
        'bodo': 'bodoh',
        'rp':' ',
        'nebeng':'ikut',
        'rupiah':' ',
        'juta':' ',
    }

    def replace_slang_and_alay(text):
        words = text.split()
        new_words = [custom_dict.get(word, word) for word in words]
        return ' '.join(new_words)


    df['full_text'] = df['full_text'].apply(replace_slang_and_alay)
    st.write('Mengubah kata tidak baku:')
    st.write(df['full_text'])
    #Stopword remove
    factory = StopWordRemoverFactory()
    stopword_sastrawi = factory.get_stop_words()

    # clean full_text jika ada data yang berupa list
    df['full_text'] = df['full_text'].apply(lambda x: ''.join(x) if isinstance(x, list) else x) #isinstance(obj, type)

    def remove_stopwords(text):
        words = text.split()
        # cek word.lower tidak ada dalam daftar insert ke filtered words / bukan stopword
        filtered_words = [word for word in words if word.lower() not in stopword_sastrawi]
        return ' '.join(filtered_words) #dijadikan 1 string dengan spasi

    df['full_text'] = df['full_text'].apply(remove_stopwords)
    st.write('menghapus stopwords:')
    st.write(df['full_text'])
    # Filter hanya untuk label 'positif' dan 'negatif'
    df = df[df['label'].isin(['positif', 'negatif'])]

    # Tokenize dan pad sequences
    X = tokenizer.texts_to_sequences(df['full_text'].values)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=200)

    # Convert label ke bentuk one-hot encoding
    Y = pd.get_dummies(df['label']).values

    # Periksa apakah ada nilai None di X atau Y
    if np.any(pd.isnull(X)):
        st.error("Fitur (X) mengandung nilai None.")
        return
    if np.any(pd.isnull(Y)):
        st.error("Label (Y) mengandung nilai None.")
        return

    # Prediksi seluruh data
    predictions = model.predict(X)
    predicted_classes = predictions.argmax(axis=-1)

    # Map predicted classes ke label
    labels = pd.get_dummies(df['label']).columns
    predicted_labels = [labels[i] for i in predicted_classes]

    # Buat DataFrame untuk hasil klasifikasi
    results_df = pd.DataFrame({
        'full_text': df['full_text'].values,
        'predicted_label': predicted_labels,
        'actual_label': df['label'].values
    })

    # Hitung akurasi secara manual
    actual_classes = Y.argmax(axis=-1)
    accuracy = np.mean(predicted_classes == actual_classes)
    st.write('Hasil Klasifikasi:')

    # Tampilkan hasil klasifikasi
    st.write(results_df)

    # Simpan hasil klasifikasi
    save_classification_results(results_df)

    # Simpan hasil ke session state agar persist
    st.session_state.classification_results = results_df

def classification_page():
    st.title("Klasifikasi Data Pengujian")
        # Show the available comments if no results in session state
    st.write(get_comments())

    # Only show the button to classify if there are no results in session state
    if st.button("Klasifikasi Data") and 'classification_results' not in st.session_state:
        classification_process()

classification_page()
