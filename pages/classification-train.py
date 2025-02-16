import streamlit as st
import os
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
import nltk
from nltk.corpus import stopwords

# --- TensorFlow dan Keras ---
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split

import pickle

# Koneksi database
def get_postgres_connection():
    return psycopg2.connect(
        dbname="postgres-streamlit",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )

# Ambil komentar
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

# Simpan pelatihan
def save_classification_results(results_df):
    conn = get_postgres_connection()
    try:
        engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres-streamlit")
        results_df.to_sql('tb_result_train', engine, if_exists='append', index=False)
    except Exception as e:
        st.error(f"Gagal menyimpan hasil klasifikasi: {e}")
    finally:
        conn.close()

# Klasifikasi
def classification_process():
    with st.spinner("Memproses data..."):
        df = get_comments()
        if df.empty:
            st.error("Data tidak tersedia untuk klasifikasi.")
            return

        if 'full_text' not in df.columns or 'label' not in df.columns:
            st.error("Kolom 'full_text' atau 'label' tidak ditemukan dalam data.")
            return

        data = get_comments()
        # Hapus data kosong
        data.dropna(subset=['full_text', 'label'], inplace=True)
        # Pilih label positif negatif
        data = data[data['label'] != 'Neutral']
        # Hapus duplikasi data
        data.drop_duplicates(inplace=True)
        # Sinkronkan indeks dataframe setelah preprocessing
        data.reset_index(drop=True, inplace=True)

        # Pre-Processing
        # Mengubah semua teks menjadi huruf kecil
        df['clean_text'] = df['full_text'].apply(lambda x: x.lower())
        st.write('Mengubah teks menjadi huruf kecil:')
        st.write(df['clean_text'])
        # Menghapus URL
        df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'http[s]?://\S+|www\.\S+', '', x))
        st.write('Menghapus url/tautan:')
        st.write(df['clean_text'])
        # Menghapus username/mention
        df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'@@[A-Za-z0-9_-]+|@[A-Za-z0-9_-]+', '', x))
        st.write('Menghapus username/mention:')
        st.write(df['clean_text'])
        # Menghapus angka
        df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\d+', '', x))
        st.write('Menghapus karakter non huruf/angka:')
        st.write(df['clean_text'])
        # Menghapus semua simbol punctuation kecuali spasi
        df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        st.write('Menghapus semua tanda baca:')
        st.write(df['clean_text'])

        kamus = {
            'gpp': 'nggak apa-apa',
            'gini': 'begini',
            'gak': 'tidak',
            'yaaah': 'ya',
            'mna': 'mana',
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
            'pr': 'tugas',
            'hp': 'ponsel',
            'tsb': 'tersebut',
            'ttg': 'tentang',
            'nyaaaa': 'nya',
            'pliiisss': 'tolong',
            'trimakasih': 'terima kasih',
            'anggep': 'anggap',
            'sbnernya': 'sebenarnya',
            'gwe': 'aku',
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
            'common': 'umum',
            'gin': 'gini',
            'stunt be real hih': '',
            'simak lengkap thread ikut': '',
            'tolil': 'tolol',
            'goblog': 'goblok',
            'gblg': 'goblok',
            'gblk': 'goblok',
            'bodo': 'bodoh',
            'rp': ' ',
            'nebeng': 'ikut',
            'rupiah': ' ',
            'juta': ' ',
            'lohhh': 'loh',
            'thx': 'makasih',
            'aj': 'saja'
        }
        def replace_slang(text):
            words = text.split()  # Memecah teks menjadi list
            new_words = [kamus.get(word, word) for word in words]
            return ' '.join(new_words)

        df['clean_text'] = df['clean_text'].apply(replace_slang)
        st.write('Mengganti kata tidak baku:')
        st.write(df['clean_text'])

        def tokenize_text(text):
            tokens = nltk.tokenize.word_tokenize(text)
            return tokens

        df['token'] = df['clean_text'].apply(tokenize_text)
        st.write('Menjadikan kalimat ke token:')
        st.write(df['token'])

        factory = StopWordRemoverFactory()
        stopwords = factory.get_stop_words()

        def stopwords_text(tokens):
            cleaned_tokens = []
            for token in tokens:
                if token not in stopwords:
                    cleaned_tokens.append(token)
            return cleaned_tokens

        df['stopword'] = df['token'].apply(stopwords_text)
        st.write('Menghapus stopwords:')
        st.write(df['stopword'])

        stem_factory = StemmerFactory()
        stemmer = stem_factory.create_stemmer()

        def stemming_text(tokens):
            stem = [stemmer.stem(token) for token in tokens]
            return stem

        df['stemming'] = df['stopword'].apply(stemming_text)
        st.write('Stemming:')
        st.write(df['stemming'])

        # Prepare data
        max_features = 3000
        max_sequence_length = 200
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(df['stopword'].values)

        X = tokenizer.texts_to_sequences(df['stopword'].values)
        X = pad_sequences(X, maxlen=max_sequence_length)
        Y = pd.get_dummies(df['label']).values

        token_df = pd.DataFrame(dict(tokenized_text=[' '.join(map(str, seq)) for seq in X]))
        st.write("Mengubah teks menjadi token:")
        st.write(token_df)

        X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(
            X, Y, np.arange(len(X)), test_size=0.2, random_state=128
        )
        st.write("Split Data:")
        st.write("Data Train:", X_train.shape)
        st.write("Data Test:", X_test.shape)

        # Build model
        model = Sequential([
            Embedding(max_features, 32, input_length=max_sequence_length),
            SpatialDropout1D(0.5),
            LSTM(16, dropout=0.5, recurrent_dropout=0.5),
            Dropout(0.5),
            Dense(2, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Define callbacks
        model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

        # Train model
        epochs = 15
        batch_size = 64
        progress_bar = st.progress(0)

        for epoch in range(epochs):
            history = model.fit(
                X_train, Y_train,
                epochs=1,
                batch_size=batch_size,
                validation_data=(X_test, Y_test),
                callbacks=[model_checkpoint],
                verbose=0
            )

            # Update the progress bar
            progress_bar.progress((epoch + 1) / epochs)

            # Display accuracy and loss
            accuracy = history.history['accuracy'][-1]
            loss = history.history['loss'][-1]
            val_accuracy = history.history['val_accuracy'][-1]
            val_loss = history.history['val_loss'][-1]
            st.write(f"Epoch {epoch + 1}/{epochs}")
            st.write(f"Akurasi: {accuracy:.4f}, Loss: {loss:.4f}")
            st.write(f"Test Akurasi: {val_accuracy:.4f}, Test Loss: {val_loss:.4f}")
            st.markdown("---")

        # Evaluate model
        predictions = model.predict(X_test)
        predicted_classes = predictions.argmax(axis=-1)

        test_indices = indices_test

        # Results
        labels = df['label'].unique()
        predicted_labels = [labels[i] for i in predicted_classes]
        results_df = pd.DataFrame({
            'full_text': df.loc[test_indices, 'full_text'].values,
            'predicted_label': predicted_labels,
            'actual_label': df.loc[test_indices, 'label'].values
        })

        save_classification_results(results_df)
        st.success("Pelatihan selesai!")

        # Display results in Streamlit
        st.write("Hasil Klasifikasi:")
        st.dataframe(results_df)

        # Simpan model dan tokenizer yang sudah dilatih
        model = model  # Gunakan model yang sudah dilatih
        tokenizer = tokenizer  # Gunakan tokenizer yang sudah dilatih

        # Path direktori tujuan
        save_dir = r"C:\streamlit-postgres\model_train"

        # Pastikan direktori ada
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Path file untuk model dan tokenizer
        model_file = os.path.join(save_dir, "model_trained.pkl")
        tokenizer_file = os.path.join(save_dir, "tokenizer.pkl")

        # Simpan model dan tokenizer
        try:
            # Simpan model
            with open(model_file, 'wb') as model_pkl:
                pickle.dump(model, model_pkl)

            # Simpan tokenizer
            with open(tokenizer_file, 'wb') as tokenizer_pkl:
                pickle.dump(tokenizer, tokenizer_pkl)

            # Tampilkan pesan sukses
            st.success(f"Model berhasil disimpan ke direktori: {save_dir}")
            st.write(f"File model: {model_file}")
            st.write(f"File tokenizer: {tokenizer_file}")
        except Exception as e:
            st.error(f"Gagal menyimpan file: {e}")

def classification_page():
    st.title("Pelatihan Data Opini")
    st.write(get_comments())
    if st.button("Klasifikasi"):
        classification_process()

classification_page()
