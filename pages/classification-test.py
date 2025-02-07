import re
import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sqlalchemy import create_engine
import psycopg2
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Fungsi untuk koneksi ke database PostgreSQL
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

# Fungsi untuk memuat tokenizer
@st.cache_resource
def load_tokenizer():
    try:
        with open('model/tokenizer.pkl', 'rb') as file:
            tokenizer = pickle.load(file)
        return tokenizer
    except Exception as e:
        st.error(f"Gagal memuat tokenizer: {e}")
        return None

# Fungsi untuk memuat model yang sudah dilatih
@st.cache_resource
def load_trained_model():
    try:
        with open('model/model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

tokenizer = load_tokenizer()
model = load_trained_model()

# Fungsi untuk mengambil data dari database
@st.cache_data
def get_comments():
    try:
        query = "SELECT full_text FROM tb_test;"
        df = pd.read_sql(query, conn)
        if df.empty:
            st.warning("Tidak ada data ditemukan.")
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data: {e}")
        return pd.DataFrame()

# Fungsi untuk menyimpan hasil klasifikasi ke PostgreSQL
def save_classification_results(results_df):
    try:
        engine = create_engine('postgresql+psycopg2://postgres:admin@localhost:5432/postgres-streamlit')
        results_df.to_sql('tb_result', con=engine, if_exists='append', index=False)
        st.success("Hasil klasifikasi berhasil disimpan di tb_result.")
    except Exception as e:
        st.error(f"Gagal menyimpan hasil klasifikasi: {e}")

# Proses klasifikasi data
def classification_process_test_data():
    data = get_comments()
    if data.empty:
        st.error("Data tidak tersedia untuk klasifikasi.")
        return

    # Pre-Processing
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
    # Menghapus semua simbol punctuation kecuali spasi
    df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    st.write('Menghapus semua tanda baca:')
    st.write(df['full_text'])

    # Ganti kata tidak baku (jika ada kamus slang)
    def replace_slang(text):
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
        words = text.split()
        return " ".join([kamus.get(word, word) for word in words])

    df['full_text'] = df['full_text'].apply(replace_slang)

    # Tokenisasi
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    stemmer = StemmerFactory().create_stemmer()

    def preprocess_text(text):
        tokens = text.split()
        tokens = [token for token in tokens if token not in stopwords]
        # tokens = [stemmer.stem(token) for token in tokens]
        return " ".join(tokens)

    df['full_text'] = df['full_text'].apply(preprocess_text)

    # Konversi teks ke sequences
    if tokenizer:
        X = tokenizer.texts_to_sequences(df['full_text'].values)
        X = pad_sequences(X, maxlen=200)

        # Prediksi menggunakan model
        if model:
            predictions = model.predict(X)
            predicted_classes = predictions.argmax(axis=-1)
            labels = ['negatif', 'positif']
            predicted_labels = [labels[i] for i in predicted_classes]

            # Simpan hasil klasifikasi ke DataFrame
            results_df = pd.DataFrame({
                'full_text': data['full_text'],
                'predicted_label': predicted_labels
            })

            # Tampilkan hasil
            st.write("Hasil Prediksi:")
            st.dataframe(results_df)

            # Simpan hasil ke database
            save_classification_results(results_df)
        else:
            st.error("Model tidak ditemukan.")
    else:
        st.error("Tokenizer tidak ditemukan.")

# Halaman utama
def classification_page():
    st.title("Pengujian Data Opini")
    st.write(get_comments())

    if st.button("Klasifikasi Data"):
        classification_process_test_data()

classification_page()
