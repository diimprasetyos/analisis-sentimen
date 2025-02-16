import re
import string
import numpy as np
import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load model dan token dari database
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

# load model dan tokenizer yg sudah dilatih
tokenizer = load_tokenizer()
model = load_trained_model()

# fungsi preprocessing teks
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
    words = text.split()
    new_words = [kamus.get(word, word) for word in words]
    return ' '.join(new_words)

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

def stopwords_text(tokens):
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    return cleaned_tokens

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def stemming_text(tokens):
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    # mengubah ke huruf kecil
    text = text.lower()
    # menghapus URL
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    # menghapus username/mention
    text = re.sub(r'@@[A-Za-z0-9_-]+|@[A-Za-z0-9_-]+', '', text)
    # menghapus angka
    text = re.sub(r'\d+', '', text)
    # menghapus semua simbol punctuation kecuali spasi
    text = re.sub(r'[^\w\s]', '', text)
    # mengganti kata tidak baku
    text = replace_slang(text)
    # tokenisasi
    tokens = word_tokenize(text)
    # menghapus stopwords
    tokens = stopwords_text(tokens)
    # stemming
    tokens = stemming_text(tokens)
    return ' '.join(tokens)

# fungsi klasifikasi teks
def single_sentence_classification(sentence):
    # preprocessing input user
    preprocessed_sentence = preprocess_text(sentence)
    st.write('Hasil preprocessing:')

    # tokenize dan pad sequences
    sequence = tokenizer.texts_to_sequences([preprocessed_sentence])
    sequence = pad_sequences(sequence, maxlen=200)

    # prediksi kelas
    prediction = model.predict(sequence)
    confidence = np.max(prediction) * 100
    predicted_class = prediction.argmax(axis=-1)[0]

    # map predicted class ke label
    labels = ['negatif', 'positif']
    predicted_label = labels[predicted_class]

    return predicted_label, confidence

def classification_page():
    st.markdown("### Cek Sentimen Kalimat")
    st.write("""
    masukkan kalimat untuk dilakukan klasifikasi.
    """)

    # input teks
    user_input = st.text_input("Kalimat:", "")

    if st.button("Cek Sentimen"):
        if user_input:
            try:
                result, confidence = single_sentence_classification(user_input)
                st.info(f"kalimat memiliki sentimen **{result}** sebesar **{confidence:.2f}%**.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses kalimat: {e}")
        else:
            st.warning("Mohon masukkan kalimat terlebih dahulu.")

classification_page()
