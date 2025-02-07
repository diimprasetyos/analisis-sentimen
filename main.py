import streamlit as st

pages = {
    "": [
        st.Page("pages/home.py", title="Beranda"),
        st.Page("pages/train.py", title="Data Pelatihan"),
        st.Page("pages/test.py", title="Data Pengujian"),
        st.Page("pages/classification-train.py", title="Pelatihan Data Opini"),
        st.Page("pages/result-train.py", title="Evaluasi Hasil Pelatihan"),
        st.Page("pages/classification-test.py", title="Pengujian Data Opini"),
        st.Page("pages/result-test.py", title="Evaluasi Pengujian"),
        st.Page("pages/identify.py", title="Cek Kalimat"),
    ]
}

pg = st.navigation(pages)
pg.run()