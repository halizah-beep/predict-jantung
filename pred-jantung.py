import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load Model SVM
rf_model = joblib.load("rf_model.pkl")

st.markdown("""
    <style>
    /* Styling untuk tombol prediksi */
    .stButton>button {
        border: 2px solid #eaeaea !important;
        color: black !important;
        padding: 5px 15px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Tampilan utama
st.markdown("<h1 style='text-align: left;'>Sistem Prediksi Diagnosa Penyakit Jantung</h1>", unsafe_allow_html=True)

# Sidebar untuk input parameter
st.sidebar.header("Input Parameter")

umur = st.sidebar.number_input("Umur", min_value=1, max_value=120, value=50, step=1)
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 1 else "Laki-Laki")
kunjungan = st.sidebar.selectbox("Kunjungan Pasien", [0, 1], format_func=lambda x: "Baru" if x == 0 else "Lama")
kasus = st.sidebar.selectbox("Jenis Kasus", [0, 1], format_func=lambda x: "Baru" if x == 0 else "Lama")

# Tampilkan input dalam bentuk tabel
st.subheader("Input Parameters")
data = {"Umur": [umur], "Jenis Kelamin": [jenis_kelamin], "Kunjungan": [kunjungan], "Kasus": [kasus]}
df = pd.DataFrame(data)
st.dataframe(df.style.set_table_styles([
    {'selector': 'thead th', 'props': [('text-align', 'center')]},  
    {'selector': 'tbody td', 'props': [('text-align', 'center')]},  
]), hide_index=True)

# Tombol Prediksi
if st.button("Prediksi Penyakit Jantung"):
    input_data = np.array([[jenis_kelamin, kunjungan, kasus, umur]])

    # Prediksi dengan model SVM
    prediction = rf_model.predict(input_data)
    probability = rf_model.predict_proba(input_data)[0][1] 

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.error("**Pasien Berisiko Terdiagnosa Penyakit Jantung!**", icon="⚠️")
    else:
        st.success("**Pasien Tidak Terdiagnosa Penyakit Jantung.**", icon="✅")
    st.subheader("📈 Probabilitas Diagnosa")
    st.progress(probability)
    