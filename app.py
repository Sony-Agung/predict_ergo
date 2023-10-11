import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Memuat model dari file pickle
with open('model/logistic_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Fungsi untuk melakukan prediksi pada data baru
def predict_efficiency(data):
    prediction = loaded_model.predict(data)
    probability = loaded_model.predict_proba(data)[:, 1]
    return prediction, probability

# Tampilan aplikasi Streamlit
def main():
    st.title('Prediksi Efisiensi Mesin CNC')
    st.sidebar.title('Input Data')

    # Form input data
    jenis_mesin = st.sidebar.selectbox('Jenis Mesin', ('CNC Lathe', 'CNC Milling'))
    waktu_operasi = st.sidebar.number_input('Waktu Operasi (menit)', min_value=0, max_value=1000, value=35)
    kelelahan_operator = st.sidebar.slider('Kelelahan Operator (skala 1-10)', 1, 10, 6)
    pencahayaan = st.sidebar.number_input('Pencahayaan (lux)', min_value=0, max_value=2000, value=750)
    suhu = st.sidebar.number_input('Suhu (°C)', min_value=-50, max_value=100, value=23)
    ketinggian_meja = st.sidebar.number_input('Ketinggian Meja (cm)', min_value=0, max_value=200, value=88)

    # Menyesuaikan jenis mesin dengan label encoding
    label_mapping = {'CNC Milling': 1, 'CNC Lathe': 0}
    jenis_mesin_encoded = label_mapping.get(jenis_mesin)

    # Menyiapkan data baru untuk prediksi
    data_new = [[jenis_mesin_encoded, waktu_operasi, kelelahan_operator, pencahayaan, suhu, ketinggian_meja]]

    if st.sidebar.button('Prediksi', key='prediksi_button'):
        prediction_label, probability = predict_efficiency(data_new)
        st.subheader('Hasil Prediksi Efisiensi Mesin')
        if prediction_label[0] == 1:
            st.success(f'Efisien (Probabilitas: {probability[0]*100:.2f}%)')
        else:
            st.error(f'Tidak Efisien (Probabilitas: {probability[0]*100:.2f}%)')

        # Menggambar plot bar dengan sns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot pengaruh fitur
        data_for_plot = {
            'Fitur': ['Waktu Operasi', 'Kelelahan Operator', 'Pencahayaan', 'Suhu', 'Ketinggian Meja'],
            'Pengaruh': loaded_model.coef_[0][1:]  # Mengambil pengaruh dari fitur kedua hingga terakhir
        }
        df_for_plot = pd.DataFrame(data_for_plot)
        sns.barplot(x='Pengaruh', y='Fitur', data=df_for_plot, orient='h', ax=ax1)
        ax1.set_title('Pengaruh Fitur Terhadap Prediksi')

        # Plot perbandingan efisiensi dan ketidak-efisiensi
        df_comparison = pd.DataFrame({
            'Kategori': ['Efisien', 'Tidak Efisien'],
            'Jumlah': [sum(prediction_label == 1), sum(prediction_label == 0)]
        })
        sns.barplot(x='Kategori', y='Jumlah', data=df_comparison, ax=ax2)
        ax2.set_title('Perbandingan Efisiensi dan Ketidak-Efisiensi')

        st.pyplot(fig)

    # Menu Info ke GitHub Anda dengan tampilan yang lebih menarik
    st.sidebar.markdown('## Info')
    st.sidebar.markdown('<a href="https://github.com/Sony-Agung/ergonomic_predict" target="_blank" style="text-decoration: none; color: #0366d6;">'
                        ' Link ke GitHub</a>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
