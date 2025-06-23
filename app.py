
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# Konfigurasi halaman
st.set_page_config(page_title="Sistem Pembagian Kelas", layout="centered")
st.title("📚 Sistem Pembagian Kelas")

# Form input
with st.form("form_siswa"):
    nis = st.text_input("NIS")
    nama = st.text_input("Nama").strip().upper()
    asal = st.text_input("Asal Sekolah")

    st.markdown("**Nilai Rapor:**")
    pa = st.number_input("PA/BP", min_value=0.0, max_value=100.0, step=0.1)
    bind = st.number_input("Bahasa Indonesia", min_value=0.0, max_value=100.0, step=0.1)
    mat = st.number_input("Matematika", min_value=0.0, max_value=100.0, step=0.1)
    bing = st.number_input("Bahasa Inggris", min_value=0.0, max_value=100.0, step=0.1)

    submitted = st.form_submit_button("Proses")

if submitted:
    if not nis or not nama or not asal or pa == 0.0 or bind == 0.0 or mat == 0.0 or bing == 0.0:
        st.warning("⚠️ Mohon lengkapi semua data terlebih dahulu sebelum diproses.")
    else:
        try:
            # Membaca data latih
            df = pd.read_csv("data_latih3.csv")

            # Samakan format nama untuk pencocokan
            df["NAMA_CLEAN"] = df["NAMA"].str.strip().str.upper()

            if nama in df["NAMA_CLEAN"].values:
                data_siswa = df[df["NAMA_CLEAN"] == nama].iloc[0]
                kategori = data_siswa["Kategori"]
                kelas = data_siswa["Kelas"]
                sumber = "📌 Data berhasil diproses"
            else:
                # Proses clustering jika nama tidak ada
                fitur = ['PA/BP', 'Bahasa Indonesia', 'Matematika', 'Bahasa Inggris']
                data_latih = df[fitur]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(data_latih)

                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(X_scaled)

                data_baru = np.array([[pa, bind, mat, bing]])
                data_baru_scaled = scaler.transform(data_baru)
                klaster = kmeans.predict(data_baru_scaled)[0]

                mean_vals = pd.DataFrame(X_scaled, columns=fitur).groupby(kmeans.labels_).mean()
                cluster_order = mean_vals.mean(axis=1).sort_values().index

                kategori_map = {
                    cluster_order[0]: "Rendah",
                    cluster_order[1]: "Sedang",
                    cluster_order[2]: "Tinggi"
                }
                kategori = kategori_map[klaster]
                kelas = {
                    "Rendah": "Kelas C",
                    "Sedang": "Kelas B",
                    "Tinggi": "Kelas A"
                }[kategori]
                sumber = "🧠 Prediksi berdasarkan model KMeans"

            # Tampilkan hasil
            st.success("📋 Hasil:")
            st.write(f"**Nama:** {nama}")
            st.write(f"**NIS:** {nis}")
            st.write(f"**Asal Sekolah:** {asal}")
            st.write(f"**Kategori Prestasi:** {kategori}")
            st.write(f"**Kelas Direkomendasikan:** {kelas}")
            st.caption(sumber)

        except FileNotFoundError:
            st.error("❌ File 'data_latih.csv' tidak ditemukan.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")

# Footer
st.markdown("---")
st.caption("© 2025 Sistem Pembagian Kelas | by Annisa Iksanti")
