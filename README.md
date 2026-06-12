# Klasifikasi Motif Batik Nusantara dengan GLCM + SVM

## Cara Menjalankan

1. Install dependencies:
   pip install -r requirements.txt

2. (Opsional) Latih ulang model:
   python train_model.py
   - Pastikan dataset ada di folder `dataset/raw_batik_v2/raw_batik_v2/{train,test}/<kelas>/*.jpg`
   - Sesuaikan path DATASET_DIR di train_model.py

3. Jalankan aplikasi:
   python app.py

4. Buka browser: http://localhost:5000

## Struktur Proyek
- app.py            -> Flask app & API prediksi
- glcm_utils.py      -> Modul ekstraksi fitur GLCM
- train_model.py     -> Skrip training SVM dari dataset
- model/             -> Model SVM, label encoder, metadata (sudah dilatih)
- templates/         -> Halaman HTML (Jinja2)
- static/css, static/js -> Tampilan (berbasis template CryptoVault)
- static/samples/    -> Contoh gambar tiap kelas batik untuk galeri

## Fitur
- Dashboard: ringkasan dataset & akurasi model
- Klasifikasi Batik: upload gambar -> ekstraksi fitur GLCM -> prediksi motif (Top-5)
- Tentang Metode: penjelasan GLCM, parameter, dan rumus fitur
