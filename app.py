import re
import numpy as np
import csv
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from nlp_id.lemmatizer import Lemmatizer
from flask import Flask, request, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle

# Inisialisasi Flask
app = Flask(__name__, static_folder="static")

# Fungsi untuk melakukan preprocessing pada teks
def preprocess_text(text):
    text = text.lower()  # Case folding
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Hapus karakter non-alfanumerik
    
    lemmatizer = Lemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(lemmatized_words)
    
    return text

# Fungsi untuk menghitung skor kesamaan
def grade_essay(jawaban_siswa, kunci_jawaban):
    jawaban_siswa = jawaban_siswa.decode('utf-8')
    preprocessed_jawaban_siswa = preprocess_text(jawaban_siswa)
    preprocessed_kunci_jawaban = preprocess_text(kunci_jawaban)
    
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-large-p2")
    model = AutoModel.from_pretrained("indobenchmark/indobert-large-p2")
    
    encoded_jawaban_siswa = tokenizer(preprocessed_jawaban_siswa, return_tensors="pt", padding=True, truncation=True)
    encoded_kunci_jawaban = tokenizer(preprocessed_kunci_jawaban, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        jawaban_siswa_embeddings = model(**encoded_jawaban_siswa).last_hidden_state[:, 0, :].numpy()
        kunci_jawaban_embeddings = model(**encoded_kunci_jawaban).last_hidden_state[:, 0, :].numpy()

    similarity_score = cosine_similarity(jawaban_siswa_embeddings, kunci_jawaban_embeddings)[0][0]
    
    # Mengonversi nilai similarity ke dalam rentang 0-100
    similarity_percentage = min(100, max(0, round(similarity_score * 100)))
    
    return similarity_percentage

@app.route('/')
def dashboard():
    return render_template('dashboard.html')
@app.route('/grading', methods=['GET', 'POST'])
def index():
    result_data = []  # List untuk menyimpan informasi nama, kelas, dan nilai similarity

    if request.method == 'POST':
        kunci_jawaban = request.form['kunci_jawaban']
        
        # Mengecek apakah pengguna mengunggah file CSV
        if 'csv_file' in request.files:
            csv_file = request.files['csv_file']
            if csv_file.filename != '':
                # Membaca file CSV dengan pemisah titik koma (;)
                df = pd.read_csv(csv_file, delimiter=';')
                
                # Check if the "Jawaban Siswa" column exists in the DataFrame
                if 'Jawaban Siswa' in df.columns:
                    df['Jawaban Siswa'] = df['Jawaban Siswa'].fillna('')
                    
                    # Convert DataFrame column values to strings
                    jawaban_siswa_values = df['Jawaban Siswa'].astype(str).values
                    
                    # Iterate through each row in the DataFrame
                    for index, row in df.iterrows():
                        jawaban_siswa = jawaban_siswa_values[index]
                        # Konversi teks jawaban_siswa menjadi bytes
                        jawaban_siswa = jawaban_siswa.encode('utf-8')

                        # Hitung skor kesamaan
                        similarity_score = grade_essay(jawaban_siswa, kunci_jawaban)
                        
                        # Menambahkan informasi ke result_data
                        result_data.append({
                            'Nama Siswa': row['Nama Siswa'],  # Mengambil nama siswa dari kolom 'Nama Siswa'
                            'Kelas': row['Kelas'],  # Mengambil kelas dari kolom 'Kelas'
                            'Similarity Score': similarity_score
                        })
                else:
                    print("Column 'Jawaban Siswa' not found in the CSV file.")
        else:
            print("No CSV file uploaded.")
        
        # Jika tidak ada unggahan CSV atau tidak ada kolom 'jawaban', gunakan kunci jawaban manual
        if not result_data:
            kunci_jawaban_manual = request.form['kunci_jawaban']
            # Iterate through each row in the DataFrame
            for index, row in df.iterrows():
                jawaban_siswa = jawaban_siswa_values[index]
                similarity_score = grade_essay(jawaban_siswa, kunci_jawaban_manual)
            similarity_score = round(similarity_score * 100)

                
                # Menambahkan informasi ke result_data
            result_data.append({
                    'Nama Siswa': row['Nama Siswa'],  # Atur nama siswa manual sesuai kebutuhan
                    'Kelas': row['Kelas'],  # Atur kelas manual sesuai kebutuhan
                    'Similarity Score': similarity_score
                })
    
    return render_template('grading.html', result_data=result_data)

# Routing untuk mengunduh template CSV
@app.route('/download_template', methods=['GET'])
def download_template():
    # Data contoh yang akan ditulis ke template CSV
    sample_data = [
        {'Nama Siswa': 'Nama Siswa 1', 'Kelas': 'Kelas 1', 'Jawaban Siswa': 'Jawaban Siswa 1'},
        {'Nama Siswa': 'Nama Siswa 2', 'Kelas': 'Kelas 2', 'Jawaban Siswa': 'Jawaban Siswa 2'}
        # Tambahkan data contoh lainnya sesuai kebutuhan
    ]

    # Header CSV
    csv_header = ['Nama Siswa', 'Kelas', 'Jawaban Siswa']

    # Buat nama file CSV
    csv_filename = 'template.csv'

    # Buat file CSV dengan data contoh
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        csv_writer.writeheader()
        csv_writer.writerows(sample_data)

    # Kembalikan file CSV sebagai tautan unduhan
    return send_file(csv_filename, as_attachment=True)




# Fungsi untuk mengubah data masukan
def transform_data(form_data):
    def tr_nilai_matkul_bahasa_inggris_1(x):
        if x >= 80:
            return 7
        elif x >= 75:
            return 6
        elif 70 > x >= 65:
            return 5
        elif 75 > x >= 70:
            return 4
        elif x >= 60:
            return 3
        elif x >= 55:
            return 2
        elif x >= 45:
            return 1
        else:
            return 0

    def tr_nilai_matematika_sma_kelas_12(x):
        return float(x)

    def tr_memiliki_beasiswa(x):
        unique_value = ["Tidak", "Iya"]
        return unique_value.index(x)

    def tr_waktu_khusus_belajar(x):
        unique_value = ["Tidak", "Iya"]
        return unique_value.index(x)

    def tr_suka_lomba(x):
        unique_value = ["Sangat tidak suka", "Tidak suka", 'Netral', 'Suka', 'Sangat suka']
        return unique_value.index(x)

    def tr_total_mengikuti_lomba(x):
        unique_value = ['0', '1-3', '4-10', '11-20', '>20']
        return unique_value.index(x)

    def tr_pendidikan_ibu(x):
        unique_value = ['Tidak lulus SD', 'SD/sederajat', 'SMP/sederajat', 'SMA/sederajat', 'D1-D3', 'D4/Sarjana Terapan', 'S1/sederajat', 'S2/sederajat']
        return unique_value.index(x)

    def tr_pendidikan_ayah(x):
        unique_value = ['Tidak lulus SD', 'SD/sederajat', 'SMP/sederajat', 'SMA/sederajat', 'D1-D3', 'D4/Sarjana Terapan', 'S1/sederajat', 'S2/sederajat']
        return unique_value.index(x)

    def tr_penghasilan_orang_tua(x):
        unique_value = ['< 2.000.000', '2.000.000 - 4.000.000', '4.000.000 - 8.000.000', '8.000.000 - 40.000.000', '> 40.000.000']
        return unique_value.index(x)

    def tr_estimasi_waktu_perjalanan_ke_kampus(x):
        unique_value = ['<15', '15-30', '30-60', '60-120', '>120']
        return unique_value.index(x)

    def tr_tempat_tinggal_kuliah_kos(x):
        unique_value = ["Tidak", "Iya"]
        return unique_value.index(x)

    form_data['nilai_matkul_bahasa_inggris_1'] = tr_nilai_matkul_bahasa_inggris_1(form_data['nilai_matkul_bahasa_inggris_1'])
    form_data['nilai_matematika_sma_kelas_12'] = tr_nilai_matematika_sma_kelas_12(form_data['nilai_matematika_sma_kelas_12'])
    form_data['memiliki_beasiswa'] = tr_memiliki_beasiswa(form_data['memiliki_beasiswa'])
    form_data['waktu_khusus_belajar'] = tr_waktu_khusus_belajar(form_data['waktu_khusus_belajar'])
    form_data['suka_lomba'] = tr_suka_lomba(form_data['suka_lomba'])
    form_data['total_mengikuti_lomba'] = tr_total_mengikuti_lomba(form_data['total_mengikuti_lomba'])
    form_data['pendidikan_ibu'] = tr_pendidikan_ibu(form_data['pendidikan_ibu'])
    form_data['pendidikan_ayah'] = tr_pendidikan_ayah(form_data['pendidikan_ayah'])
    form_data['penghasilan_orang_tua'] = tr_penghasilan_orang_tua(form_data['penghasilan_orang_tua'])
    form_data['estimasi_waktu_perjalanan_ke_kampus'] = tr_estimasi_waktu_perjalanan_ke_kampus(form_data['estimasi_waktu_perjalanan_ke_kampus'])
    form_data['tempat_tinggal_kuliah_kos'] = tr_tempat_tinggal_kuliah_kos(form_data['tempat_tinggal_kuliah_kos'])

    return form_data

# Fungsi untuk memprediksi cluster
def predict_cluster(data):
    with open('./tools/scaler.pkl', 'rb') as scaler_path:
        scaler = pickle.load(scaler_path)
    with open('./model/model_kprototype.pkl', 'rb') as model_path:
        model_kprototype = pickle.load(model_path)

    scaled_columns = [
        'memiliki_beasiswa',
        'nilai_matkul_bahasa_inggris_1',
        'nilai_matematika_sma_kelas_12',
        'suka_lomba',
        'total_mengikuti_lomba',
        'pendidikan_ibu',
        'pendidikan_ayah',
        'penghasilan_orang_tua',
        'estimasi_waktu_perjalanan_ke_kampus',
    ]

    data[scaled_columns] = scaler.transform(data[scaled_columns])
    data = data.drop(columns=['nama'])
    if 'cluster' in data.columns:
        data = data.drop(columns=['cluster'])

    cluster = model_kprototype.predict(data, categorical=[9])
    del scaler
    del model_kprototype
    return cluster

@app.route('/personalisasi')
def personalisasi():
    return render_template('personalisasi.html')

@app.route('/submit', methods=['POST'])
def submit():
    form_data = {
        'nama': request.form['nama'],
        'nilai_matkul_bahasa_inggris_1': float(request.form['nilai_matkul_bahasa_inggris_1']),
        'nilai_matematika_sma_kelas_12': float(request.form['nilai_matematika_sma_kelas_12']),
        'memiliki_beasiswa': request.form['memiliki_beasiswa'],
        'waktu_khusus_belajar': request.form['waktu_khusus_belajar'],
        'suka_lomba': request.form['suka_lomba'],
        'total_mengikuti_lomba': request.form['total_mengikuti_lomba'],
        'pendidikan_ibu': request.form['pendidikan_ibu'],
        'pendidikan_ayah': request.form['pendidikan_ayah'],
        'penghasilan_orang_tua': request.form['penghasilan_orang_tua'],
        'estimasi_waktu_perjalanan_ke_kampus': request.form['estimasi_waktu_perjalanan_ke_kampus'],
        'tempat_tinggal_kuliah_kos': request.form['tempat_tinggal_kuliah_kos'],
    }

    # Ubah data formulir menggunakan fungsi transform_data
    transformed_data = transform_data(form_data)

    # Masukkan data yang telah diubah ke dalam DataFrame
    df = pd.DataFrame([transformed_data])

    # Prediksi cluster untuk data tersebut
    df['cluster'] = predict_cluster(df)

    # Visualisasi analisis cluster
    sns.set(style="whitegrid")  # Gaya plot seaborn

    # Hitung jumlah data dalam setiap cluster
    cluster_counts = df['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']

    # Buat plot bar untuk jumlah data dalam setiap cluster
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Cluster', y='Count', data=cluster_counts, palette='viridis')
    ax.set(xlabel='Cluster', ylabel='Jumlah Data', title='Distribusi Data dalam Setiap Cluster')

    # Simpan plot sebagai gambar
    plot_filename = 'cluster_distribution.png'
    plt.savefig(plot_filename, format='png')

    # Tampilkan hasil personalisasi di halaman hasil bersama dengan gambar plot
    return render_template('hasil.html', data=transformed_data, cluster=df['cluster'].values[0], plot_filename=plot_filename)

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
