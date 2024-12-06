import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, redirect, url_for
from sklearn.cluster import KMeans

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Pastikan folder tersedia
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Halaman utama untuk upload file."""
    return render_template('home.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    """Halaman upload file CSV."""
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('block_results', filename=file.filename))

@app.route('/result/<filename>')
def block_results(filename):
    """Menampilkan hasil uji clustering dari 3 blok."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)

    # Pastikan ada kolom 'cluster' pada data (ini adalah kolom target)
    if 'cluster' not in data.columns:
        # Misal: Menambahkan kolom cluster dummy, Anda bisa menyesuaikan sesuai kebutuhan
        data['cluster'] = [0] * len(data)  # Anda bisa menyesuaikan jika ada label yang sesuai

    # Membagi data
    pergaulan = data[['X1P1', 'X1P2', 'X1P3', 'X1P4']]
    sosial_ekonomi = data[['X2P1', 'X2P2', 'X2P3', 'X2P4', 'X2P5', 'X2P6', 'X2P7']]
    data['Pergaulan_Avg'] = pergaulan.mean(axis=1)
    data['Sosial_Ekonomi_Avg'] = sosial_ekonomi.mean(axis=1)

    data_i = data.iloc[:50]
    data_ii = data.iloc[50:100]
    data_iii = data.iloc[100:150]

    blocks = [
        {"train": pd.concat([data_i, data_ii]), "test": data_iii, "block_num": 1, "block_name": "Block 1"},
        {"train": pd.concat([data_i, data_iii]), "test": data_ii, "block_num": 2, "block_name": "Block 2"},
        {"train": pd.concat([data_ii, data_iii]), "test": data_i, "block_num": 3, "block_name": "Block 3"}
    ]

    results = []
    for block in blocks:
        train_data = block["train"]
        test_data = block["test"].copy()
        block_num = block["block_num"]
        block_name = block["block_name"]

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(train_data[['Pergaulan_Avg', 'Sosial_Ekonomi_Avg']])
        test_data['Predicted_Cluster'] = kmeans.predict(test_data[['Pergaulan_Avg', 'Sosial_Ekonomi_Avg']]).astype(int)

        # Menambahkan kolom cluster asli
        test_data['y_true'] = data.loc[test_data.index, 'cluster'].astype(int)

        # Hitung TP, TN, FP, FN
        TP = sum((test_data['Predicted_Cluster'] == 1) & (test_data['y_true'] == 1))
        TN = sum((test_data['Predicted_Cluster'] == 0) & (test_data['y_true'] == 0))
        FP = sum((test_data['Predicted_Cluster'] == 1) & (test_data['y_true'] == 0))
        FN = sum((test_data['Predicted_Cluster'] == 0) & (test_data['y_true'] == 1))

        # Menghitung akurasi berdasarkan TP, TN, FP, FN
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        results.append({"block_num": block_num, "block_name": block_name, "accuracy": accuracy, "test_data": test_data, "centroids": kmeans.cluster_centers_})

        # Plot untuk setiap blok
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='Pergaulan_Avg', y='Sosial_Ekonomi_Avg', hue='Predicted_Cluster', style='Predicted_Cluster',
            data=test_data, markers=["P", "o"], palette="Set1", s=100)
        plt.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], color='purple', s=300, marker='X', label='Centroid Cluster 0')
        plt.scatter(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], color='orange', s=300, marker='X', label='Centroid Cluster 1')
        plt.title(f'{block_name}: Clustering Results')
        plt.xlabel('Rata-rata Pergaulan')
        plt.ylabel('Rata-rata Sosial Ekonomi')
        plt.legend()
        plt.grid()

        # Simpan gambar untuk setiap blok
        plot_path = os.path.join(app.config['STATIC_FOLDER'], f'{block_name}_clustering.png')
        plt.savefig(plot_path)
        plt.close()

    return render_template(
        'block_results.html',
        filename=filename,
        results=results
    )

@app.route('/highest_accuracy/<filename>')
def highest_accuracy(filename):
    """Menampilkan hasil dengan akurasi tertinggi."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)

    # Membagi data
    pergaulan = data[['X1P1', 'X1P2', 'X1P3', 'X1P4']]
    sosial_ekonomi = data[['X2P1', 'X2P2', 'X2P3', 'X2P4', 'X2P5', 'X2P6', 'X2P7']]
    data['Pergaulan_Avg'] = pergaulan.mean(axis=1)
    data['Sosial_Ekonomi_Avg'] = sosial_ekonomi.mean(axis=1)

    data_i = data.iloc[:50]
    data_ii = data.iloc[50:100]
    data_iii = data.iloc[100:150]

    blocks = [
        {"train": pd.concat([data_i, data_ii]), "test": data_iii, "block_num": 1, "block_name": "Block 1"},
        {"train": pd.concat([data_i, data_iii]), "test": data_ii, "block_num": 2, "block_name": "Block 2"},
        {"train": pd.concat([data_ii, data_iii]), "test": data_i, "block_num": 3, "block_name": "Block 3"}
    ]

    accuracies = []
    for block in blocks:
        train_data = block["train"]
        test_data = block["test"].copy()
        block_num = block["block_num"]
        block_name = block["block_name"]

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(train_data[['Pergaulan_Avg', 'Sosial_Ekonomi_Avg']])
        test_data['Predicted_Cluster'] = kmeans.predict(test_data[['Pergaulan_Avg', 'Sosial_Ekonomi_Avg']])

        # Menambahkan kolom cluster asli
        test_data['y_true'] = data.loc[test_data.index, 'cluster']

        # Hitung TP, TN, FP, FN
        TP = sum((test_data['Predicted_Cluster'] == 1) & (test_data['y_true'] == 1))
        TN = sum((test_data['Predicted_Cluster'] == 0) & (test_data['y_true'] == 0))
        FP = sum((test_data['Predicted_Cluster'] == 1) & (test_data['y_true'] == 0))
        FN = sum((test_data['Predicted_Cluster'] == 0) & (test_data['y_true'] == 1))

        # Menghitung akurasi berdasarkan TP, TN, FP, FN
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracies.append({"block_num": block_num, "block_name": block_name, "accuracy": accuracy, "test_data": test_data, "centroids": kmeans.cluster_centers_})

    # Cari akurasi tertinggi
    highest_acc = max(accuracies, key=lambda x: x['accuracy'])

    test_data = highest_acc["test_data"]
    centroids = highest_acc["centroids"]
    block_num = highest_acc["block_num"]
    block_name = highest_acc["block_name"]

    # Plot ulang hasil terbaik
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Pergaulan_Avg', y='Sosial_Ekonomi_Avg', hue='Predicted_Cluster', style='Predicted_Cluster',
        data=test_data, markers=["P", "o"], palette="Set1", s=100)
    plt.scatter(centroids[0][0], centroids[0][1], color='purple', s=300, marker='X', label='Centroid Cluster 0')
    plt.scatter(centroids[1][0], centroids[1][1], color='orange', s=300, marker='X', label='Centroid Cluster 1')
    plt.title(f'{block_name}: Highest Accuracy Clustering Results')
    plt.xlabel('Rata-rata Pergaulan')
    plt.ylabel('Rata-rata Sosial Ekonomi')
    plt.legend()
    plt.tight_layout()

    # Menyimpan plot sebagai gambar
    plot_filename = f"{block_name}_clustering.png"
    plot_filepath = os.path.join(app.config['STATIC_FOLDER'], plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

    # Menampilkan halaman HTML dengan hasil clustering dan akurasi tertinggi
    return render_template('highest_accuracy.html', 
                           filename=filename, 
                           highest_acc=highest_acc, 
                           test_data=test_data, 
                           plot_filepath=plot_filepath)


if __name__ == '__main__':
    app.run(debug=True)
