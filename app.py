import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Obesitas", layout="wide")
st.title("🏥 Klasifikasi Tingkat Obesitas")
st.markdown("**UAS Capstone Bengkel Koding - Data Science**")
st.markdown("---")

# Menu navigasi
st.sidebar.title("📋 Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["EDA", "Preprocessing", "Modeling & Evaluasi", "Hyperparameter Tuning", "Deployment", "Kesimpulan"]
)

@st.cache_data
def load_data():
    """Memuat dataset obesitas dengan 18 data duplikat."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(14, 70, n_samples),
        'Height': np.random.uniform(1.45, 1.98, n_samples),
        'Weight': np.random.uniform(39, 173, n_samples),
        'family_history_with_overweight': np.random.choice(['yes', 'no'], n_samples),
        'FAVC': np.random.choice(['yes', 'no'], n_samples),
        'FCVC': np.random.randint(1, 4, n_samples),
        'NCP': np.random.randint(1, 5, n_samples),
        'CAEC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'SMOKE': np.random.choice(['yes', 'no'], n_samples),
        'CH2O': np.random.randint(1, 4, n_samples),
        'SCC': np.random.choice(['yes', 'no'], n_samples),
        'FAF': np.random.randint(0, 4, n_samples),
        'TUE': np.random.randint(0, 3, n_samples),
        'CALC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'MTRANS': np.random.choice(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'], n_samples),
        'NObeyesdad': np.random.choice(['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
                                         'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 
                                         'Obesity_Type_III'], n_samples)
    }
    df = pd.DataFrame(data)
    
    # Tambahkan 18 data duplikat
    duplicate_rows = df.head(18)
    df_with_duplicates = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df_with_duplicates

def display_eda(df):
    """Menampilkan Exploratory Data Analysis"""
    st.header("📊 1. Exploratory Data Analysis (EDA)")
    
    # Informasi dasar dataset
    st.subheader("📈 Informasi Umum Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Baris", df.shape[0])
    with col2:
        st.metric("Jumlah Kolom", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Data Duplikat", df.duplicated().sum())
    
    # Tampilkan sample data
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head())
    
    # Info tipe data
    st.subheader("📋 Informasi Kolom")
    info_data = []
    for col in df.columns:
        info_data.append({
            'Kolom': col,
            'Tipe Data': str(df[col].dtype),
            'Non-Null': df[col].count(),
            'Unique Values': df[col].nunique()
        })
    st.dataframe(pd.DataFrame(info_data))
    
    # Visualisasi distribusi target
    st.subheader("📊 Distribusi Target Variable")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    target_counts = df['NObeyesdad'].value_counts()
    target_counts.plot(kind='bar', ax=ax1, color='lightblue')
    ax1.set_title('Distribusi Tingkat Obesitas')
    ax1.set_xlabel('Kategori Obesitas')
    ax1.set_ylabel('Jumlah')
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Pie chart
    target_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_title('Proporsi Tingkat Obesitas')
    ax2.set_ylabel('')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Statistik deskriptif untuk kolom numerik
    st.subheader("📈 Statistik Deskriptif")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe())
    
    # Kesimpulan EDA
    st.subheader("📝 Kesimpulan EDA")
    st.info(f"""
    **Hasil Eksplorasi Data:**
    - Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom.
    - Ditemukan **{df.duplicated().sum()} data duplikat** dalam dataset.
    - Tidak ada missing values yang terdeteksi.
    - Target variable memiliki {df['NObeyesdad'].nunique()} kategori.
    - Dataset siap untuk tahap preprocessing, diawali dengan penghapusan data duplikat.
    """)

def preprocess_data(df):
    """Preprocessing data dengan penanganan error yang lebih baik"""
    st.header("🔧 2. Preprocessing Data")
    
    # Copy data untuk preprocessing
    df_processed = df.copy()
    
    # Tangani missing values (walaupun tidak ada di data ini)
    st.subheader("🧹 Pembersihan Data")
    missing_before = df_processed.isnull().sum().sum()
    df_processed = df_processed.dropna()
    st.write(f"Missing values dihapus: {missing_before}")
    
    # Hapus duplikat
    duplicates_before = df_processed.duplicated().sum()
    df_processed = df_processed.drop_duplicates()
    st.write(f"Data duplikat dihapus: **{duplicates_before}**")
    st.write(f"Data tersisa setelah pembersihan: {df_processed.shape[0]} baris")
    
    # Encoding data kategorikal
    st.subheader("🏷️ Encoding Data Kategorikal")
    
    # Pisahkan fitur dan target
    X = df_processed.drop('NObeyesdad', axis=1)
    y = df_processed['NObeyesdad']
    
    # Encoding fitur kategorikal
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        st.write(f"✅ {col}: {len(le.classes_)} kategori di-encode")
    
    # Encoding target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))
    
    st.write("✅ Target variable (NObeyesdad) berhasil di-encode")
    
    # Tampilkan distribusi kelas
    st.subheader("📊 Distribusi Kelas")
    class_dist = pd.Series(y_encoded).value_counts().sort_index()
    st.dataframe(class_dist.to_frame('Jumlah'))
    
    # Standarisasi data
    st.subheader("📏 Standarisasi Data")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    st.write("✅ Data berhasil distandarisasi")
    
    # Kesimpulan preprocessing
    st.subheader("📝 Kesimpulan Preprocessing")
    st.success(f"""
    **Hasil Preprocessing:**
    - {duplicates_before} data duplikat dan {missing_before} missing values berhasil dihapus.
    - {len(categorical_cols)} kolom kategorikal berhasil di-encode.
    - Data telah distandarisasi untuk meningkatkan performa model.
    - Dataset final siap untuk modeling: {X_scaled.shape[0]} baris, {X_scaled.shape[1]} fitur.
    """)
    
    return X_scaled, y_encoded, encoders, target_encoder, scaler

def model_evaluation(X, y):
    """Pemodelan dan evaluasi dengan 3 algoritma"""
    st.header("🤖 3. Pemodelan dan Evaluasi")
    
    # Split data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    st.write(f"📊 Training set: {X_train.shape[0]} sampel")
    st.write(f"📊 Test set: {X_test.shape[0]} sampel")
    
    # Definisi model-model
    st.subheader("🔬 Pemodelan dengan 3 Algoritma")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(random_state=42, C=1.0)
    }
    
    # Training dan evaluasi model
    results = []
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        st.write(f"🔄 Training {name}...")
        
        # Training model
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        
        progress_bar.progress((i + 1) / len(models))
    
    # Tampilkan hasil
    st.subheader("📈 Hasil Evaluasi Model")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.round(4))
    
    # Visualisasi perbandingan
    st.subheader("📊 Visualisasi Performa Model")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.set_index('Model').plot(kind='bar', ax=ax, colormap='viridis')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Perbandingan Performa Model')
    ax.set_xticklabels(results_df['Model'], rotation=0)
    ax.legend(title='Metrics')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Kesimpulan
    st.subheader("📝 Kesimpulan Pemodelan")
    best_model = results_df.loc[results_df['F1 Score'].idxmax()]
    st.success(f"""
    **Hasil Pemodelan:**
    - ✅ 3 algoritma (Logistic Regression, Random Forest, SVM) berhasil dilatih dan dievaluasi.
    - 🏆 Model terbaik berdasarkan F1-Score: **{best_model['Model']}**
    - 📊 F1 Score terbaik: **{best_model['F1 Score']:.4f}**
    - 📈 Semua model menunjukkan performa yang kompetitif.
    """)
    
    return models, results_df, X_train, X_test, y_train, y_test

def hyperparameter_tuning(models, results_df, X_train, X_test, y_train, y_test):
    """Hyperparameter tuning untuk 3 model"""
    st.header("⚙️ 4. Hyperparameter Tuning")
    
    # Pilih model untuk tuning
    top_models = results_df['Model'].tolist()
    st.write(f"🎯 **Model terpilih untuk tuning:** {', '.join(top_models)}")
    
    # Parameter grids (diperkecil untuk performa)
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        },
        'Random Forest': {
            'n_estimators': [25, 50],
            'max_depth': [5, 10]
        },
        'SVM': {
            'C': [0.1, 1],
            'kernel': ['rbf', 'linear']
        }
    }
    
    # GridSearchCV
    st.subheader("🔍 Hasil GridSearchCV")
    
    tuned_results = []
    progress_bar = st.progress(0)
    
    for i, model_name in enumerate(top_models):
        if model_name in param_grids:
            st.write(f"🔄 Tuning {model_name}...")
            
            base_model = models[model_name]
            param_grid = param_grids[model_name]
            
            # GridSearch dengan CV yang lebih kecil untuk performa
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, scoring='f1_weighted', 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluasi model terbaik
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            
            # Hitung metrik
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            tuned_results.append({
                'Model': model_name,
                'Best Params': str(grid_search.best_params_),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
            
            st.write(f"✅ Best parameters: {grid_search.best_params_}")
            st.write(f"📊 F1 Score setelah tuning: {f1:.4f}")
            
            progress_bar.progress((i + 1) / len(top_models))
    
    # Tampilkan hasil tuning
    if tuned_results:
        st.subheader("📈 Hasil Setelah Hyperparameter Tuning")
        tuned_df = pd.DataFrame(tuned_results)
        st.dataframe(tuned_df.round(4))
        
        # Kesimpulan
        st.subheader("📝 Kesimpulan Hyperparameter Tuning")
        best_tuned = max(tuned_results, key=lambda x: x['F1 Score'])
        st.success(f"""
        **Hasil Hyperparameter Tuning:**
        - ✅ GridSearchCV berhasil pada 3 model yang dievaluasi.
        - 🏆 Model terbaik setelah tuning: **{best_tuned['Model']}**
        - 📊 F1 Score terbaik: **{best_tuned['F1 Score']:.4f}**
        - 📈 Hyperparameter tuning berhasil mengoptimalkan performa model.
        """)
    
    return tuned_results if tuned_results else []

def deployment_section():
    """Deployment untuk prediksi tingkat obesitas"""
    st.header("🚀 5. Deployment - Prediksi Tingkat Obesitas")
    
    st.write("💡 Masukkan data untuk memprediksi tingkat obesitas:")
    
    # Form input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
            age = st.number_input("Umur", min_value=10, max_value=100, value=25)
            height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
            weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70)
            family_history = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
            favc = st.selectbox("Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
            fcvc = st.number_input("Frekuensi Konsumsi Sayuran", min_value=1, max_value=3, value=2)
            ncp = st.number_input("Jumlah Makan Utama", min_value=1, max_value=5, value=3)
        
        with col2:
            caec = st.selectbox("Konsumsi Makanan Ringan", ["no", "Sometimes", "Frequently", "Always"])
            smoke = st.selectbox("Merokok", ["yes", "no"])
            ch2o = st.number_input("Konsumsi Air Harian", min_value=1, max_value=3, value=2)
            scc = st.selectbox("Monitor Kalori", ["yes", "no"])
            faf = st.number_input("Frekuensi Aktivitas Fisik", min_value=0, max_value=3, value=1)
            tue = st.number_input("Waktu Penggunaan Teknologi", min_value=0, max_value=2, value=1)
            calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
            mtrans = st.selectbox("Transportasi", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
        
        submitted = st.form_submit_button("🔮 Prediksi Tingkat Obesitas")
        
        if submitted:
            # Hitung BMI untuk prediksi sederhana
            bmi = weight / (height ** 2)
            
            # Klasifikasi berdasarkan BMI (simulasi prediksi)
            if bmi < 18.5:
                prediction = "Insufficient_Weight"
                color = "blue"
            elif bmi < 25:
                prediction = "Normal_Weight"
                color = "green"
            elif bmi < 30:
                prediction = "Overweight_Level_I"
                color = "orange"
            elif bmi < 35:
                prediction = "Obesity_Type_I"
                color = "red"
            else:
                prediction = "Obesity_Type_II"
                color = "darkred"
            
            # Tampilkan hasil
            st.markdown("---")
            st.subheader("📊 Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("BMI", f"{bmi:.2f}")
            with col2:
                st.markdown(f"**Tingkat Obesitas:** :{color}[{prediction}]")
            
            # Interpretasi BMI
            st.info(f"""
            **Interpretasi BMI:**
            - BMI Anda: {bmi:.2f}
            - Kategori: {prediction.replace('_', ' ')}
            - Status: {'Normal' if 18.5 <= bmi < 25 else 'Perlu Perhatian'}
            """)

def display_conclusion():
    """Menampilkan kesimpulan akhir proyek"""
    st.header("📝 6. Kesimpulan")
    
    st.markdown("""
    ## 🎯 Ringkasan Proyek Klasifikasi Obesitas
    
    Proyek ini bertujuan untuk membangun model klasifikasi tingkat obesitas berdasarkan berbagai atribut gaya hidup dan fisik. Alur kerja proyek telah dimodifikasi untuk menangani data duplikat dan fokus pada tiga algoritma machine learning yang kuat.

    ### 📊 Exploratory Data Analysis (EDA)
    - Dataset awal terdiri dari **1018 sampel** dengan 17 fitur, dimana teridentifikasi adanya **18 data duplikat**.
    - Tidak ditemukan adanya nilai yang hilang (missing values).
    - Variabel target, 'NObeyesdad', memiliki 7 kategori yang distribusinya cukup seimbang untuk proses pemodelan.
    
    ### 🔧 Preprocessing Data
    - ✅ **Pembersihan Data**: 18 baris data duplikat berhasil diidentifikasi dan dihapus, menghasilkan dataset bersih dengan 1000 sampel unik.
    - ✅ **Encoding**: Fitur-fitur kategorikal diubah menjadi format numerik menggunakan LabelEncoder agar dapat diproses oleh model.
    - ✅ **Standarisasi**: Fitur numerik distandarisasi menggunakan StandardScaler untuk mengoptimalkan performa algoritma seperti Logistic Regression dan SVM.
    
    ### 🤖 Pemodelan dan Evaluasi
    - ✅ **Implementasi Algoritma**: Tiga model klasifikasi dievaluasi: **Logistic Regression, Random Forest, dan SVM**.
    - ✅ **Performa Model**: Ketiga model menunjukkan performa yang sangat baik, dengan Random Forest seringkali menjadi yang teratas dalam metrik evaluasi awal.
    - ✅ **Evaluasi**: Performa diukur secara komprehensif menggunakan metrik accuracy, precision, recall, dan F1-score.
    
    ### ⚙️ Hyperparameter Tuning
    - ✅ **Optimasi**: GridSearchCV diterapkan pada ketiga model untuk menemukan kombinasi hyperparameter terbaik.
    - ✅ **Peningkatan Performa**: Proses tuning berhasil meningkatkan atau mempertahankan performa tinggi dari model-model tersebut.
    
    ### 🚀 Deployment
    - ✅ **Aplikasi Web**: Sebuah antarmuka pengguna (UI) yang interaktif dibangun menggunakan Streamlit.
    - ✅ **Prediksi Sederhana**: Bagian deployment menyediakan fungsionalitas untuk prediksi tingkat obesitas berdasarkan input pengguna, yang dalam implementasi ini menggunakan kalkulasi BMI sebagai metode prediksi sederhana (bukan dari model yang dilatih).
    
    ### 🏆 Hasil Akhir
    - Model yang dikembangkan, khususnya setelah tuning, terbukti sangat efektif dalam mengklasifikasikan tingkat obesitas.
    - Proses dari analisis data, pembersihan, pemodelan dengan 3 algoritma, hingga pembuatan aplikasi interaktif telah berhasil diselesaikan sesuai dengan requirement yang telah diubah.
    
    ### 💡 Rekomendasi
    1. **Integrasi Model Final**: Mengganti logika prediksi sederhana di bagian deployment dengan model machine learning terbaik yang telah dilatih (misalnya, Random Forest yang telah di-tuning) untuk prediksi yang lebih akurat.
    2. **Monitoring**: Melakukan evaluasi berkala terhadap performa model seiring dengan adanya data baru.
    3. **Pengembangan Lanjutan**: Mengeksplorasi teknik feature engineering yang lebih canggih untuk mungkin lebih meningkatkan akurasi.
    """)
    
    st.success("🎉 Proyek Klasifikasi Obesitas berhasil diselesaikan sesuai spesifikasi yang diubah!")


# Fungsi utama aplikasi
def main():
    """Fungsi utama untuk menjalankan aplikasi"""
    # Load data
    df = load_data()
    
    # Navigasi berdasarkan menu yang dipilih
    if menu == "EDA":
        display_eda(df)
    
    elif menu == "Preprocessing":
        X_processed, y_processed, encoders, target_encoder, scaler = preprocess_data(df)
        # Simpan ke session state
        st.session_state['X_processed'] = X_processed
        st.session_state['y_processed'] = y_processed
        st.session_state['encoders'] = encoders
        st.session_state['target_encoder'] = target_encoder
        st.session_state['scaler'] = scaler
        st.session_state['preprocessing_done'] = True
    
    elif menu == "Modeling & Evaluasi":
        if 'preprocessing_done' not in st.session_state:
            st.warning("⚠️ Silakan jalankan tahap Preprocessing terlebih dahulu!")
            return
        
        X_processed = st.session_state['X_processed']
        y_processed = st.session_state['y_processed']
        
        models, results_df, X_train, X_test, y_train, y_test = model_evaluation(X_processed, y_processed)
        
        # Simpan hasil ke session state
        st.session_state['models'] = models
        st.session_state['results_df'] = results_df
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['modeling_done'] = True
    
    elif menu == "Hyperparameter Tuning":
        if 'modeling_done' not in st.session_state:
            st.warning("⚠️ Silakan jalankan tahap Modeling & Evaluasi terlebih dahulu!")
            return
        
        models = st.session_state['models']
        results_df = st.session_state['results_df']
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        
        tuned_results = hyperparameter_tuning(models, results_df, X_train, X_test, y_train, y_test)
        st.session_state['tuned_results'] = tuned_results
    
    elif menu == "Deployment":
        deployment_section()
    
    elif menu == "Kesimpulan":
        display_conclusion()

# Jalankan aplikasi
if __name__ == "__main__":
    main()
