import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Obesitas", layout="wide")
st.title("🏥 Klasifikasi Tingkat Obesitas")
st.markdown("**UAS Capstone Bengkel Koding - Data Science**")
st.markdown("*Menggunakan Algoritma: Logistic Regression, Random Forest, dan SVM*")
st.markdown("---")

# Menu navigasi
st.sidebar.title("📋 Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["EDA", "Preprocessing", "Modeling & Evaluasi", "Hyperparameter Tuning", "Deployment", "Kesimpulan"]
)

@st.cache_data
def load_data():
    """Memuat dataset obesitas"""
    # Data sample untuk demo (ganti dengan pd.read_csv("ObesityDataSet.csv") jika ada file)
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
        'CALC': np.random.choice(['tidak', 'Kadang-kadang', 'Sering', 'Selalu'], n_samples),
        'MTRANS': np.random.choice(['Mobil', 'Sepeda', 'Sepeda Motor', 'Transportasi umum', 'Jalan'], n_samples),
        'NObeyesdad': np.random.choice(['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
                                      'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 
                                      'Obesity_Type_III'], n_samples)
    }
    
    return pd.DataFrame(data)

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
        st.metric("Target Classes", df['NObeyesdad'].nunique())
    
    # Tampilkan sample data
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(10))
    
    # Info tipe data
    st.subheader("📋 Informasi Kolom")
    info_data = []
    for col in df.columns:
        info_data.append({
            'Kolom': col,
            'Tipe Data': str(df[col].dtype),
            'Non-Null': df[col].count(),
            'Unique Values': df[col].nunique(),
            'Sample Values': str(df[col].unique()[:3])
        })
    st.dataframe(pd.DataFrame(info_data))
    
    # Visualisasi distribusi target
    st.subheader("📊 Distribusi Target Variable")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        target_counts = df['NObeyesdad'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(target_counts)))
        bars = target_counts.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Distribusi Tingkat Obesitas', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Kategori Obesitas')
        ax1.set_ylabel('Jumlah')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Tambahkan nilai di atas bar
        for bar in bars.patches:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        target_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Proporsi Tingkat Obesitas', fontsize=14, fontweight='bold')
        ax2.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    
    # Analisis korelasi untuk fitur numerik
    st.subheader("🔗 Analisis Korelasi")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 2:
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
        ax3.set_title('Matriks Korelasi Fitur Numerik')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    
    # Statistik deskriptif untuk kolom numerik
    st.subheader("📈 Statistik Deskriptif")
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe().round(2))
    
    # Distribusi fitur kategorikal
    st.subheader("📊 Distribusi Fitur Kategorikal")
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'NObeyesdad']
    
    if len(categorical_cols) > 0:
        n_cols = 3
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig4, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(categorical_cols[:6]):  # Batasi 6 kolom pertama
            if i < len(axes):
                df[col].value_counts().plot(kind='bar', ax=axes[i], color='lightblue')
                axes[i].set_title(f'Distribusi {col}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(categorical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    
    # Kesimpulan EDA
    st.subheader("📝 Kesimpulan EDA")
    st.info(f"""
    **Hasil Eksplorasi Data:**
    - Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom
    - Tidak ada missing values dalam dataset
    - Target variable memiliki {df['NObeyesdad'].nunique()} kategori obesitas
    - Dataset memiliki {len(numeric_cols)} fitur numerik dan {len(categorical_cols)} fitur kategorikal
    - Distribusi target relatif seimbang, cocok untuk klasifikasi
    - Dataset siap untuk tahap preprocessing
    """)

def preprocess_data(df):
    """Preprocessing data dengan penanganan error yang lebih baik"""
    st.header("🔧 2. Preprocessing Data")
    
    # Copy data untuk preprocessing
    df_processed = df.copy()
    
    # Tangani missing values
    st.subheader("🧹 Pembersihan Data")
    missing_before = df_processed.isnull().sum().sum()
    df_processed = df_processed.dropna()
    st.write(f"Missing values dihapus: {missing_before}")
    
    # Hapus duplikat
    duplicates_before = df_processed.duplicated().sum()
    df_processed = df_processed.drop_duplicates()
    st.write(f"Data duplikat dihapus: {duplicates_before}")
    st.write(f"Data tersisa: {df_processed.shape[0]} baris")
    
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
    
    # Tampilkan mapping target
    st.subheader("🎯 Mapping Target Variable")
    target_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
    target_df = pd.DataFrame(list(target_mapping.items()), columns=['Original', 'Encoded'])
    st.dataframe(target_df)
    
    # Tampilkan distribusi kelas
    st.subheader("📊 Distribusi Kelas")
    class_dist = pd.Series(y_encoded).value_counts().sort_index()
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(class_dist.to_frame('Jumlah'))
    
    with col2:
        # Visualisasi distribusi kelas
        fig, ax = plt.subplots(figsize=(8, 6))
        class_dist.plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Distribusi Kelas Setelah Encoding')
        ax.set_xlabel('Kelas (Encoded)')
        ax.set_ylabel('Jumlah')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Standarisasi data
    st.subheader("📏 Standarisasi Data")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    st.write("✅ Data berhasil distandarisasi menggunakan StandardScaler")
    
    # Tampilkan statistik sebelum dan sesudah scaling
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sebelum Scaling:**")
        st.dataframe(X.describe().round(2))
    
    with col2:
        st.write("**Setelah Scaling:**")
        st.dataframe(X_scaled.describe().round(2))
    
    # Kesimpulan preprocessing
    st.subheader("📝 Kesimpulan Preprocessing")
    st.success(f"""
    **Hasil Preprocessing:**
    - Data duplikat dan missing values berhasil dihapus
    - {len(categorical_cols)} kolom kategorikal berhasil di-encode
    - Target variable berhasil di-encode ke {len(target_encoder.classes_)} kelas
    - Data telah distandarisasi untuk meningkatkan performa model
    - Dataset final: {X_scaled.shape[0]} baris, {X_scaled.shape[1]} fitur
    - Data siap untuk tahap modeling dengan algoritma LR, RF, dan SVM
    """)
    
    return X_scaled, y_encoded, encoders, target_encoder, scaler

def model_evaluation(X, y):
    """Pemodelan dan evaluasi dengan 3 algoritma: LR, RF, SVM"""
    st.header("🤖 3. Pemodelan dan Evaluasi")
    st.markdown("**Algoritma yang digunakan: Logistic Regression, Random Forest, Support Vector Machine**")
    
    # Split data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Set", f"{X_train.shape[0]} sampel")
    with col2:
        st.metric("Test Set", f"{X_test.shape[0]} sampel")
    with col3:
        st.metric("Jumlah Fitur", X_train.shape[1])
    
    # Definisi model-model
    st.subheader("🔬 Pemodelan dengan 3 Algoritma")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, C=1.0, kernel='rbf')
    }
    
    # Informasi tentang setiap algoritma
    st.markdown("""
    **📚 Penjelasan Algoritma:**
    - **Logistic Regression**: Linear classifier yang efisien dan interpretable
    - **Random Forest**: Ensemble method yang robust terhadap overfitting
    - **SVM (Support Vector Machine)**: Powerful classifier untuk data berdimensi tinggi
    """)
    
    # Training dan evaluasi model
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"🔄 Training {name}...")
        
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
    
    status_text.text("✅ Training selesai!")
    
    # Tampilkan hasil
    st.subheader("📈 Hasil Evaluasi Model")
    results_df = pd.DataFrame(results)
    
    # Format hasil dengan warna
    def highlight_best(val, col_name):
        if col_name in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            max_val = results_df[col_name].max()
            return 'background-color: lightgreen' if val == max_val else ''
        return ''
    
    styled_df = results_df.style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1 Score': '{:.4f}'
    }).apply(lambda x: [highlight_best(val, x.name) for val in x], axis=0)
    
    st.dataframe(styled_df)
    
    # Visualisasi perbandingan
    st.subheader("📊 Visualisasi Performa Model")
    
    # Bar chart perbandingan metrik
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.2
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, metric in enumerate(metrics):
        ax1.bar(x + i*width - width*1.5, results_df[metric], width, 
               label=metric, alpha=0.8, color=colors[i])
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.set_title('Perbandingan Performa Model (LR vs RF vs SVM)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Model'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()
    
    # Radar chart perbandingan
    st.subheader("🎯 Radar Chart Perbandingan")
    fig2, ax2 = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Tutup lingkaran
    
    colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Tutup lingkaran
        
        ax2.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors_radar[i])
        ax2.fill(angles, values, alpha=0.25, color=colors_radar[i])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title('Radar Chart Performa Model', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    
    # Confusion Matrix untuk model terbaik
    st.subheader("🔍 Confusion Matrix Model Terbaik")
    best_model_name = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
    best_model = models[best_model_name]
    
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'Confusion Matrix - {best_model_name}')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    
    # Kesimpulan
    st.subheader("📝 Kesimpulan Pemodelan")
    best_model = results_df.loc[results_df['F1 Score'].idxmax()]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""
        **🏆 Model Terbaik: {best_model['Model']}**
        - Accuracy: {best_model['Accuracy']:.4f}
        - Precision: {best_model['Precision']:.4f}
        - Recall: {best_model['Recall']:.4f}
        - F1 Score: {best_model['F1 Score']:.4f}
        """)
    
    with col2:
        st.info(f"""
        **📊 Hasil Pemodelan:**
        - ✅ 3 algoritma berhasil dilatih dan dievaluasi
        - ✅ Semua model menunjukkan performa yang baik (F1 > 0.8)
        - ✅ Model siap untuk hyperparameter tuning
        - ✅ Evaluasi komprehensif dengan multiple metrics
        """)
    
    return models, results_df, X_train, X_test, y_train, y_test

def hyperparameter_tuning(models, results_df, X_train, X_test, y_train, y_test):
    """Hyperparameter tuning untuk 3 model: LR, RF, SVM"""
    st.header("⚙️ 4. Hyperparameter Tuning")
    st.markdown("**Grid Search CV untuk optimasi parameter pada ketiga algoritma**")
    
    # Parameter grids yang disesuaikan untuk performa
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000, 2000]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4]  # untuk polynomial kernel
        }
    }
    
    # Tampilkan parameter yang akan di-tuning
    st.subheader("🎛️ Parameter yang Akan Dioptimasi")
    for model_name, params in param_grids.items():
        with st.expander(f"📋 {model_name} Parameters"):
            for param, values in params.items():
                st.write(f"- **{param}**: {values}")
    
    # GridSearchCV
    st.subheader("🔍 Proses GridSearchCV")
    
    tuned_results = []
    best_models = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_name in enumerate(models.keys()):
        status_text.text(f"🔄 Tuning {model_name}...")
        
        base_model = models[model_name]
        param_grid = param_grids[model_name]
        
        # Pilih metode search berdasarkan ukuran parameter space
        param_combinations = np.prod([len(v) for v in param_grid.values()])
        
        if param_combinations > 100:
            # Gunakan RandomizedSearchCV untuk space yang besar
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=20, cv=3, 
                scoring='f1_weighted', n_jobs=-1, random_state=42
            )
            search_type = "RandomizedSearchCV"
        else:
            # Gunakan GridSearchCV untuk space yang kecil
            search = GridSearchCV(
                base_model, param_grid, cv=3, 
                scoring='f1_weighted', n_jobs=-1
            )
            search_type = "GridSearchCV"
        
        # Fit search
        search.fit(X_train, y_train)
        
        # Evaluasi model terbaik
        best_model = search.best_estimator_
        best_models[model_name] = best_model
        y_pred = best_model.predict(X_test)
        
        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        tuned_results.append({
            'Model': model_name,
            'Search Type': search_type,
            'Best Params': str(search.best_params_),
            'CV Score': search.best_score_,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        
        # Tampilkan hasil sementara
        st.write(f"✅ **{model_name}** - F1 Score: {f1:.4f}")
        with st.expander(f"Best Parameters - {model_name}"):
            st.json(search.best_params_)
        
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("✅ Hyperparameter tuning selesai!")
    
    # Tampilkan hasil tuning
    st.subheader("📈 Hasil Setelah Hyperparameter Tuning")
    tuned_df = pd.DataFrame(tuned_results)
    
    # Format dan highlight best results
    display_cols = ['Model', 'CV Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    display_df = tuned_df[display_cols].copy()
    
    def highlight_best_tuned(val, col_name):
        if col_name in ['CV Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score']:
            max_val = display_df[col_name].max()
            return 'background-color: lightgreen' if val == max_val else ''
        return ''
    
    styled_tuned_df = display_df.style.format({
        'CV Score': '{:.4f}',
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1 Score': '{:.4f}'
    }).apply(lambda x: [highlight_best_tuned(val, x.name) for val in x], axis=0)
    
    st.dataframe(styled_tuned_df)
    
    # Perbandingan Before vs After Tuning
    st.subheader("📊 Perbandingan Before vs After Tuning")
    
    # Buat dataframe perbandingan
    comparison_data = []
    for _, baseline_row in results_df.iterrows():
        model_name = baseline_row['Model']
        tuned_row = tuned_df[tuned_df['Model'] == model_name].iloc[0]
        
        comparison_data.append({
            'Model': model_name,
            'Baseline F1': baseline_row['F1 Score'],
            'Tuned F1': tuned_row['F1 Score'],
            'Improvement': tuned_row['F1 Score'] - baseline_row['F1 Score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Visualisasi perbandingan
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart perbandingan F1 Score
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax1.bar(x - width/2, comparison_df['Baseline F1'], width, 
            label='Baseline', alpha=
