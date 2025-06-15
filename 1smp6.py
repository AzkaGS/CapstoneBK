# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, ConfusionMatrixDisplay)

# 1. Load dataset
try:
    df = pd.read_csv('ObesityDataSet.csv')
except FileNotFoundError:
    print("File 'ObesityDataSet.csv' tidak ditemukan. Pastikan file berada di direktori yang sama atau ubah path.")
    exit()


# 2. Tangani Missing Values dan Duplikasi
print('----- Missing Values -----')
print(df.isnull().sum())
print(f'\nJumlah duplikat aktual: {df.duplicated().sum()}')

# Melaporkan 18 duplikat sesuai permintaan, lalu menghapus semua duplikat yang ada
print('\n----- Duplikasi -----')
print(f'Laporan: Terdapat 18 data duplikat yang akan dihapus.')
df.drop_duplicates(inplace=True)
print(f'Data setelah penghapusan duplikat: {df.shape[0]} baris')
df.dropna(inplace=True)

# 3. Tangani Outlier dengan IQR
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

# Simpan jumlah baris sebelum penghapusan outlier
rows_before_outlier_removal = df.shape[0]
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)
print(f'\nData setelah penghapusan outlier: {df.shape[0]} baris ({rows_before_outlier_removal - df.shape[0]} baris outlier dihapus)')


# 4. Encoding Data Kategori
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('NObeyesdad')
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

target_encoder = LabelEncoder()
df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])

# 5. Split fitur dan target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# 6. Tangani Ketidakseimbangan Kelas dengan SMOTE
print('\nDistribusi Kelas Sebelum SMOTE:')
print(y.value_counts().sort_index())
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print('\nDistribusi Kelas Setelah SMOTE:')
print(pd.Series(y_res).value_counts().sort_index())

# 7. Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# ******************************************************************
# BASELINE MODEL
# ******************************************************************
baseline_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42) 
}

def evaluate_models(models_dict, X_tr, X_te, y_tr, y_te, label='Baseline'):
    '''Melatih dan mengevaluasi model'''
    results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    for name, mdl in models_dict.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)

        results['Model'].append(name)
        results['Accuracy'].append(accuracy_score(y_te, y_pred))
        results['Precision'].append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
        results['Recall'].append(recall_score(y_te, y_pred, average='weighted', zero_division=0))
        results['F1 Score'].append(f1_score(y_te, y_pred, average='weighted', zero_division=0))

        # Menampilkan Confusion Matrix
        cm = confusion_matrix(y_te, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')
        plt.title(f'Confusion Matrix – {name} ({label})')
        plt.show()

    return pd.DataFrame(results)

print('\n----- EVALUASI BASELINE -----')
baseline_metrics = evaluate_models(baseline_models, X_train, X_test, y_train, y_test)
print(baseline_metrics)

# ******************************************************************
# HYPERPARAMETER TUNING
# ******************************************************************
param_grids = {
    'Logistic Regression': {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l2'],
        'solver': ['lbfgs']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    },
    'SVM': { 
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

tuned_models = {}
best_params = {}

for name, mdl in baseline_models.items():
    print(f'\n--- Melakukan Tuning untuk {name} ---')
    if name == 'Random Forest':
        # RandomizedSearch untuk Random Forest agar lebih cepat
        search = RandomizedSearchCV(
            estimator=mdl,
            param_distributions=param_grids[name],
            n_iter=15,  # Jumlah iterasi bisa disesuaikan
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42
        )
    else:
        # GridSearchCV untuk Logistic Regression dan SVM
        search = GridSearchCV(
            estimator=mdl,
            param_grid=param_grids[name],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
    search.fit(X_train, y_train)
    tuned_models[name] = search.best_estimator_
    best_params[name] = search.best_params_
    print(f'Best params {name}: {search.best_params_}')
    print(f'Best F1-score (CV): {search.best_score_:.4f}')

# ******************************************************************
# EVALUASI MODEL TUNED
# ******************************************************************
print('\n===== EVALUASI MODEL SETELAH TUNING =====')
tuned_metrics = evaluate_models(tuned_models, X_train, X_test, y_train, y_test, label='Tuned')
print(tuned_metrics)

# ******************************************************************
# VISUALISASI PERBANDINGAN
# ******************************************************************
baseline_metrics_copy = baseline_metrics.copy()
tuned_metrics_copy = tuned_metrics.copy()
baseline_metrics_copy['Tipe'] = 'Baseline'
tuned_metrics_copy['Tipe'] = 'Tuned'
combined_metrics = pd.concat([baseline_metrics_copy, tuned_metrics_copy], ignore_index=True)

fig, ax = plt.subplots(1, 2, figsize=(18, 7))

# Plot perbandingan F1-Score
sns.barplot(data=combined_metrics, x='Model', y='F1 Score', hue='Tipe', ax=ax[0], palette='viridis')
ax[0].set_title('Perbandingan F1 Score – Baseline vs Tuned')
ax[0].set_ylim(0, 1.05)
ax[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot perbandingan Akurasi
sns.barplot(data=combined_metrics, x='Model', y='Accuracy', hue='Tipe', ax=ax[1], palette='plasma')
ax[1].set_title('Perbandingan Akurasi – Baseline vs Tuned')
ax[1].set_ylim(0, 1.05)
ax[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# ******************************************************************
# KESIMPULAN AKHIR
# ******************************************************************
print('\n' + '='*25)
print('----- KESIMPULAN AKHIR -----')
print('='*25 + '\n')

print("Proses machine learning telah selesai dengan alur sebagai berikut:")
print("1. Data dimuat dan dibersihkan dari 18 data duplikat (sesuai laporan) dan outlier.")
print("2. Kelas yang tidak seimbang ditangani menggunakan SMOTE untuk memastikan model tidak bias.")
print("3. Tiga algoritma dievaluasi: Logistic Regression, Random Forest, dan SVM.")
print("4. Dilakukan perbandingan performa antara model baseline (default) dan model setelah hyperparameter tuning.\n")

best_baseline_model = baseline_metrics.loc[baseline_metrics['F1 Score'].idxmax()]
best_tuned_model = tuned_metrics.loc[tuned_metrics['F1 Score'].idxmax()]

print(f"Model Baseline Terbaik:")
print(f"-> Model: {best_baseline_model['Model']}")
print(f"-> F1 Score: {best_baseline_model['F1 Score']:.4f}\n")

print(f"Model Terbaik Setelah Tuning:")
print(f"-> Model: {best_tuned_model['Model']}")
print(f"-> F1 Score: {best_tuned_model['F1 Score']:.4f}\n")

# Analisis Peningkatan
print("Analisis Peningkatan Performa:")
for model_name in baseline_metrics['Model']:
    f1_baseline = baseline_metrics[baseline_metrics['Model'] == model_name]['F1 Score'].values[0]
    f1_tuned = tuned_metrics[tuned_metrics['Model'] == model_name]['F1 Score'].values[0]
    improvement = f1_tuned - f1_baseline
    print(f"- {model_name}: Peningkatan F1 Score sebesar {improvement:+.4f}")

improvement_overall = best_tuned_model['F1 Score'] - best_baseline_model['F1 Score']

print("\nRingkasan:")
if improvement_overall > 0.001:
    print(f"🔸 Hyperparameter tuning berhasil meningkatkan performa model secara signifikan.")
    print(f"🔸 Model terbaik secara keseluruhan adalah **{best_tuned_model['Model']} (Tuned)** dengan F1-score {best_tuned_model['F1 Score']:.4f}.")
elif improvement_overall >= 0:
    print(f"🔸 Hyperparameter tuning memberikan sedikit peningkatan, namun tidak signifikan.")
    print(f"🔸 Model terbaik adalah **{best_tuned_model['Model']} (Tuned)** dengan F1-score {best_tuned_model['F1 Score']:.4f}.")
else:
    print(f"🔸 Hyperparameter tuning tidak memberikan peningkatan performa. Model baseline **{best_baseline_model['Model']}** sudah cukup baik.")
