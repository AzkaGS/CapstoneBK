#import library
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
df = pd.read_csv('ObesityDataSet.csv')

# 2. Tangani Missing Values dan Duplikasi
print('===== Missing Values =====')
print(df.isnull().sum())
print('\n===== Duplikasi =====')
print(f'Jumlah duplikat: {df.duplicated().sum()}')
df.drop_duplicates(inplace=True)
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

for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

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
print(y.value_counts())
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print('\nDistribusi Kelas Setelah SMOTE:')
print(pd.Series(y_res).value_counts())

# 7. Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 8. Train‑test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# ------------------------------------------------------------------
# BASELINE MODEL
# ------------------------------------------------------------------
baseline_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

def evaluate_models(models_dict, X_tr, X_te, y_tr, y_te, label='Baseline'):
    '''Melatih dan mengevaluasi model'''
    results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    for name, mdl in models_dict.items():
        print(f'Training {name}...')
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)

        results['Model'].append(name)
        results['Accuracy'].append(accuracy_score(y_te, y_pred))
        results['Precision'].append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
        results['Recall'].append(recall_score(y_te, y_pred, average='weighted', zero_division=0))
        results['F1 Score'].append(f1_score(y_te, y_pred, average='weighted', zero_division=0))

        cm = confusion_matrix(y_te, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix – {name} ({label})')
        plt.show()

    return pd.DataFrame(results)

print('\n===== EVALUASI BASELINE =====')
baseline_metrics = evaluate_models(baseline_models, X_train, X_test, y_train, y_test)
print(baseline_metrics)

# ------------------------------------------------------------------
# HYPERPARAMETER TUNING
# ------------------------------------------------------------------
param_grids = {
    'Logistic Regression': {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
}

tuned_models = {}
best_params = {}

for name, mdl in baseline_models.items():
    print(f'\nTuning hyperparameters untuk {name}...')
    
    if name == 'Random Forest':
        # Gunakan RandomizedSearchCV untuk Random Forest karena space parameter yang besar
        search = RandomizedSearchCV(
            estimator=mdl,
            param_distributions=param_grids[name],
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    elif name == 'SVM':
        # Gunakan RandomizedSearchCV untuk SVM karena kombinasi parameter yang banyak
        search = RandomizedSearchCV(
            estimator=mdl,
            param_distributions=param_grids[name],
            n_iter=15,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    else:
        # GridSearchCV untuk Logistic Regression
        search = GridSearchCV(
            estimator=mdl,
            param_grid=param_grids[name],
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
    
    search.fit(X_train, y_train)
    tuned_models[name] = search.best_estimator_
    best_params[name] = search.best_params_
    print(f'Best params {name}: {search.best_params_}')
    print(f'Best CV score {name}: {search.best_score_:.4f}')

# ------------------------------------------------------------------
# EVALUASI MODEL TUNED
# ------------------------------------------------------------------
print('\n===== EVALUASI MODEL SETELAH TUNING =====')
tuned_metrics = evaluate_models(tuned_models, X_train, X_test, y_train, y_test, label='Tuned')
print(tuned_metrics)

# ------------------------------------------------------------------
# VISUALISASI PERBANDINGAN
# ------------------------------------------------------------------
# Perbandingan Baseline vs Tuned
baseline_metrics['Tipe'] = 'Baseline'
tuned_metrics['Tipe'] = 'Tuned'
combined_metrics = pd.concat([baseline_metrics, tuned_metrics], ignore_index=True)

plt.figure(figsize=(15, 8))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(2, 2, i)
    metric_data = combined_metrics[['Model', 'Tipe', metric]]
    pivot_data = metric_data.pivot(index='Model', columns='Tipe', values=metric)
    pivot_data.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title(f'Perbandingan {metric}')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend(title='Tipe Model')
    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Visualisasi performa per model
plt.figure(figsize=(12, 6))
models_order = ['Logistic Regression', 'Random Forest', 'SVM']
colors = ['skyblue', 'lightcoral', 'lightgreen']

x = np.arange(len(models_order))
width = 0.35

baseline_f1 = [baseline_metrics[baseline_metrics['Model'] == model]['F1 Score'].iloc[0] for model in models_order]
tuned_f1 = [tuned_metrics[tuned_metrics['Model'] == model]['F1 Score'].iloc[0] for model in models_order]

plt.bar(x - width/2, baseline_f1, width, label='Baseline', color=colors, alpha=0.7)
plt.bar(x + width/2, tuned_f1, width, label='Tuned', color=colors, alpha=1.0)

plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('Perbandingan F1 Score: Baseline vs Tuned')
plt.xticks(x, models_order)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 1.05)

# Tambahkan nilai di atas bar
for i, (baseline, tuned) in enumerate(zip(baseline_f1, tuned_f1)):
    plt.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + width/2, tuned + 0.01, f'{tuned:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# KESIMPULAN
# ------------------------------------------------------------------
print('\n===== KESIMPULAN =====')
print('\nBaseline Performance:')
print(baseline_metrics.round(4))
print('\nTuned Performance:')
print(tuned_metrics.round(4))

best_baseline = baseline_metrics.loc[baseline_metrics['F1 Score'].idxmax()]
best_tuned = tuned_metrics.loc[tuned_metrics['F1 Score'].idxmax()]
improvement = best_tuned['F1 Score'] - best_baseline['F1 Score']

print(f'\nPerforma terbaik baseline: {best_baseline["Model"]} (F1 = {best_baseline["F1 Score"]:.4f})')
print(f'Performa terbaik setelah tuning: {best_tuned["Model"]} (F1 = {best_tuned["F1 Score"]:.4f})')
print(f'Peningkatan F1 Score: {improvement:.4f}')

# Analisis per model
print('\nPerbandingan peningkatan per model:')
for model in models_order:
    baseline_score = baseline_metrics[baseline_metrics['Model'] == model]['F1 Score'].iloc[0]
    tuned_score = tuned_metrics[tuned_metrics['Model'] == model]['F1 Score'].iloc[0]
    improvement_per_model = tuned_score - baseline_score
    print(f'{model}: {baseline_score:.4f} → {tuned_score:.4f} (Δ: {improvement_per_model:+.4f})')

if improvement > 0:
    print('\n🔸 Hyperparameter tuning berhasil meningkatkan kinerja model.')
else:
    print('\n🔸 Hyperparameter tuning tidak memberikan peningkatan signifikan; pertimbangkan teknik lain (mis. feature engineering).')

print('\n===== BEST HYPERPARAMETERS =====')
for model_name, params in best_params.items():
    print(f'\n{model_name}:')
    for param, value in params.items():
        print(f'  {param}: {value}')
