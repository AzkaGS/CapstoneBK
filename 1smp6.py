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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, ConfusionMatrixDisplay)

# 1. Load dataset
df = pd.read_csv('C:/bengkod/ObesityDataSet.csv')

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
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
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

        cm = confusion_matrix(y_te, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix – {name} ({label})')
        plt.show()

    return pd.DataFrame(results)

print('\n===== EVALUASI BASELINE =====')
baseline_metrics = evaluate_models(baseline_models, X_train, X_test, y_train, y_test)

# ------------------------------------------------------------------
# HYPERPARAMETER TUNING
# ------------------------------------------------------------------
param_grids = {
    'Logistic Regression': {
        'C': np.logspace(-3, 3, 10),
        'penalty': ['l2'],
        'solver': ['lbfgs']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 400, 600],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    'KNN': {
        'n_neighbors': list(range(3, 21, 2)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
}

tuned_models = {}
best_params = {}

for name, mdl in baseline_models.items():
    if name == 'Random Forest':
        search = RandomizedSearchCV(
            estimator=mdl,
            param_distributions=param_grids[name],
            n_iter=30,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42
        )
    else:
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
    print(f'\nBest params {name}: {search.best_params_}')

# ------------------------------------------------------------------
# EVALUASI MODEL TUNED
# ------------------------------------------------------------------
print('\n===== EVALUASI MODEL SETELAH TUNING =====')
tuned_metrics = evaluate_models(tuned_models, X_train, X_test, y_train, y_test, label='Tuned')

# ------------------------------------------------------------------
# VISUALISASI PERBANDINGAN
# ------------------------------------------------------------------
baseline_metrics['Tipe'] = 'Baseline'
tuned_metrics['Tipe'] = 'Tuned'
combined_metrics = pd.concat([baseline_metrics, tuned_metrics], ignore_index=True)

plt.figure(figsize=(12, 6))
metrics_melted = combined_metrics.melt(
    id_vars=['Model', 'Tipe'], var_name='Metric', value_name='Score'
)
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted)
plt.title('Perbandingan Performa Model – Baseline vs Tuned')
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.show()

# ------------------------------------------------------------------
# KESIMPULAN
# ------------------------------------------------------------------
print('\n===== KESIMPULAN =====')
best_baseline = baseline_metrics.loc[baseline_metrics['F1 Score'].idxmax()]
best_tuned = tuned_metrics.loc[tuned_metrics['F1 Score'].idxmax()]
improvement = best_tuned['F1 Score'] - best_baseline['F1 Score']

print(f'Performa terbaik baseline: {best_baseline["Model"]} (F1 = {best_baseline["F1 Score"]:.4f})')
print(f'Performa terbaik setelah tuning: {best_tuned["Model"]} (F1 = {best_tuned["F1 Score"]:.4f})')
print(f'Peningkatan F1 Score: {improvement:.4f}')

if improvement > 0:
    print('🔸 Hyperparameter tuning berhasil meningkatkan kinerja model.')
else:
    print('🔸 Hyperparameter tuning tidak memberikan peningkatan signifikan; pertimbangkan teknik lain (mis. feature engineering).')
