"""
Практическое задание №3: прогнозирование погодных условий с использованием XGBoost и кластеризации.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import warnings
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Игнорируем предупреждения
warnings.filterwarnings('ignore')

def main():
    # Авторизация в Kaggle
    api = KaggleApi()
    api.authenticate()

    # Создаем папку для датасетов, если её нет
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    # Загрузка датасета прогнозирования погодных условий
    dataset_path = 'datasets/daily-climate-dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        logging.info("Загружаем датасет прогнозирования погодных условий...")
        api.dataset_download_files('sumanthvrao/daily-climate-time-series-data', path=dataset_path, unzip=True)
        logging.info(f"Датасет загружен в {dataset_path}")
    else:
        logging.info(f"Датасет уже загружен в {dataset_path}")

    logging.info("Начало выполнения практического задания №3")

    files = os.listdir(dataset_path)
    csv_files = [f for f in files if f.endswith('.csv')]

    if csv_files:
        df = pd.read_csv(f"{dataset_path}/{csv_files[0]}")
        logging.info("Датасет успешно загружен в DataFrame")
        
        logging.info("--- Базовая информация о наборе данных ---")
        logging.debug(f"Форма датасета (строки, столбцы): {df.shape}")
        logging.debug(f"Первые 5 строк датасета:\n{df.head()}")
        logging.debug(f"Статистическое описание числовых данных:\n{df.describe()}")
        logging.debug(f"Информация о типах данных и непустых значениях:\n{df.info()}")
        logging.debug(f"Количество пустых значений в каждом столбце:\n{df.isnull().sum()}")

        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            logging.info(f"Преобразование колонки даты {date_col}...")
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)

        logging.info("--- Создаем визуализации данных ---")

        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        logging.info("Создаем гистограммы распределения...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols[:3]):
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Гистограмма распределения: {col}')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/histogram_climate_{i}.png')
            plt.close()
        logging.info("Гистограммы распределения сохранены")

        logging.info("Создаем графики распределения данных...")
        for i, col in enumerate(numeric_cols[:3]):
            plt.figure(figsize=(10, 6))
            sns.kdeplot(df[col])
            plt.title(f'График распределения: {col}')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/dist_plot_climate_{i}.png')
            plt.close()
        logging.info("Графики распределения сохранены")

        logging.info("Создаем диаграммы размаха...")
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df[numeric_cols])
        plt.title('Диаграмма размаха для числовых данных')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/boxplot_climate.png')
        plt.close()
        logging.info("Диаграммы размаха сохранены")

        if hasattr(df.index, 'date') or hasattr(df.index, 'time'):
            logging.info("Создаем график временной динамики...")
            for i, col in enumerate(numeric_cols[:2]):
                plt.figure(figsize=(14, 8))
                plt.plot(df.index, df[col])
                plt.title(f'Временная динамика: {col}')
                plt.xlabel('Дата')
                plt.ylabel(col)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{plots_dir}/time_series_climate_{i}.png')
                plt.close()
            logging.info("Графики временной динамики сохранены")

        logging.info("--- Анализ особенностей набора данных ---")

        corr = df.corr()
        logging.debug(f"Матрица корреляции:\n{corr}")

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Тепловая карта корреляций')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/correlation_heatmap_climate.png')
        plt.close()

        logging.info("--- Кластеризация данных ---")

        features_for_clustering = numeric_cols.tolist()

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[features_for_clustering])

        inertia = []
        k_range = range(1, 11)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_scaled)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertia, 'o-')
        plt.title('Метод локтя для определения оптимального числа кластеров')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Инерция')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/elbow_method_climate.png')
        plt.close()

        optimal_k = 3

        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(df_scaled)

        df['Cluster'] = clusters

        if len(features_for_clustering) >= 2:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=df[features_for_clustering[0]], y=df[features_for_clustering[1]], hue=df['Cluster'], palette='viridis')
            plt.title('Кластеры данных')
            plt.xlabel(features_for_clustering[0])
            plt.ylabel(features_for_clustering[1])
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/clusters_climate.png')
            plt.close()

        logging.info("Анализ кластеров:")
        for i in range(optimal_k):
            logging.debug(f"Кластер {i}:\n{df[df['Cluster'] == i].describe().mean()}")

        logging.info("--- Обучение модели XGBoost Regressor ---")

        target_variable = numeric_cols[0]
        logging.info(f"Целевая переменная для прогнозирования: {target_variable}")

        if hasattr(df.index, 'date') or hasattr(df.index, 'time'):
            for lag in range(1, 4):
                df[f'{target_variable}_lag_{lag}'] = df[target_variable].shift(lag)
            df.dropna(inplace=True)

        features = [col for col in df.columns if col != target_variable and col != 'Cluster']
        X = df[features]
        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        logging.info("XGBoost Regressor:")
        logging.info(f"  R^2 на обучающей выборке: {train_r2:.4f}")
        logging.info(f"  R^2 на тестовой выборке: {test_r2:.4f}")

        plt.figure(figsize=(12, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Реальные значения')
        plt.ylabel('Предсказанные значения')
        plt.title(f'XGBoost: Реальные vs Предсказанные значения (R^2={test_r2:.4f})')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/xgboost_predictions_climate.png')
        plt.close()

        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
        plt.title('Важность признаков в модели XGBoost')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/feature_importance_climate.png')
        plt.close()

        logging.info("Практическое задание 3 выполнено успешно! Все визуализации сохранены в папке 'plots'")
    else:
        logging.info("CSV-файл не найден в загруженном датасете")

if __name__ == "__main__":
    main()
