"""
Практическое задание №5: сегментация покупателей методом кластеризации с PCA.
"""

import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import warnings

def main():
    # Игнорируем предупреждения
    warnings.filterwarnings('ignore')

    # Авторизация в Kaggle
    api = KaggleApi()
    api.authenticate()

    # Создаем папку для датасетов, если её нет
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    # Загрузка датасета по сегментации покупателей
    dataset_path = 'datasets/customer-personality-analysis'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        logging.info("Загружаем датасет по сегментации покупателей...")
        api.dataset_download_files('imakash3011/customer-personality-analysis', path=dataset_path, unzip=True)
        logging.info(f"Датасет загружен в {dataset_path}")
    else:
        logging.info(f"Датасет уже загружен в {dataset_path}")

    # Загружаем датасет в pandas
    files = os.listdir(dataset_path)
    csv_files = [f for f in files if f.endswith('.csv')]

    if csv_files:
        df = pd.read_csv(f"{dataset_path}/{csv_files[0]}")
        logging.info("Датасет успешно загружен в DataFrame")
        
        logging.info("--- Базовая информация о наборе данных ---")
        logging.debug("Форма датасета (строки, столбцы):")
        logging.debug(df.shape)
        
        logging.debug("Первые 5 строк датасета:")
        logging.debug(df.head())
        
        logging.debug("Статистическое описание числовых данных:")
        logging.debug(df.describe())
        
        logging.debug("Информация о типах данных и непустых значениях:")
        logging.debug(df.info())
        
        logging.debug("Количество пустых значений в каждом столбце:")
        logging.debug(df.isnull().sum())
        
        logging.info("--- Создаем визуализации данных ---")
        
        # Создаем папку для визуализаций, если её нет
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        logging.info("Создаем графики распределения признаков...")
        
        # Выбираем только числовые колонки для визуализации
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Создаем несколько графиков распределения для числовых признаков
        for i, col in enumerate(numeric_cols[:5]):  # Ограничиваем до 5 колонок для наглядности
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Распределение признака: {col}')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/distribution_customer_{i}.png')
            plt.close()
        logging.info("Графики распределения признаков сохранены")
        
        logging.info("Создаем тепловую карту корреляций...")
        plt.figure(figsize=(16, 12))
        corr = df.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Тепловая карта корреляций')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/correlation_heatmap_customer.png')
        plt.close()
        logging.info("Тепловая карта корреляций сохранена")
        
        logging.info("--- Анализ распределения индивидуальных качеств покупателей ---")
        
        # Анализ демографических характеристик
        if 'Age' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['Age'], kde=True, bins=30)
            plt.title('Распределение возраста покупателей')
            plt.xlabel('Возраст')
            plt.ylabel('Частота')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/age_distribution.png')
            plt.close()
            logging.info("Распределение возраста покупателей сохранено")
        
        # Анализ семейного положения, если есть соответствующие колонки
        marital_status_col = [col for col in df.columns if 'marital' in col.lower()]
        if marital_status_col:
            marital_col = marital_status_col[0]
            plt.figure(figsize=(10, 6))
            sns.countplot(x=df[marital_col])
            plt.title('Распределение по семейному положению')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/marital_status_distribution.png')
            plt.close()
            logging.info("Распределение по семейному положению сохранено")
        
        # Анализ образования, если есть соответствующие колонки
        education_col = [col for col in df.columns if 'education' in col.lower()]
        if education_col:
            edu_col = education_col[0]
            plt.figure(figsize=(12, 6))
            sns.countplot(x=df[edu_col])
            plt.title('Распределение по уровню образования')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/education_distribution.png')
            plt.close()
            logging.info("Распределение по уровню образования сохранено")
        
        logging.info("--- Предобработка данных ---")
        
        # Обработка пропущенных значений
        if df.isnull().sum().sum() > 0:
            logging.info(f"Обнаружено {df.isnull().sum().sum()} пропущенных значений")
            # Заполнение пропущенных значений в числовых столбцах медианой
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Заполнение пропущенных значений в категориальных столбцах модой
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    
            logging.info(f"После обработки осталось пропущенных значений: {df.isnull().sum().sum()}")
        
        # Кодирование категориальных данных
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            logging.info(f"Кодирование {len(categorical_cols)} категориальных признаков...")
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            logging.info(f"Форма датасета после кодирования: {df_encoded.shape}")
        else:
            df_encoded = df.copy()
        
        # Стандартизация данных
        logging.info("Стандартизация данных...")
        # Выбираем только числовые колонки для стандартизации
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        
        scaler = StandardScaler()
        df_scaled = df_encoded.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
        
        logging.info(f"Форма датасета после стандартизации: {df_scaled.shape}")
        
        logging.info("--- Снижение размерности набора данных ---")
        
        # Применение метода главных компонент (PCA)
        pca = PCA(n_components=0.95)  # сохраняем 95% дисперсии
        X_pca = pca.fit_transform(df_scaled[numeric_cols])
        
        logging.info(f"Форма данных после снижения размерности: {X_pca.shape}")
        logging.info(f"Количество компонент: {pca.n_components_}")
        logging.info(f"Объясненная дисперсия: {sum(pca.explained_variance_ratio_):.4f}")
        
        # Визуализация объясненной дисперсии
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        plt.xlabel('Номер компоненты')
        plt.ylabel('Доля объясненной дисперсии')
        plt.title('Объясненная дисперсия по компонентам')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/pca_explained_variance.png')
        plt.close()
        
        # Визуализация данных в пространстве первых двух главных компонент
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)
        plt.xlabel('Первая главная компонента')
        plt.ylabel('Вторая главная компонента')
        plt.title('Данные в пространстве первых двух главных компонент')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/pca_scatter.png')
        plt.close()
        
        logging.info("--- Обучение модели кластеризации методом k-средних ---")
        
        # Определение оптимального количества кластеров с помощью метода локтя
        inertia = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_pca)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))
        
        # Визуализация метода локтя
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertia, 'o-')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Инерция')
        plt.title('Метод локтя для определения оптимального числа кластеров')
        plt.grid(True)
        
        # Визуализация коэффициента силуэта
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'o-')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Коэффициент силуэта')
        plt.title('Коэффициент силуэта для разного числа кластеров')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/kmeans_elbow_silhouette.png')
        plt.close()
        
        # Определение оптимального количества кластеров
        optimal_k = k_range[np.argmax(silhouette_scores)]
        logging.info(f"Оптимальное количество кластеров по коэффициенту силуэта: {optimal_k}")
        
        # Применение кластеризации с оптимальным количеством кластеров
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        
        # Добавление меток кластеров к исходному датафрейму
        df['Cluster'] = cluster_labels
        
        # Визуализация результатов кластеризации
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
        plt.colorbar(scatter, label='Кластер')
        plt.xlabel('Первая главная компонента')
        plt.ylabel('Вторая главная компонента')
        plt.title('Результаты кластеризации методом k-средних')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/kmeans_clusters.png')
        plt.close()
        
        logging.info("--- Разведочный анализ данных по кластерам ---")
        
        # Анализ средних значений признаков по кластерам
        cluster_means = df.groupby('Cluster').mean()
        logging.info("Средние значения признаков по кластерам:")
        logging.info(f"\n{cluster_means}")
        
        # Визуализация средних значений нескольких ключевых признаков по кластерам
        key_features = numeric_cols[:5]  # Выбираем первые 5 числовых признаков для наглядности
        
        plt.figure(figsize=(15, 10))
        cluster_means[key_features].plot(kind='bar', figsize=(15, 8))
        plt.title('Средние значения ключевых признаков по кластерам')
        plt.xlabel('Кластер')
        plt.ylabel('Среднее значение (стандартизированное)')
        plt.legend(loc='upper right')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/cluster_means.png')
        plt.close()
        
        # Распределение размеров кластеров
        plt.figure(figsize=(10, 6))
        cluster_sizes = df['Cluster'].value_counts().sort_index()
        plt.bar(cluster_sizes.index, cluster_sizes.values)
        plt.title('Распределение размеров кластеров')
        plt.xlabel('Кластер')
        plt.ylabel('Количество покупателей')
        plt.xticks(cluster_sizes.index)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/cluster_sizes.png')
        plt.close()
        
        # Визуализация кластеров в виде параллельных координат для ключевых признаков
        plt.figure(figsize=(15, 8))
        pd.plotting.parallel_coordinates(
            df[key_features + ['Cluster']].sample(n=min(1000, len(df))),
            'Cluster',
            colormap='viridis'
        )
        plt.title('Параллельные координаты для ключевых признаков по кластерам')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/parallel_coordinates.png')
        plt.close()
        
        # Боксплоты для распределения ключевых признаков по кластерам
        for feature in key_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Cluster', y=feature, data=df)
            plt.title(f'Распределение признака {feature} по кластерам')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/boxplot_{feature}_by_cluster.png')
            plt.close()
        
        logging.info("Характеристика кластеров:")
        for cluster_id in range(optimal_k):
            logging.info(f"\nКластер {cluster_id}:")
            cluster_df = df[df['Cluster'] == cluster_id]
            
            # Размер кластера
            logging.info(f"Размер кластера: {len(cluster_df)} покупателей ({len(cluster_df) / len(df) * 100:.2f}% от общего числа)")
            
            # Средние значения ключевых признаков
            logging.info("Средние значения ключевых признаков:")
            for feature in key_features:
                logging.info(f"- {feature}: {cluster_df[feature].mean():.4f} (общее среднее: {df[feature].mean():.4f})")
        
        logging.info("Практическое задание 5 выполнено успешно! Все визуализации сохранены в папке 'plots'")
    else:
        logging.info("CSV-файл не найден в загруженном датасете")

if __name__ == "__main__":
    main()
