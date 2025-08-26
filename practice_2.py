"""
Практическое задание №2: анализ и классификация данных для прогнозирования инфаркта.
"""
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from kaggle.api.kaggle_api_extended import KaggleApi
import os

def main():
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    dataset_path = 'datasets/heart-attack-dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        logging.info("Загружаем датасет по прогнозированию вероятности инфаркта...")
        api.dataset_download_files('rashikrahmanpritom/heart-attack-analysis-prediction-dataset', path=dataset_path, unzip=True)
        logging.info(f"Датасет загружен в {dataset_path}")
    else:
        logging.info(f"Датасет уже загружен в {dataset_path}")

    files = os.listdir(dataset_path)
    csv_files = [f for f in files if f.endswith('.csv')]

    if csv_files:
        df = pd.read_csv(f"{dataset_path}/{csv_files[0]}")
        logging.info("Датасет успешно загружен в DataFrame")

        logging.info("--- Базовая информация о наборе данных ---")
        logging.info("Форма датасета (строки, столбцы):")
        logging.info(df.shape)

        logging.debug("Первые 5 строк датасета:")
        logging.debug(df.head())

        logging.debug("Статистическое описание числовых данных:")
        logging.debug(df.describe())

        logging.debug("Информация о типах данных и непустых значениях:")
        buffer = []
        df.info(buf=buffer)
        # Since df.info prints to stdout or buffer, to capture info we can use a workaround:
        # But here, to keep it simple, just log that info was displayed
        logging.debug("Информация о типах данных и непустых значениях выведена")

        logging.debug("Количество пустых значений в каждом столбце:")
        logging.debug(df.isnull().sum())

        logging.info("--- Создаем визуализации данных ---")

        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        logging.info("Создаем парные диаграммы...")
        selected_cols = df.columns[:5].tolist() + [df.columns[-1]]
        pairplot = sns.pairplot(df[selected_cols], hue=df.columns[-1])
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/pairplot_heart.png')
        plt.close()
        logging.info("Парные диаграммы сохранены")

        logging.info("Создаем тепловую карту корреляций...")
        plt.figure(figsize=(12, 10))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Тепловая карта корреляций')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/correlation_heatmap_heart.png')
        plt.close()
        logging.info("Тепловая карта корреляций сохранена")

        logging.info("Создаем гистограммы...")
        for i, col in enumerate(df.columns[:3]):
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Гистограмма распределения: {col}')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/histogram_heart_{i}.png')
            plt.close()
        logging.info("Гистограммы сохранены")

        logging.info("Создаем диаграмму размаха (ящик с усами)...")
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df.select_dtypes(include=[np.number]))
        plt.title('Диаграмма размаха для числовых данных')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/boxplot_heart.png')
        plt.close()
        logging.info("Диаграмма размаха сохранена")

        logging.info("Создаем графики распределения...")
        target_col = df.columns[-1]
        for i, col in enumerate(df.columns[:3]):
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=df, x=col, hue=target_col, common_norm=False)
            plt.title(f'Распределение {col} по классам')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/dist_plot_heart_{i}.png')
            plt.close()
        logging.info("Графики распределения сохранены")

        logging.info("--- Анализ влияния признаков на вероятность инфаркта ---")

        correlation_with_target = corr[target_col].sort_values(ascending=False)
        logging.info("Корреляция признаков с целевой переменной:")
        logging.info(correlation_with_target)

        logging.info("Топ-5 признаков с наибольшей корреляцией:")
        logging.info(correlation_with_target.drop(target_col).head(5))

        logging.info("--- Предобработка данных ---")

        low_corr_threshold = 0.1
        low_corr_features = correlation_with_target[abs(correlation_with_target) < low_corr_threshold].index.tolist()

        if target_col in low_corr_features:
            low_corr_features.remove(target_col)

        logging.info(f"Признаки с корреляцией ниже {low_corr_threshold}:")
        logging.info(low_corr_features)

        df_clean = df.drop(columns=low_corr_features)
        logging.info(f"Размер датафрейма после удаления признаков с низкой корреляцией: {df_clean.shape}")

        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        logging.info(f"Размер обучающей выборки: {X_train.shape}")
        logging.info(f"Размер тестовой выборки: {X_test.shape}")

        logging.info("--- Обучение моделей и оценка точности ---")

        models = {
            'Метод опорных векторов': SVC(random_state=42),
            'Логистическая регрессия': LogisticRegression(random_state=42),
            'Дерево решений': DecisionTreeClassifier(random_state=42),
            'Случайный лес': RandomForestClassifier(random_state=42)
        }

        results = {}

        for name, model in models.items():
            logging.info(f"Обучение модели: {name}")
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)

            y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            results[name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            }

            logging.info(f"{name}:")
            logging.info(f"  Точность на обучающей выборке: {train_accuracy:.4f}")
            logging.info(f"  Точность на тестовой выборке: {test_accuracy:.4f}")

        plt.figure(figsize=(12, 8))

        model_names = list(results.keys())
        train_accuracies = [results[model]['train_accuracy'] for model in model_names]
        test_accuracies = [results[model]['test_accuracy'] for model in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width/2, train_accuracies, width, label='Обучающая выборка')
        plt.bar(x + width/2, test_accuracies, width, label='Тестовая выборка')

        plt.xlabel('Модель')
        plt.ylabel('Точность')
        plt.title('Сравнение точности моделей')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/model_comparison_heart.png')
        plt.close()

        logging.info("Практическое задание 2 выполнено успешно! Все визуализации сохранены в папке 'plots'")
    else:
        logging.info("CSV-файл не найден в загруженном датасете")

if __name__ == "__main__":
    main()
