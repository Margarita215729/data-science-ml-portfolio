"""
Практическое задание №4: прогнозирование стоимости недвижимости с использованием регрессий и полиномиальных моделей.
"""

import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from kaggle.api.kaggle_api_extended import KaggleApi
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings

# Игнорируем предупреждения

warnings.filterwarnings('ignore')

def main():
    logging.info("Авторизация в Kaggle")
    api = KaggleApi()
    api.authenticate()

    # Создаем папку для датасетов, если её нет
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    # Загрузка датасета прогнозирования стоимости недвижимости
    dataset_path = 'datasets/house-sales-prediction'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        logging.info("Загружаем датасет прогнозирования стоимости недвижимости...")
        api.dataset_download_files('harlfoxem/housesalesprediction', path=dataset_path, unzip=True)
        logging.info(f"Датасет загружен в {dataset_path}")
    else:
        logging.info(f"Датасет уже загружен в {dataset_path}")

    # Загружаем датасет в pandas
    files = os.listdir(dataset_path)
    csv_files = [f for f in files if f.endswith('.csv')]

    if csv_files:
        df = pd.read_csv(f"{dataset_path}/{csv_files[0]}")
        logging.info("Датасет успешно загружен в DataFrame")
        
        logging.info("Базовая информация о наборе данных")
        logging.debug(f"Форма датасета (строки, столбцы): {df.shape}")
        logging.debug(f"Первые 5 строк датасета:\n{df.head()}")
        logging.debug(f"Статистическое описание числовых данных:\n{df.describe()}")
        logging.debug("Информация о типах данных и непустых значениях:")
        logging.debug(df.info())
        logging.debug(f"Количество пустых значений в каждом столбце:\n{df.isnull().sum()}")

        logging.info("Создаем визуализации данных")
        # Создаем папку для визуализаций, если её нет
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Определение целевой переменной (цены)
        price_column = 'price'
        if price_column not in df.columns:
            # Если цена называется по-другому, выбираем колонку с наибольшими значениями
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            price_column = df[numeric_cols].mean().idxmax()  # Предполагаем, что это будет цена

        # 1. Гистограммы
        logging.info("Создаем гистограммы")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols[:3]):  # Ограничиваем до 3 колонок
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Гистограмма распределения: {col}')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/histogram_house_{i}.png')
            plt.close()
        logging.info("Гистограммы сохранены")

        # 2. Ящик с усами (диаграмма размаха)
        logging.info("Создаем диаграмму размаха (ящик с усами)")
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df[numeric_cols])
        plt.title('Диаграмма размаха для числовых данных')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/boxplot_house.png')
        plt.close()
        logging.info("Диаграмма размаха сохранена")

        # 3. Точечные 3D-графики распределения признаков
        logging.info("Создаем точечные 3D-графики")
        if len(numeric_cols) >= 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            x_col, y_col = numeric_cols[1], numeric_cols[2]  # Выбираем два признака
            ax.scatter(df[x_col], df[y_col], df[price_column], c=df[price_column], cmap='viridis', s=30, alpha=0.5)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(price_column)
            ax.set_title(f'3D график: {x_col} vs {y_col} vs {price_column}')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/3d_scatter_house.png')
            plt.close()
            logging.info("3D график сохранен")

        # 4. Тепловая карта корреляционной матрицы
        logging.info("Создаем тепловую карту корреляций")
        plt.figure(figsize=(12, 10))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Тепловая карта корреляций')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/correlation_heatmap_house.png')
        plt.close()
        logging.info("Тепловая карта корреляций сохранена")

        logging.info("Анализ влияния характеристик на стоимость недвижимости")
        # Корреляция с ценой
        price_correlation = corr[price_column].sort_values(ascending=False)
        logging.info("Корреляция признаков с ценой недвижимости:\n" + str(price_correlation))
        # Топ-5 самых влияющих признаков
        top_features = price_correlation.drop(price_column).head(5).index.tolist()
        logging.info("Топ-5 самых влияющих на цену признаков:")
        for feature in top_features:
            logging.info(f"- {feature}: {price_correlation[feature]:.4f}")
        # Визуализация зависимости цены от топ-3 признаков
        for i, feature in enumerate(top_features[:3]):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[feature], y=df[price_column])
            plt.title(f'Зависимость цены от {feature}')
            plt.xlabel(feature)
            plt.ylabel('Цена')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/price_vs_{feature}.png')
            plt.close()

        logging.info("Предобработка данных")
        # Обработка даты, если она есть
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            logging.info(f"Преобразование колонки даты {date_col}")
            df[date_col] = pd.to_datetime(df[date_col])
            # Добавление года и месяца как отдельных признаков
            df['sale_year'] = df[date_col].dt.year
            df['sale_month'] = df[date_col].dt.month
            # Удаление исходной колонки даты
            df.drop(columns=[date_col], inplace=True)
        # Объединение значений по периодам в дате постройки и капремонта
        date_build_cols = [col for col in df.columns if 'build' in col.lower() or 'yr_built' in col.lower()]
        date_renov_cols = [col for col in df.columns if 'renov' in col.lower() or 'yr_renovated' in col.lower()]
        # Группировка по периодам для даты постройки
        if date_build_cols:
            build_col = date_build_cols[0]
            df['build_period'] = pd.cut(df[build_col], 
                                       bins=[df[build_col].min()-1, 1950, 1970, 1990, 2000, df[build_col].max()+1],
                                       labels=['pre-1950', '1950s-1960s', '1970s-1980s', '1990s', '2000s'])
        # Группировка по периодам для даты ремонта
        if date_renov_cols:
            renov_col = date_renov_cols[0]
            if df[renov_col].min() == 0:
                # Создаем категорию "Без ремонта"
                df['renovation_status'] = np.where(df[renov_col] == 0, "No Renovation", "Renovated")
                # Для домов с ремонтом группируем по периодам
                renov_mask = df[renov_col] > 0
                df.loc[renov_mask, 'renovation_period'] = pd.cut(df.loc[renov_mask, renov_col], 
                                                               bins=[df.loc[renov_mask, renov_col].min()-1, 1980, 1990, 2000, df[renov_col].max()+1],
                                                               labels=['pre-1980', '1980s', '1990s', '2000s'])
                df.loc[~renov_mask, 'renovation_period'] = 'No Renovation'
        # Нормализация данных
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if price_column in numeric_cols:
            numeric_cols.remove(price_column)
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        # Подготовка данных для обучения моделей
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        # Разделение на признаки и целевую переменную
        X = df_encoded.drop(columns=[price_column])
        y = df_encoded[price_column]
        # Разбиение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Размер обучающей выборки: {X_train.shape}")
        logging.info(f"Размер тестовой выборки: {X_test.shape}")

        logging.info("Обучение моделей и оценка точности")
        # Список моделей для обучения
        models = {
            'Линейная регрессия': LinearRegression(),
            'Регрессия LASSO': Lasso(alpha=0.1, random_state=42),
            'Регрессия Ridge': Ridge(alpha=1.0, random_state=42),
            'Полиномиальная регрессия': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
        }
        metrics = {
            'R^2 (коэф. детерминации)': r2_score,
            'Средняя абсолютная ошибка (MAE)': mean_absolute_error,
            'Средняя квадратичная ошибка (MSE)': mean_squared_error
        }
        results = {}
        for name, model in models.items():
            logging.info(f"Обучение модели: {name}")
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            model_metrics = {
                'train': {},
                'test': {}
            }
            for metric_name, metric_func in metrics.items():
                train_score = metric_func(y_train, y_train_pred)
                test_score = metric_func(y_test, y_test_pred)
                model_metrics['train'][metric_name] = train_score
                model_metrics['test'][metric_name] = test_score
            results[name] = model_metrics
            logging.info(f"{name}:")
            for metric_name in metrics:
                logging.info(f"  {metric_name} на обучающей выборке: {model_metrics['train'][metric_name]:.4f}")
                logging.info(f"  {metric_name} на тестовой выборке: {model_metrics['test'][metric_name]:.4f}")
        # Визуализация результатов (R^2)
        plt.figure(figsize=(12, 8))
        model_names = list(results.keys())
        train_r2 = [results[model]['train']['R^2 (коэф. детерминации)'] for model in model_names]
        test_r2 = [results[model]['test']['R^2 (коэф. детерминации)'] for model in model_names]
        x = np.arange(len(model_names))
        width = 0.35
        plt.bar(x - width/2, train_r2, width, label='Обучающая выборка')
        plt.bar(x + width/2, test_r2, width, label='Тестовая выборка')
        plt.xlabel('Модель')
        plt.ylabel('R^2 (коэф. детерминации)')
        plt.title('Сравнение точности моделей по R^2')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/model_comparison_r2_house.png')
        plt.close()
        # Визуализация результатов (MAE)
        plt.figure(figsize=(12, 8))
        train_mae = [results[model]['train']['Средняя абсолютная ошибка (MAE)'] for model in model_names]
        test_mae = [results[model]['test']['Средняя абсолютная ошибка (MAE)'] for model in model_names]
        plt.bar(x - width/2, train_mae, width, label='Обучающая выборка')
        plt.bar(x + width/2, test_mae, width, label='Тестовая выборка')
        plt.xlabel('Модель')
        plt.ylabel('MAE')
        plt.title('Сравнение точности моделей по MAE')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/model_comparison_mae_house.png')
        plt.close()

        logging.info("Кросс-валидация моделей")
        cv_results = {}
        for name, model in models.items():
            logging.info(f"Кросс-валидация модели: {name}")
            r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            mse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            cv_results[name] = {
                'R^2': r2_scores.mean(),
                'MAE': mae_scores.mean(),
                'MSE': mse_scores.mean()
            }
            logging.info(f"{name} (после кросс-валидации):")
            logging.info(f"  Средний R^2: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
            logging.info(f"  Средний MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
            logging.info(f"  Средний MSE: {mse_scores.mean():.4f} ± {mse_scores.std():.4f}")
        # Визуализация результатов кросс-валидации
        plt.figure(figsize=(12, 8))
        model_names = list(cv_results.keys())
        cv_r2 = [cv_results[model]['R^2'] for model in model_names]
        plt.bar(x, cv_r2, width=0.5)
        plt.xlabel('Модель')
        plt.ylabel('R^2 (коэф. детерминации)')
        plt.title('Сравнение точности моделей по R^2 после кросс-валидации')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/model_comparison_cv_r2_house.png')
        plt.close()

        logging.info("Поиск оптимальных гиперпараметров")
        best_model_name = max(cv_results, key=lambda k: cv_results[k]['R^2'])
        logging.info(f"Модель с наилучшими показателями: {best_model_name}")
        if best_model_name == 'Регрессия LASSO':
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
            }
            grid_model = Lasso(random_state=42)
        elif best_model_name == 'Регрессия Ridge':
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
            }
            grid_model = Ridge(random_state=42)
        elif best_model_name == 'Полиномиальная регрессия':
            param_grid = {
                'poly__degree': [1, 2, 3]
            }
            grid_model = Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', LinearRegression())
            ])
        else:  # Линейная регрессия
            logging.info("Для линейной регрессии не требуется настройка гиперпараметров")
            logging.info("Практическое задание 4 выполнено успешно! Все визуализации сохранены в папке 'plots'")
            return
        grid_search = GridSearchCV(grid_model, param_grid, cv=5, scoring='r2')
        grid_search.fit(X, y)
        logging.info("Результаты поиска оптимальных гиперпараметров:")
        logging.info(f"Лучшие параметры: {grid_search.best_params_}")
        logging.info(f"Лучший R^2: {grid_search.best_score_:.4f}")
        if best_model_name in ['Регрессия LASSO', 'Регрессия Ridge']:
            results_df = pd.DataFrame(grid_search.cv_results_)
            plt.figure(figsize=(10, 6))
            plt.errorbar(results_df['param_alpha'], results_df['mean_test_score'], 
                        yerr=results_df['std_test_score'])
            plt.xlabel('alpha')
            plt.ylabel('R^2 (кросс-валидация)')
            plt.title(f'Влияние параметра alpha на R^2 для {best_model_name}')
            plt.xscale('log')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/grid_search_alpha_{best_model_name.lower().replace(" ", "_")}.png')
            plt.close()
        elif best_model_name == 'Полиномиальная регрессия':
            results_df = pd.DataFrame(grid_search.cv_results_)
            plt.figure(figsize=(10, 6))
            plt.errorbar(results_df['param_poly__degree'], results_df['mean_test_score'], 
                        yerr=results_df['std_test_score'])
            plt.xlabel('Степень полинома')
            plt.ylabel('R^2 (кросс-валидация)')
            plt.title('Влияние степени полинома на R^2')
            plt.xticks(results_df['param_poly__degree'])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/grid_search_poly_degree.png')
            plt.close()
        logging.info("Практическое задание 4 выполнено успешно! Все визуализации сохранены в папке 'plots'")
    else:
        logging.error("CSV-файл не найден в загруженном датасете")


if __name__ == "__main__":
    main()
