"""
Практическое задание №1: анализ и визуализация мирового населения.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def main():
    # Авторизация в Kaggle
    api = KaggleApi()
    api.authenticate()

    # Создаем папку для датасетов, если её нет
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    # Загрузка датасета по населению мира
    dataset_path = 'datasets/world-population-dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        logging.info("Загружаем датасет по населению мира...")
        api.dataset_download_files('iamsouravbanerjee/world-population-dataset', path=dataset_path, unzip=True)
        logging.info(f"Датасет загружен в {dataset_path}")
    else:
        logging.info(f"Датасет уже загружен в {dataset_path}")

    # Загружаем датасет в pandas
    files = os.listdir(dataset_path)
    csv_files = [f for f in files if f.endswith('.csv')]

    if csv_files:
        df = pd.read_csv(f"{dataset_path}/{csv_files[0]}")
        print("Датасет успешно загружен в DataFrame")

        print("\n--- Базовая информация о наборе данных ---")
        print("\nФорма датасета (строки, столбцы):")
        print(df.shape)

        print("\nПервые 5 строк датасета:")
        print(df.head())

        print("\nСтатистическое описание числовых данных:")
        print(df.describe())

        print("\nИнформация о типах данных и непустых значениях:")
        print(df.info())

        print("\nКоличество пустых значений в каждом столбце:")
        print(df.isnull().sum())

        print("\n--- Создаем визуализации данных ---")

        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        print("Создаем парные диаграммы...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 5:
            numeric_cols = numeric_cols[:5]

        if numeric_cols:
            pairplot = sns.pairplot(df[numeric_cols])
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/pairplot.png')
            plt.close()
            print("Парные диаграммы сохранены")

        print("Создаем тепловую карту корреляций...")
        plt.figure(figsize=(12, 10))
        corr = df.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Тепловая карта корреляций')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/correlation_heatmap.png')
        plt.close()
        print("Тепловая карта корреляций сохранена")

        print("Создаем диаграмму рассеяния...")
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]])
            plt.title(f'Диаграмма рассеяния: {numeric_cols[0]} vs {numeric_cols[1]}')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/scatter_plot.png')
            plt.close()
            print("Диаграмма рассеяния сохранена")

        print("Создаем гистограммы...")
        for i, col in enumerate(numeric_cols[:3]):
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Гистограмма распределения: {col}')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/histogram_{i}.png')
            plt.close()
        print("Гистограммы сохранены")

        print("Создаем диаграмму размаха (ящик с усами)...")
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df[numeric_cols])
        plt.title('Диаграмма размаха для числовых данных')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/boxplot.png')
        plt.close()
        print("Диаграмма размаха сохранена")

        print("Создаем скрипичные диаграммы...")
        for i, col in enumerate(numeric_cols[:2]):
            plt.figure(figsize=(10, 8))
            sns.violinplot(y=df[col])
            plt.title(f'Скрипичная диаграмма для {col}')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/violinplot_{i}.png')
            plt.close()
        print("Скрипичные диаграммы сохранены")

        print("\n--- Анализ динамики населения по регионам ---")

        if 'Region' in df.columns and '2022 Population' in df.columns:
            region_population = df.groupby('Region')['2022 Population'].sum().sort_values(ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x=region_population.index, y=region_population.values)
            plt.title('Население по регионам (2022)')
            plt.xlabel('Регион')
            plt.ylabel('Население')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/region_population_barplot.png')
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.pie(region_population, labels=region_population.index, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Процентное соотношение населения по регионам (2022)')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/region_population_pie.png')
            plt.close()

            print("Диаграммы по регионам сохранены")

        population_columns = [col for col in df.columns if 'Population' in col and col != 'World Population Percentage']

        if len(population_columns) > 1:
            years = []
            total_population = []

            for col in population_columns:
                if col.startswith('19') or col.startswith('20'):
                    year = col.split(' ')[0]
                    years.append(int(year))
                    total_population.append(df[col].sum())

            pop_df = pd.DataFrame({'Year': years, 'Total Population': total_population})
            pop_df = pop_df.sort_values(by='Year')

            plt.figure(figsize=(12, 8))
            plt.plot(pop_df['Year'], pop_df['Total Population'] / 1_000_000_000, marker='o', linestyle='-')
            plt.title('Динамика мирового населения')
            plt.xlabel('Год')
            plt.ylabel('Население (миллиарды)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/population_dynamics.png')
            plt.close()

            pop_df['Percent Change'] = pop_df['Total Population'].pct_change() * 100

            plt.figure(figsize=(12, 8))
            plt.bar(pop_df['Year'][1:], pop_df['Percent Change'][1:])
            plt.title('Процентное изменение мирового населения по годам')
            plt.xlabel('Год')
            plt.ylabel('Изменение в %')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/population_percent_change.png')
            plt.close()

            print("Диаграммы динамики населения сохранены")

        print("\nПрактическое задание 1 выполнено успешно! Все визуализации сохранены в папке 'plots'")
    else:
        print("CSV-файл не найден в загруженном датасете.")


if __name__ == "__main__":
    main()
