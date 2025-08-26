#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для загрузки данных о населении мира
"""

import pandas as pd
import numpy as np
from urllib.request import urlopen
import json
import os
import sys
import logging
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

logger.info("Загрузка данных о населении мира...")

# Создаем датафрейм из доступных данных по населению мира
# Пользуемся открытым API World Bank для получения данных
try:
    # Список стран
    countries_url = "http://api.worldbank.org/v2/country?format=json&per_page=300"
    with urlopen(countries_url) as response:
        countries_data = json.loads(response.read())[1]
    
    countries = []
    for country in countries_data:
        # Пропускаем агрегированные данные и регионы
        if country.get('region', {}).get('id') == "":
            continue
            
        countries.append({
            'Country': country['name'],
            'ISO3': country.get('id', ''),
            'Continent': country.get('region', {}).get('value', ''),
            'Region': country.get('incomeLevel', {}).get('value', '')
        })
    
    # Создаем базовую таблицу
    df = pd.DataFrame(countries)
    
    # Загрузим данные о населении за несколько лет
    # Годы для анализа
    years = [1970, 1980, 1990, 2000, 2010, 2020, 2022]
    
    for year in years:
        indicator_url = f"http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?date={year}&format=json&per_page=300"
        try:
            with urlopen(indicator_url) as response:
                population_data = json.loads(response.read())[1]
            
            # Добавляем данные за год в основной датафрейм
            pop_dict = {}
            for entry in population_data:
                if entry.get('value') is not None:
                    country_code = entry['country']['id']
                    pop_dict[country_code] = entry['value']
            
            # Добавляем столбец с годом
            df[f'{year} Population'] = df['ISO3'].map(pop_dict)
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных за {year} год: {e}")
    
    # Добавляем прирост населения
    df['Growth Rate'] = (df['2022 Population'] - df['1970 Population']) / df['1970 Population'] * 100
    
    # Сортируем по населению 2022 года
    df = df.sort_values(by='2022 Population', ascending=False)
    
    # Сохраняем в CSV
    df.to_csv('world_population.csv', index=False)
    
    # Пример ML-задачи: предсказание населения на 2022 год по данным за 1970-2010
    features = [f'{year} Population' for year in [1970, 1980, 1990, 2000, 2010]]
    target = '2022 Population'

    df_clean = df.dropna(subset=features + [target])
    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    logger.info(f"Mean Squared Error (Random Forest): {mse:.2f}")

    logger.info("Данные успешно загружены и сохранены в файл world_population.csv")
    
except Exception as e:
    logger.error(f"Ошибка при загрузке данных: {e}")
    
    # Создадим минимальный пример данных о населении мира
    # для тестирования скрипта, если API недоступен
    logger.info("Создаем тестовые данные для примера...")
    
    # Список крупнейших стран по населению
    countries = [
        {'Country': 'China', 'ISO3': 'CHN', 'Continent': 'East Asia & Pacific', 'Region': 'Upper middle income'},
        {'Country': 'India', 'ISO3': 'IND', 'Continent': 'South Asia', 'Region': 'Lower middle income'},
        {'Country': 'United States', 'ISO3': 'USA', 'Continent': 'North America', 'Region': 'High income'},
        {'Country': 'Indonesia', 'ISO3': 'IDN', 'Continent': 'East Asia & Pacific', 'Region': 'Lower middle income'},
        {'Country': 'Pakistan', 'ISO3': 'PAK', 'Continent': 'South Asia', 'Region': 'Lower middle income'},
        {'Country': 'Brazil', 'ISO3': 'BRA', 'Continent': 'Latin America & Caribbean', 'Region': 'Upper middle income'},
        {'Country': 'Nigeria', 'ISO3': 'NGA', 'Continent': 'Sub-Saharan Africa', 'Region': 'Lower middle income'},
        {'Country': 'Bangladesh', 'ISO3': 'BGD', 'Continent': 'South Asia', 'Region': 'Lower middle income'},
        {'Country': 'Russia', 'ISO3': 'RUS', 'Continent': 'Europe & Central Asia', 'Region': 'Upper middle income'},
        {'Country': 'Mexico', 'ISO3': 'MEX', 'Continent': 'Latin America & Caribbean', 'Region': 'Upper middle income'}
    ]
    
    # Создадим датафрейм
    df = pd.DataFrame(countries)
    
    # Добавим данные о населении за разные годы
    population_data = {
        'CHN': {'1970': 818315000, '1980': 981235000, '1990': 1135185000, '2000': 1262645000, '2010': 1337705000, '2020': 1407745000, '2022': 1412175000},
        'IND': {'1970': 553943000, '1980': 696783000, '1990': 870133000, '2000': 1059634000, '2010': 1234281000, '2020': 1380004000, '2022': 1417173000},
        'USA': {'1970': 207053000, '1980': 227726000, '1990': 252653000, '2000': 282398000, '2010': 309011000, '2020': 335942000, '2022': 338289000},
        'IDN': {'1970': 114835000, '1980': 147490000, '1990': 181437000, '2000': 211540000, '2010': 241834000, '2020': 270626000, '2022': 275501000},
        'PAK': {'1970': 58142000, '1980': 78054000, '1990': 107608000, '2000': 138523000, '2010': 179425000, '2020': 220892000, '2022': 235825000},
        'BRA': {'1970': 95113000, '1980': 121618000, '1990': 149648000, '2000': 174790000, '2010': 196796000, '2020': 212559000, '2022': 215313000},
        'NGA': {'1970': 55982000, '1980': 73698000, '1990': 95212000, '2000': 122352000, '2010': 158503000, '2020': 206140000, '2022': 218541000},
        'BGD': {'1970': 65048000, '1980': 83929000, '1990': 103172000, '2000': 127658000, '2010': 147575000, '2020': 164689000, '2022': 171186000},
        'RUS': {'1970': 130404000, '1980': 138127000, '1990': 147531000, '2000': 146405000, '2010': 142849000, '2020': 144104000, '2022': 143449000},
        'MEX': {'1970': 51493000, '1980': 67705000, '1990': 83943000, '2000': 97873000, '2010': 112532000, '2020': 126014000, '2022': 127504000}
    }
    
    # Добавляем данные в датафрейм
    years = [1970, 1980, 1990, 2000, 2010, 2020, 2022]
    for year in years:
        df[f'{year} Population'] = df['ISO3'].apply(lambda x: population_data.get(x, {}).get(str(year), np.nan))
    
    # Добавляем прирост населения
    df['Growth Rate'] = (df['2022 Population'] - df['1970 Population']) / df['1970 Population'] * 100
    
    # Сохраняем в CSV
    df.to_csv('world_population.csv', index=False)
    
    # Пример ML-задачи: предсказание населения на 2022 год по данным за 1970-2010
    features = [f'{year} Population' for year in [1970, 1980, 1990, 2000, 2010]]
    target = '2022 Population'

    df_clean = df.dropna(subset=features + [target])
    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    logger.info(f"Mean Squared Error (Random Forest): {mse:.2f}")

    logger.info("Тестовые данные успешно созданы и сохранены в файл world_population.csv")

logger.info("\nДля продолжения выполнения задания запустите скрипт practice1_solution.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт загрузки и анализа данных о населении мира")
    parser.add_argument('--verbose', action='store_true', help="Выводить подробные логи")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
