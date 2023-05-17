import numpy as np
import pandas as pd


def csv_to_dict(filename: str) -> {list, dict}:
    """Convierte un archivo CSV en un diccionario.

    Lee un archivo CSV con las columnas "Country", "Area", "GDP", "Inflation", "Life.expect", "Military",
    "Pop.growth" y "Unemployment", y crea un diccionario donde la clave es el país y el valor es un
    array NumPy con los valores de las columnas restantes.

    Args:
        filename (str): Ruta del archivo CSV.

    Returns:
        dict: Diccionario donde la clave es el país y el valor es un array NumPy con los valores
              de las columnas restantes.
    """
    data = pd.read_csv(filename)
    titles = data.columns.tolist()
    countries = data['Country'].tolist()
    country_data = {}

    for _, row in data.iterrows():
        country = row['Country']
        values = row[['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']].to_numpy()
        country_data[country] = values

    return countries, titles, country_data


def standardize_data(country_data: dict) -> dict:
    """Estandariza los datos en el diccionario country_data.

    Calcula la media y la desviación estándar de cada columna en el diccionario country_data
    y estandariza los datos restando la media y dividiendo por la desviación estándar.

    Args:
        country_data (dict): Diccionario con los datos de los países.

    Returns:
        dict: Diccionario estandarizado con los datos de los países.
    """
    standardized_data = {}

    for country, values in country_data.items():
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        standardized_values = (values - mean) / std
        standardized_data[country] = standardized_values

    return standardized_data


