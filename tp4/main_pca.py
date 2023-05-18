import copy
from statistics import mean, stdev

import pandas as pd

from tp4.pca_plots import *

def get_csv_data(file_path):
    csv = pd.read_csv(file_path)
    countries = csv.values[:, 0]
    labels = list(csv.columns)[1:]
    csv.set_index('Country', drop=True, inplace=True)
    data = csv.values
    return data, countries, labels

def standarize_data2(data):
   data_standarized = copy.deepcopy(data)
   for i in range(len(data[0])):
      aux = data_standarized[:, i]
      mean_aux = mean(aux)
      stdev_aux = stdev(aux)
      data_standarized[:, i] = (data_standarized[:, i] - mean_aux) / stdev_aux
   return data_standarized

def main():
    country_data, countries, labels = get_csv_data("./europe.csv")
    standardized_matrix = standarize_data2(country_data)

    # plot_biplot(standardized_matrix, countries, labels)
    plot_biplot_with_sklearn(standardized_matrix, countries, labels)
    plot_PCA1_barchart_with_sklearn(standardized_matrix, countries)

if __name__ == '__main__':
    main()
