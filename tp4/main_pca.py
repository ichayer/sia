from tp4.pca_plots import *
from tp4.tools import csv_to_dict, standardize_data

def main():
    countries, labels, country_data = csv_to_dict("./europe.csv")
    standardized_country_data = standardize_data(country_data)

    labels.pop(0) # Drop country label

    standardized_matrix = np.array(list(standardized_country_data.values()), dtype=np.float64)
    not_standardized_matrix = np.array(list(country_data.values()), dtype=np.float64)

    # plot_biplot(standardized_matrix, countries, labels)
    plot_biplot_with_sklearn(standardized_matrix, countries, labels)
    plot_PCA1_barchart_with_sklearn(standardized_matrix, countries)
    plot_boxplot(standardized_matrix, 'Boxplot con datos estandarizados', labels)
    plot_boxplot(not_standardized_matrix, 'Boxplot con datos no estandarizados', labels)

if __name__ == '__main__':
    main()
