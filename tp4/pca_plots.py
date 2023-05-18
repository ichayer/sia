import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def plot_biplot(standardized_country_data, countries, labels):

    # Calculamos la matriz de correlaciones
    corr_matrix = np.corrcoef(standardized_country_data, rowvar=False)

    # Calculamos los autovalores y autovectores de la matriz
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

    # Ordenamos los autovectores de mayor a menor autovalor
    index = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[index]
    eigenvectors_sorted = eigenvectors[:, index]

    # Construimos la matriz R tomando los autovectores correspondientes a los mayores autovalores
    R = eigenvectors_sorted[:, :2] # Número de componentes principales deseados

    # Calculamos Y como combinacion lineal de las originales
    Y = np.dot(standardized_country_data, R)

    # Graficar los data points
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1])

    for i, country in enumerate(countries):
        plt.annotate(country, (Y[i, 0] * 1, Y[i, 1] * 1), color='b')

    # Graficar las caracteristicas o variables como flechas
    for i, label in enumerate(labels):
        plt.arrow(0, 0, R[i, 0], R[i, 1], color='r', alpha=0.5)
        plt.text(R[i, 0] * 1.15, R[i, 1] * 1.15, label, color='r')

    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Biplot de las dos primeras componentes principales')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_biplot_with_sklearn(data_standarized, countries, labels):
    pca = PCA()
    data_pca = pca.fit_transform(data_standarized)
    pca_components = pca.components_
    fig, ax = plt.subplots()

    # Graficar las proyecciones de los datos
    ax.scatter(data_pca[:, 0], data_pca[:, 1])

    # Graficar las direcciones de los vectores de carga
    scale_factor = 2
    text_scale_factor = scale_factor * 1.2
    for i, label in enumerate(labels):
        ax.arrow(0, 0, pca_components[0, i] * scale_factor, pca_components[1, i] * scale_factor, color='r', alpha=0.5, head_width=0.2, head_length=0.2)
        ax.text(pca_components[0, i] * text_scale_factor, pca_components[1, i] * text_scale_factor, label, color='r', ha='center', va='center', fontsize=8)

    # Etiquetar los países
    for country, x, y in zip(countries, data_pca[:, 0], data_pca[:, 1]):
        ax.text(x, y, country)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.title('Biplot de países con PCA')
    plt.grid(True)
    plt.show()

def plot_PCA1_barchart_with_sklearn(data_standarized, countries):
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(data_standarized)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(countries)), principal_components[:, 0], align='center')
    ax.set_xticks(np.arange(len(countries)))
    ax.set_xticklabels(countries, rotation='vertical')
    ax.set_xlabel('Countries')
    ax.set_ylabel('Componente principal 1')
    ax.set_title('PCA1 Bar Chart por pais')
    plt.tight_layout()
    plt.show()