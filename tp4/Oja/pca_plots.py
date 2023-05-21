import numpy as np
from adjustText import adjust_text
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def plot_biplot(standardized_country_data, countries, labels):
    """
    Plotea un biplot de las dos primeras componentes principales de los datos sin usar sklearn.
    """

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
    """
    Plotea un biplot de las dos primeras componentes principales de los datos usando sklearn.
    """

    pca = PCA()
    data_pca = pca.fit_transform(data_standarized)
    pca_components = pca.components_
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    # Graficar las proyecciones de los datos
    ax.scatter(data_pca[:, 0], data_pca[:, 1])

    # Graficar las direcciones de los vectores de carga
    scale_factor = 2
    text_scale_factor = scale_factor * 1.2
    for i, label in enumerate(labels):
        ax.arrow(0, 0, pca_components[0, i] * scale_factor, pca_components[1, i] * scale_factor, color='r', alpha=0.5, head_width=0.12, head_length=0.08)
        ax.text(pca_components[0, i] * text_scale_factor, pca_components[1, i] * text_scale_factor, label, color='r', ha='center', va='center', fontsize=11)

    # Etiquetar los países con adjust_text
    texts = []
    for country, x, y in zip(countries, data_pca[:, 0], data_pca[:, 1]):
        texts.append(ax.text(x, y, country, color='b', fontsize=11))

    # Ajustar automáticamente la posición de las etiquetas para evitar superposiciones
    adjust_text(texts)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.title('Biplot de países con PCA')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_PCA1_barchart_with_sklearn(data_standarized, countries, labels):
    """
    Plotea un bar chart de la primera componente principal de los datos usando sklearn.
    """

    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(data_standarized)

    print("Printing eigenvalues for the first component:")
    print(pca.components_[0])
    print("Printing labels for the first component:")
    print(labels)

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(countries)), principal_components[:, 0], align='center')
    ax.set_xticks(np.arange(len(countries)))
    ax.set_xticklabels(countries, rotation='vertical')
    ax.set_xlabel('Countries')
    ax.set_ylabel('Componente principal 1')
    ax.set_title('PCA1 Bar Chart por pais')
    plt.tight_layout()
    plt.show()

def plot_boxplot(data, box_plot_title, labels):
    """
    Plotea un boxplot de los datos.
    """
    plt.title(box_plot_title)
    plt.boxplot(data, labels=labels, widths=0.5, boxprops=dict(color='black'), whiskerprops=dict(color='black'), medianprops=dict(color='red', linewidth=2))
    plt.xticks(fontsize=8, horizontalalignment='center')
    plt.tight_layout()
    plt.show()