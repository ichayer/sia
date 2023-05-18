import numpy as np
import plotly.express as px
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
    principal_components = pca.fit_transform(data_standarized)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig = px.scatter(principal_components, x=0, y=1, text=countries, color=countries)
    fig.update_traces(textposition='top center')

    for i, label in enumerate(np.array(list(labels))):
        fig.add_shape(type='line', x0=0, y0=0, x1=loadings[i, 0], y1=loadings[i, 1])
        fig.add_annotation(x=loadings[i, 0], y=loadings[i, 1], ax=0, ay=0,
                           xanchor="center", yanchor="bottom", text=label)

    fig.update_xaxes(dict(title=f'PCA 1 - variance {pca.explained_variance_ratio_[0] * 100:.2f}%', ))
    fig.update_yaxes(dict(title=f'PCA 2 - variance {pca.explained_variance_ratio_[1] * 100:.2f}%'))
    fig.show()

def plot_biplot_with_sklearn2(data_standarized, countries, labels):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_standarized)

    # Represents the directions and magnitudes of the original variables in the transformed space (arrows).
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots()
    ax.scatter(principal_components[:, 0], principal_components[:, 1])

    for i, label in enumerate(labels):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.01, head_length=0.2, fc='r', ec='r')
        ax.annotate(label, (loadings[i, 0], loadings[i, 1]), xytext=(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2),
                    textcoords='offset points', ha='center', va='bottom')

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Biplot')

    # Add country names as labels for the data points
    for i, country in enumerate(countries):
        ax.text(principal_components[i, 0], principal_components[i, 1], country, ha='center', va='bottom')

    plt.show()

def plot_PCA1_barchart_with_sklearn(data_standarized, countries):
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(data_standarized)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(countries)), principal_components[:, 0], align='center')
    ax.set_xticks(np.arange(len(countries)))
    ax.set_xticklabels(countries, rotation='vertical')
    ax.set_xlabel('Countries')
    ax.set_ylabel('Principal Component 1')
    ax.set_title('PCA1 Bar Chart per countries')
    plt.tight_layout()
    plt.show()