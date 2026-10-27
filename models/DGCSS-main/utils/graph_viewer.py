import os

import matplotlib.pyplot as plt
import networkx as nx


def plot_weights(G, labels, folder_path=None, filename="example.jpg"):
    weights = [edata["weight"] for u, v, edata in G.edges(data=True)]

    pos = nx.spring_layout(G, seed=42)
    # Crie um mapa de cores com base nos pesos das arestas
    edge_color_map = plt.cm.get_cmap("Reds")  # Escolha o colormap desejado
    edge_colors = [edge_color_map(weight) for weight in weights]

    # Dicionario de plot dos vertices
    # dict_color = {0: 'red', 1: 'blue', 2: 'green'}

    # label_color_mapping = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "black"}

    # node_colors = []
    # for node_features in features:
    #     node_color = node_features.tolist().index(1)
    #     node_colors.append(label_color_mapping[node_color])

    # Desenhe o grafo com cores representando os pesos das arestas
    nx.draw(
        G,
        with_labels=False,
        edge_color=edge_colors,
        edge_cmap=edge_color_map,
        node_color=labels,
        font_weight="bold",
        arrows=False,
        width=3.0,
        pos=pos,
        node_size=100,
    )

    # Adicione uma barra de cores para representar os valores
    sm = plt.cm.ScalarMappable(
        cmap=edge_color_map, norm=plt.Normalize(vmin=min(weights), vmax=max(weights))
    )

    sm._A = []  # hack para evitar o bug do matplotlib
    plt.colorbar(sm, label="Pesos das Arestas", ax=plt.gca())

    if folder_path:
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        plt.savefig(folder_path + "/" + filename)
        plt.close()

    else:
        plt.show()
