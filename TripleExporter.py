import os
from typing import List

from os.path import isdir
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from urllib import parse


rdfs_uri = "http://www.w3.org/TR/rdf-schema/"


def data_to_graph(triples, file_name: str,
                  save: bool = False, show: bool = True):
    source = triples[0]
    relations = triples[1]
    target = triples[2]
    if isdir(file_name):
        file_name = file_name + os.sep + "plot.png"
    kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})

    # create a directed-graph from a dataframe
    G = nx.from_pandas_edgelist(kg_df, "source", "target",
                                edge_attr=True, create_using=nx.MultiDiGraph())

    # G = nx.from_pandas_edgelist(kg_df[kg_df['edge'] == "won"], "source", "target",
    #                            edge_attr=True, create_using=nx.MultiDiGraph())
    if show:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.5)  # k regulates the distance between nodes
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)
        plt.show()
    if save:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.5)  # k regulates the distance between nodes
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)
        plt.savefig(fname=file_name.replace("txt", "png"), format="png")


def data_to_n3(triples, file_name: str,
               uri_prefix: str = "https://www.disit.dinfo.it",
               uri_separator: str = "/"):
    if isdir(file_name):
        file_name = file_name + os.sep + "output" + os.sep + "triple.n3"
    file = open(file_name.replace("txt", "n3"), "w+")
    source = triples[0]
    relations = triples[1]
    target = triples[2]
    prefix = f"{uri_prefix}{uri_separator}"
    for i in range(len(source)):
        src = parse.quote(source[i])
        rel = parse.quote(relations[i])
        targ = parse.quote(target[i])
        # skip empty element
        if src == "" or rel == "" or targ == "":
            continue

        file.write(f"<{prefix}{src}> <{prefix}{rel}> <{prefix}{targ}> .\n")
        file.write(f"<{prefix}{src}> <{rdfs_uri}label> \"{source[i]}\" .\n")
        file.write(f"<{prefix}{rel}> <{rdfs_uri}label> \"{relations[i]}\" .\n")
        file.write(f"<{prefix}{targ}> <{rdfs_uri}label> \"{target[i]}\" .\n")