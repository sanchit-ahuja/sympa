import os
import pickle

import numpy as np


# convert an adjcaency matrix to a list of edges
def get_dense_list(adj_matrix):
    # get the number of nodes
    num_nodes = adj_matrix.shape[0]
    # get the number of edges
    num_edges = np.sum(adj_matrix)
    # get the list of edges
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                edge_list.append((i, j, 1))
    # return the list of edges
    return edge_list


def main():
    # open pheme_dgl_full_roberta.pkl file
    with open("pheme_dgl_full_roberta.pkl", "rb") as f:
        # load the pickle file
        data = pickle.load(f)
    list_file_name = "graph_list"
    file_dir = "data"
    for i, ds in enumerate(data):
        print("Processing dataset {}".format(i))
        # print(ds)
        adj_matrix = ds.adj(scipy_fmt="csr")
        adj_matrix = adj_matrix.toarray()
        edge_list = get_dense_list(adj_matrix)
        # Write edge_list to a file
        # create a directory of list_file_name+str(i)
        if not os.path.exists(file_dir + "/" + list_file_name + str(i)):
            os.makedirs(file_dir + "/" + list_file_name + str(i))
        # write the edge_list to a file
        with open(
            file_dir
            + "/"
            + list_file_name
            + str(i)
            + "/"
            + list_file_name
            + str(i)
            + ".edges",
            "w",
        ) as f:
            for edge in edge_list:
                f.write(str(edge[0]) + " " + str(edge[1]) + " " + str(edge[2]) + "\n")


if __name__ == "__main__":
    # load the adjacency matrix
    # adj_matrix = np.load('../data/adj_matrix.npy')
    # get the list of edges
    main()
    # edge_list = get_dense_list(adj_matrix)
    # write the edge_list to a text file
