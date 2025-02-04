import copy
import os
import random
import networkx as nx
import numpy as np
from tools import create_folder
from scipy.stats import zipfian

# generate synthetic graph

def generate_dataset(seed: int, keywords_per_vertex_num: int, all_keyword_num: int,
                      node_num: int, neighbor_num: int, add_edge_probability: float,
                      distribution:str):
    random.seed(seed)
    np.random.seed(seed)
    # 1. Generate a graph
    target_graph = nx.newman_watts_strogatz_graph(n=node_num, k=neighbor_num, p=add_edge_probability)
    # 1.1. Make sure the graph is connected
    while not nx.is_connected(target_graph):
        target_graph = nx.newman_watts_strogatz_graph(node_num, neighbor_num, add_edge_probability)
    # 2. Compute infos of graph
    # edges_num = target_graph.number_of_edges()
    average_degree = sum(
        d for v, d in nx.degree(target_graph)) / target_graph.number_of_nodes()
    print(target_graph)
    print('Average Degree: ', average_degree)
    print('Max Degree: ', max(d for v, d in nx.degree(target_graph)))
    print('Min Degree: ', min(d for v, d in nx.degree(target_graph)))

    # 2.1. Delete some edges until edge num equals node_num * neighbor_num / 2
    while target_graph.number_of_edges() > node_num * neighbor_num:
        (u, v) = random.sample(target_graph.edges, 1)[0]
        G = copy.deepcopy(target_graph)
        G.remove_edge(u, v)
        if nx.is_connected(G):
            target_graph.remove_edge(u, v)

    # Print infos
    # edges_num = target_graph.number_of_edges()
    average_degree = sum(
        d for v, d in nx.degree(target_graph)) / target_graph.number_of_nodes()
    print(target_graph)
    print('Average Degree: ', average_degree)
    print('Max Degree: ', max(d for v, d in nx.degree(target_graph)))
    print('Min Degree: ', min(d for v, d in nx.degree(target_graph)))

    # 2.2. Add the keyword set to each vertex
    if distribution == "zipf":
        # Zipf
        a = 0.8  # param
        size = target_graph.number_of_nodes()
        keywords_set = zipfian.rvs(a,size*keywords_per_vertex_num)
        
        # keywords_set = zipf_dist.rvs(size*keywords_per_vertex_num)
    elif distribution == "gau":
        # Gaussian
        # mean = 10  # 均值
        # stddev = 6  # 标准差
        if all_keyword_num == 10:
            mean = 5  # 均值
            stddev = 3  # 标准差
        elif all_keyword_num == 20:
            mean = 10  # 均值
            stddev = 6  # 标准差
        elif all_keyword_num == 50:
            mean = 25  # 均值
            stddev = 15  # 标准差
        elif all_keyword_num == 80:
            mean = 40  # 均值
            stddev = 24  # 标准差
        size = target_graph.number_of_nodes()
        keywords_set = np.random.normal(mean, stddev, size)
    elif distribution == "uni":
        size = target_graph.number_of_nodes()
        keywords_set = np.random.uniform(0, all_keyword_num, size=size).astype(int)
    print("ok")
    keywords_set = np.clip(keywords_set, 0, all_keyword_num-1).astype(int)

    label_counter = [0 for _ in range(all_keyword_num)]
    for i, node in enumerate(target_graph):
        keyword_num = np.random.randint(max(keywords_per_vertex_num - 1, 1),
                                        keywords_per_vertex_num + 2)  # the num is key_per ± 1
        # print(len(set(keywords_set)), len(keywords_set))
        # unique_keywords = np.unique(keywords_set)
        # keywords_set =unique_keywords
        keywords = np.random.choice(keywords_set, keyword_num)
        while len(set(keywords)) != len(keywords):
            # print(len(set(keywords)), len(keywords))
            keywords = np.random.choice(keywords_set, keyword_num)
            # print(f"kale:{i}")
        for keyword in keywords:
            label_counter[keyword] += 1
        keywords_int = [int(x) for x in keywords]
        target_graph.nodes[node]['keywords'] = keywords_int

    print([{i: label_counter[i]} for i in range(all_keyword_num)])

    folder_name = os.path.join(
        "./dataset",
        "precompute",
        "synthetic",
        "{}-{}-{}-{}".format(
            target_graph.number_of_nodes(),
            target_graph.number_of_edges(),
            all_keyword_num,
            keywords_per_vertex_num
        )
    )

    create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)
    nx.write_gml(target_graph, 'G-'+str(distribution)+'.gml')
    print(folder_name, 'G-'+str(distribution)+'.gml', 'saved successfully!')
    os.chdir(initial_directory)

if __name__ == "__main__":
    # Set the parameters to generate dataset.
    seed = 2025
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=10000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=10000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="gau"
    )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=10000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=25000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=25000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="gau"
    )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=25000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=50000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=50000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="gau"
    )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=50000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )


    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=100000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=100000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="gau"
    )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=100000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=250000,  # 10K, 25K, 50K, 100K, 250K
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=250000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="gau"
    )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=250000,  # 10K, 25K, 50K, 100K, 250K
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )