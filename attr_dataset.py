from torch_geometric.datasets import AttributedGraphDataset, EllipticBitcoinDataset
from torch_geometric.datasets import Yelp, AmazonProducts, Planetoid
from numpy import *
import numpy as np
import networkx as nx
import os
from tools import create_folder
from collections import defaultdict

if __name__ ==  '__main__':
    save_name = "Ntweibo" # pubmed, facebook, Nel, Npubmed, Nfacebook
    # Nel
    # dataset = AttributedGraphDataset(root='./dataset',name='Facebook')
    dataset = AttributedGraphDataset(root='./dataset',name='TWeibo')
    # dataset = Yelp("./dataset/yelp")
    # dataset = AmazonProducts(root='./dataset/amazonP')
    # dataset = Planetoid(root='./dataset', name="PubMed")
    # dataset = EllipticBitcoinDataset(root='./dataset/EL')
    data = dataset._data
    key_num = []
    # print(data.x)
    # print(data.edge_index)

    G = nx.Graph()

    temp_data = data.edge_index[0]
    temp_data_2 = data.edge_index[1]
    print(temp_data)
    print(temp_data_2)
    l = len(data.edge_index[0])
    count = 0
    k = 20
    for i in range(l):
        
        v1 = temp_data[i]
        v2 = temp_data_2[i]
        # print(type(v1))
        G.add_edge(int(v1), int(v2))
        if 'keywords' not in G.nodes[int(v1)]:
            node_attributes = data.x[int(v1)]
            keywords = []
            if node_attributes.sum() > 0:
                # keywords = [str(i+1) for i, val in enumerate(node_attributes) if val != 0]
                if not isinstance(node_attributes, np.ndarray):
                    node_attributes = np.array(node_attributes)
                topk_indices = np.argsort(node_attributes)[-k:][::-1]
                # 生成 keywords
                keywords = [str(i+1) for i in topk_indices]
            else:
                keywords = "0"
            G.nodes[int(v1)]['keywords'] = keywords
        if 'keywords' not in G.nodes[int(v2)]:
            node_attributes = data.x[int(v2)] 
            keywords = []
            if node_attributes.sum() != 0:
                # keywords = [str(i+1) for i, val in enumerate(node_attributes) if val != 0]
                if not isinstance(node_attributes, np.ndarray):
                    node_attributes = np.array(node_attributes)
                topk_indices = np.argsort(node_attributes)[-k:][::-1]
                # 生成 keywords
                keywords = [str(i+1) for i in topk_indices]
            else:
                keywords = "0"
            G.nodes[int(v2)]['keywords'] = keywords
        count += 1
        if count % 10000 == 0:
            print("{} / {}".format(count, l))

    # keyword_freq = defaultdict(int)

    # for node_id in range(data.x.shape[0]):
    #     if node_id not in G:
    #         print(f"节点 {node_id} 不存在于图中。")
    #         continue
    #     node_attributes = data.x[node_id]
    #     keywords = []
    #     if node_attributes.sum() > 0:
    #         keywords = [str(i+1) for i, val in enumerate(node_attributes) if val != 0]
    #     else:
    #         keywords = ["0"]
    #     G.nodes[node_id]['keywords'] = keywords

    #     for keyword in keywords:
    #         keyword_freq[keyword] += 1
        
    # print("keywords frequency:", dict(keyword_freq))


    print(data.x[1])
    print(G.nodes[1]['keywords'])

    folder_name = os.path.join(
        "dataset",
        "precompute",
        "real",
        save_name
    )
    sucess_bool = create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)
    nx.write_gml(G, 'G-{}-{}.gml'.format(
            G.number_of_nodes(),
            G.number_of_edges(),))
    print(folder_name, 'G-{}-{}.gml'.format(
            G.number_of_nodes(),
            G.number_of_edges(),), 'saved successfully!')
    os.chdir(initial_directory)
    # si = len(data.x)
    # sj = len(data.x[0])
    # d = np.array(data.x)
    # for i in range(si):
    #     temp = 0
    #     temp = np.sum(d[i] == 1)
    #     key_num.append(temp)
    # # print(data.edge_index)
    # print(len(data.x), len(data.x[0]))
    # print(len(data.y))
    # print(mean(key_num))
    # print(max(key_num))
    # print(min(key_num))




