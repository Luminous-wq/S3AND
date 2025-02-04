import pandas as pd
import networkx as nx
import os
import time
from infomation import Info
import random
import json
import copy

def construct_index_balanced(Graph: nx.Graph, info: Info) -> list:
    nodes = list(Graph.nodes())
    k = info.partition_number
    print(k)
    start_time = time.time()
    root_node = root_tree(nodes_list=nodes, num_partition=k, level=0, Graph=Graph, info=info)
    print("Tree index is finished")
    print("Construct cost time:{}".format(time.time()-start_time))
    if len(info.dataset_name.split("-")) > 4:
        f_name = info.dataset_name.rsplit("-", 1)[0]
        d_name = info.dataset_name.rsplit("-", 1)[1]
        folder_name = os.path.join("dataset", "precompute", "synthetic", f_name)
        initial_directory = os.getcwd()
        os.chdir(folder_name)
        save_index_as_json(root_node, 'G+-'+d_name+'_index.json')
        os.chdir(initial_directory)
    else:
        folder_name = os.path.join("dataset", "precompute", "real", info.dataset_name)
        initial_directory = os.getcwd()
        os.chdir(folder_name)
        save_index_as_json(root_node, info.dataset_name+'_index.json')
        os.chdir(initial_directory)
    # print_index_tree(root_node)
    # level_counts = {}
    # count_nodes_per_level(root_node, level_counts, 0)
    # for level, count in level_counts.items():
    #     print(f"Level {level}: {count} nodes")

    return root_node, Graph

def root_tree(nodes_list: list, num_partition: int, level: int, Graph: nx.Graph, info: Info):
    # print("len nodes before if: {}, num : {}".format(len(nodes_list), num_partition))
    if len(nodes_list) <= num_partition:
        print("len nodes in if: {}, num : {}".format(len(nodes_list), num_partition))
        # leaf nodes are vertices
        return [
            {
                "P": node,
                "Aux": Graph.nodes[node]['Aux'],
                "T": True, # leaf node
                "L": level 

            } for node in nodes_list
        ]
        
        # leaf nodes are vertex set
        # aggregate_aux = {
        # "BV": [0] * info.group_number,
        # "NBV": [0] * info.group_number,
        # "nk": 0
        # }
        # for node in nodes_list:
        #     for group in range(info.group_number):
        #         aggregate_aux['BV'][group] = aggregate_aux['BV'][group] | Graph.nodes[node]['Aux']['BV'][group]
        #         aggregate_aux['NBV'][group] = aggregate_aux['NBV'][group] | Graph.nodes[node]['Aux']['NBV'][group]

        #     if aggregate_aux['nk'] < Graph.nodes[node]['Aux']['nk']:
        #         aggregate_aux['nk'] = Graph.nodes[node]['Aux']['nk']

        # return [{
        #     "P": nodes_list,
        #     "Aux": aggregate_aux,
        #     "T": True,
        #     "L": level
        # }]
    
    partition_ans, centers_bv, Graph1 = initialize_partition(Graph=Graph, nodes_list=nodes_list, 
                                                num_partition=info.partition_number, m=info.group_number)
    final_partition, _, Graph = cost_model(Graph=Graph1, partition=partition_ans, t=info.iteration_number, m=info.group_number,
                                            centers_bv=centers_bv, nodes_list=nodes_list, num_partition=info.partition_number,
                                            info=info)
    
    aggregated_child_entry_list = []

    aggregate_aux = {
        "BV": [0] * info.group_number,
        "NBV": [0] * info.group_number,
        "nk": 0
    }

    aggregated_synopsis = aggregate_aux
    # aggregated_synopsis.append(aggregate_aux)

    for i in range(1, num_partition+1):
        partition_node_array = final_partition[i]
        # if len(partition_node_array) < num_partition:
        #     child_entry_list = [
        #         {
        #             "P": node,
        #             "Aux": Graph.nodes[node]['Aux'],
        #             "T": True,
        #             "L": level + 1
        #         } for node in partition_node_array
        #     ]
        # else:
        child_entry_list = root_tree(partition_node_array, num_partition, level+1, Graph, info)

        for child_entry in child_entry_list:
            for group in range(info.group_number):
                # print(aggregated_synopsis)
                # print(aggregated_synopsis['BV'][group])
                # print(group)
                # print(child_entry)
                # print(child_entry['Aux'])
                aggregated_synopsis['BV'][group] = aggregated_synopsis['BV'][group] | int(child_entry['Aux']['BV'][group])
                aggregated_synopsis['NBV'][group] = aggregated_synopsis['NBV'][group] | int(child_entry['Aux']['NBV'][group])

            if aggregated_synopsis['nk'] < child_entry['Aux']['nk']:
                aggregated_synopsis['nk'] = child_entry['Aux']['nk']
        
        aggregated_child_entry_list.append(child_entry_list)

    return [
        {
            "P": child_entry_list,
            "Aux": aggregated_synopsis,
            "T": False,
            "L": level
        } for child_entry_list in aggregated_child_entry_list
    ]

def save_index_as_json(index, filename):
    with open(filename, 'w') as file:
        json.dump(index, file)


def load_index_from_json(filename):
    with open(filename, 'r') as file:
        index = json.load(file)
    return index


def print_index_tree(node, depth=0):
    if isinstance(node, list):
        for subnode in node:
            print_index_tree(subnode, depth)
    elif isinstance(node, dict):
        print("  " * depth + "Node:")
        print("  " * (depth + 1) + f"P: {node['P']}")
        print("  " * (depth + 1) + f"Aux: {node['Aux']}")
        print("  " * (depth + 1) + f"T: {node['T']}")
        if 'P' in node:
            print_index_tree(node['P'], depth + 1)


def count_nodes_per_level(root, level_counts=None, level=0):
    if level in level_counts:
        level_counts[level] += len(root)
    else:
        level_counts[level] = len(root)

    for entry in root:
        if not entry['T']:
            child_entry_list = entry['P']
            count_nodes_per_level(child_entry_list, level_counts, level + 1)

    return level_counts

def initialize_partition(Graph: nx.Graph, nodes_list: list, num_partition: int, m):
    # print("len nodes: {}".format(len(nodes_list)))
    centers = random.sample(nodes_list, num_partition)
    
    # TODO: 把center的标签换成数字，比如[a,b,c,d,e,f]或者数字[1,2,3,4,5]
    # 主要是为了不改变center，只改变它对应的keywords / m bit vectors

    mapping = {}
    for i, center in enumerate(centers, start=1):
        mapping[center] = i
    
    partition = {mapping[center]: [center] for center in centers} # {1: [xx, xxx]; 2: [xxx, xxxx]}

    centers_bv = {}
    for center in centers:
        centers_bv[mapping[center]] = Graph.nodes[center]['Aux'].get('BV', [])
        Graph.nodes[center]['partition'] = mapping[center]

    for node in nodes_list:
        if node in centers:
            continue
        min_distance = float('inf')
        best_center = None

        for center in centers:
            # balanced limit
            if len(partition[mapping[center]]) > len(nodes_list)/num_partition:
                continue
            total_distance = 0
            center_bv = centers_bv[mapping[center]]
            node_bv = Graph.nodes[node]['Aux'].get('BV', [])
            for group_index in range(m):
                # xor
                xor_result = int(center_bv[group_index]) ^ int(node_bv[group_index])
                distance = bin(xor_result).count('1')
                total_distance += distance
            if total_distance < min_distance:
                min_distance = total_distance
                best_center = mapping[center] # 这是number，center的编号
        partition[best_center].append(node)
        Graph.nodes[node]['partition'] = best_center
        if best_center == None:
            print("oh no: {}".format(node))
    # for center_index, nodes in partition.items():
    #     for node in nodes:
    #         Graph.nodes[node]['partition'] = center_index
    partition_sizes = {center_index: len(nodes) for center_index, nodes in partition.items()}
    # print(partition_sizes)

    return partition, centers_bv, Graph

def cost_model(Graph:nx.Graph, partition: dict, t: int, m: int, 
               centers_bv: dict, nodes_list: list, num_partition: int, info: Info):
    cost_score = cost(Graph=Graph, P=partition, m=m, centers_bv=centers_bv)
    # print("initilaization cost score: {}".format(cost_score))
    unchanged_rounds = 0
    max_unchanged_rounds = 5  # converge iterations
    group_size = info.keyword_domain // m
    remainder = info.keyword_domain % m
    keywords_domain = range(info.keyword_domain)
    groups = [[] for _ in range(m)]
    index = 0
    for i in range(m):
        group_size_i = group_size + (1 if i < remainder else 0)
        groups[i] = keywords_domain[index:index + group_size_i]
        index += group_size_i
    
    for _ in range(t):
        # update center bit vectors
        new_centers_bv = []
        cost_centers_bv = {}
        for center_index in partition: # each partition
            results = [] 
            # group_frequency_vectors = []
            for group_index in range(m): # each group
                result = 0 
                b = len(groups[group_index]) # group size: b
                for bit in range(b):
                    one_count = 0
                    for node in partition[center_index]:
                        node_bv = Graph.nodes[node]['Aux'].get('BV', [])
                        if (int(node_bv[group_index]) >> bit) & 1:
                            one_count += 1
                    if one_count >= len(partition[center_index]) / 2:
                        result |= 1 << bit
                
                results.append(result)
            new_centers_bv.append([center_index, results])
            cost_centers_bv[center_index] = results
        # print("new center with bv: {}".format(new_centers_bv))

        partition2 = {center_index: [] for center_index in partition}

        for node in nodes_list:
            min_distance = float('inf')
            best_center = None

            for center_index in partition:
                if len(partition2[center_index]) > len(nodes_list)/num_partition*(1+0.2):
                    continue
                total_distance = 0
                center_bv = cost_centers_bv[center_index]
                node_bv = Graph.nodes[node]['Aux'].get('BV', [])
                for group_index in range(m):
                    # xor
                    xor_result = int(center_bv[group_index]) ^ int(node_bv[group_index])
                    distance = bin(xor_result).count('1')
                    total_distance += distance
                if total_distance < min_distance:
                    min_distance = total_distance
                    best_center = center_index
            partition2[best_center].append(node)
            if best_center == None:
                print("oh no: {} in cost model".format(node))
        
        # partition2_sizes = {center_index: len(nodes) for center_index, nodes in partition2.items()}
        # print(partition2_sizes)

        if partition2 == partition:
            return partition, new_centers_bv, Graph
        else:
            new_cost = cost(Graph=Graph, P=partition2, m=m, centers_bv=cost_centers_bv)
            # print("new partitioning cost: {}".format(new_cost))
            if new_cost < cost_score:
                unchanged_rounds = 0
                partition = copy.deepcopy(partition2)
                cost_score = new_cost
                # print("new cost: {} is good ! we accept it".format(new_cost))
            else:
                # test = random.random()
                # if test <= 0.2: # beta
                #     partition = copy.deepcopy(partition2)
                #     cost_score = new_cost
                #     print("new cost: {} is pool! but we also accept it".format(new_cost))
                # else:
                unchanged_rounds += 1
                if unchanged_rounds >= max_unchanged_rounds:
                    break
                # print("no change, continue!")

    partition_sizes = {center: len(nodes) for center, nodes in partition.items()}
    # print(partition_sizes)
    return partition, new_centers_bv, Graph

def cost(Graph: nx.Graph, P: dict, m: int, centers_bv: dict):
    # intra-distance
    intra = []
    for center_index, nodes in P.items():
        total_distance = 0
        for node in nodes:
            node_bv = Graph.nodes[node]['Aux'].get('BV', [])
            center_bv = centers_bv[center_index]
            for group_index in range(m):
                # 异或操作
                xor_result = int(center_bv[group_index]) ^ int(node_bv[group_index])
                distance = bin(xor_result).count('1')
                total_distance += distance
        intra.append(total_distance)
    intra_score = sum(intra)

    # inter-distance
    inter = []
    num_partition = len(P)
    for i in range(1, num_partition+1):
        total_distance = 0
        center_bv_i = centers_bv[i]
        for j in range(i+1, num_partition+1):
            center_bv_j = centers_bv[j]
            for group_index in range(m):
                # 异或操作
                xor_result = int(center_bv_i[group_index]) ^ int(center_bv_j[group_index])
                distance = bin(xor_result).count('1')
                total_distance += distance
        inter.append(total_distance)
    inter_score = sum(inter)

    score = intra_score / (inter_score+1)
    return score