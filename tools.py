import networkx as nx
import os
import time
from collections import deque
import json
import random
import copy

random.seed(42)

def read_query_keywords(file_path):
    keywords = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉换行符并按冒号分割
            node, keywords_str = line.strip().split(':')
            node = int(node)  # 将顶点编号转换为整数
            keywords_list = [kw.strip() for kw in keywords_str.split(',')]  # 提取关键词列表
            keywords[node] = keywords_list
    return keywords

def read_query_edges(file_path):
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉换行符并按空格分割
            u, v = map(int, line.strip().split())
            edges.append((u, v))
    return edges

def generate_query(G:nx.Graph, n:int, p:float, keyword_domain):
    # subgraph
    
    subgraph = get_random_subgraph(G, n)
    query_graph = remove_edges(subgraph=subgraph, p=p)
    # query_graph = sample_keywords_subset(query_graph, keywords_domain=keyword_domain)
    return query_graph

def query_keywords_to_BV(G_q:nx.Graph, keywords_domain:str, m:int):
    num_keywords = len(keywords_domain)
    group_size = num_keywords // m
    remainder = num_keywords % m
    
    groups = [[] for _ in range(m)]
    index = 0
    for i in range(m):
        group_size_i = group_size + (1 if i < remainder else 0)
        groups[i] = keywords_domain[index:index + group_size_i]
        index += group_size_i

    keyword_to_index_per_group = [
        {keyword: index for index, keyword in enumerate(group)} for group in groups
    ]

    # print("keyword_to_index_per_group:{}".format(keyword_to_index_per_group))

    for node in G_q.nodes():
        G_q.nodes[node]['Aux'] = {}
        bit_vectors = []
        keywords = G_q.nodes[node].get('keywords', []) # get访问，默认空
        for group_index in range(m):
            bv = 0
            for keyword in keywords:
                if keyword in keyword_to_index_per_group[group_index]:
                    index = keyword_to_index_per_group[group_index][keyword]
                    bv |= (1 << index)
            bit_vectors.append(bv)

        G_q.nodes[node]['Aux']['BV'] = bit_vectors
    
    return G_q

# def get_random_subgraph(G:nx.Graph, n):
#     nodes = list(G.nodes())
#     while True:
#         selected_nodes = random.sample(nodes, n)
#         subgraph = G.subgraph(selected_nodes)
#         if nx.is_connected(subgraph):
#             return subgraph

# def get_random_subgraph(G:nx.Graph, n:int):
#     while True:
#         start_node = random.choice(list(G.nodes()))
#         selected_nodes = {start_node} 

#         while len(selected_nodes) < n:
#             neighbors = set()
#             for node in selected_nodes:
#                 neighbors.update(G.neighbors(node))

#             neighbors -= selected_nodes

#             if not neighbors:
#                 print("Cannot expand the subgraph while maintaining connectivity.")
#                 break
            
#             new_node = random.choice(list(neighbors))
#             selected_nodes.add(new_node)
#         if len(selected_nodes) == n:
#             return nx.Graph(G.subgraph(selected_nodes))
#         else:
#             continue

import random
import networkx as nx

def get_random_subgraph(G: nx.Graph, n: int):
    while True:
        # 随机选择一个起始节点，且该节点的 keywords 不为 "0"
        valid_nodes = [node for node in G.nodes() if G.nodes[node].get('keywords', '1') != "0"]
        if not valid_nodes:
            raise ValueError("No valid starting node found (all nodes have keywords='0').")
        
        start_node = random.choice(valid_nodes)
        selected_nodes = {start_node}

        while len(selected_nodes) < n:
            neighbors = set()
            for node in selected_nodes:
                # 获取邻居节点，并排除 keywords 为 "0" 的节点
                neighbors.update(neighbor for neighbor in G.neighbors(node) if G.nodes[neighbor].get('keywords', '1') != "0")

            neighbors -= selected_nodes

            if not neighbors:
                print("Cannot expand the subgraph while maintaining connectivity.")
                break

            new_node = random.choice(list(neighbors))
            selected_nodes.add(new_node)

        if len(selected_nodes) == n:
            return nx.Graph(G.subgraph(selected_nodes))
        else:
            continue

def remove_edges(subgraph:nx.Graph, p=0.3):
    """
    随机裁剪子图的边，以概率 p 对每条边进行裁剪，同时保证子图的连通性。
    """
    edges = list(subgraph.edges())
    for edge in edges:
        if random.random() < p:
            subgraph.remove_edge(*edge)
            if not nx.is_connected(subgraph):
                subgraph.add_edge(*edge)
    return subgraph

def sample_keywords_subset(subgraph:nx.Graph, keywords_domain):
    """
    对子图中每个节点的 keywords 属性进行子集采样。
    对于每个节点，随机生成一个 m (1 <= m <= len(keywords)),
    然后从 keywords 中随机选择 m 个作为子集。
    """
    # k_domain = keywords_domain
    # k_domain.remove("wasBornIn")
    # k_domain.remove("hasGender")
    # k_domain.remove("playsFor")
    for node in subgraph.nodes():
        if 'keywords' in subgraph.nodes[node]:
            keywords = subgraph.nodes[node]['keywords']
            if len(keywords) > 3:
                sampled_keywords = random.sample(keywords, 3)
                subgraph.nodes[node]['keywords'] = sampled_keywords
            # if len(keywords) < 2:
            #     sampled_keywords = random.sample(k_domain, 2)
            #     subgraph.nodes[node]['keywords'] = sampled_keywords
            # elif len(keywords) <= 1:
            #     sampled_keywords = random.sample(keywords_domain, 2)
            #     subgraph.nodes[node]['keywords'] = sampled_keywords


            # if "wasBornIn" in keywords:
            #     subgraph.nodes[node]['keywords'].remove("wasBornIn")
            # if "hasGender" in keywords:
            #     subgraph.nodes[node]['keywords'].remove("hasGender")
            
            # if not subgraph.nodes[node]['keywords']:
            #     k_domain = keywords_domain
            #     k_domain.remove("wasBornIn")
            #     k_domain.remove("hasGender")
            #     sampled_keywords = random.sample(k_domain, 2)
            #     subgraph.nodes[node]['keywords'] = sampled_keywords
            # else:
            #     if len(subgraph.nodes[node]['keywords']) > 2:
            #         k_sub = subgraph.nodes[node]['keywords']
            #         sampled_keywords = random.sample(k_sub, 2)
            #         subgraph.nodes[node]['keywords'] = sampled_keywords

    return subgraph

def save_Graph(Graph:nx.Graph, dataset_name:str, is_Aux:bool) -> bool:
    folder_name = os.path.join(
        "dataset",
        "precompute",
        "real",
        dataset_name
    )
    create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)
    # G: 经过pre，没有aux；G+：经过offline，具备aux
    if is_Aux:
        nx.write_gml(Graph, 'G+-{}-{}.gml'.format(
                Graph.number_of_nodes(),
                Graph.number_of_edges(),))
        print(folder_name, 'G+-{}-{}.gml'.format(
                Graph.number_of_nodes(),
                Graph.number_of_edges(),), 'saved successfully!')
    else:
        nx.write_gml(Graph, 'G-{}-{}.gml'.format(
                Graph.number_of_nodes(),
                Graph.number_of_edges(),))
        print(folder_name, 'G-{}-{}.gml'.format(
                Graph.number_of_nodes(),
                Graph.number_of_edges(),), 'saved successfully!')
    os.chdir(initial_directory)

def save_Graph_synthetic(Graph:nx.Graph, dataset_name:str, distribution:str) -> bool:
    folder_name = os.path.join(
        "dataset",
        "precompute",
        "synthetic",
        dataset_name
    )
    create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)
    # G: 经过pre，没有aux；G+：经过offline，具备aux

    nx.write_gml(Graph, 'G+-{}.gml'.format(distribution))
    print(folder_name, 'G+-{}.gml'.format(distribution), 'saved successfully!')
    os.chdir(initial_directory)

def create_folder(folder_name: str) -> bool:
    # base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # folder_path = os.path.join(base_path, folder_name)
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_name)
    # return True
    folder_path = os.path.abspath(folder_name)
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"folder create sucess: {folder_path}")
            return True
        except Exception as e:
            print(f"folder create fail: {e}")
            return False
    else:
        print(f"folder alrealy exists: {folder_path}")
        return True


def info_file_save(info, dataset_name: str) -> bool:
    infor_path = "S3AND/Result/info"
    create_folder(folder_name=infor_path)
    base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    stat_file = 'information-' + dataset_name + "-" + time.strftime('%m%d-%H%M%S', time.localtime())+ '.txt'
    info.output_info_file_name = stat_file
    result_stat_file = open(os.path.join(base_path, infor_path, stat_file), 'w')
    result_stat_file.write(info.get_S3AND_result())
    result_stat_file.close()
    print(info.output_info_file_name, "saved successfully!")
    return True

def load_index_from_json(filename):
    with open(filename, 'r') as file:
        index = json.load(file)
    return index

def pruning_keyword_set(entry, query_bv, m) -> bool:
    node_bv = entry['Aux']['BV']
    for i in range(m):
        xor = int(node_bv[i]) & int(query_bv[i])
        if xor != int(query_bv[i]):
            return False
    return True

def pruning_LB_ND_phase_1(entry, N_q_j, q_size, sigma, sigmaAttr) -> bool:
    lb_ND = max(0, N_q_j-entry["Aux"]["nk"])
    if lb_ND > sigma:
            return False
    else:
        return True
    
    # if sigmaAttr == "MAX":
    #     if lb_ND > sigma:
    #         return False
    #     else:
    #         return True
    # elif sigmaAttr == "SUM":
    #     if lb_ND <= (sigma/q_size):
    #         return True # 注意这里是需要retrieval
    #     else: # 注意这里不是被pruning
    #         return False 

    return None

def pruning_LB_ND_phase_2(entry, N_q_j, q_size, sigma, sigmaAttr, m, N_q, G_q:nx.Graph):
    lb_ND = N_q_j
    for nq in N_q:
        flag = True
        for i in range(m):
            nq_bv = int(G_q.nodes[nq]["Aux"]["BV"][i])
            xor = nq_bv & int(entry["Aux"]["NBV"][i])
            if xor != nq_bv:
                flag = False
                break
        if flag:
            lb_ND = lb_ND - 1

    if lb_ND > sigma:
        return False
    else:
        return True

    # if sigmaAttr == "MAX":
    #     if lb_ND > sigma:
    #         return False
    #     else:
    #         return True
    # elif sigmaAttr == "SUM":
    #     if lb_ND <= (sigma/q_size):
    #         return True # 注意这里是需要retrieval
    #     else: # 注意这里不是被pruning
    #         return False 

    return None

def AND(q:nx.Graph, s:nx.Graph, tra_list:dict, sigmaAttr:str) -> float:
    # tra_list: 遍历顺序，按照节点匹配顺序而来
    # res = 0
    # for i in tra_list:
    #     neighbor_set_q = set(q.neighbors(q.nodes[i]))
    #     neighbor_set_s = set(s.neighbors(s.nodes[i]))
    #     if sigmaAttr == "SUM":
    #         res += len(neighbor_set_q-neighbor_set_s)
    #     elif sigmaAttr == "MAX":
    #         res = max(len(neighbor_set_q-neighbor_set_s), res)
    
    # 还可以考虑用字典dict，格式为 {u: u'}，表示 q 中的节点 u 匹配 s 中的节点 u'
    res = 0
    for u in q.nodes:
        neighbors_q = set(q.neighbors(u))
        u_prime = tra_list.get(u)
        if u_prime is None:
            raise ValueError(f"Node {u} in q cannot match one of in s")
        neighbors_s = set(s.neighbors(u_prime))
        diff = neighbors_q - neighbors_s
        if sigmaAttr == "SUM":
            res += len(diff)
        elif sigmaAttr == "MAX":
            res = max(len(diff), res)
    return res

from typing import List, Dict, Set, Tuple

def select_query_order_with_pivot(query_graph: nx.Graph, V_cand: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    根据候选集大小和邻居关系选择查询顶点的遍历顺序，并生成 pivot 顶点。

    参数:
        query_graph (nx.Graph): 查询图 \( q \)。
        V_cand (Dict[str, List[str]]): 每个查询顶点对应的候选顶点集合。

    返回:
        Tuple[List[str], List[str]]: 查询顶点的遍历顺序和对应的 pivot 顶点。
    """
    # 初始化有序列表 Q 和 pivot 列表
    Q = []
    pivot = []
    # 未处理的查询顶点集合
    unprocessed = set(query_graph.nodes())

    # 第一步：选择候选集最小的查询顶点作为起点
    start_vertex = min(unprocessed, key=lambda v: len(V_cand[v]))
    Q.append(start_vertex)
    pivot.append(None)  # 第一个顶点没有 pivot
    unprocessed.remove(start_vertex)

    # 第二步：迭代选择邻居
    while unprocessed:
        # 找到 Q 中顶点的所有未处理的邻居
        neighbors = set()
        for q in Q:
            neighbors.update(query_graph.neighbors(q))
        neighbors.intersection_update(unprocessed)

        # 选择候选集最小的邻居
        next_vertex = min(neighbors, key=lambda v: len(V_cand[v]))

        # 找到 next_vertex 在 Q 中的第一个邻居作为 pivot
        for q in Q:
            if query_graph.has_edge(q, next_vertex):
                pivot.append(q)
                break

        Q.append(next_vertex)
        unprocessed.remove(next_vertex)

    return Q, pivot

def select_query_order(query_graph: nx.Graph, V_cand: Dict[str, List[str]]) ->List[str]:
    # 初始化有序列表 Q 和 pivot 列表
    Q = []
    # 未处理的查询顶点集合
    unprocessed = set(query_graph.nodes())

    # 第一步：选择候选集最小的查询顶点作为起点
    start_vertex = min(unprocessed, key=lambda v: len(V_cand[v]))
    Q.append(start_vertex)
    unprocessed.remove(start_vertex)

    # 第二步：迭代选择邻居
    while unprocessed:
        # 找到 Q 中顶点的所有未处理的邻居
        neighbors = set()
        for q in Q:
            neighbors.update(query_graph.neighbors(q))
        neighbors.intersection_update(unprocessed)
        # 选择候选集最小的邻居
        next_vertex = min(neighbors, key=lambda v: len(V_cand[v]))
        Q.append(next_vertex)
        unprocessed.remove(next_vertex)

    return Q

def CSSaR(Graph:nx.Graph, V_cand:list, G_q:nx.Graph, node, 
          sigma:float, sigmaAttr:str, q_nodes:list, 
          q_nodes_bv:list, min_ind:int) -> list:
    # Candidate Subgraph Search and Aggregate
    entry = {}
    entry["P"] = node
    entry["U"] = [node]
    entry["Map"] = {q_nodes[min_ind]:node} # 与query vertices 一一映射的字典
    entry["I"] = [min_ind]
    # Vis = [node]
    s_cand = []
    stack = deque()
    stack.append(entry)

    while stack:
        pop_entry = stack.pop()
        if len(pop_entry["U"]) == len(q_nodes):
            s = Graph.subgraph(pop_entry["U"])
            AND_s_q = AND(q=G_q, s=s, tra_list=pop_entry["Map"], sigmaAttr=sigmaAttr)
            if AND_s_q <= sigma:
                s_cand.append(pop_entry["U"])
        elif len(pop_entry["U"]) < G_q.number_of_nodes():
            for v_l in Graph.neighbors(pop_entry["P"]):
                if v_l not in pop_entry["U"]: # 无candidate vertex重合
                    if sigmaAttr == "MAX":
                        for j in range(len(V_cand)):
                            if j not in pop_entry["I"]: # 无keyword重合
                                if v_l in V_cand[j]:
                                    v_l_entry = copy.deepcopy(pop_entry) # 需要深拷贝，pop是变量
                                    v_l_entry["P"] = v_l
                                    v_l_entry["U"].append(v_l)
                                    v_l_entry["I"].append(j)
                                    v_l_entry["Map"][q_nodes[j]] = v_l
                                    stack.append(v_l_entry)
        else:
            print("pop_entry[U] out of number")



        #     temp_pop_entry = copy.deepcopy(pop_entry)
        #     if i not in temp_pop_entry["I"]:
        #         if temp_pop_entry["P"] in V_cand[i]:
        #             temp_pop_entry["U"].append(temp_pop_entry["P"])
        #             temp_pop_entry["I"].append(i)
        #             temp_pop_entry["Map"][q_nodes[i]] = temp_pop_entry["P"]
        #             # 扩散
        #             if len(temp_pop_entry["U"]) == len(q_nodes):
        #                 s = Graph.subgraph(temp_pop_entry["U"])
        #                 AND_s_q = AND(q=G_q, s=s, tra_list=temp_pop_entry["Map"], sigmaAttr=sigmaAttr)
        #                 if AND_s_q <= sigma:
        #                     s_cand.append(temp_pop_entry["U"])
        #             else:
        #                 for v_l in Graph.neighbors(temp_pop_entry["P"]):
        #                     if sigmaAttr == "MAX":
        #                         if v_l not in Vis:
        #                             if any(v_l in row for row in V_cand):
        #                                 v_l_entry = {}
        #                                 v_l_entry["P"] = v_l
        #                                 v_l_entry["U"] = temp_pop_entry["U"]
        #                                 v_l_entry["I"] = temp_pop_entry["I"]
        #                                 v_l_entry["Map"] = temp_pop_entry["Map"]
        #                                 stack.append(v_l_entry)
        #                     elif sigmaAttr == "SUM":
        #                         if v_l not in Vis:
        #                             v_l_entry = Graph.nodes[v_l]
        #                             for j in range(len(q_nodes)):
        #                                 if j not in pop_entry["I"]:
        #                                     j_bv = q_nodes_bv[j]
        #                                     match_t = True
        #                                     pop_entry_bv = Graph.nodes[pop_entry["P"]]["Aux"]["BV"]
        #                                     for z in range(len(j_bv)):
        #                                         if j_bv[z] & pop_entry_bv[z] != j_bv[z]:
        #                                             match_t = False
        #                                             break
        #                                     if match_t:
        #                                         v_l_entry["U"] = pop_entry["U"]
        #                                         v_l_entry["I"] = pop_entry["I"]
        #                                         v_l_entry["Map"] = pop_entry["Map"]
        #                                         stack.append(v_l_entry)      
        #     Vis.append(pop_entry["P"])           
        # else:
        #     print("pop_entry[U] out of number")

    return s_cand


# from typing import List, Dict, Set

# def CCSandR(data_graph: nx.Graph, query_graph: nx.Graph, query_plan: List[Dict], subgraphs: Set[tuple], 
#             matching: List[int], depth: int, threshold: float) -> None:
#     """
#     Candidate Subgraph Search and Refinement (CCSandR) algorithm.

#     Parameters:
#         data_graph (nx.Graph): The data graph \(G\).
#         query_graph (nx.Graph): The query graph \(q\).
#         query_plan (List[Dict]): The query plan \(Q\), containing candidate vertices for each query vertex.
#         subgraphs (Set[tuple]): The set \(S\) of subgraphs that satisfy the AND constraints.
#         matching (List[int]): The current matching \(M\) of query vertices to data graph vertices.
#         depth (int): The current recursion depth \(n\).
#         threshold (float): The aggregate threshold \(\sigma_{\mathcal{f}}\).
#     """
#     # Base case: if all query vertices are matched
#     if depth == len(query_plan):
#         if AND_constraint(query_graph, matching) <= threshold:
#             subgraphs.add(tuple(matching))  # Add the matching to the subgraphs set
#         return

#     # Initialize candidate set for the current depth
#     candidate_set = set()

#     if depth == 0:
#         # For the first query vertex, add all candidate vertices
#         candidate_set.update(query_plan[depth]['V_cand'])
#     else:
#         # For subsequent query vertices, filter candidates based on edge constraints
#         for v in query_plan[depth]['V_cand']:
#             if v not in matching:
#                 valid = True
#                 for i in range(depth):
#                     # Check if edges exist between the candidate and already matched vertices
#                     if not data_graph.has_edge(matching[i], v):
#                         valid = False
#                         break
#                 if valid:
#                     candidate_set.add(v)

#     # Recursively search for valid subgraphs
#     for v in candidate_set:
#         matching[depth] = v  # Assign the candidate vertex to the current query vertex
#         CCSandR(data_graph, query_graph, query_plan, subgraphs, matching, depth + 1, threshold)


# def AND_constraint(query_graph: nx.Graph, matching: List[int]) -> float:
    """
    Compute the AND constraint value for the current matching.

    Parameters:
        query_graph (nx.Graph): The query graph \(q\).
        matching (List[int]): The current matching \(M\).

    Returns:
        float: The AND constraint value, which is the sum of the differences in neighbors.
    """
    total_difference = 0

    for u in range(len(matching)):
        if matching[u] is not None:
            # Get neighbors of query vertex u in the query graph
            query_neighbors = set(query_graph.neighbors(u))
            # Get neighbors of matched vertex in the matching M
            matched_neighbors = set()
            for i in range(len(matching)):
                if matching[i] is not None and i != u and query_graph.has_edge(u, i):
                    matched_neighbors.add(i)

            # Calculate the difference between query neighbors and matched neighbors
            difference = query_neighbors - matched_neighbors
            total_difference += len(difference)

    return total_difference