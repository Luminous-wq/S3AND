from infomation import Info
import networkx as nx
import heapq
import time
from tools import pruning_keyword_set, pruning_LB_ND_phase_1, pruning_LB_ND_phase_2, AND, select_query_order_with_pivot, select_query_order
from refinement import CCSandR, generate_bn
from refine_max import refine

class IndexEntry:
    def __init__(self, key, index_N, idx):
        self.key = key  # AND lower bound
        self.index_N = index_N
        self.idx = idx

    def __gt__(self, other):
        if self.index_N["L"] == other.index_N["L"]:
            return self.key > other.key
        else:
            return self.index_N["L"] < other.index_N["L"]

def S3AND(
        Graph: nx.Graph,
        G_q: nx.Graph,
        sigma: float,
        sigmaAttr: str,
        root: list,
        info: Info) -> list:
    
    # TODO: 优化
    # query bit vector with m
    initiaize_time = time.time()
    q_nodes = []
    q_neighbor = []
    q_neighbor_size = []
    q_nodes_bv = []

    for node in G_q.nodes():
        q_nodes.append(node)
        q_bv = G_q.nodes[node]['Aux']['BV']
        q_nodes_bv.append(q_bv)
        q_N = list(G_q.neighbors(node))
        q_neighbor.append(q_N)
        q_N_size = len(q_N)
        q_neighbor_size.append(q_N_size)
    q_size = len(q_nodes)
    # print("query node bit vectors: {}".format(q_nodes_bv))
    # print("query nodes: {}".format(q_nodes))
    # print("query node neighbor: {}".format(q_neighbor))
    # print("query node neighbor size: {}".format(q_neighbor_size))
    print("****************************************************")
    info.initialization_time += (time.time()-initiaize_time)

    V_cand = {q_node:[] for q_node in q_nodes}
    # S_cand = []
    # S = []
    H = []
    root_Q = q_nodes
    for idx, entry in enumerate(root):
        info.entry_node_visit_counter += 1
        entry["Q"] = []
        for q_j in root_Q:
            if pruning_keyword_set(entry=entry, query_bv=q_nodes_bv[q_nodes.index(q_j)], m=info.group_number):
                if pruning_LB_ND_phase_1(entry=entry, N_q_j=q_neighbor_size[q_nodes.index(q_j)], q_size=q_size, sigma=sigma, sigmaAttr=sigmaAttr):
                    if pruning_LB_ND_phase_2(entry=entry, N_q_j=q_neighbor_size[q_nodes.index(q_j)],
                                            q_size=q_size, sigma=sigma, sigmaAttr=sigmaAttr, m=info.group_number,
                                            N_q=q_neighbor[q_nodes.index(q_j)], G_q=G_q):
                        entry["Q"].append(q_j)
        if entry["Q"]:
            heapq.heappush(H, IndexEntry(key=entry["Aux"]["nk"], index_N=entry, idx=idx))
        else:
            info.entry_pruning_counter += 1

    while len(H) > 0:
        start_time = time.time()
        now_entry_H = heapq.heappop(H)
        now_entry = now_entry_H.index_N
        # print(now_entry["Q"])
        heapq.heapify(H)

        if now_entry["T"]:
            leaf_node_start_time = time.time()
            # the set, IND, of index matching
            v_bv = now_entry["Aux"].get('BV', [])
            IND = []
            for i in range(q_size):
                query_BV = q_nodes_bv[i]
                has_match = True
                for j in range(info.group_number):
                    if int(query_BV[j]) & int(v_bv[j]) != int(query_BV[j]):
                        has_match = False
                        break
                if has_match:
                    IND.append(i)
            for node_index in IND:
                # lb_ND pruning
                if pruning_keyword_set(entry=now_entry, query_bv=q_nodes_bv[node_index], m=info.group_number):
                    info.vertex_pruning_counter_BV += 1
                    if pruning_LB_ND_phase_1(entry=now_entry, N_q_j=q_neighbor_size[node_index],q_size=q_size,
                                                sigma=sigma, sigmaAttr=sigmaAttr):
                        info.vertex_pruning_counter_LB_ND_1 += 1
                        if pruning_LB_ND_phase_2(entry=now_entry, N_q_j=q_neighbor_size[node_index], q_size=q_size,
                                                    sigma=sigma, sigmaAttr=sigmaAttr, m=info.group_number, 
                                                    N_q=q_neighbor[node_index], G_q=G_q):
                            V_cand[q_nodes[node_index]].append(now_entry["P"])
                            info.vertex_visit_counter += 1
                            info.vertex_pruning_counter_LB_ND_2
                            # V_cand[q_nodes[node_index]].append(now_entry)
                        # else:
                        #     info.vertex_pruning_counter_LB_ND_2 += 1
                #     else:
                #         info.vertex_pruning_counter_LB_ND_1 += 1
                # else:
                #     info.vertex_pruning_counter_BV += 1

            info.leaf_node_traverse_time += (time.time()-leaf_node_start_time)
        else:
            info.entry_node_visit_counter += 1
            non_leaf_node_start_time = time.time()
            for child_entry in now_entry["P"]:
                child_entry["Q"] = []
                for q_j in now_entry["Q"]:
                    if pruning_keyword_set(entry=child_entry, query_bv=q_nodes_bv[q_nodes.index(q_j)], m=info.group_number):

                        if pruning_LB_ND_phase_1(entry=child_entry, N_q_j=q_neighbor_size[q_nodes.index(q_j)], 
                                q_size=q_size, sigma=sigma, sigmaAttr=sigmaAttr):
                            
                            if pruning_LB_ND_phase_2(entry=child_entry, N_q_j=q_neighbor_size[q_nodes.index(q_j)], 
                                    q_size=q_size, sigma=sigma, sigmaAttr=sigmaAttr, m=info.group_number,
                                    N_q=q_neighbor[q_nodes.index(q_j)], G_q=G_q):
                                
                                child_entry["Q"].append(q_j)
                if child_entry["Q"]:
                    heapq.heappush(H, IndexEntry(key=child_entry["Aux"]["nk"], index_N=child_entry, idx=None))
                else:
                    info.entry_pruning_counter += 1
            info.non_leaf_node_traverse_time += (time.time()-non_leaf_node_start_time)

    cand_lengths = {key: len(value) for key, value in V_cand.items()}
    print("vertex candidate length: {}".format(cand_lengths))
    pw = (int(info.nodes_num)-max(cand_lengths.values()))/int(info.nodes_num)
    print("pruning power: {}".format(pw))
    # return [pw]
    # if min(cand_lengths.values()) > 1000:
    #     return []
    # order = select_query_order(query_graph=G_q, V_cand=V_cand)
    order, pivot = select_query_order_with_pivot(query_graph=G_q, V_cand=V_cand)

    # Initialize variables
    S = set()
    matching = {u: None for u in V_cand}  # Initialize matching as a dictionary
    threshold = info.thresholds_max
    output_limit_num = 5000

    bn, bn_count = generate_bn(G_q, order, pivot)
    refine_time = time.time()
    candidate_subgraphs = 0
    CCSandR(Graph, G_q, V_cand, S, matching, 0, threshold, output_limit_num, bn, bn_count, order, 
            pivot, f=info.function, candidate_subgraphs=candidate_subgraphs)
    # S = refine(G=Graph, q=G_q, Q=V_cand, sigma=sigma, order=order, f=info.function)
    info.refine_time = (time.time()-refine_time)
    # Print results
    # print("Valid subgraphs:")
    # for subgraph in S:
    #     print(subgraph)
    print(len(S))
    print("candidate_subgraphs:{}".format(candidate_subgraphs))
    # t = min(
    #     enumerate(V_cand),
    #     key = lambda x : len(x[1])
    # )[0]
    # # print(enumerate(V_cand), t)
    # for node in V_cand[t]:
    #     # aggregate and refine candidate subgraphs
    #     s_cand = CSSaR(Graph=Graph, V_cand=V_cand, G_q=G_q, node=node, sigma=sigma,
    #                    sigmaAttr=sigmaAttr, q_nodes=q_nodes, q_nodes_bv=q_nodes_bv, min_ind=t)
    #     for s in s_cand:
    #         S.append(s)
    return S

