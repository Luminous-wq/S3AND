import networkx as nx
from typing import List, Dict, Set, Tuple, Optional

def generate_bn(query_graph: nx.Graph, order: List[str], pivot: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Generate neighbor information for each query vertex.

    Parameters:
        query_graph (nx.Graph): The query graph \(q\).
        order (List[str]): The matching order.
        pivot (List[str]): The pivot vertices.

    Returns:
        bn (Dict[str, List[str]]): Neighbor information for each query vertex.
        bn_count (Dict[str, int]): Number of neighbors for each query vertex.
    """
    query_vertices = query_graph.nodes()
    bn_count = {v: 0 for v in query_vertices}
    bn = {v: [] for v in query_vertices}

    visited_vertices = set()
    visited_vertices.add(order[0])

    for i in range(1, len(order)):
        vertex = order[i]
        nbrs = list(query_graph.neighbors(vertex))
        for nbr in nbrs:
            if nbr in visited_vertices and nbr != pivot[i]:
                bn[vertex].append(nbr)
                bn_count[vertex] += 1
        visited_vertices.add(vertex)

    return bn, bn_count


def generate_valid_candidates(query_graph: nx.Graph, data_graph: nx.Graph, depth: int, embedding: Dict[str, str],
                              idx_count: Dict[str, int], valid_candidate: Dict[str, List[str]], visited_vertices: Set[str],
                              bn: Dict[str, List[str]], bn_count: Dict[str, int], order: List[str], pivot: List[str], query_plan) -> None:
    """
    Generate valid candidates for the current depth.

    Parameters:
        query_graph (nx.Graph): The query graph \(q\).
        data_graph (nx.Graph): The data graph \(G\).
        depth (int): The current recursion depth.
        embedding (Dict[str, str]): The current matching \(M\).
        idx_count (Dict[str, int]): The number of valid candidates for each depth.
        valid_candidate (Dict[str, List[str]]): The valid candidates for each depth.
        visited_vertices (Set[str]): Whether a vertex in the data graph has been visited.
        bn (Dict[str, List[str]]): Neighbor information for each query vertex.
        bn_count (Dict[str, int]): Number of neighbors for each query vertex.
        order (List[str]): The matching order.
        pivot (List[str]): The pivot vertices.
    """
    u = order[depth]
    p = embedding[pivot[depth]]
    # nbrs = list(data_graph.neighbors(p))

    idx_count[u] = 0
    valid_candidate[u] = []

    for v in query_plan[u]:
        if v not in embedding.values():  # Ensure v is not already in the current matching M
            if v not in visited_vertices:  # Ensure v is not visited
                valid = True
                for u_nbr in bn[u]:
                    u_nbr_v = embedding[u_nbr]
                    if not data_graph.has_edge(v, u_nbr_v):
                        valid = False
                        break
                if valid:
                    valid_candidate[u].append(v)
                    idx_count[u] += 1



def CCSandR(data_graph: nx.Graph, query_graph: nx.Graph, query_plan: Dict[str, List[str]], subgraphs: Set[tuple],
            matching: Dict[str, Optional[str]], depth: int, threshold: float, output_limit_num: int,
            bn: Dict[str, List[str]], bn_count: Dict[str, int], order: List[str], pivot: List[str], f:str, candidate_subgraphs:int) -> None:
    """
    Candidate Subgraph Search and Refinement (CCSandR) algorithm with optimizations.

    Parameters:
        data_graph (nx.Graph): The data graph \(G\).
        query_graph (nx.Graph): The query graph \(q\).
        query_plan (Dict[str, List[str]]): The query plan \(Q\), containing candidate vertices for each query vertex.
        subgraphs (Set[tuple]): The set \(S\) of subgraphs that satisfy the AND constraints.
        matching (Dict[str, Optional[str]]): The current matching \(M\) of query vertices to data graph vertices.
        depth (int): The current recursion depth \(n\).
        threshold (float): The aggregate threshold \(\sigma_{\mathcal{f}}\).
        output_limit_num (int): The maximum number of matches to find.
        bn (Dict[str, List[str]]): Neighbor information for each query vertex.
        bn_count (Dict[str, int]): Number of neighbors for each query vertex.
        order (List[str]): The matching order.
        pivot (List[str]): The pivot vertices.
    """
    if depth == len(order):
        candidate_subgraphs += 1
        if AND_constraint(query_graph, data_graph, matching, threshold,f):
            subgraphs.add(tuple(matching.items()))
            # if len(subgraphs) >= output_limit_num:
            #     return
        return

    u = order[depth]
    idx_count = {u: 0}
    valid_candidate = {u: []}
    visited_vertices = set(matching.values())  # Track visited vertices in the data graph

    if depth == 0:
        valid_candidate[u] = query_plan[u].copy()
        idx_count[u] = len(valid_candidate[u])
    else:
        generate_valid_candidates(query_graph, data_graph, depth, matching, idx_count, valid_candidate,
                                  visited_vertices, bn, bn_count, order, pivot, query_plan=query_plan)

    for v in valid_candidate[u]:
        matching[u] = v
        CCSandR(data_graph, query_graph, query_plan, subgraphs, matching, depth + 1, threshold, output_limit_num,
                bn, bn_count, order, pivot, f=f, candidate_subgraphs=candidate_subgraphs)
        # if len(subgraphs) >= output_limit_num:
        #     return
        matching[u] = None  # Backtrack


# def AND_constraint(query_graph: nx.Graph, matching: Dict[str, Optional[str]]) -> float:
#     """
#     Compute the AND constraint value for the current matching.

#     Parameters:
#         query_graph (nx.Graph): The query graph \(q\).
#         matching (Dict[str, Optional[str]]): The current matching \(M\).

#     Returns:
#         float: The AND constraint value.
#     """
#     total_difference = 0
#     for u in matching:
#         if matching[u] is not None:
#             query_neighbors = set(query_graph.neighbors(u))
#             matched_neighbors = set()
#             for v in matching:
#                 if matching[v] is not None and v != u and query_graph.has_edge(u, v):
#                     matched_neighbors.add(v)
#             difference = query_neighbors - matched_neighbors
#             total_difference += len(difference)
#     return total_difference


def AND_constraint(query_graph:nx.Graph, G:nx.Graph, matching:dict, sigma, f:str) -> bool:
    total_difference = 0
    subgraph = G.subgraph(list(matching.values()))
    if not nx.is_connected(subgraph):
        # print("no connected:{}".format(list(matching.values())))
        return False
    for q_node in matching:
        matching_neighbors = set(subgraph.neighbors(matching[q_node]))
        # print(matching[q_node], matching_neighbors)
        query_neighbors = set(query_graph.neighbors(q_node))
        query_neighbors_new_set = {matching[element] for element in query_neighbors if element in matching}
        # print(query_neighbors_new_set)
        difference = query_neighbors_new_set - matching_neighbors
        total_difference += len(difference)
        if f == "SUM": # SUM
            total_difference += len(difference)
        elif f == "MAX": # MAX
            total_difference = max(total_difference, len(difference))
        if total_difference / 2 > sigma:
            return False 
    return total_difference/2 <= sigma

# Example usage
if __name__ == "__main__":
    data_graph = nx.Graph()
    data_graph.add_edges_from([("Ben_Luz", "David_A._Christian"), ("male", "Roland_Schwegler"),
                               ("male", "Sebastián_Hernández"), ("male", "David_A._Christian")])
    # data_graph.add_node("Ben_Luz")
    data_graph.nodes["Ben_Luz"]["label"] = "person"
    data_graph.nodes["male"]["label"] = "gender"
    data_graph.nodes["Roland_Schwegler"]["label"] = "person"
    data_graph.nodes["Sebastián_Hernández"]["label"] = "person"
    data_graph.nodes["David_A._Christian"]["label"] = "person"

    # Define query graph
    query_graph = nx.Graph()
    query_graph.add_edges_from([("u1", "u2"), ("u2", "u3")])
    query_graph.nodes["u1"]["label"] = "person"
    query_graph.nodes["u2"]["label"] = "gender"
    query_graph.nodes["u3"]["label"] = "person"

    # Define query plan (candidate vertices for each query vertex)
    query_plan = {
        "u1": ["Roland_Schwegler", "Ben_Luz", "Sebastián_Hernández", "David_A._Christian"],  # Candidates for query vertex u1
        "u2": ["male"],     # Candidates for query vertex u2
        "u3": ["Roland_Schwegler",  "Ben_Luz", "Sebastián_Hernández", "David_A._Christian"]  # Candidates for u3
    }

    # Initialize variables
    subgraphs = set()
    matching = {u: None for u in query_plan}  # Initialize matching as a dictionary
    threshold = 1.0
    output_limit_num = 50

    # Generate matching order and pivot vertices
    order = ["u2", "u3", "u1"]
    pivot = ["u2", "u2", "u3"]

    # Generate neighbor information
    bn, bn_count = generate_bn(query_graph, order, pivot)

    # Run CCSandR algorithm
    CCSandR(data_graph, query_graph, query_plan, subgraphs, matching, 0, threshold, output_limit_num, bn, bn_count, order, pivot, f="MAX")

    # Print results
    print("Valid subgraphs:")
    for subgraph in subgraphs:
        print(subgraph)