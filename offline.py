import networkx as nx
from tools import save_Graph, save_Graph_synthetic
import time

def aux_graph(path:str, m:int, keywords_domain:str, dataset_name:str, distribution="uni") -> nx.Graph:
    Graph = nx.read_gml(path=path)
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

    for node in Graph.nodes():
        Graph.nodes[node]['Aux'] = {}
        bit_vectors = []
        keywords = Graph.nodes[node].get('keywords', []) # get访问，默认空
        for group_index in range(m):
            bv = 0
            for keyword in keywords:
                if keyword in keyword_to_index_per_group[group_index]:
                    index = keyword_to_index_per_group[group_index][keyword]
                    bv |= (1 << index)
            bit_vectors.append(bv)

        Graph.nodes[node]['Aux']['BV'] = bit_vectors
        # print("node:{}; keywords:{}; bit vector:{}".format(node, keywords, bit_vectors))
        # print("keyword_to_index_per_group:{}".format(keyword_to_index_per_group))

    for node in Graph.nodes():
        neighbor_bit_vectors = []
        neighbors = list(Graph.neighbors(node))

        for group_index in range(m):
            group_aggregate_bv = 0
            for neighbor in neighbors:
                neighbor_bv = Graph.nodes[neighbor]['Aux'].get('BV', [])[group_index]
                group_aggregate_bv |= neighbor_bv
            neighbor_bit_vectors.append(group_aggregate_bv)
        Graph.nodes[node]['Aux']['NBV'] = neighbor_bit_vectors
    
        distinct_keywords_count = 0
        for group_aggregate_bv in neighbor_bit_vectors:
            binary_str = bin(group_aggregate_bv)[2:]
            count = binary_str.count('1')
            # distinct_keywords_count.append(count)
            distinct_keywords_count += count
        Graph.nodes[node]['Aux']['nk'] = distinct_keywords_count

        # distinct_keywords_count = 0
        # for group_index in range(m):
        #     group_keywords = set()
        #     for neighbor in neighbors:
        #         neighbor_keywords = Graph.nodes[neighbor].get('BV', [])
        #         relevant_neighbor_keywords = [k for k in neighbor_keywords if k in keyword_to_index_per_group[group_index]]
        #         group_keywords.update(relevant_neighbor_keywords)
        #     distinct_keywords_count += len(group_keywords)
        #     # 计算聚合位向量
        #     bv = 0
        #     for keyword in group_keywords:
        #         index = keyword_to_index_per_group[group_index][keyword]
        #         bv |= (1 << index)
        #     neighbor_bit_vectors.append(bv)
        # Graph.nodes[node]['NBV'] = neighbor_bit_vectors
        # Graph.nodes[node]['nk'] = distinct_keywords_count

    # if not synthetic
    # save_Graph(Graph=Graph, dataset_name=dataset_name, is_Aux=True)


    save_Graph_synthetic(Graph=Graph, dataset_name=dataset_name,distribution=distribution)
    return Graph


def read_keywords(path):
    keywords_domains = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            keyword = line.split()[0]
            keywords_domains.append(keyword)
    
    return keywords_domains

if __name__=="__main__":
    # ----------------------YAGO3-----------------------
    # path = "./dataset/precompute/real/YAGO3/G-123182-778561.gml"
    # m = 3
    # keywords_domain_path = "./dataset/precompute/real/YAGO3/keywords_domain.txt"

    # keywords_domains = read_keywords(keywords_domain_path)
    # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="YAGO3")
    # print(time.time()-time1)
    # # for node in graph.nodes():
    # #     print(f"Node {node}: bit_vectors = {graph.nodes[node]['BV']}")

    # ----------------------PubMed-----------------------
    # path = "./dataset/precompute/real/Npubmed/G-19717-44324.gml"
    # m = 5
    # keywords_domains_list = range(501)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="Npubmed")
    # print(time.time()-time1)

    # ----------------------Facebook-----------------------
    # path = "./dataset/precompute/real/Nfacebook/G-4039-88234.gml"
    # m = 5
    # keywords_domains_list = range(1284)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="Nfacebook")
    # print(time.time()-time1)

    # ----------------------Elliptc-----------------------
    # path = "./dataset/precompute/real/Nel/G-203769-234355.gml"
    # m = 5
    # keywords_domains_list = range(166)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="Nel")
    # print(time.time()-time1)

    # ----------------------Tweibo-----------------------
    # path = "./dataset/precompute/real/Ntweibo/G-1944589-50133382.gml"
    # m = 5
    # keywords_domains_list = range(1658)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="Ntweibo")
    # print(time.time()-time1)

    # ----------------------DBLP-----------------------
    # path = "./dataset/precompute/real/dblpV14/G-2956012-29560065.gml"
    # m = 5
    # keywords_domains_list = range(7990611)
    # keywords_domains = [str(i) for i in keywords_domains_list]
    # # print(keywords_domains)

    # time1 = time.time()
    # graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="dblpV14")
    # print(time.time()-time1)

    # ----------------------Synthetic-----------------------
    path = "./dataset/precompute/synthetic/50000-124812-50-3/G-uni.gml"
    path = "./dataset/precompute/synthetic/50000-124812-50-3/G-uni_2.gml"
    # path = "./dataset/precompute/synthetic/10000-24979-50-3/G-uni.gml"
    # path = "./dataset/precompute/synthetic/25000-62440-50-3/G-uni.gml"
    # path = "./dataset/precompute/synthetic/100000-249807-50-3/G-uni.gml"
    # path = "./dataset/precompute/synthetic/250000-624992-50-3/G-uni.gml"
    m = 5
    keywords_domains = range(50)
    # keywords_domains = [i for i in keywords_domains_list]
    print(keywords_domains)
    time1 = time.time()
    graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="50000-124812-50-2", distribution="uni")
    print(time.time()-time1)

    # path = "./dataset/precompute/synthetic/50000-124812-50-3/G-gau.gml"
    # path = "./dataset/precompute/synthetic/10000-24979-50-3/G-gau.gml"
    # path = "./dataset/precompute/synthetic/25000-62440-50-3/G-gau.gml"
    # path = "./dataset/precompute/synthetic/100000-249807-50-3/G-gau.gml"
    # path = "./dataset/precompute/synthetic/250000-624992-50-3/G-gau.gml"
    # m = 5
    # keywords_domains = range(50)
    # # keywords_domains = [i for i in keywords_domains_list]
    # print(keywords_domains)
    # time1 = time.time()
    # graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="250000-624992-50-3", distribution="gau")
    # print(time.time()-time1)

    # # path = "./dataset/precompute/synthetic/50000-124812-50-3/G-zipf.gml"
    # # path = "./dataset/precompute/synthetic/10000-24979-50-3/G-zipf.gml"
    # # path = "./dataset/precompute/synthetic/25000-62440-50-3/G-zipf.gml"
    # # path = "./dataset/precompute/synthetic/100000-249807-50-3/G-zipf.gml"
    # path = "./dataset/precompute/synthetic/250000-624992-50-3/G-zipf.gml"
    # m = 5
    # keywords_domains = range(50)
    # # keywords_domains = [i for i in keywords_domains_list]
    # print(keywords_domains)
    # time1 = time.time()
    # graph = aux_graph(path=path, m=m, keywords_domain=keywords_domains, dataset_name="250000-624992-50-3", distribution="zipf")
    # print(time.time()-time1)

