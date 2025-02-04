import numpy as np
from argparser import args_parser
from infomation import Info
from tools import info_file_save, load_index_from_json, read_query_keywords, read_query_edges, generate_query, query_keywords_to_BV
import time
import random
from offline import read_keywords, aux_graph
import networkx as nx
# from construct_index import construct_index_balanced, initialize_partition, cost_model
from c_index import construct_index_balanced, initialize_partition, cost_model
from online import S3AND
import logging
random.seed(42)
np.random.seed(42)


def make_seed(seed):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    # make_seed(2025)
    args = args_parser()
    info = Info(
        input_file_folder=args.input,
        dataset_name=args.dataset,
        query_graph_edges=args.queryEdges,
        query_graph_size=args.querySize,
        query_graph_keywords=args.queryKeywords,
        keyword_domain=args.keywordDomain,
        group_number=args.groupNumber,
        partition_number=args.partitionNumber,
        iteration_number=args.iterationNumber,
        thresholds_sum=args.thresholdSUM,
        thresholds_max=args.thresholdMAX,
        F=args.aggregateF
    )
    info.start_time = time.time()
    print(args)
    logging.info("This is an info message")
    logging.info(args)
    #-------------- Data Graph -----------------#
    G = nx.read_gml(path=info.input_file_name)
    print("Data Graph: {}".format(G))

    #-------------- Query Graph -----------------#
    '''
    用文件生成
    '''
    # keywords = read_query_keywords(info.query_graph_keywords)
    # edges = read_query_edges(info.query_graph_edges)
    # G_q = nx.Graph()
    # for node, node_keywords in keywords.items():
    #     G_q.add_node(node, keywords=list(node_keywords))
    # G_q.add_edges_from(edges)
    # print("Query Graph: {}".format(G_q))
    '''
    子图采样生成
    n in [3, 5, 8, 10]
    '''
    # YAGO3
    # keywords_domain_path = "./dataset/precompute/real/YAGO3/keywords_domain.txt"
    # keywords_domains = read_keywords(keywords_domain_path)

    # real
    # keywords_domains_list = range(info.keyword_domain)
    # keywords_domains = [str(i) for i in keywords_domains_list]

    # synthetic
    keywords_domains = range(info.keyword_domain)

    #-------------- Index -----------------#
    time2 = time.time()
    root, G = construct_index_balanced(Graph=G, info=info)
    info.index_time = time.time()-time2
    
    index_root = load_index_from_json(info.index_file_name)
    print("Index loading is complete.\nStart online query.")
    

    G_forq = nx.read_gml("./dataset/precompute/synthetic/50000-124812-50-3/G+-uni.gml")
    iter_test = 100
    avg_time = 0
    
    pw_ans = 0
    for i in range(iter_test):
        online_start = time.time()
        G_q = generate_query(G=G_forq, n=info.query_graph_size, p=0.3, keyword_domain=keywords_domains)
        # print("Query Graph: {}".format(G_q))
        G_q = query_keywords_to_BV(G_q=G_q, keywords_domain=keywords_domains, m=info.group_number)
        print("Query Graph Nodes:", G_q.nodes())
        print("Query Graph Edges:", G_q.edges())
        for node in G_q.nodes():
            print(f"Node {node} Keywords:", G_q.nodes[node]['keywords'])
            print(f"Node {node} BV:", G_q.nodes[node]["Aux"]["BV"])
        

        #-------------- Online Query -----------------#
        
        S = S3AND(Graph=G, G_q=G_q, sigma=info.thresholds_max, sigmaAttr=info.function,
                root=index_root, info=info)
        # print(len(S))
        info.online_time = time.time()-online_start
        avg_time += info.online_time
        S = list(S)
        if len(S) > 100:
            info.S3AND_result = S[:100]
        else:
            info.S3AND_result = S
        info.finish_time = time.time()
        print(info.online_time)
        # pw_ans += S[0]
    print("online average time:{}, iter. number: {}".format(avg_time/iter_test, iter_test))
    print("index time: {}".format(info.index_time))
    # print("avg pruning power {}".format(pw_ans/iter_test))
    info_file_save(info, args.dataset)