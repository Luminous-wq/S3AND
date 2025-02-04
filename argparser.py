import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, help="path of graph input file",
                        default="./dataset/precompute/real/dblpV14/G+-2956012-29560065.gml")
    parser.add_argument("-DS", "--dataset", type=str, 
                        help="the name of dataset", 
                        default="dblpV14")
    parser.add_argument("-qe", "--queryEdges", type=str, 
                        help="the query edges file", 
                        default="./dataset/precompute/real/YAGO3/query_edges.txt")
    parser.add_argument("-qs", "--querySize", type=int, 
                        help="the query vertex set size", 
                        default=5)
    parser.add_argument("-qk", "--queryKeywords", type=str, 
                        help="the query vertex keywords file", 
                        default="./dataset/precompute/real/YAGO3/query_keywords.txt")
    parser.add_argument("-s", "--keywordDomain", type=int, 
                        help="the keyword domain size", 
                        default=7990611)
    parser.add_argument("-m", "--groupNumber", type=int, 
                        help="the number of group", 
                        default=5)
    parser.add_argument("-p", "--partitionNumber", type=int, 
                        help="the number of partitioning", 
                        default=16)
    parser.add_argument("-t", "--iterationNumber", type=int, 
                        help="the number of iteration", 
                        default=100)
    parser.add_argument("-tSUM", "--thresholdSUM", type=float, 
                        help="the sum threshold for AND", 
                        default=5.0)
    parser.add_argument("-tMAX", "--thresholdMAX", type=float, 
                        help="the max threshold for AND", 
                        default=1.0)
    parser.add_argument("-f", "--aggregateF", type=str, 
                        help="the aggregate function for AND", 
                        default="MAX")
    # parser.print_help()
    args = parser.parse_args()
    return args
