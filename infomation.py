import os
import time

class Info:
    def __init__(
            self,
            input_file_folder: str,
            dataset_name: str,
            query_graph_edges: str,
            query_graph_size: int,
            query_graph_keywords: str,
            keyword_domain: int,
            group_number: int,
            partition_number: int,
            iteration_number: int,
            thresholds_sum: float,
            thresholds_max: float,
            F: str
    ) -> None:
        self.input_file_name = os.path.join(input_file_folder)
        
        self.output_info_file_name = os.path.join(input_file_folder, "information.txt")
        self.dataset_name = dataset_name
        
        self.S3AND_result = []
        
        input_info = input_file_folder.split('/')[-1].split('-')
        if len(dataset_name.split("-")) < 4:
            self.nodes_num = input_info[1]
            self.edges_num = input_info[2].split('.')[0]
            self.index_file_name = os.path.join(os.path.dirname(input_file_folder), dataset_name+"_index.json")
        else:
            self.nodes_num = dataset_name.split('-')[0]
            self.edges_num = dataset_name.split('-')[1]
            self.index_file_name = os.path.join(input_file_folder.rsplit(".",1)[0]+"_index.json")
        # self.all_keyword_num = 0
        # self.distribution = "uniform"

        self.query_graph_edges = query_graph_edges
        self.query_graph_size = query_graph_size
        self.query_graph_keywords = query_graph_keywords
        self.keyword_domain = keyword_domain
        self.group_number = group_number
        self.partition_number = partition_number
        self.iteration_number = iteration_number
        self.thresholds_sum = thresholds_sum
        self.thresholds_max = thresholds_max
        self.function = F

        self.start_time = 0
        self.finish_time = 0
        self.offline_time = 0
        self.index_time = 0
        self.non_leaf_node_traverse_time = 0
        self.leaf_node_traverse_time = 0
        self.initialization_time = 0
        self.online_time = 0
        self.refine_time = 0
        # self.select_greatest_entry_in_H_time = 0
        # self.leaf_node_traverse_time = 0
        # self.non_leaf_node_traverse_time = 0
        # self.traversal_compute_influential_score_time = 0
        # self.mod_compute_influential_score_time = 0
        # self.modify_result_set_time = 0
        self.vertex_visit_counter = 0
        self.vertex_pruning_counter_LB_ND_2 = 0
        self.vertex_pruning_counter_LB_ND_1 = 0
        self.vertex_pruning_counter_BV = 0
        self.entry_pruning_counter = 0
        self.entry_node_visit_counter = 0
        # self.leaf_node_counter = 0
        # self.leaf_node_visit_counter = 0

        # self.compute_inf_count_traversal = 0
        # self.compute_inf_count_mid = 0
        # self.need_mid_number = 0

        # self.influence_score_result = 0

    def get_S3AND_result(self) -> str:
        result = ""
        result += "INFORMATION RESULTS\n"
        result += "------------------ FILE INFO ------------------\n"
        result += "Input File: {}\n".format(self.input_file_name)
        result += "Index File: {}\n".format(self.index_file_name)
        result += "Output Info File: {}\n".format(self.output_info_file_name)
        result += "Graph Total Nodes Number: {}\n".format(self.nodes_num)
        result += "Graph Total Edges Number: {}\n".format(self.edges_num)
        result += "Keyword Domain: {}\n".format(self.keyword_domain)
        result += "Grouping Number: {}\n".format(self.group_number)
        result += "Construction Index Iteration Number: {}\n".format(self.iteration_number)
        # result += "Keywords Per Vertex: {}\n".format(self.keywords_pre_vertex)
        # result += "Distribution: {}\n".format(self.distribution)
        result += "\n"
        result += "------------------ ANSWER INFO ------------------\n"
        result += "S3AND Result (S): {}\n".format(self.S3AND_result)
        result += "Leaf Node Visit Counter: {}\n".format(self.vertex_visit_counter)
        result += "Non-leaf Node Visit Counter: {}\n".format(self.entry_node_visit_counter)
        result += "\n"
        result += "------------------ QUERY INFO ------------------\n"
        result += "Query Keywords File: {}\n".format(self.query_graph_keywords)
        result += "Query Size: {}\n".format(self.query_graph_size)
        result += "Query Edges File: {}\n".format(self.query_graph_edges)
        result += "SUM AND Threshold: {}\n".format(self.thresholds_sum)
        result += "MAX AND Threshold: {}\n".format(self.thresholds_max)
        result += "\n"
        result += "------------------ PRUNING INFO ------------------\n"
        result += "Pruning Vertices: {}\n".format(int(self.nodes_num)-self.vertex_visit_counter)
        result += "Pruning Vertices By BV: {}\n".format(self.vertex_pruning_counter_BV)
        result += "Pruning Vertices By LB_ND_1: {}\n".format(self.vertex_pruning_counter_LB_ND_1)
        result += "Pruning Vertices By LB_ND_2: {}\n".format(self.vertex_pruning_counter_LB_ND_2)
        result += "Pruning Entries: {}\n".format(self.entry_pruning_counter)
        result += "\n"
        result += "------------------ TIME INFO ------------------\n"
        result += "Started at: {} \tFinished at: {}\n".format(self.start_time, self.finish_time)
        result += "Total Time: {}\n".format(self.finish_time - self.start_time)
        result += "Initialization Time: {}\n".format(self.initialization_time)
        result += "Offline Calculate Time: {}\n".format(self.offline_time)
        result += "Index Construct Time: {}\n".format(self.index_time)
        result += "Non-leaf Nodes Trasverse Time: {}\n".format(self.non_leaf_node_traverse_time)
        result += "Leaf Nodes Trasverse Time: {}\n".format(self.leaf_node_traverse_time)
        result += "Online Time: {}\n".format(self.online_time)
        result += "Refinement Time: {}\n".format(self.refine_time)
        # result += "Select Greatest Entry in Heap time: {}\n".format(self.select_greatest_entry_in_H_time)
        # result += "Leaf Node Traverse time: {}\n".format(self.leaf_node_traverse_time)
        # result += "NonLeaf Node Traverse time: {}\n".format(self.non_leaf_node_traverse_time)
        # result += "Traversal Compute Influential Score time: {}\n".format(self.traversal_compute_influential_score_time)
        # result += "Modify Compute Influential Score time: {}\n".format(self.mod_compute_influential_score_time)
        # result += "Modify Result Set time: {}\n".format(self.modify_result_set_time)
        result += "\n"
        # result += "------------------COUNT INFO------------------\n"
        # result += "Leaf Nodes Visit: {}\n".format(self.leaf_node_visit_counter)
        # result += "Influential count: {}\n".format(self.compute_inf_count_traversal+self.compute_inf_count_mid)
        # result += "Influential count traversal: {}\n".format(self.compute_inf_count_traversal)
        # result += "Influential count mid: {}\n".format(self.compute_inf_count_mid)
        # result += "Need modify community count: {}\n".format(self.need_mid_number)
        # result += "\n"
        return result


