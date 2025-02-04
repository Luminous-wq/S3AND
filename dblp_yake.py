import ijson
import networkx as nx
import yake
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from tools import save_Graph
# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 从DBLP数据集中构建图并提取关键词
def get_graph_dblp(file_path, output_authors_path, output_titles_path):
    """
    从DBLP数据集中读取数据，构建作者合作图，并逐项保存作者和标题。
    """
    G = nx.Graph()
    author_titles = defaultdict(list)
    t = 0

    # 打开文件并逐项读取
    with open(file_path, "r", encoding="utf-8") as f, \
         open(output_authors_path, "w", encoding="utf-8") as authors_file, \
         open(output_titles_path, "w", encoding="utf-8") as titles_file:
        
        objects = ijson.items(f, "item")
        for item in objects:
            try:
                # 只处理2010年及以后的论文
                if "year" in item and int(item["year"]) >= 2010:
                    if "title" in item:
                        tit = item["title"].strip()
                        if tit:  # 过滤空标题
                            titles_file.write(f"{tit}\n")  # 保存标题
                            t += 1
                            if t % 10000 == 0:
                                logging.info(f"Processed {t} titles")

                            # 处理作者信息
                            paper_authors = item.get("authors", [])
                            author_names = [author.get("name", "").strip() for author in paper_authors]
                            author_names = [name for name in author_names if name]  # 过滤空作者名

                            for author_name in author_names:
                                author_titles[author_name].append(tit)
                                authors_file.write(f"{author_name}\n")  # 保存作者

                            # 添加合作者边
                            for i in range(len(author_names)):
                                for j in range(i + 1, len(author_names)):
                                    G.add_edge(author_names[i], author_names[j])
            except (ijson.common.IncompleteJSONError, KeyError, ValueError) as e:
                logging.warning(f"Skipping item due to error: {e}")
                continue

    logging.info("Finished processing DBLP data")
    return G, author_titles

# 处理单个作者的关键词
def process_author(author, titles, kw_extractor):
    """
    提取单个作者的关键词。
    """
    text = ". ".join(titles)
    keywords = kw_extractor.extract_keywords(text)
    return author, [kw[0] for kw in keywords[:5]]

# 并行处理所有作者的关键词
def process_authors_keywords(author_titles, kw_extractor, output_keywords_path):
    """
    使用多线程并行提取所有作者的关键词，并逐项保存结果。
    """
    author_keywords = {}
    all_keywords = set()
    t = 0

    cpu_count = os.cpu_count()
    print(f"CPU cores: {cpu_count}")

    # 根据 CPU 核心数设置线程数
    max_workers = cpu_count if cpu_count else 64

    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
         open(output_keywords_path, "w", encoding="utf-8") as keywords_file:
        
        # 提交任务到线程池
        futures = [
            executor.submit(process_author, author, titles, kw_extractor)
            for author, titles in author_titles.items()
        ]

        # 处理任务结果
        for future in futures:
            author, keywords = future.result()
            author_keywords[author] = keywords
            all_keywords.update(keywords)
            keywords_file.write(f"{author}: {', '.join(keywords)}\n")  # 逐项保存关键词
            t += 1
            if t % 1000 == 0:
                logging.info(f"Processed {t} authors")

    logging.info("Finished extracting keywords")
    return author_keywords, all_keywords

# 保存图数据到GML文件
def save_graph_to_file(G, output_graph_path):
    """
    保存带有关键词的图到GML文件。
    """
    logging.info("Saving graph to file...")
    nx.write_gml(G, output_graph_path)
    logging.info(f"Graph saved to {output_graph_path}")

# 从文件中读取作者和标题数据
def load_authors_and_titles(authors_path, titles_path):
    """
    从文件中加载作者和标题数据，并重建 author_titles 字典。
    """
    author_titles = defaultdict(list)
    authors = set()
    titles = set()

    # 读取作者文件
    with open(authors_path, "r", encoding="utf-8") as authors_file:
        for line in authors_file:
            author = line.strip()
            if author:
                authors.add(author)

    # 读取标题文件
    with open(titles_path, "r", encoding="utf-8") as titles_file:
        for line in titles_file:
            title = line.strip()
            if title:
                titles.add(title)

    # 重建 author_titles 字典
    with open(authors_path, "r", encoding="utf-8") as authors_file, \
         open(titles_path, "r", encoding="utf-8") as titles_file:
        
        authors_list = list(authors)
        titles_list = list(titles)
        
        # 假设作者和标题的顺序是对应的
        for author, title in zip(authors_list, titles_list):
            author_titles[author].append(title)

    logging.info(f"Loaded {len(authors)} authors and {len(titles)} titles")
    return author_titles

def load_keywords_from_file(keywords_path):
    """
    从 keywords_domain.txt 文件中加载作者及其关键词。
    """
    author_keywords = {}
    with open(keywords_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    author, keywords_str = line.split(": ", 1)
                    keywords = keywords_str.split(", ")
                    author_keywords[author] = keywords
                except ValueError as e:
                    logging.warning(f"Skipping malformed line: {line}. Error: {e}")
    return author_keywords

# 主程序
if __name__ == '__main__':
    # 配置路径和参数
    dblp_path = "./dataset/dblpv14/dblp_v14.json"  # DBLP数据集路径
    output_authors_path = "./dataset/precompute/real/dblpV14/authors.txt"  # 作者保存路径
    output_titles_path = "./dataset/precompute/real/dblpV14/titles.txt"  # 标题保存路径
    output_keywords_path = "./dataset/precompute/real/dblpV14/keywords_domain.txt"  # 关键词保存路径

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_authors_path), exist_ok=True)

    # 初始化YAKE!关键词提取器
    kw_extractor = yake.KeywordExtractor(
        lan="en",  # 文档语言
        n=3,  # N-grams
        dedupLim=0.9,  # 筛选阈值
        dedupFunc='seqm',
        windowsSize=1,
        top=20  # 最大关键词数量
    )

    # # 从DBLP数据集中构建图并提取数据
    # G, author_titles = get_graph_dblp(dblp_path, output_authors_path, output_titles_path)
    # logging.info(f"Graph nodes: {len(G.nodes)}, Graph edges: {len(G.edges)}")

    # 从文件中加载作者和标题数据
    author_titles = load_authors_and_titles(output_authors_path, output_titles_path)
    
    keywords_path = "./dataset/precompute/real/dblpV14/keywords_domain.txt"
    # 提取每个作者的关键词
    # author_keywords, all_keywords = process_authors_keywords(author_titles, kw_extractor, output_keywords_path)
    # logging.info(f"Total unique keywords: {len(all_keywords)}")
    author_keywords = load_keywords_from_file(keywords_path)

    # 重建图并添加关键词属性
    G = nx.Graph()
    added_nodes = set()

    # 添加节点（仅当 keywords 不为空）
    for author, titles in author_titles.items():
        if author in author_keywords and author_keywords[author]:  # 检查 keywords 是否为空
            G.add_node(author, keywords=author_keywords[author])
            added_nodes.add(author)

    print("add done")
    print(len(added_nodes))

    # 将 added_nodes 转换为集合（如果还不是集合）
    added_nodes_set = set(added_nodes)

    edge_count = 0
    k = 10  # 50 20 10

    for author, titles in author_titles.items():
        if author in added_nodes_set:  # 确保 author 在图中
            # 计数器，记录当前作者已添加的边数
            added_edges_for_author = 0
            
            # 遍历所有可能的合作者
            for co_author in added_nodes_set:
                if co_author != author:  # 排除自己
                    G.add_edge(author, co_author)
                    edge_count += 1
                    added_edges_for_author += 1
                    
                    # 每添加 1000 条边输出一次日志
                    if edge_count % 10000 == 0:
                        logging.info(f"Added {edge_count} edges so far.")
                    
                    # 如果当前作者已经添加了 k 条边，退出内层循环
                    if added_edges_for_author >= k:
                        break

    # 添加边（仅当两个节点都在图中）
    # for author, titles in author_titles.items():
    #     if author in added_nodes:  # 确保 author 在图中
    #         for co_author in author_titles:
    #             if co_author in added_nodes and author != co_author:  # 确保 co_author 在图中且不是自己
    #                 G.add_edge(author, co_author)

    print(f"Number of nodes: {len(G.nodes)}")
    print(f"Number of edges: {len(G.edges)}")
    data_name = "dblpV14"
    ok = save_Graph(Graph=G, dataset_name=data_name, is_Aux=False)
    # 保存带有关键词的图
    # save_graph_to_file(G, output_graph_path)