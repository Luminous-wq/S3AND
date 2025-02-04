import networkx as nx
import random

def main(file, W_number):
    # 读取图文件
    G = nx.read_gml(file)
    
    # 计算与3的差距
    dis = W_number - 3
    
    # 遍历图中的每个节点
    for node in G.nodes():
        keywords = G.nodes[node]['keywords']  # 获取当前节点的关键词列表
        
        if dis < 0:
            # 如果差距小于0，随机移除关键词，但至少保留一个
            num_to_remove = min(abs(dis), len(keywords) - 1)  # 确保至少保留一个关键词
            if num_to_remove > 0:
                keywords = random.sample(keywords, len(keywords) - num_to_remove)
        elif dis > 0:
            # 如果差距大于0，随机添加关键词，并确保不重复
            num_to_add = dis
            existing_keywords = set(keywords)  # 将已有关键词转换为集合，方便去重
            available_keywords = [k for k in range(50) if k not in existing_keywords]  # 从0-49中选择未使用的关键词
            if available_keywords:
                new_keywords = random.sample(available_keywords, min(num_to_add, len(available_keywords)))  # 随机选择不重复的关键词
                keywords.extend(new_keywords)
        
        # 更新节点的关键词
        G.nodes[node]['keywords'] = keywords
    
    # 保存更新后的图
    output_file = file.replace(".gml", f"_{W_number}.gml")
    nx.write_gml(G, output_file)
    print(f"Modified graph saved to {output_file}")

# 示例调用
if __name__ == "__main__":
    main("./dataset/precompute/synthetic/50000-124812-50-3/G-uni.gml", 2)  # 假设输入文件为 example.gml，W_number 为 4