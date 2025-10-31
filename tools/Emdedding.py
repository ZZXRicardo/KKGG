import requests
import numpy as np
from typing import List, Union
BATCH_SIZE = 10

class Embedding:
    """千问Embedding类，用于生成文本嵌入向量"""
    
    # 千问API配置 - 北京地域
    API_CONFIG = {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",  # 北京地域URL
        "model": "text-embedding-v3",  # 使用text-embedding-v3模型
        "key": "sk-ecf819b71fae427bb1ca8be81a257509"  # 测试密钥，直接写入代码
    }
    
    def __init__(self, input_texts: Union[str, List[str]]):
        """
        初始化Embedding类
        
        Args:
            input_texts: 输入的文本或文本列表
        """
        self.input_texts = input_texts
    def _chunks(self, lst, n):
        """将列表按 n 个一组切片"""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    
    def embedding_call(self):
        """调用千问 Embedding API 生成文本嵌入向量（自动分批，单批最多 10 条）"""

        if not self.input_texts:
            raise ValueError("输入文本不能为空")

        headers = {
            "Authorization": f"Bearer {self.API_CONFIG['key']}",
            "Content-Type": "application/json"
        }

        # 统一成列表输入
        if isinstance(self.input_texts, str):
            inputs = [self.input_texts]
        else:
            inputs = list(self.input_texts)

        # 分批调用
        combined_data = []
        combined_usage = None  # 用于合并 usage 信息
        base_url = self.API_CONFIG["url"]
        model = self.API_CONFIG["model"]

        def chunks(lst, n):
            """按 n 大小切分列表"""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        for batch_start, batch in enumerate(chunks(inputs, 10)):  # 千问上限是 10
            payload = {
                "model": model,
                "input": batch
            }

            response = requests.post(base_url, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"千问 Embedding API 错误: {response.text}")
                return {
                    "error": f"API 请求失败，状态码：{response.status_code}",
                    "details": response.text
                }

            jr = response.json()
            data = jr.get("data", [])

            for k, item in enumerate(data):
                global_index = batch_start * 10 + k
                item["index"] = global_index
                combined_data.append(item)

            usage = jr.get("usage")
            if usage:
                if combined_usage is None:
                    combined_usage = usage
                else:
                    for key, val in usage.items():
                        if isinstance(val, (int, float)) and isinstance(combined_usage.get(key), (int, float)):
                            combined_usage[key] += val

        combined_resp = {"data": sorted(combined_data, key=lambda x: x["index"])}
        if combined_usage:
            combined_resp["usage"] = combined_usage

        return combined_resp

    def extract_embeddings(self, api_response):
        """从API响应中提取嵌入向量"""
        if "error" in api_response:
            return f"错误: {api_response['error']}"
        
        try:
            # 提取嵌入向量 - 使用OpenAI兼容格式
            embeddings = []
            for item in api_response['data']:
                embeddings.append(item['embedding'])
            
            # 如果是单文本输入，直接返回第一个嵌入向量
            if isinstance(self.input_texts, str) and len(embeddings) == 1:
                return embeddings[0]
            else:
                return embeddings
                
        except (KeyError, IndexError, TypeError) as e:
            return f"解析嵌入向量失败: {str(e)}"
    
    def get_embedding_dimension(self, api_response):
        """获取嵌入向量的维度"""
        embeddings = self.extract_embeddings(api_response)
        
        if isinstance(embeddings, str) and embeddings.startswith("错误"):
            return embeddings
        
        if isinstance(embeddings, list) and len(embeddings) > 0:
            return len(embeddings[0])
        elif isinstance(embeddings, list):
            return 0
        else:
            return len(embeddings)
    
    def get_usage_info(self, api_response):
        """获取API使用情况"""
        if "error" in api_response:
            return f"错误: {api_response['error']}"
        
        try:
            return api_response.get('usage', {})
        except (KeyError, TypeError) as e:
            return f"解析使用信息失败: {str(e)}"
    
    def cosine_similarity(self, vector1, vector2):
        """
        计算两个向量之间的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            float: 余弦相似度，范围[-1, 1]
        """
        # 转换为numpy数组
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # 计算点积
        dot_product = np.dot(v1, v2)
        
        # 计算模长
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算余弦相似度
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def calculate_similarities(self, api_response):
        """
        计算多个句子之间的余弦相似度矩阵
        
        Args:
            api_response: API响应结果
            
        Returns:
            list: 相似度矩阵的二维列表
        """
        # 提取嵌入向量
        embeddings = self.extract_embeddings(api_response)
        
        if isinstance(embeddings, str) and embeddings.startswith("错误"):
            return {"error": embeddings}
        
        # 确保我们有多个句子
        if isinstance(embeddings, list) and len(embeddings) < 2:
            return {"error": "需要至少两个句子来计算相似度"}
        
        # 计算相似度矩阵
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i][j] = self.cosine_similarity(embeddings[i], embeddings[j])
        
        # 只返回相似度矩阵
        return similarity_matrix.tolist()

# -------------------------------------------------------------------
# 示例用法
if __name__ == "__main__":
    # -----------------------------------------------------------------
# 修改示例用法的测试数据部分
# -----------------------------------------------------------------
# 更新：使用您提供的新术语和解释
# -----------------------------------------------------------------
    relation_arrays = [
    ["属于: 表明一个事物归于哪种类型或范围。", "归类为: 将某对象分配到事先确定的种类或群体中。", "是一种: 描述某具体事物在本质上与哪个大类概念相同。"],
    ["包含: 描述一个整体内部拥有哪些组成部分或要素。", "下辖: 用于行政或组织结构，指代其管辖的附属机构或区域。", "设有: 描述一个组织机构内部配备了哪些具体的部门或职能单位。"],
    ["相关: 指出两个或多个事物之间存在着某种联系或交集。", "涉及: 描述某个问题或事件牵扯到了哪些人和事。", "关联到: 说明一个信息或数据通过某种逻辑连接指向了另一个信息或数据。"],
    ["位于: 标明一个物体在空间坐标上的具体位置。", "坐落于: 描述大型建筑、城市或地标处于某个地理环境之中。", "分布于: 说明某种事物或现象在特定区域内散布开来的情况。"],
    ["导致: 表明前一个动作或事件直接产生了后面的结果。", "引发: 描述一个事件或行为触发了后续一系列的反应或后果。", "促使: 描述某种原因推动了某人或某事向特定方向发展或做出决定。"],
    ["任职于: 表明某人在特定的机构或岗位上承担职务。", "效力于: 描述某人为了特定的组织或目标尽心尽力工作。", "在...工作: 记录某个人进行劳动和领取报酬的具体场所。"],
    ["出生于: 记录一个生命体来到世界时的具体地点。", "籍贯: 标明某个人祖先的起源地或家庭历史所在的地区。", "源于: 追溯某个概念、传统或事物的最初发端和出处。"],
    ["具有: 描述一个对象身上带有某种特征、能力或性质。", "特点是: 描述某个事物在同类中区别于其他事物的显著方面。", "拥有: 描述一个主体掌握着某种财富、资源或能力。"],
    ["了解: 描述一个人对某个信息或事理达到了一定程度的认知。", "精通: 描述一个人对某项技能或知识掌握得极其深入和熟练。", "知晓: 表示一个人已经被告知某个消息或事实。"],
    ["由...制成: 说明制造某个成品所使用的基本物质原料。", "材质为: 直接指明组成某个物品的物理材料种类。", "采用: 选择并开始使用某种方法、方案或系统。"]
]
    # -----------------------------------------------------------------
    
    # 将二维数组展开为一维列表，用于嵌入计算 (只取解释部分)
    all_texts = []
    for row in relation_arrays:
        for item in row:
            # 提取冒号后面的解释部分作为输入文本
            # 格式: "术语: 解释" -> 提取 "解释"
            explanation_text = item.split(': ', 1)[1]
            all_texts.append(explanation_text)
    
    # 1. 计算所有文本的嵌入向量
    embedding = Embedding(input_texts=all_texts)
    result = embedding.embedding_call()
    if "error" in result:
        print(f"API调用失败: {result}")
        exit(1)
    
    # 2. 提取嵌入向量 (后续计算组内相似度时需要)
    embeddings = embedding.extract_embeddings(result)
    if isinstance(embeddings, str) and embeddings.startswith("错误"):
        print(f"提取嵌入向量失败: {embeddings}")
        exit(1)

    # -----------------------------------------------------------------
    # 任务 1: 输出所有术语的解释之间的相似度 (All-Pairs Similarity)
    # -----------------------------------------------------------------
    print("## (任务1) 所有术语解释之间的相似度矩阵 (All-Pairs Similarity Matrix)")
    
    # 使用 calculate_similarities 函数计算完整的矩阵 (现在是 30x30)
    all_pairs_matrix = embedding.calculate_similarities(result)
    
    if isinstance(all_pairs_matrix, dict) and "error" in all_pairs_matrix:
        print(f"计算全体相似度矩阵失败: {all_pairs_matrix['error']}")
    else:
        # 准备Markdown表格
        
        # 表头 (T1, T2, ..., T30)
        headers = ["(索引)"] + [f"T{i+1}" for i in range(len(all_texts))]
        print("| " + " | ".join(headers) + " |")
        
        # 分隔符
        print("|---" * (len(all_texts) + 1) + "|")
        
        # 打印矩阵的每一行
        for i, row in enumerate(all_pairs_matrix):
            # 获取行标题 (T1: 术语名)
            # 使用原始 relation_arrays 中对应的第一个元素来获取术语名
            term_name = relation_arrays[i//3][i%3].split(":")[0] 
            row_header = f"**T{i+1}** ({term_name})"
            
            # 格式化相似度数值
            row_data = [f"{val:.4f}" for val in row]
            
            # 打印行
            print(f"| {row_header} | " + " | ".join(row_data) + " |")

    # -----------------------------------------------------------------
    # 任务 2: 给出每一组之间的相似度 (Within-Group Similarity)
    # -----------------------------------------------------------------
    print("\n\n## (任务2) 每一组内部的相似度对比")
    
    # 输出Markdown表格
    print("| 关系类型 | 解释1-解释2相似度 | 解释1-解释3相似度 | 解释2-解释3相似度 |")
    print("|---|---|---|---|")
    
    for i in range(len(relation_arrays)):
        # 获取当前行的三个解释在嵌入向量中的索引
        idx1 = i * 3
        idx2 = i * 3 + 1
        idx3 = i * 3 + 2
        
        # 提取三个解释的嵌入向量 (使用之前提取的 embeddings 列表)
        vec1 = embeddings[idx1]
        vec2 = embeddings[idx2]
        vec3 = embeddings[idx3]
        
        # 计算两两之间的相似度
        sim12 = embedding.cosine_similarity(vec1, vec2)
        sim13 = embedding.cosine_similarity(vec1, vec3)
        sim23 = embedding.cosine_similarity(vec2, vec3)
        
        # 获取关系类型（从第一个解释中提取术语名称）
        relation_type = relation_arrays[i][0].split(":")[0]
        
        print(f"| {relation_type} | {sim12:.4f} | {sim13:.4f} |")
    
    # -----------------------------------------------------------------
    # 附录: 详细解释内容
    # -----------------------------------------------------------------
    print("\n\n## 附录：详细解释内容")
    for i, row in enumerate(relation_arrays):
        relation_type = row[0].split(":")[0]
        # (T1, T2, T3), (T4, T5, T6), etc.
        print(f"\n### {relation_type} (T{i*3+1} / T{i*3+2} / T{i*3+3})")
        print(f"1. (T{i*3+1}) {row[0]}")
        print(f"2. (T{i*3+2}) {row[1]}")
        print(f"3. (T{i*3+3}) {row[2]}")