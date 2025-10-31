import requests
import numpy as np
import os
from pathlib import Path
from typing import List, Union
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 已加载环境变量文件: {env_path}")

BATCH_SIZE = 10

class Embedding:
    """千问Embedding类，用于生成文本嵌入向量"""
    
    def __init__(self, input_texts: Union[str, List[str]]):
        """
        初始化Embedding类
        
        Args:
            input_texts: 输入的文本或文本列表
        """
        self.input_texts = input_texts
        
        # 验证环境变量
        self.api_key = os.getenv("QIANWEN_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量: QIANWEN_API_KEY")
        
        self.API_CONFIG = {
            "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            "model": "text-embedding-v3"
        }

    def _chunks(self, lst, n):
        """将列表按 n 个一组切片"""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def embedding_call(self):
        """调用千问 Embedding API 生成文本嵌入向量（自动分批，单批最多 10 条）"""

        if not self.input_texts:
            raise ValueError("输入文本不能为空")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 统一成列表输入
        if isinstance(self.input_texts, str):
            inputs = [self.input_texts]
        else:
            inputs = list(self.input_texts)

        # 分批调用
        combined_data = []
        combined_usage = None
        base_url = self.API_CONFIG["url"]
        model = self.API_CONFIG["model"]

        for batch_start, batch in enumerate(self._chunks(inputs, 10)):
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

# 示例用法
if __name__ == "__main__":
    relation_arrays = [
        ["属于: 表明一个事物归于哪种类型或范围。", "归类为: 将某对象分配到事先确定的种类或群体中。", "是一种: 描述某具体事物在本质上与哪个大类概念相同。"],
        ["包含: 描述一个整体内部拥有哪些组成部分或要素。", "下辖: 用于行政或组织结构，指代其管辖的附属机构或区域。", "设有: 描述一个组织机构内部配备了哪些具体的部门或职能单位。"],
        ["相关: 指出两个或多个事物之间存在着某种联系或交集。", "涉及: 描述某个问题或事件牵扯到了哪些人和事。", "关联到: 说明一个信息或数据通过某种逻辑连接指向了另一个信息或数据。"],
    ]
    
    # 将二维数组展开为一维列表，用于嵌入计算 (只取解释部分)
    all_texts = []
    for row in relation_arrays:
        for item in row:
            # 提取冒号后面的解释部分作为输入文本
            explanation_text = item.split(': ', 1)[1]
            all_texts.append(explanation_text)
    
    # 1. 计算所有文本的嵌入向量
    embedding = Embedding(input_texts=all_texts)
    result = embedding.embedding_call()
    if "error" in result:
        print(f"API调用失败: {result}")
        exit(1)
    
    # 2. 提取嵌入向量
    embeddings = embedding.extract_embeddings(result)
    if isinstance(embeddings, str) and embeddings.startswith("错误"):
        print(f"提取嵌入向量失败: {embeddings}")
        exit(1)

    # 任务 1: 输出所有术语的解释之间的相似度
    print("## 所有术语解释之间的相似度矩阵")
    
    all_pairs_matrix = embedding.calculate_similarities(result)
    
    if isinstance(all_pairs_matrix, dict) and "error" in all_pairs_matrix:
        print(f"计算全体相似度矩阵失败: {all_pairs_matrix['error']}")
    else:
        # 准备Markdown表格
        headers = ["(索引)"] + [f"T{i+1}" for i in range(len(all_texts))]
        print("| " + " | ".join(headers) + " |")
        print("|---" * (len(all_texts) + 1) + "|")
        
        # 打印矩阵的每一行
        for i, row in enumerate(all_pairs_matrix):
            term_name = relation_arrays[i//3][i%3].split(":")[0] 
            row_header = f"**T{i+1}** ({term_name})"
            row_data = [f"{val:.4f}" for val in row]
            print(f"| {row_header} | " + " | ".join(row_data) + " |")

    # 任务 2: 给出每一组之间的相似度
    print("\n\n## 每一组内部的相似度对比")
    print("| 关系类型 | 解释1-解释2相似度 | 解释1-解释3相似度 | 解释2-解释3相似度 |")
    print("|---|---|---|---|")
    
    for i in range(len(relation_arrays)):
        idx1 = i * 3
        idx2 = i * 3 + 1
        idx3 = i * 3 + 2
        
        vec1 = embeddings[idx1]
        vec2 = embeddings[idx2]
        vec3 = embeddings[idx3]
        
        sim12 = embedding.cosine_similarity(vec1, vec2)
        sim13 = embedding.cosine_similarity(vec1, vec3)
        sim23 = embedding.cosine_similarity(vec2, vec3)
        
        relation_type = relation_arrays[i][0].split(":")[0]
        print(f"| {relation_type} | {sim12:.4f} | {sim13:.4f} | {sim23:.4f} |")