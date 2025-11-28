# 文件：解释聚类削歧_类封装版_批量优化版.py
# ✅ 优化：批量解释术语，提升效率

from LLM import LLM
from Embedding import Embedding
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field


# =========================================================
# 主类：术语消歧器（批量优化版）
# =========================================================
class TermDisambiguator:
    """术语消歧与聚类系统 - 批量优化版"""
    
    # =========================================================
    # 内部数据容器类
    # =========================================================
    @dataclass
    class TermEntry:
        term: str = ""
        explanation: str = ""
        embedding: List[float] = field(default_factory=list)
        synonyms: List[str] = field(default_factory=list)
    
class TermDisambiguator:
    """术语消歧与聚类系统 - 批量优化版"""
    
    # =========================================================
    # 内部数据容器类（保持原嵌套结构）
    # =========================================================
    @dataclass
    class TermEntry:
        term: str = ""
        explanation: str = ""
        embedding: List[float] = field(default_factory=list)
        synonyms: List[str] = field(default_factory=list)
    
    # =========================================================
    # 修改初始化方法（仅增加参数和属性）
    # =========================================================
    def __init__(self, api_provider: str = "qianwen", base_terms_dir: Optional[str] = None, process_id: Optional[int] = None):
        """
        初始化消歧器
        
        Args:
            api_provider: API提供商，默认为"qianwen"
            base_terms_dir: 术语库基础目录路径
            process_id: 进程ID，用于PID隔离
        """
        self.api_provider = api_provider
        self.base_terms_dir = Path(base_terms_dir) if base_terms_dir else None
        self.process_id = process_id
        self.term_entries: List[TermDisambiguator.TermEntry] = []
        self.term_entry_map: Dict[str, TermDisambiguator.TermEntry] = {}
    # =========================================================
    # 新增的私有路径生成方法（保持缩进）
    # =========================================================
    def _get_entity_json_path(self) -> Path:
        """获取实体术语库路径"""
        if self.base_terms_dir:
            entity_dir = self.base_terms_dir / "Entity"
            entity_dir.mkdir(parents=True, exist_ok=True)
            if self.process_id:
                return entity_dir / f"Entity_{self.process_id}.json"
            return entity_dir / "Entity.json"
        return Path(r"E:\KKGG\output\terms\Entity.json")
    
    def _get_relation_json_path(self) -> Path:
        """获取关系术语库路径"""
        if self.base_terms_dir:
            relation_dir = self.base_terms_dir / "Relation"
            relation_dir.mkdir(parents=True, exist_ok=True)
            if self.process_id:
                return relation_dir / f"Relation_{self.process_id}.json"
            return relation_dir / "Relation.json"
        return Path(r"E:\KKGG\output\terms\Relation.json")
    
    def _get_cluster_json_path(self) -> Path:
        """获取聚类三元组库路径（存于entity_cluster_triples目录）"""
        if self.base_terms_dir:
            cluster_dir = self.base_terms_dir / "entity_cluster_triples"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            if self.process_id:
                return cluster_dir / f"entity_cluster_triples_{self.process_id}.json"
            return cluster_dir / "entity_cluster_triples.json"
        return Path(r"E:\KKGG\output\terms\entity_cluster_triples\entity_cluster_triples.json")       
    def _get_concept_entity_json_path(self) -> Path:
        """概念聚类专用：获取概念实体术语库路径（存于definitions目录）"""
        if self.base_terms_dir:
            concept_dir = self.base_terms_dir / "definitions"
            concept_dir.mkdir(parents=True, exist_ok=True)
            if self.process_id:
                return concept_dir / f"Entity_{self.process_id}.json"
            return concept_dir / "Entity.json"
        return Path(r"E:\KKGG\output\terms\definitions\Entity.json")
    # =========================================================
    # 辅助方法：余弦相似度
    # =========================================================

    @staticmethod
    def _cos(a: list, b: list) -> float:
        """计算两个向量的余弦相似度"""
        if not a or not b:
            return 0.0
        m = min(len(a), len(b))
        s_ab = sum(a[k] * b[k] for k in range(m))
        s_aa = sum(a[k] * a[k] for k in range(m)) ** 0.5
        s_bb = sum(b[k] * b[k] for k in range(m)) ** 0.5
        if s_aa == 0 or s_bb == 0:
            return 0.0
        return s_ab / (s_aa * s_bb)
    
    # =========================================================
    # JSON 读写方法
    # =========================================================
    def _load_json_terms(self, json_path: Optional[str | Path]) -> Tuple[List[Dict], Dict[str, Dict]]:
        """读取 JSON，返回 (数组, {term: item_dict})"""
        if not json_path:
            return [], {}
        p = Path(json_path)
        if not p.exists():
            return [], {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return [], {}
            term_map = {}
            for item in data:
                if isinstance(item, dict):
                    t = (item.get("term") or "").strip()
                    if t:
                        if "synonyms" not in item:
                            item["synonyms"] = []
                        term_map[t] = item
                        # 同时为所有同义词建立映射（指向同一个对象）
                        for syn in item.get("synonyms", []):
                            term_map[syn] = item
            return data, term_map
        except Exception as e:
            print(f"[JSON读取错误] {e}")
            return [], {}
    
    def _save_json_terms(self, json_path: Optional[str | Path], data: List[Dict]):
        """保存 JSON"""
        if not json_path:
            return
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        for item in data:
            if "synonyms" not in item:
                item["synonyms"] = []
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[JSON保存] 已保存至 {json_path}")
    
    # =========================================================
    # 🆕 批量生成解释和嵌入 - 核心优化
    # =========================================================
    def generate_batch_explanations_and_embeddings(
        self,
        terms: List[str],
        shared_context: Optional[str] = None
    ) -> Dict[str, Tuple[str, List[float]]]:
        """
        批量生成多个术语的解释和嵌入向量（核心优化方法）
        
        Args:
            terms: 待解释的术语列表
            shared_context: 包含所有术语的共享上下文
        
        Returns:
            Dict[术语, (解释, 嵌入向量)]
        """
        if not terms:
            return {}
        
        print(f"\n[批量解释] 开始批量处理 {len(terms)} 个术语")
        
        # 构建批量解释的prompt
        base_instruction = (
            "你是给高中生写词条定义的老师，请用『一句中文』解释给定的每个术语，"
            "不要出现术语本身，也不要用「X 是……」的句式；直接给出定义内容。"
            "不要在解释的时候引入他的同义词，比如A是B的另一种称呼。"
            "做这些解释的时候不要参考你做的其他解释，每个词语独立解释，当作每次只处理一个任务。"
            "如果提供了共享上下文，请从上下文中提取该术语的语义进行消歧。\n"
            "【示例-正确】「一种能自动处理信息的电子设备。」\n"
            "【示例-错误】「电脑是一种能自动处理信息的电子设备。」（包含术语）"
            "同时，在解释非关系词时请确保解释体现术语的本质特征，避免使用上位词和归类句式。\n\n"
    
            "【核心要求】\n"
            "1. 🔍 挖掘本质：聚焦该术语最独特的、区别于其他同类事物的特征\n"
            "2. 🚫 避免上位词：不要使用『一种XX』、『属于XX』等归类句式\n"
            "3. 🎯 特征导向：直接描述其核心属性、功能、形态或作用机制\n"
            "4. 📝 句式多样：避免所有解释使用相同句式结构\n"
            "5. ❌ 禁止出现：术语本身、『是……』句式、同义词引用\n\n"
    
            "【错误示例】\n"
            "苹果：一种水果，富含维生素C ❌（使用上位词）\n"
            "橙子：一种富含维生素C的水果 ❌（句式重复）\n"
            "升：一种容量的计量单位 ❌（过于笼统）\n"
            "毫升：空间计量单位 ❌（特征不准确）\n\n"
    
            "【正确示例】\n"
            "苹果：果皮多为红/绿色，果肉脆甜多汁，核心有籽 ✔\n"
            "橙子：柑橘类果实，果皮橙黄易剥，果瓣多汁酸甜 ✔\n"
            "升：等于立方分米，常用于衡量液体体积的基本单位 ✔\n"
            "毫升：千分之一升，适用于小容量液体的精确计量 ✔\n"
            "机器学习：通过数据训练让计算机自动改进决策能力 ✔\n"
            "区块链：由按时间顺序连接的不可篡改数据块构成 ✔\n\n"
        )
        
        # 🆕 批量解释的新提示词
        batch_instruction = (
            "\n【批量解释任务】\n"
            "现在给你一组术语列表，请按照上述要求，为每个术语独立生成解释。\n"
            "请严格按照以下JSON格式输出，不要输出任何其他内容：\n"
            "{\n"
            '  "术语1": "解释1",\n'
            '  "术语2": "解释2",\n'
            '  "术语3": "解释3"\n'
            "}\n\n"
            "【重要】\n"
            "- 必须为列表中的每个术语都生成解释\n"
            "- JSON的键必须与给定的术语完全一致\n"
            "- 每个解释都要独立完成，不要相互参考\n"
            "- 只输出JSON，不要包含任何其他文字、标记或格式符号\n"
        )
        
        # 构建术语列表字符串
        terms_str = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])
        
        if shared_context:
            prompt = (
                f"{base_instruction}\n"
                f"{batch_instruction}\n"
                f"【待解释术语列表】\n{terms_str}\n\n"
                f"【共享上下文（包含多个术语，请从中提取各术语的语义）】\n{shared_context}\n\n"
                "请按照JSON格式输出所有术语的解释："
            )
        else:
            prompt = (
                f"{base_instruction}\n"
                f"{batch_instruction}\n"
                f"【待解释术语列表】\n{terms_str}\n\n"
                "请按照JSON格式输出所有术语的解释："
            )
        
        # 调用LLM生成批量解释
        print(f"  📤 发送批量解释请求...")
        llm_instance = LLM(prompt=prompt, api_provider=self.api_provider)
        result = llm_instance.llm_call()
        response_text = llm_instance.extract_response(result)
        
        # 解析JSON响应
        try:
            # 清理可能的markdown代码块标记
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            explanations_dict = json.loads(response_text)
            print(f"  ✅ 成功解析批量解释JSON")
        except Exception as e:
            print(f"  ❌ 解析JSON失败: {e}")
            print(f"  响应内容: {response_text[:200]}...")
            # 失败时返回空字典
            return {}
        
        # 为每个解释生成嵌入向量
        print(f"  📊 开始生成嵌入向量...")
        result_dict = {}
        
        # 收集所有解释文本用于批量embedding
        explanation_texts = []
        valid_terms = []
        
        for term in terms:
            explanation = explanations_dict.get(term, "").strip()
            if explanation:
                explanation_texts.append(explanation)
                valid_terms.append(term)
            else:
                print(f"  ⚠️  术语 '{term}' 缺少解释")
        
        # 批量生成嵌入向量
        if explanation_texts:
            try:
                embedding = Embedding(input_texts=explanation_texts)
                embedding_result = embedding.embedding_call()
                vectors = embedding.extract_embeddings(embedding_result)
                
                if isinstance(vectors, list) and len(vectors) == len(valid_terms):
                    for i, term in enumerate(valid_terms):
                        result_dict[term] = (explanations_dict[term], vectors[i])
                    print(f"  ✅ 批量生成 {len(result_dict)} 个嵌入向量")
                else:
                    print(f"  ⚠️  嵌入向量数量不匹配: {len(vectors)} vs {len(valid_terms)}")
            except Exception as e:
                print(f"  ❌ 批量生成嵌入向量失败: {e}")
        
        return result_dict
    
    # =========================================================
    # 生成单个术语的解释和嵌入（保留用于兼容性）
    # =========================================================
    def generate_explanation_and_embedding(
        self,
        term: str,
        shared_context: Optional[str] = None
    ) -> Tuple[str, List[float]]:
        """
        为单个术语生成解释和嵌入向量
        注意：此方法保留用于向后兼容，但建议使用批量方法
        
        Args:
            term: 待解释的术语
            shared_context: 包含所有术语的共享上下文
        """
        # 调用批量方法处理单个术语
        batch_result = self.generate_batch_explanations_and_embeddings([term], shared_context)
        
        if term in batch_result:
            return batch_result[term]
        else:
            return "", []
    
    # =========================================================
    # 一词多义拆分方法
    # =========================================================
    def disambiguate_term_to_two(
        self,
        term: str,
        explanation_a: str,
        explanation_b: str
    ) -> Tuple[str, str]:
        """将多义词拆分为两个更具体的术语名称"""
        rules = (
            "你需要把一个多义词拆分为两个更具体的新术语名称，用以清晰区分两种不同含义。"
            "名称应简洁明确，可使用「（类别）」、「公司」、「品牌」、「地名」等限定词进行消歧。"
            "两个名称必须分别与各自的解释严格匹配，语义互不重叠。"
        )
        io = (
            "仅输出 JSON，必须且只包含键：term_a, term_b。"
            "不要输出除 JSON 以外的任何字符。"
        )
        prompt = f"""
{rules}

【原始术语】'{term}'

【解释A】'{explanation_a}'
【解释B】'{explanation_b}'

请分别给出两个更具体的新术语名称，用以区分不同含义。
- 示例："{term}（水果）"、"{term}公司"、"{term}（地名）" 等
- 两个名称必须与各自解释严格匹配。

{io}
"""
        llm = LLM(prompt=prompt, api_provider=self.api_provider)
        resp = llm.llm_call()
        text = llm.extract_response(resp)
        
        obj = json.loads(text)
        term_a = (obj.get("term_a") or "").strip()
        term_b = (obj.get("term_b") or "").strip()
        if not term_a or not term_b:
            raise RuntimeError(f"JSON缺少term_a/term_b或为空：{text!r}")
        return term_a, term_b
    
    # =========================================================
    # 核心：增量处理术语（逐个判断是否为同义词）
    # =========================================================
    def process_term_incrementally(
        self,
        term: str,
        explanation: str,
        embedding_vec: List[float],
        shared_context: Optional[str],
        json_data: List[Dict],
        json_map: Dict[str, Dict],
        synonym_threshold_low: float = 0.73,
        synonym_threshold_high: Optional[float] = 0.85,
        polysemy_threshold: float = 0.73,
        force_polysemy_check: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        增量处理单个术语（已修复一词多义检测）- 🆕 优化：接收预生成的解释和嵌入
        返回: (是否为同义词, 代表术语)
        - 如果是同义词: (True, 代表术语)
        - 如果不是同义词: (False, None)
        """
        print(f"\n[处理术语] {term}")
        
        # 1. 检查是否已存在（作为主术语或同义词）
        if term in json_map:
            existing = json_map[term]
            main_term = existing["term"]
            
            # 🔧 修复点1：不直接返回，而是先检查是否可能是多义词
            if force_polysemy_check and shared_context:
                print(f"  ℹ️  术语已存在为'{main_term}'{'的同义词' if main_term != term else '（主术语）'}，检查是否为多义词...")
                
                # 使用已提供的解释和嵌入
                new_explanation = explanation
                new_embedding = embedding_vec
                
                if not new_embedding:
                    print(f"  ⚠️  无法生成新嵌入向量，保持原有术语")
                    return True, main_term
                
                # 获取已有解释
                old_explanation = existing.get("explanation", "")
                old_embedding = existing.get("embedding", [])
                
                if not old_embedding:
                    print(f"  ⚠️  旧术语缺少嵌入向量，保持原有术语")
                    return True, main_term
                
                # 计算相似度
                sim = self._cos(new_embedding, old_embedding)
                print(f"  📊 新旧解释相似度: {sim:.4f}")
                print(f"    旧解释: {old_explanation[:60]}...")
                print(f"    新解释: {new_explanation[:60]}...")
                
                # 🔧 修复点2：根据相似度判断
                if sim < polysemy_threshold:
                    print(f"  🎯 检测到一词多义！(相似度 {sim:.4f} < 阈值 {polysemy_threshold})")
                    
                    # 触发多义词拆分
                    try:
                        term_a, term_b = self.disambiguate_term_to_two(
                            term, old_explanation, new_explanation
                        )
                        print(f"  ✂️  拆分为: '{term_a}' 和 '{term_b}'")
                        
                        # 更新已有术语名称
                        existing["term"] = term_a
                        # 从json_map中移除旧的映射
                        if term in json_map:
                            del json_map[term]
                        # 更新同义词中的引用
                        for syn in existing.get("synonyms", []):
                            if syn in json_map:
                                del json_map[syn]
                        # 重新建立映射
                        json_map[term_a] = existing
                        for syn in existing.get("synonyms", []):
                            json_map[syn] = existing
                        
                        # 创建新术语
                        new_item = {
                            "term": term_b,
                            "explanation": new_explanation,
                            "embedding": new_embedding,
                            "synonyms": []
                        }
                        json_data.append(new_item)
                        json_map[term_b] = new_item
                        
                        print(f"  ✨ 多义拆分完成: {term_a} 和 {term_b}")
                        return False, None  # 表示发生了拆分
                        
                    except Exception as e:
                        print(f"  ❌ 拆分失败: {e}")
                        print(f"  ⚠️  保持为'{main_term}'的同义词")
                        return True, main_term
                else:
                    # 相似度高，确认为同义词
                    print(f"  ✅ 确认为'{main_term}'的同义词 (相似度 {sim:.4f} >= {polysemy_threshold})")
                    return True, main_term
            else:
                # 不检查多义词，直接跳过
                if main_term == term:
                    print(f"  ⚠️  术语已存在为主术语，跳过")
                else:
                    print(f"  ⚠️  术语已存在为'{main_term}'的同义词，跳过")
                return True, main_term
        
        # 2. 术语不存在，使用已提供的解释和嵌入
        if not embedding_vec:
            print(f"  ❌ 无法生成嵌入向量，跳过")
            return False, None
        
        print(f"  解释: {explanation[:50]}...")
        
        # 3. 与已有术语比较，寻找同义词
        best_match = None
        best_similarity = 0.0
        
        for existing_item in json_data:
            existing_term = existing_item["term"]
            existing_embedding = existing_item.get("embedding", [])
            existing_explanation = existing_item.get("explanation", "")
            
            if not existing_embedding:
                continue
            
            sim = self._cos(embedding_vec, existing_embedding)
            print(f"  与'{existing_term}'相似度: {sim:.4f}")
            
            # 高阈值：直接判定为同义词
            if synonym_threshold_high and sim >= synonym_threshold_high:
                print(f"  ✅ 高阈值判定: {term} ≈ {existing_term}")
                best_match = existing_term
                best_similarity = sim
                break
            
            # 低阈值：需要LLM确认
            if sim >= synonym_threshold_low and sim > best_similarity:
                prompt = f"""
请判断以下两个术语是否为『同义词』（表达同一概念的不同叫法）。

【术语A】'{existing_term}'
【解释A】'{existing_explanation}'

【术语B】'{term}'
【解释B】'{explanation}'

【相似度】'{sim:.4f}'

请回答"是"或"否"。只输出一个字。
"""
                llm = LLM(prompt=prompt, api_provider=self.api_provider)
                resp = llm.llm_call()
                ans = llm.extract_response(resp).strip()
                
                if "是" in ans:
                    print(f"  ✅ LLM确认: {term} ≈ {existing_term} (相似度: {sim:.4f})")
                    best_match = existing_term
                    best_similarity = sim
        
        # 4. 处理结果
        if best_match:
            # 添加为同义词
            existing_item = json_map[best_match]
            if term not in existing_item["synonyms"]:
                existing_item["synonyms"].append(term)
            json_map[term] = existing_item
            print(f"  ✅ 添加同义词: {term} → {best_match}")
            return True, best_match
        else:
            # 创建新条目
            new_item = {
                "term": term,
                "explanation": explanation,
                "embedding": embedding_vec,
                "synonyms": []
            }
            json_data.append(new_item)
            json_map[term] = new_item
            print(f"  ✨ 创建新术语: {term}")
            return False, None
    
    # =========================================================
    # 一词多义检测（同一术语多次出现）
    # =========================================================
    def check_polysemy(
        self,
        term: str,
        shared_context: Optional[str],
        indices: List[int],
        threshold: float = 0.73
    ) -> Optional[Tuple[str, str]]:
        """
        检测同一术语是否有多义
        
        Args:
            term: 待检测的术语
            shared_context: 包含所有术语的共享上下文
            indices: 该术语在原列表中的索引位置（用于多次出现）
            threshold: 多义判定阈值
        
        Returns:
            None 或 (新术语A, 新术语B)
        """
        if len(indices) < 2:
            return None
        
        print(f"\n[多义检测] {term} 出现 {len(indices)} 次")
        
        # 为每个出现生成解释（使用相同的共享上下文）
        # 注意：这里必须单独调用，因为需要多次解释同一个术语以检测多义性
        explanations = []
        embeddings = []
        
        for idx in indices:
            exp, emb = self.generate_explanation_and_embedding(term, shared_context)
            if exp and emb:
                explanations.append(exp)
                embeddings.append(emb)
        
        if len(embeddings) < 2:
            return None
        
        # 比较不同出现的相似度
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cos(embeddings[i], embeddings[j])
                print(f"  比较出现 {i+1} 和出现 {j+1}: 相似度={sim:.4f}")
                
                if sim < threshold:
                    print(f"  🎯 检测到多义: 相似度={sim:.4f} < 阈值={threshold}")
                    
                    # 拆分
                    try:
                        term_a, term_b = self.disambiguate_term_to_two(
                            term, explanations[i], explanations[j]
                        )
                        print(f"  ✂️  拆分为: '{term_a}' 和 '{term_b}'")
                        return term_a, term_b
                    except Exception as e:
                        print(f"  ❌ 拆分失败: {e}")
        
        print(f"  ✅ 无多义")
        return None
    
    # =========================================================
    # 生成def聚类三元组
    # =========================================================
    def _generate_def_cluster_triples(
        self,
        json_data: List[Dict],
        cluster_json_path: str | Path,
        old_json_data: List[Dict]
    ):
        """
        比较新旧术语数据，为有变化的术语生成"def聚类"三元组
        
        Args:
            json_data: 当前的术语数据
            cluster_json_path: 三元组JSON文件路径
            old_json_data: 处理前的术语数据（用于比对变化）
        """
        cluster_path = Path(cluster_json_path)
        
        # 读取已有的三元组
        existing_triples = {}
        if cluster_path.exists():
            try:
                existing_triples = json.loads(cluster_path.read_text(encoding="utf-8"))
                if not isinstance(existing_triples, dict):
                    existing_triples = {}
            except:
                existing_triples = {}
        
        # 构建旧数据的术语->同义词映射
        old_term_map = {}
        for item in old_json_data:
            main_term = item.get("term", "")
            synonyms = item.get("synonyms", [])
            if synonyms:
                old_term_map[main_term] = set(synonyms)
        
        # 检查哪些术语需要更新
        terms_to_update = []
        new_synonyms_count = 0
        new_terms_with_synonyms = 0
        
        for item in json_data:
            main_term = item["term"]
            current_synonyms = item.get("synonyms", [])
            
            # 只处理有同义词的术语
            if not current_synonyms:
                # 如果该术语之前在三元组中，但现在没有同义词了，删除它
                if main_term in existing_triples:
                    del existing_triples[main_term]
                    print(f"  🗑️  删除无同义词术语: {main_term}")
                continue
            
            # 检查是否有变化
            old_synonyms = old_term_map.get(main_term, set())
            current_synonyms_set = set(current_synonyms)
            
            if main_term not in old_term_map:
                # 新增带同义词的术语
                terms_to_update.append(main_term)
                new_terms_with_synonyms += 1
                print(f"  ✨ 新增带同义词术语: {main_term} (同义词: {len(current_synonyms)})")
            elif current_synonyms_set != old_synonyms:
                # 同义词列表有变化
                terms_to_update.append(main_term)
                added = current_synonyms_set - old_synonyms
                removed = old_synonyms - current_synonyms_set
                if added:
                    new_synonyms_count += len(added)
                    print(f"  🔄 更新术语: {main_term} (新增同义词: {added})")
                if removed:
                    print(f"  🔄 更新术语: {main_term} (删除同义词: {removed})")
        
        # 为需要更新的术语生成三元组
        if terms_to_update:
            print(f"\n  需要更新的术语数: {len(terms_to_update)}")
            
            for main_term in terms_to_update:
                # 找到对应的术语项
                item = next((x for x in json_data if x["term"] == main_term), None)
                if not item:
                    continue
                
                synonyms = item.get("synonyms", [])
                if not synonyms:
                    continue
                
                # 生成链式三元组：term -> syn1 -> syn2 -> ...
                all_terms = [main_term] + synonyms
                triples_for_term = []
                
                for i in range(len(all_terms) - 1):
                    triple = {
                        "head": all_terms[i],
                        "relation": "def聚类",
                        "tail": all_terms[i + 1]
                    }
                    triples_for_term.append(triple)
                
                existing_triples[main_term] = triples_for_term
        
        # 保存
        cluster_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_path.write_text(
            json.dumps(existing_triples, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\n[三元组保存] 已保存至 {cluster_json_path}")
        print(f"  总术语数（有同义词）: {len(existing_triples)}")
        print(f"  新增带同义词术语: {new_terms_with_synonyms}")
        print(f"  新增同义词数: {new_synonyms_count}")
    
    # =========================================================
    # 🆕 完整流程：增量处理（批量优化版）
    # =========================================================
    def process_terms_pipeline(
        self,
        terms: List[str],
        json_path: Optional[str | Path] = None,
        synonym_threshold_low: float = 0.73,
        synonym_threshold_high: Optional[float] = 0.85,
        polysemy_threshold: float = 0.73,
        shared_context: Optional[str] = None,
        force_polysemy_check: bool = True
    ) -> dict:
        """
        增量处理术语列表（批量优化版）
        
        Args:
            terms: 待处理的术语列表
            json_path: JSON文件路径
            synonym_threshold_low: 同义词判定低阈值
            synonym_threshold_high: 同义词判定高阈值
            polysemy_threshold: 多义词判定阈值
            shared_context: 包含所有术语的共享上下文
            force_polysemy_check: 是否强制检查一词多义
        """
        print("\n" + "=" * 60)
        print("术语处理流程 - 批量优化版")
        print("=" * 60)
        print(f"输入术语数: {len(terms)}")
        print(f"同义词判定阈值: {synonym_threshold_low} ~ {synonym_threshold_high or '无上限'}")
        print(f"多义词拆分阈值: {polysemy_threshold}")
        print(f"一词多义检测: {'启用' if force_polysemy_check else '禁用'}")
        
        # 1. 加载已有数据
        json_data, json_map = self._load_json_terms(json_path)
        print(f"JSON中已有术语: {len(json_data)}")
        
        # 2. 检测同一术语在输入中多次出现
        term_indices: Dict[str, List[int]] = {}
        for idx, t in enumerate(terms):
            term_indices.setdefault(t, []).append(idx)
        
        # 对多次出现的术语进行多义检测
        disambiguations = []
        for t, indices in term_indices.items():
            if len(indices) > 1:
                result = self.check_polysemy(t, shared_context, indices, polysemy_threshold)
                if result:
                    term_a, term_b = result
                    disambiguations.append((t, term_a, term_b))
                    # 替换原列表中的术语
                    for i, idx in enumerate(indices):
                        terms[idx] = term_a if i == 0 else term_b
        
        # 3. 🆕 批量生成解释和嵌入（核心优化）
        print("\n" + "=" * 60)
        print("🚀 批量生成解释和嵌入向量")
        print("=" * 60)
        
        # 收集需要生成解释的术语（排除已存在的）
        terms_to_explain = []
        for term in terms:
            if term not in json_map:
                terms_to_explain.append(term)
        
        print(f"需要生成解释的新术语: {len(terms_to_explain)}")
        
        # 批量生成解释和嵌入
        explanations_embeddings = {}
        if terms_to_explain:
            explanations_embeddings = self.generate_batch_explanations_and_embeddings(
                terms_to_explain, shared_context
            )
        
        # 4. 增量处理每个术语
        print("\n" + "=" * 60)
        print("开始增量处理术语")
        print("=" * 60)
        
        new_terms = []
        synonym_pairs = []
        
        for term in terms:
            # 获取预生成的解释和嵌入，如果没有则为空
            if term in explanations_embeddings:
                explanation, embedding_vec = explanations_embeddings[term]
            elif term in json_map:
                # 已存在的术语，从json_map获取
                existing = json_map[term]
                explanation = existing.get("explanation", "")
                embedding_vec = existing.get("embedding", [])
            else:
                # 批量生成失败的情况，单独生成
                print(f"  ⚠️  术语 '{term}' 批量生成失败，尝试单独生成")
                explanation, embedding_vec = self.generate_explanation_and_embedding(term, shared_context)
            
            is_synonym, representative = self.process_term_incrementally(
                term=term,
                explanation=explanation,
                embedding_vec=embedding_vec,
                shared_context=shared_context,
                json_data=json_data,
                json_map=json_map,
                synonym_threshold_low=synonym_threshold_low,
                synonym_threshold_high=synonym_threshold_high,
                polysemy_threshold=polysemy_threshold,
                force_polysemy_check=force_polysemy_check
            )
            
            if not is_synonym:
                new_terms.append(term)
            else:
                if representative:
                    synonym_pairs.append((term, representative))
        
        # 5. 保存更新
        if json_path:
            self._save_json_terms(json_path, json_data)
        
        # 6. 统计输出
        print("\n" + "=" * 60)
        print("处理完成")
        print("=" * 60)
        print(f"新增术语: {len(new_terms)}")
        print(f"同义词对数: {len(synonym_pairs)}")
        print(f"多义拆分: {len(disambiguations)}")
        
        return {
            "json_data": json_data,
            "json_map": json_map,
            "new_terms": new_terms,
            "synonym_pairs": synonym_pairs,
            "disambiguations": disambiguations
        }
    
    # =========================================================
    # Clusterer 函数：处理实体数组（无关系数组）
    # =========================================================
    def clusterer(
        self,
        terms: List[str],
        shared_context: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        概念聚类 - 只处理实体术语，返回术语的三元组
        
        Args:
            terms: 实体术语列表
            shared_context: 实体的共享上下文（可选）
        
        Returns:
            Dict[术语, 三元组列表]
        """
        # 关键修改：使用概念实体专用路径
        entity_json_path = self._get_concept_entity_json_path()  # ← 改为新方法
        cluster_json_path = self._get_cluster_json_path()
        
        print(f"\n" + "=" * 60)
        print(f"Clusterer 函数 - 概念聚类")
        print(f"概念实体路径: {entity_json_path}")
        print(f"聚类输出路径: {cluster_json_path}")
        print("=" * 60)
        
        # 保存处理前的数据
        old_json_data, _ = self._load_json_terms(entity_json_path)
        
        # 1. 处理实体数组
        print("\n" + "-" * 60)
        print("处理概念实体")
        print("-" * 60)
        print(f"概念实体路径: {entity_json_path}")
        print(f"输入实体数: {len(terms)}")
        
        result = self.process_terms_pipeline(
            terms=terms,
            json_path=entity_json_path,
            synonym_threshold_low=0.69,
            synonym_threshold_high=0.84,
            polysemy_threshold=0.75,
            shared_context=shared_context,
            force_polysemy_check=True
        )
        
        # 2. 检查更新并生成def聚类三元组
        print("\n" + "=" * 60)
        print("检查更新并生成三元组")
        print("=" * 60)
        self._generate_def_cluster_triples(
            json_data=result["json_data"],
            cluster_json_path=cluster_json_path,
            old_json_data=old_json_data
        )
        
        print("\n" + "=" * 60)
        print("Clusterer 完成")
        print("=" * 60)
        
        return result
    # Disambiguate 函数：处理实体和关系数组
    # =========================================================
    def Disambiguate(
        self,
        entity_terms: List[str],
        relation_terms: List[str],
        entity_shared_context: Optional[str] = None,
        relation_shared_context: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        处理实体和关系术语数组，并返回消歧后的数组
        
        Args:
            entity_terms: 实体术语数组
            relation_terms: 关系术语数组
            entity_shared_context: 实体的共享上下文（可选）
            relation_shared_context: 关系的共享上下文（可选）
        
        Returns:
            (更新后的实体数组, 更新后的关系数组)
        """
        # 使用辅助方法生成路径（仅这两行修改）
        entity_json_path = self._get_entity_json_path()
        relation_json_path = self._get_relation_json_path()
        
        print(f"\n" + "=" * 60)
        print(f"Disambiguate 函数 - 实体与关系消歧")
        print(f"实体JSON路径: {entity_json_path}")
        print(f"关系JSON路径: {relation_json_path}")
        print("=" * 60)
        
        # =========================================================
        # 1. 处理实体数组
        # =========================================================
        print("\n" + "-" * 60)
        print("处理实体术语")
        print("-" * 60)
        print(f"实体JSON路径: {entity_json_path}")
        print(f"输入实体数: {len(entity_terms)}")
        
        entity_result = self.process_terms_pipeline(
            terms=entity_terms,
            json_path=entity_json_path,
            synonym_threshold_low=0.69,
            synonym_threshold_high=0.84,
            polysemy_threshold=0.75,
            shared_context=entity_shared_context,
            force_polysemy_check=True
        )
        
        # =========================================================
        # 2. 处理关系数组
        # =========================================================
        print("\n" + "-" * 60)
        print("处理关系术语")
        print("-" * 60)
        print(f"关系JSON路径: {relation_json_path}")
        print(f"输入关系数: {len(relation_terms)}")
        
        relation_result = self.process_terms_pipeline(
            terms=relation_terms,
            json_path=relation_json_path,
            synonym_threshold_low=0.62,
            synonym_threshold_high=0.92,
            polysemy_threshold=0.72,
            shared_context=relation_shared_context,
            force_polysemy_check=True
        )
        
        # =========================================================
        # 3. 构建映射并替换数组中的术语
        # =========================================================
        print("\n" + "-" * 60)
        print("替换数组中的同义词为主术语")
        print("-" * 60)
        
        # 读取最新的JSON数据
        entity_json_data, entity_json_map = self._load_json_terms(entity_json_path)
        relation_json_data, relation_json_map = self._load_json_terms(relation_json_path)
        
        # 替换实体数组
        updated_entity_terms = []
        entity_replacements = 0
        for term in entity_terms:
            if term in entity_json_map:
                main_term = entity_json_map[term]["term"]
                updated_entity_terms.append(main_term)
                if main_term != term:
                    entity_replacements += 1
            else:
                updated_entity_terms.append(term)
        
        # 替换关系数组
        updated_relation_terms = []
        relation_replacements = 0
        for term in relation_terms:
            if term in relation_json_map:
                main_term = relation_json_map[term]["term"]
                updated_relation_terms.append(main_term)
                if main_term != term:
                    relation_replacements += 1
            else:
                updated_relation_terms.append(term)
        
        # =========================================================
        # 4. 输出统计
        # =========================================================
        print("\n" + "=" * 60)
        print("Disambiguate 完成")
        print("=" * 60)
        print(f"实体术语:")
        print(f"  输入数量: {len(entity_terms)}")
        print(f"  输出数量: {len(updated_entity_terms)}")
        print(f"  替换数量: {entity_replacements}")
        print(f"关系术语:")
        print(f"  输入数量: {len(relation_terms)}")
        print(f"  输出数量: {len(updated_relation_terms)}")
        print(f"  替换数量: {relation_replacements}")
        
        return updated_entity_terms, updated_relation_terms
# =========================================================
# 测试主函数（可选）
# =========================================================
if __name__ == "__main__":
    print("术语消歧器 - 批量优化版测试")
    print("=" * 60)
    
    disambiguator = TermDisambiguator(api_provider="qianwen")
    
    # 测试批量解释功能
    test_terms = ["苹果", "橙子", "香蕉", "机器学习", "深度学习"]
    
    result = disambiguator.generate_batch_explanations_and_embeddings(
        terms=test_terms,
        shared_context="这是一些水果和AI术语"
    )
    
    print("\n批量解释结果:")
    for term, (explanation, embedding) in result.items():
        print(f"  {term}: {explanation}")
        print(f"    嵌入维度: {len(embedding)}")
    
    print("\n测试完成")