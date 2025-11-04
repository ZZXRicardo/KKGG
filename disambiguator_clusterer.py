# 文件：解释聚类削歧_类封装版_修复一词多义.py
# ✅ 修复一词多义检测失效问题

from LLM import LLM
from Embedding import Embedding
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field


# =========================================================
# 主类：术语消歧器
# =========================================================
class TermDisambiguator:
    """术语消歧与聚类系统"""
    
    # =========================================================
    # 内部数据容器类
    # =========================================================
    @dataclass
    class TermEntry:
        term: str = ""
        explanation: str = ""
        embedding: List[float] = field(default_factory=list)
        synonyms: List[str] = field(default_factory=list)
    
    def __init__(self, api_provider: str = "qianwen"):
        """
        初始化消歧器
        
        Args:
            api_provider: API提供商，默认为"qianwen"
        """
        self.api_provider = api_provider
        self.term_entries: List[TermDisambiguator.TermEntry] = []
        self.term_entry_map: Dict[str, TermDisambiguator.TermEntry] = {}
    
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
    # 生成单个术语的解释和嵌入
    # =========================================================
    def generate_explanation_and_embedding(
        self,
        term: str,
        shared_context: Optional[str] = None
    ) -> Tuple[str, List[float]]:
        """
        为单个术语生成解释和嵌入向量
        
        Args:
            term: 待解释的术语
            shared_context: 包含所有术语的共享上下文（非单个术语的专属上下文）
        """
        base_instruction = (
            "你是给高中生写词条定义的老师，请用『一句中文』解释给定术语，"
            "不要出现术语本身，也不要用「X 是……」的句式；直接给出定义内容。"
            "不要在解释的时候引入他的同义词，比如A是B的另一种称呼。"
            "做这些解释的时候不要参考你做的其他解释，每个词语独立解释，当作每次只处理一个任务。"
            "如果提供了共享上下文，请从上下文中提取该术语的语义进行消歧。\n"
            "【示例-正确】「一种能自动处理信息的电子设备。」\n"
            "【示例-错误】「电脑是一种能自动处理信息的电子设备。」（包含术语）"
        )
        
        if shared_context:
            prompt = (
                f"{base_instruction}\n\n"
                f"【术语】'{term}'\n"
                f"【共享上下文（包含多个术语，请从中提取该术语的语义）】\n{shared_context}\n\n"
                "只输出一句中文解释。不要出现术语本身。"
            )
        else:
            prompt = (
                f"{base_instruction}\n\n"
                f"【术语】'{term}'\n"
                "只输出一句中文解释。不要出现术语本身。"
            )
        
        # 生成解释
        llm_instance = LLM(prompt=prompt, api_provider=self.api_provider)
        result = llm_instance.llm_call()
        explanation = llm_instance.extract_response(result)
        explanation = explanation.strip() if isinstance(explanation, str) else ""
        
        # 生成嵌入向量
        embedding_vec = []
        if explanation:
            embedding = Embedding(input_texts=[explanation])
            embedding_result = embedding.embedding_call()
            vectors = embedding.extract_embeddings(embedding_result)
            if isinstance(vectors, list) and vectors:
                embedding_vec = vectors[0]
        
        return explanation, embedding_vec
    
    # =========================================================
    # 核心：增量处理术语（逐个判断是否为同义词）- 🔧 已修复一词多义检测
    # =========================================================
    def process_term_incrementally(
        self,
        term: str,
        shared_context: Optional[str],
        json_data: List[Dict],
        json_map: Dict[str, Dict],
        synonym_threshold_low: float = 0.73,
        synonym_threshold_high: Optional[float] = 0.85,
        polysemy_threshold: float = 0.73,
        force_polysemy_check: bool = True  # 🔧 新增：强制检查多义词
    ) -> Tuple[bool, Optional[str]]:
        """
        增量处理单个术语（已修复一词多义检测）
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
                
                # 基于新上下文生成解释
                new_explanation, new_embedding = self.generate_explanation_and_embedding(
                    term, shared_context
                )
                
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
        
        # 2. 术语不存在，正常处理（原有逻辑）
        explanation, embedding_vec = self.generate_explanation_and_embedding(
            term, shared_context
        )
        
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
                    term_a, term_b = self.disambiguate_term_to_two(
                        term, explanations[i], explanations[j]
                    )
                    print(f"  ✂️  拆分为: '{term_a}' 和 '{term_b}'")
                    return term_a, term_b
        
        print(f"  ✅ 无多义")
        return None
    
    # =========================================================
    # 生成 def聚类 三元组（只包含有同义词的术语，增量更新）
    # =========================================================
    def _generate_def_cluster_triples(
        self,
        json_data: List[Dict],
        cluster_json_path: str | Path,
        old_json_data: List[Dict]
    ):
        """
        生成 def聚类 三元组，只包含有同义词的术语，并进行增量更新
        
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
    # 完整流程：增量处理
    # =========================================================
    def process_terms_pipeline(
        self,
        terms: List[str],
        json_path: Optional[str | Path] = None,
        synonym_threshold_low: float = 0.73,
        synonym_threshold_high: Optional[float] = 0.85,
        polysemy_threshold: float = 0.73,
        shared_context: Optional[str] = None,
        force_polysemy_check: bool = True  # 🔧 新增参数
    ) -> dict:
        """
        增量处理术语列表
        
        Args:
            terms: 待处理的术语列表
            json_path: JSON文件路径
            synonym_threshold_low: 同义词判定低阈值
            synonym_threshold_high: 同义词判定高阈值
            polysemy_threshold: 多义词判定阈值
            shared_context: 包含所有术语的共享上下文
            force_polysemy_check: 是否强制检查一词多义（新增）
        """
        print("\n" + "=" * 60)
        print("术语处理流程 - 增量版")
        print("=" * 60)
        print(f"输入术语数: {len(terms)}")
        print(f"同义词判定阈值: {synonym_threshold_low} ~ {synonym_threshold_high or '无上限'}")
        print(f"多义词拆分阈值: {polysemy_threshold}")
        print(f"一词多义检测: {'启用' if force_polysemy_check else '禁用'}")  # 🔧 显示状态
        
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
        
        # 3. 增量处理每个术语
        print("\n" + "=" * 60)
        print("开始增量处理术语")
        print("=" * 60)
        
        new_terms = []
        synonym_pairs = []
        
        for term in terms:
            is_synonym, representative = self.process_term_incrementally(
                term=term,
                shared_context=shared_context,
                json_data=json_data,
                json_map=json_map,
                synonym_threshold_low=synonym_threshold_low,
                synonym_threshold_high=synonym_threshold_high,
                polysemy_threshold=polysemy_threshold,
                force_polysemy_check=force_polysemy_check  # 🔧 传递参数
            )
            
            if is_synonym:
                synonym_pairs.append((term, representative))
            else:
                new_terms.append(term)
        
        # 4. 保存
        self._save_json_terms(json_path, json_data)
        
        print("\n" + "=" * 60)
        print("处理完成")
        print("=" * 60)
        print(f"总术语数（JSON中）: {len(json_data)}")
        print(f"新增术语: {len(new_terms)}")
        print(f"识别为同义词: {len(synonym_pairs)}")
        print(f"多义拆分: {len(disambiguations)}")
        
        return {
            "json_data": json_data,
            "new_terms": new_terms,
            "synonym_pairs": synonym_pairs,
            "disambiguations": disambiguations
        }
    
    # =========================================================
    # Clusterer 函数：聚类 + 三元组生成（完整流程）
    # =========================================================
    def clusterer(
        self,
        terms: List[str],
        shared_context: Optional[str] = None
    ) -> dict:
        """
        Clusterer 函数：对术语进行聚类，识别同义词，并生成 def聚类 三元组
        
        Args:
            terms: 待处理的术语列表
            shared_context: 包含所有术语的共享上下文
        
        Returns:
            处理结果字典
        """
        # 固定参数
        json_path = r"E:\KKGG\output\terms\definitions.json"
        cluster_json_path = r"E:\KKGG\output\KG\def_cluster.json"
        synonym_threshold_low = 0.73
        synonym_threshold_high = 0.85
        polysemy_threshold = 0.73
        
        print("\n" + "=" * 60)
        print("Clusterer 函数 - 术语聚类与三元组生成")
        print("=" * 60)
        print(f"术语定义JSON: {json_path}")
        print(f"三元组JSON: {cluster_json_path}")
        
        # 0. 保存处理前的JSON数据用于比对
        old_json_data, _ = self._load_json_terms(json_path)
        print(f"处理前术语数: {len(old_json_data)}")
        
        # 1. 调用流程函数处理术语
        result = self.process_terms_pipeline(
            terms=terms,
            json_path=json_path,
            synonym_threshold_low=synonym_threshold_low,
            synonym_threshold_high=synonym_threshold_high,
            polysemy_threshold=polysemy_threshold,
            shared_context=shared_context,
            force_polysemy_check=True  # 🔧 启用一词多义检测
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
    
    # =========================================================
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
            entity_shared_context: 实体的共享上下文（可选，包含所有实体术语）
            relation_shared_context: 关系的共享上下文（可选，包含所有关系术语）
        
        Returns:
            (更新后的实体数组, 更新后的关系数组)
            - 如果术语作为同义词出现，则替换为主术语
        """
        print("\n" + "=" * 60)
        print("Disambiguate 函数 - 实体与关系消歧")
        print("=" * 60)
        
        # 固定参数
        entity_json_path = r"E:\KKGG\output\terms\Entity.json"
        relation_json_path = r"E:\KKGG\output\terms\Relation.json"
        
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
            synonym_threshold_low=0.73,
            synonym_threshold_high=0.89,
            polysemy_threshold=0.73,
            shared_context=entity_shared_context,
            force_polysemy_check=True  # 🔧 启用一词多义检测
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
            synonym_threshold_low=0.73,
            synonym_threshold_high=0.85,
            polysemy_threshold=0.73,
            shared_context=relation_shared_context,
            force_polysemy_check=True  # 🔧 启用一词多义检测
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
# 测试主函数 - 🔧 已添加一词多义跨批次测试
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("术语消歧与聚类系统 - 三大功能测试")
    print("=" * 60)
    
    # =========================================================
    # 测试1: Clusterer 函数测试
    # =========================================================
    print("\n" + "=" * 60)
    print("测试1: Clusterer 函数 - 术语聚类与三元组生成")
    print("=" * 60)
    
    clusterer_disambiguator = TermDisambiguator(api_provider="qianwen")
    
    # 测试术语和上下文
    test1_terms = ["机器学习", "ML", "深度学习", "神经网络", "人工智能", "AI"]
    test1_shared_context = (
        "机器学习是人工智能的一个分支，让计算机通过数据学习。"
        "ML是Machine Learning的缩写，指机器学习。"
        "深度学习使用多层神经网络进行学习。"
        "神经网络模拟人脑神经元结构。"
        "人工智能是计算机科学的一个领域。"
        "AI是Artificial Intelligence的简称。"
    )
    
    print(f"输入术语: {test1_terms}")
    print(f"共享上下文长度: {len(test1_shared_context)} 字符")
    
    # 调用clusterer函数
    result1 = clusterer_disambiguator.clusterer(
        terms=test1_terms,
        shared_context=test1_shared_context
    )
    
    # 查看结果
    definitions_path = r"E:\KKGG\output\terms\definitions.json"
    cluster_path = r"E:\KKGG\output\KG\def_cluster.json"
    
    print("\n[术语定义JSON - 前5个]")
    try:
        with open(definitions_path, "r", encoding="utf-8") as f:
            definitions_data = json.load(f)
        for item in definitions_data[:5]:
            print(f"  术语: {item['term']}")
            print(f"  解释: {item.get('explanation', '')[:50]}...")
            print(f"  同义词: {item['synonyms']}")
            print()
    except Exception as e:
        print(f"  读取失败: {e}")
    
    print("\n[三元组JSON - 前3个术语]")
    try:
        with open(cluster_path, "r", encoding="utf-8") as f:
            cluster_data = json.load(f)
        for term, triples in list(cluster_data.items())[:3]:
            print(f"  术语: {term}")
            for triple in triples:
                print(f"    ({triple['head']}, {triple['relation']}, {triple['tail']})")
            print()
    except Exception as e:
        print(f"  读取失败: {e}")
    
    print("✅ 测试1完成 - Clusterer函数")
    
    # =========================================================
    # 🔧 新增测试1.5: 跨批次一词多义检测
    # =========================================================
    print("\n" + "=" * 60)
    print("测试1.5: 跨批次一词多义检测 - ML作为容量单位")
    print("=" * 60)
    
    # 第二批：ML作为容量单位
    test1_5_terms = ["毫升", "ML", "升", "立方厘米"]
    test1_5_shared_context = (
        "毫升是容量单位，常用于液体测量。"
        "ML是milliliter的英文缩写，表示毫升。"
        "升是公制容量单位，1升等于1000毫升。"
        "立方厘米是体积单位，1立方厘米等于1毫升。"
    )
    
    print(f"输入术语: {test1_5_terms}")
    print(f"共享上下文长度: {len(test1_5_shared_context)} 字符")
    
    # 调用clusterer函数（使用相同的实例和JSON文件）
    result1_5 = clusterer_disambiguator.clusterer(
        terms=test1_5_terms,
        shared_context=test1_5_shared_context
    )
    
    print("\n[更新后的术语定义JSON - 所有]")
    try:
        with open(definitions_path, "r", encoding="utf-8") as f:
            definitions_data = json.load(f)
        for item in definitions_data:
            print(f"  术语: {item['term']}")
            print(f"  解释: {item.get('explanation', '')[:60]}...")
            print(f"  同义词: {item['synonyms']}")
            print()
    except Exception as e:
        print(f"  读取失败: {e}")
    
    print("✅ 测试1.5完成 - 跨批次一词多义检测")
    
    # =========================================================
    # 测试2: Disambiguate 函数测试
    # =========================================================
    print("\n" + "=" * 60)
    print("测试2: Disambiguate 函数 - 实体与关系术语消歧")
    print("=" * 60)
    
    disambiguate_instance = TermDisambiguator(api_provider="qianwen")
    
    # 测试数据
    test2_entity_terms = ["苹果公司", "Apple", "微软", "Microsoft", "谷歌", "Google"]
    test2_relation_terms = ["创建", "建立", "成立", "拥有", "持有", "开发", "研发"]
    
    test2_entity_shared_context = (
        "苹果公司是一家美国科技公司，生产iPhone和Mac。"
        "Apple是全球知名的科技企业，总部在加州。"
        "微软是软件公司，开发Windows系统。"
        "Microsoft开发了Office办公软件。"
        "谷歌是搜索引擎公司。"
        "Google提供互联网服务。"
    )
    
    test2_relation_shared_context = (
        "史蒂夫·乔布斯创建了苹果公司。"
        "史蒂夫·乔布斯建立了苹果公司。"
        "比尔·盖茨成立了微软。"
        "比尔·盖茨拥有微软股份。"
        "投资者持有公司股票。"
        "苹果开发了iOS系统。"
        "微软研发了Azure云平台。"
    )
    
    print(f"输入实体术语: {test2_entity_terms}")
    print(f"输入关系术语: {test2_relation_terms}")
    
    # 调用 Disambiguate 函数
    updated_entities, updated_relations = disambiguate_instance.Disambiguate(
        entity_terms=test2_entity_terms,
        relation_terms=test2_relation_terms,
        entity_shared_context=test2_entity_shared_context,
        relation_shared_context=test2_relation_shared_context
    )
    
    # 显示结果
    print("\n[更新后的数组对比]")
    print(f"原始实体 ({len(test2_entity_terms)}): {test2_entity_terms}")
    print(f"更新实体 ({len(updated_entities)}): {updated_entities}")
    print(f"\n原始关系 ({len(test2_relation_terms)}): {test2_relation_terms}")
    print(f"更新关系 ({len(updated_relations)}): {updated_relations}")
    
    # 查看JSON文件
    entity_json_path = r"E:\KKGG\output\terms\Entity.json"
    relation_json_path = r"E:\KKGG\output\terms\Relation.json"
    
    print("\n[实体JSON内容 - 所有]")
    try:
        with open(entity_json_path, "r", encoding="utf-8") as f:
            entity_data = json.load(f)
        for item in entity_data:
            print(f"  术语: {item['term']}, 同义词: {item['synonyms']}")
    except Exception as e:
        print(f"  读取失败: {e}")
    
    print("\n[关系JSON内容 - 所有]")
    try:
        with open(relation_json_path, "r", encoding="utf-8") as f:
            relation_data = json.load(f)
        for item in relation_data:
            print(f"  术语: {item['term']}, 同义词: {item['synonyms']}")
    except Exception as e:
        print(f"  读取失败: {e}")
    
    print("✅ 测试2完成 - Disambiguate函数")
    
    # =========================================================
    # 测试3: 一词多义检测测试
    # =========================================================
    print("\n" + "=" * 60)
    print("测试3: 一词多义检测 - 分批次处理同一术语")
    print("=" * 60)
    
    polysemy_disambiguator = TermDisambiguator(api_provider="qianwen")
    json_polysemy_path = r"E:\KKGG\output\terms\polysemy_test.json"
    
    # 第一批：苹果（水果）
    print("\n--- 第一批次：苹果作为水果 ---")
    batch1_terms = ["苹果", "香蕉", "橙子"]
    batch1_context = (
        "苹果是一种常见的水果，富含维生素C和膳食纤维，口感清甜爽脆。"
        "香蕉是热带水果，含有丰富的钾元素。"
        "橙子是柑橘类水果，富含维生素C。"
    )
    
    print(f"输入术语: {batch1_terms}")
    result3_batch1 = polysemy_disambiguator.process_terms_pipeline(
        terms=batch1_terms,
        json_path=json_polysemy_path,
        synonym_threshold_low=0.70,
        synonym_threshold_high=0.85,
        polysemy_threshold=0.73,
        shared_context=batch1_context,
        force_polysemy_check=True  # 🔧 启用一词多义检测
    )
    
    print(f"\n第一批结果 - 新增术语: {result3_batch1['new_terms']}")
    print(f"第一批结果 - 同义词对: {result3_batch1['synonym_pairs']}")
    
    # 第二批：苹果（公司）
    print("\n--- 第二批次：苹果作为公司 ---")
    batch2_terms = ["苹果", "微软", "谷歌"]
    batch2_context = (
        "苹果公司是全球知名的科技企业，总部位于加州库比蒂诺，主要产品包括iPhone、iPad和Mac。"
        "微软是世界最大的软件公司之一，开发了Windows操作系统。"
        "谷歌是互联网搜索引擎公司，提供各种在线服务。"
    )
    
    print(f"输入术语: {batch2_terms}")
    result3_batch2 = polysemy_disambiguator.process_terms_pipeline(
        terms=batch2_terms,
        json_path=json_polysemy_path,
        synonym_threshold_low=0.70,
        synonym_threshold_high=0.85,
        polysemy_threshold=0.73,
        shared_context=batch2_context,
        force_polysemy_check=True  # 🔧 启用一词多义检测
    )
    
    print(f"\n第二批结果 - 新增术语: {result3_batch2['new_terms']}")
    print(f"第二批结果 - 同义词对: {result3_batch2['synonym_pairs']}")
    print(f"第二批结果 - 多义拆分: {result3_batch2['disambiguations']}")
    
    # 显示最终的JSON内容
    print("\n[最终JSON内容 - 所有术语]")
    try:
        with open(json_polysemy_path, "r", encoding="utf-8") as f:
            polysemy_data = json.load(f)
        for item in polysemy_data:
            print(f"  术语: {item['term']}")
            print(f"  解释: {item.get('explanation', '')[:60]}...")
            print(f"  同义词: {item['synonyms']}")
            print()
    except Exception as e:
        print(f"  读取失败: {e}")
    
    print("✅ 测试3完成 - 一词多义检测")
    
    # =========================================================
    # 总结
    # =========================================================
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    print("测试文件保存位置:")
    print(f"  1. Clusterer - 术语定义: {definitions_path}")
    print(f"  2. Clusterer - 三元组: {cluster_path}")
    print(f"  3. Disambiguate - 实体: {entity_json_path}")
    print(f"  4. Disambiguate - 关系: {relation_json_path}")
    print(f"  5. 一词多义: {json_polysemy_path}")
    print("=" * 60)