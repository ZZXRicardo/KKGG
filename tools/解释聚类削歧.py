# 文件：解释聚类削歧_修改版.py
# ✅ 修复：同义词只保留一个代表性术语的JSON对象
# ✅ 修改1：JSON输出路径改为 E:\流程\sources\terms.json
# ✅ 修改2：添加多义词检测测试

from LLM import LLM
from Emdedding import Embedding
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field


# =========================================================
# 1) 数据容器类
# =========================================================
@dataclass
class TermEntry:
    term: str = ""
    explanation: str = ""
    embedding: List[float] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)


# 全局暂存
TERM_ENTRIES: List[TermEntry] = []
TERM_ENTRY_MAP: Dict[str, TermEntry] = {}


# =========================================================
# 2) 辅助函数：余弦相似度
# =========================================================
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
# 3) JSON 读写
# =========================================================
def _load_json_terms(json_path: Optional[str | Path]) -> Tuple[List[Dict], Dict[str, Dict]]:
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


def _save_json_terms(json_path: Optional[str | Path], data: List[Dict]):
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
# 4) 一词多义拆分
# =========================================================
def disambiguate_term_to_two(term, explanation_a, explanation_b, api_provider="qianwen"):
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
    llm = LLM(prompt=prompt, api_provider=api_provider)
    resp = llm.llm_call()
    text = llm.extract_response(resp)

    obj = json.loads(text)
    term_a = (obj.get("term_a") or "").strip()
    term_b = (obj.get("term_b") or "").strip()
    if not term_a or not term_b:
        raise RuntimeError(f"JSON缺少term_a/term_b或为空：{text!r}")
    return term_a, term_b


# =========================================================
# 5) 生成单个术语的解释和嵌入
# =========================================================
def generate_explanation_and_embedding(
    term: str,
    api_provider: str = "qianwen",
    context: Optional[str] = None
) -> Tuple[str, List[float]]:
    """为单个术语生成解释和嵌入向量"""
    base_instruction = (
        "你是给高中生写词条定义的老师，请用『一句中文』解释给定术语，"
        "不要出现术语本身，也不要用「X 是……」的句式；直接给出定义内容。"
        "不要在解释的时候引入他的同义词，比如A是B的另一种称呼。"
        "做这些解释的时候不要参考你做的其他解释，每个词语独立解释，当作每次只处理一个任务。"
        "如果提供了上下文，请以其中语境为准进行消歧。\n"
        "【示例-正确】「一种能自动处理信息的电子设备。」\n"
        "【示例-错误】「电脑是一种能自动处理信息的电子设备。」（包含术语）"
    )

    if context:
        prompt = (
            f"{base_instruction}\n\n"
            f"【术语】'{term}'\n"
            f"【上下文/出处（仅供该术语消歧使用）】\n{context}\n\n"
            "只输出一句中文解释。不要出现术语本身。"
        )
    else:
        prompt = (
            f"{base_instruction}\n\n"
            f"【术语】'{term}'\n"
            "只输出一句中文解释。不要出现术语本身。"
        )

    # 生成解释
    llm_instance = LLM(prompt=prompt, api_provider=api_provider)
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
# 6) ✅ 核心：增量处理术语（逐个判断是否为同义词）
# =========================================================
def process_term_incrementally(
    term: str,
    context: Optional[str],
    json_data: List[Dict],
    json_map: Dict[str, Dict],
    api_provider: str = "qianwen",
    synonym_threshold_low: float = 0.73,
    synonym_threshold_high: Optional[float] = 0.85,
    polysemy_threshold: float = 0.73
) -> Tuple[bool, Optional[str]]:
    """
    增量处理单个术语
    返回: (是否为同义词, 代表术语)
    - 如果是同义词: (True, 代表术语)
    - 如果不是同义词: (False, None)
    """
    print(f"\n[处理术语] {term}")
    
    # 1. 检查是否已存在（作为主术语或同义词）
    if term in json_map:
        existing = json_map[term]
        main_term = existing["term"]
        if main_term == term:
            print(f"  ⚠️  术语已存在为主术语，跳过")
        else:
            print(f"  ⚠️  术语已存在为'{main_term}'的同义词，跳过")
        return True, main_term
    
    # 2. 生成解释和嵌入
    explanation, embedding_vec = generate_explanation_and_embedding(
        term, api_provider, context
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
        
        sim = _cos(embedding_vec, existing_embedding)
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
            llm = LLM(prompt=prompt, api_provider=api_provider)
            resp = llm.llm_call()
            ans = llm.extract_response(resp).strip()
            
            if "是" in ans:
                print(f"  ✅ LLM确认: {term} ≈ {existing_term} (相似度: {sim:.4f})")
                best_match = existing_term
                best_similarity = sim
                break
    
    # 4. 如果找到同义词，添加到其synonyms列表
    if best_match:
        existing_item = json_map[best_match]
        if term not in existing_item["synonyms"]:
            existing_item["synonyms"].append(term)
            # 更新映射：新术语也指向同一个对象
            json_map[term] = existing_item
        print(f"  ➕ 添加'{term}'为'{best_match}'的同义词")
        return True, best_match
    
    # 5. 不是同义词，创建新对象
    print(f"  🆕 创建新术语对象")
    new_item = {
        "term": term,
        "explanation": explanation,
        "embedding": embedding_vec,
        "synonyms": []
    }
    json_data.append(new_item)
    json_map[term] = new_item
    
    return False, None


# =========================================================
# 7) ✅ 一词多义检测（同一术语多次出现）
# =========================================================
def check_polysemy(
    term: str,
    contexts: List[Optional[str]],
    indices: List[int],
    api_provider: str = "qianwen",
    threshold: float = 0.73
) -> Optional[Tuple[str, str]]:
    """
    检测同一术语是否有多义
    返回: None 或 (新术语A, 新术语B)
    """
    if len(indices) < 2:
        return None
    
    print(f"\n[多义检测] {term} 出现 {len(indices)} 次")
    
    # 为每个出现生成解释
    explanations = []
    embeddings = []
    
    for idx in indices:
        ctx = contexts[idx] if contexts and idx < len(contexts) else None
        exp, emb = generate_explanation_and_embedding(term, api_provider, ctx)
        if exp and emb:
            explanations.append(exp)
            embeddings.append(emb)
    
    if len(embeddings) < 2:
        return None
    
    # 比较不同出现的相似度
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = _cos(embeddings[i], embeddings[j])
            print(f"  出现{i+1} vs 出现{j+1}: {sim:.4f}")
            
            if sim < threshold:
                print(f"  ⚠️  检测到多义: {term}")
                # 拆分为两个新术语
                new_a, new_b = disambiguate_term_to_two(
                    term, explanations[i], explanations[j], api_provider
                )
                print(f"  拆分为: {new_a} | {new_b}")
                return (new_a, new_b)
    
    return None


# =========================================================
# 8) ✅ 完整流程：增量处理
# =========================================================
def process_terms_pipeline(
    terms: List[str],
    api_provider: str = "qianwen",
    json_path: Optional[str | Path] = None,
    synonym_threshold_low: float = 0.73,
    synonym_threshold_high: Optional[float] = 0.85,
    polysemy_threshold: float = 0.73,
    contexts: Optional[List[Optional[str]]] = None
) -> dict:
    """增量处理术语列表"""
    print("\n" + "=" * 60)
    print("术语处理流程 - 增量版")
    print("=" * 60)
    print(f"输入术语数: {len(terms)}")
    print(f"同义词判定阈值: {synonym_threshold_low} ~ {synonym_threshold_high or '无上限'}")
    print(f"多义词拆分阈值: {polysemy_threshold}")
    
    # 加载现有JSON
    json_data, json_map = _load_json_terms(json_path)
    print(f"JSON中已有术语: {len(json_data)}")
    
    # 规范化上下文
    if contexts is None:
        contexts = [None] * len(terms)
    else:
        contexts = list(contexts) + [None] * max(0, len(terms) - len(contexts))
    
    # 统计同一术语的出现
    term_occurrences: Dict[str, List[int]] = {}
    for i, t in enumerate(terms):
        term_occurrences.setdefault(t, []).append(i)
    
    # 处理多义词
    disambiguations = []
    terms_to_process = []
    contexts_to_process = []
    
    for term, indices in term_occurrences.items():
        if len(indices) > 1:
            # 检查是否多义
            result = check_polysemy(
                term, contexts, indices, api_provider, polysemy_threshold
            )
            if result:
                new_a, new_b = result
                disambiguations.append((term, new_a, new_b))
                # 替换原术语
                for idx in indices:
                    if idx == indices[0]:
                        terms_to_process.append(new_a)
                        contexts_to_process.append(contexts[idx])
                    elif idx == indices[1]:
                        terms_to_process.append(new_b)
                        contexts_to_process.append(contexts[idx])
                continue
        
        # 单义或无法拆分，保留原术语
        for idx in indices:
            terms_to_process.append(term)
            contexts_to_process.append(contexts[idx])
    
    # 增量处理每个术语
    print("\n" + "=" * 60)
    print("开始增量处理术语")
    print("=" * 60)
    
    synonym_pairs = []
    new_terms = []
    
    for term, ctx in zip(terms_to_process, contexts_to_process):
        is_synonym, main_term = process_term_incrementally(
            term=term,
            context=ctx,
            json_data=json_data,
            json_map=json_map,
            api_provider=api_provider,
            synonym_threshold_low=synonym_threshold_low,
            synonym_threshold_high=synonym_threshold_high,
            polysemy_threshold=polysemy_threshold
        )
        
        if is_synonym:
            synonym_pairs.append((term, main_term))
        else:
            new_terms.append(term)
    
    # 保存JSON
    if json_path:
        _save_json_terms(json_path, json_data)
    
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
# 9) ✅ 修改后的测试主函数
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("术语消歧与聚类系统 - 修改版测试")
    print("=" * 60)
    
    provider = "qianwen"
    
    # ✅ 修改1：指定JSON输出路径
    json_output_path = r"E:\流程\sources\terms.json"
    print(f"\nJSON输出路径: {json_output_path}")
    
    # 测试1: 基础同义词识别
    print("\n" + "=" * 60)
    print("测试1: 基础同义词识别")
    print("=" * 60)
    
    test1_terms = ["笔记本电脑", "手提电脑", "台式电脑"]
    
    result1 = process_terms_pipeline(
        terms=test1_terms,
        api_provider=provider,
        json_path=json_output_path,
        synonym_threshold_low=0.70,
        synonym_threshold_high=0.85
    )
    
    print("\n[JSON内容]")
    with open(json_output_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    for item in data1:
        print(f"  {item['term']}: synonyms={item['synonyms']}")
    
    print("✅ 测试1完成")
    
    # 测试2: 增量添加新同义词
    print("\n" + "=" * 60)
    print("测试2: 增量添加新同义词")
    print("=" * 60)
    
    test2_terms = ["笔记本电脑", "便携式电脑", "平板电脑"]
    
    result2 = process_terms_pipeline(
        terms=test2_terms,
        api_provider=provider,
        json_path=json_output_path,
        synonym_threshold_low=0.70,
        synonym_threshold_high=0.85
    )
    
    print("\n[JSON内容]")
    with open(json_output_path, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    for item in data2:
        print(f"  {item['term']}: synonyms={item['synonyms']}")
    
    print("✅ 测试2完成")
    
    # ✅ 修改2：测试多义词检测与处理
    print("\n" + "=" * 60)
    print("测试3: 多义词检测与处理")
    print("=" * 60)
    
    # 使用"苹果"作为多义词测试（水果 vs 公司）
    test3_terms = ["苹果", "苹果", "香蕉"]
    test3_contexts = [
        "苹果是一种常见的水果，富含维生素C和膳食纤维。",
        "苹果公司是全球知名的科技企业，总部位于加州库比蒂诺。",
        None
    ]
    
    result3 = process_terms_pipeline(
        terms=test3_terms,
        api_provider=provider,
        json_path=json_output_path,
        synonym_threshold_low=0.70,
        synonym_threshold_high=0.85,
        polysemy_threshold=0.73,
        contexts=test3_contexts
    )
    
    print("\n[多义拆分结果]")
    if result3["disambiguations"]:
        for original, term_a, term_b in result3["disambiguations"]:
            print(f"  原术语: {original}")
            print(f"  拆分为: {term_a} | {term_b}")
    else:
        print("  未检测到多义词")
    
    print("\n[最终JSON内容]")
    with open(json_output_path, "r", encoding="utf-8") as f:
        data3 = json.load(f)
    for item in data3:
        print(f"  {item['term']}: synonyms={item['synonyms']}")
    
    print("✅ 测试3完成")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print(f"最终JSON文件保存在: {json_output_path}")
    print("=" * 60)