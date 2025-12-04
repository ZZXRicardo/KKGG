# evaluation/extractor.py
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from LLM import LLM  # 你的 LLM 模塊


class EvaluationExtractor:
    """
    統一的實體/關係評估迭代修正器（支援完整過程記錄 + 最終帶歷史輸出）
    """

    def __init__(
        self,
        eval_model1: str = "qianwen",
        eval_model2: str = "deepseek",
        eval_prompt_path: str = None,
        max_iterations: int = 5,
        record_base_dir: str = "Output/record",  # 保留兼容性，實際已不再使用
    ):
        self.eval_model1 = eval_model1
        self.eval_model2 = eval_model2
        self.max_iterations = max_iterations
        self.record_base_dir = Path(record_base_dir)

        if not eval_prompt_path:
            raise ValueError("eval_prompt_path 不能為空")
        self.prompt_template = Path(eval_prompt_path).read_text(encoding="utf-8").strip()

    def _get_llm_instance(self, iteration: int) -> LLM:
        model = self.eval_model1 if iteration % 2 == 1 else self.eval_model2
        return LLM(prompt="", api_provider=model)

    def _call_llm_for_evaluation(self, iteration: int, data: Dict) -> Dict[str, Any]:
        llm = self._get_llm_instance(iteration)
        full_prompt = f"{self.prompt_template}\n\n待評估數據：\n{json.dumps(data, ensure_ascii=False, indent=2)}"
        llm.prompt = full_prompt
        response = llm.llm_call()
        content = llm.extract_response(response)

        try:
            import re
            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            json_str = match.group(1) if match else content.strip()
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(f"第 {iteration} 輪評估結果解析失敗: {e}\n原始輸出:\n{content}")
            return {"raw_output": content, "parse_error": True}

    def _extract_evaluation_list(self, eval_result: Dict) -> Tuple[List[Dict], List[str]]:
        data = eval_result
        analysis_list: List[str] = []
        results: List[Dict] = []

        # 嘗試從常見 key 中找到真正的評估陣列
        keys_to_try = [
            "evaluation", "results", "entity_evaluation", "relation_evaluation",
            "data", "analysis", "評估", "總結", "summary", "output"
        ]
        for key in keys_to_try:
            if isinstance(data, dict) and key in data:
                data = data[key]
                break

        # 處理被多包一層 list 的情況
        while isinstance(data, list) and len(data) == 1 and not isinstance(data[0], (str, int, float)):
            data = data[0]

        # 轉成統一的 list 結構
        if isinstance(data, list):
            candidate_list = data
        elif isinstance(data, dict):
            if all(str(k).isdigit() for k in data.keys()):
                candidate_list = [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]
            else:
                candidate_list = [data]
        else:
            candidate_list = []

        expected_len = len(eval_result.get("results", [])) if isinstance(eval_result, dict) else 1

        for item in candidate_list:
            if not isinstance(item, dict):
                results.append({"FP": [], "FN": []})
                analysis_list.append("解析失敗：非字典結構")
                continue

            # 提取分析說明
            analysis = next(
                (str(item.get(k, "")).strip() for k in ["analysis", "評論", "reason", "說明", "備註", "comment", "analysis_zh"] if item.get(k)),
                "無分析說明"
            )

            # 提取 FP（應刪除）
            fp = next(
                (item[k] for k in ["FP", "false_positives", "錯誤", "應刪除", "delete"] if k in item and isinstance(item.get(k), list)),
                []
            )

            # 提取 FN（應補充）
            fn = next(
                (item[k] for k in ["FN", "false_negatives", "遺漏", "應補充", "add", "missing"] if k in item and isinstance(item.get(k), list)),
                []
            )

            results.append({"FP": fp, "FN": fn})
            analysis_list.append(analysis)

        # 補齊長度（防止索引越界）
        while len(results) < expected_len:
            results.append({"FP": [], "FN": []})
            analysis_list.append("自動補位（長度不足）")

        return results, analysis_list

    def _is_all_clean(self, eval_result: Dict) -> bool:
        eval_list, _ = self._extract_evaluation_list(eval_result)
        return all(not (item.get("FP") or item.get("FN")) for item in eval_list)

    def _apply_corrections_entities(
        self, original: List[List[Dict]], eval_result: Dict
    ) -> Tuple[List[List[Dict]], List[str]]:
        results, analysis = self._extract_evaluation_list(eval_result)
        corrected: List[List[Dict]] = []

        for sent_idx, entities in enumerate(original):
            item = results[sent_idx] if sent_idx < len(results) else {"FP": [], "FN": []}
            fp_names = {e["name"] for e in item.get("FP", []) if isinstance(e, dict) and "name" in e}
            add_list = [e for e in item.get("FN", []) if isinstance(e, dict) and "name" in e]

            filtered = [e for e in entities if e.get("name") not in fp_names]
            filtered.extend(add_list)
            corrected.append(filtered)

        return corrected, analysis

    def _apply_corrections_relations(
        self, original: List[List[Dict]], eval_result: Dict
    ) -> Tuple[List[List[Dict]], List[str]]:
        results, analysis = self._extract_evaluation_list(eval_result)
        corrected: List[List[Dict]] = []

        def triple_key(t):
            if isinstance(t, dict) and "triple" in t:
                tr = t["triple"]
            elif isinstance(t, list) and len(t) >= 3:
                tr = t
            else:
                return None
            return tuple(tr[:3]) if len(tr) >= 3 else None

        for sent_idx, relations in enumerate(original):
            item = results[sent_idx] if sent_idx < len(results) else {"FP": [], "FN": []}
            fp_keys = {triple_key(x) for x in item.get("FP", []) if triple_key(x) is not None}

            filtered = [r for r in relations if triple_key(r) not in fp_keys]

            for add in item.get("FN", []):
                if isinstance(add, dict) and "triple" in add and len(add["triple"]) >= 3:
                    filtered.append({"triple": add["triple"][:3]})
                elif isinstance(add, list) and len(add) >= 3:
                    filtered.append({"triple": add[:3]})

            corrected.append(filtered)

        return corrected, analysis

    def evaluate_and_correct(
        self,
        input_file: Path,
        output_file: Path,
        record_dir: Path,
        task_type: str,
        chunk_id: str = None
    ) -> Dict[str, Any]:
        
        # 強制確保所有輸出路徑都是 .json
        input_file = Path(input_file)
        output_file = Path(output_file)
        if output_file.suffix != ".json":
            output_file = output_file.with_suffix(".json")  # 強制改成 .json

        record_dir = Path(record_dir)
        task_subdir = "entities" if task_type == "entity" else "triples"
        
        llm_dir        = record_dir / f"{task_subdir}_LLM"
        process_dir    = record_dir / f"{task_subdir}_process"
        evaluation_dir = record_dir / f"{task_subdir}_evaluation"   # 這就是你說的 entities_evaluation

        for d in [llm_dir, process_dir, evaluation_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 讀取完整原始數據（保留所有頂層字段）
        original_full_data = json.loads(input_file.read_text(encoding="utf-8"))
        current_data = original_full_data.copy()

        # 第1輪輸入記錄
        first_input_path = llm_dir / "1_input.json"
        first_input_path.write_text(json.dumps(current_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"已建立第 1 輪輸入記錄：{first_input_path}")

        num_sentences = len(current_data.get("results", []))
        correction_history = [[] for _ in range(num_sentences)]
        total_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            total_iterations = iteration
            model_used = self.eval_model1 if iteration % 2 == 1 else self.eval_model2
            logging.info(f"[{task_type.capitalize()} 評估] - 第 {iteration} 輪 (使用 {model_used})")

            eval_result = self._call_llm_for_evaluation(iteration, current_data)

            # 保存 LLM 原始輸出
            process_path = process_dir / f"{iteration}_eval_raw.json"
            process_path.write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")

            # 提取評估建議
            eval_list, analysis_list = self._extract_evaluation_list(eval_result)

            # 取出當前要修正的目標字段
            if task_type == "entity":
                raw_entities = [r.get("output", {}).get("entities", []) for r in current_data.get("results", [])]
                corrected_entities, _ = self._apply_corrections_entities(raw_entities, eval_result)
                target_key = "entities"
                old_items_list = raw_entities
                new_items_list = corrected_entities
            else:
                raw_relations = [r.get("output", {}).get("relations", []) for r in current_data.get("results", [])]
                corrected_relations, _ = self._apply_corrections_relations(raw_relations, eval_result)
                target_key = "relations"
                old_items_list = raw_relations
                new_items_list = corrected_relations

            # 構建新的 results（只改目標字段）
            new_results = []
            for i, item in enumerate(current_data.get("results", [])):
                output = item.get("output", {}).copy()

                # 關係任務必須保留 entities
                if task_type == "relation":
                    output["entities"] = output.get("entities", [])

                # 計算 diff 用於歷史記錄
                old_items = old_items_list[i]
                new_items = new_items_list[i]

                if task_type == "entity":
                    old_set = {e["name"] for e in old_items if isinstance(e, dict) and "name" in e}
                    new_set = {e["name"] for e in new_items if isinstance(e, dict) and "name" in e}
                else:
                    def tk(t): 
                        return tuple(t["triple"][:3]) if isinstance(t, dict) and "triple" in t else tuple(t[:3]) if isinstance(t, list) else None
                    old_set = {tk(t) for t in old_items if tk(t)}
                    new_set = {tk(t) for t in new_items if tk(t)}

                fp_removed = list(old_set - new_set)
                fn_added = list(new_set - old_set)

                # 記錄修正歷史
                correction_history[i].append({
                    "round": iteration,
                    "model": model_used,
                    "FP_removed": fp_removed,
                    "FN_added": fn_added,
                    "analysis": analysis_list[i] if i < len(analysis_list) else "無說明",
                    "eval_raw_file": process_path.name
                })

                # 更新目標字段
                output[target_key] = new_items
                new_results.append({
                    "input": item["input"],
                    "output": output
                })

            # 關鍵：恢復原始頂層結構，只替換 results
            current_data = original_full_data.copy()
            current_data["results"] = new_results

            # 保存當前輪修正後的完整數據
            corrected_path = evaluation_dir / f"{iteration}_corrected.json"
            corrected_path.write_text(json.dumps(current_data, ensure_ascii=False, indent=2), encoding="utf-8")

            # 檢查是否已完美
            if self._is_all_clean(eval_result):
                logging.info(f"[{task_type.capitalize()} 評估] 第 {iteration} 輪已完美，停止迭代")
                break

            # 為下一輪準備輸入
            next_input_path = llm_dir / f"{iteration + 1}_input.json"
            next_input_path.write_text(json.dumps(current_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # ====================== 最終交付 ======================
        final_delivery_data = original_full_data.copy()
        for i, item in enumerate(current_data.get("results", [])):
            clean_output = {
                "entities": item["output"].get("entities", [])
            }
            if task_type == "relation":
                clean_output["relations"] = item["output"].get("relations", [])
            final_delivery_data["results"][i]["output"] = clean_output

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(final_delivery_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        logging.info(f"[{task_type.capitalize()} 評估] 最終交付檔完成 → {output_file}")
        logging.info(f"  完整過程記錄保留在 → {record_dir}")

        return {
            "total_iterations": total_iterations,
            "record_dir": str(record_dir),
            "final_delivery_file": str(output_file),
            "correction_history": correction_history  # 可選：返回歷史給上層使用
        }