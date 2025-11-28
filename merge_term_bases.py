#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
跨文件术语库合并工具（保留所有术语版）
处理完成后：合并的同义词保留在输出文件，未合并的独立术语也保留，源文件全部删除
"""

import argparse
import json
import logging
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

sys.path.append(r"E:\KKGG\project")
from disambiguator_clusterer import TermDisambiguator
from LLM import LLM


class TermBaseMerger:
    """跨文件术语库合并器"""
    
    def __init__(self, api_provider: str = "qianwen"):
        self.api_provider = api_provider
        self.disambiguator = TermDisambiguator(api_provider=api_provider)
    
    def load_all_term_bases(self, input_dir: Path) -> Dict[str, List[Dict]]:
        """加载输入目录下的所有术语库文件"""
        term_bases = {}
        json_files = list(input_dir.glob("*.json"))
        
        logging.info(f"发现 {len(json_files)} 个术语库文件")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # 过滤掉没有embedding的无效术语
                    valid_terms = [t for t in data if isinstance(t, dict) and t.get('embedding')]
                    term_bases[json_file.name] = valid_terms
                    logging.info(f"  加载 {json_file.name}: {len(valid_terms)} 个有效术语")
                else:
                    logging.warning(f"  跳过 {json_file.name}: 不是列表格式")
            except Exception as e:
                logging.error(f"  加载失败 {json_file.name}: {e}")
        
        return term_bases
    
    def merge_terms(self, term_a: Dict, term_b: Dict):
        """将 term_b 合并到 term_a 中"""
        if 'synonyms' not in term_a:
            term_a['synonyms'] = []
        if 'synonyms' not in term_b:
            term_b['synonyms'] = []
        
        # 添加 term_b 本身作为同义词
        if term_b['term'] not in term_a['synonyms'] and term_b['term'] != term_a['term']:
            term_a['synonyms'].append(term_b['term'])
        
        # 添加 term_b 的同义词
        for syn in term_b['synonyms']:
            if syn not in term_a['synonyms'] and syn != term_a['term']:
                term_a['synonyms'].append(syn)
        
        logging.debug(f"  合并: '{term_b['term']}' → '{term_a['term']}'")
    
    def llm_confirm_merge(self, term_a: Dict, term_b: Dict, similarity: float) -> bool:
        """使用 LLM 确认两个术语是否应该合并"""
        prompt = f"""
请判断以下两个术语是否为『同义词』（表达同一概念的不同叫法）。

【术语A】'{term_a['term']}'
【解释A】'{term_a['explanation'][:100]}...'

【术语B】'{term_b['term']}'
【解释B】'{term_b['explanation'][:100]}...'

【相似度】{similarity:.4f}

请回答"是"或"否"。只输出一个字。
"""
        try:
            llm = LLM(prompt=prompt, api_provider=self.api_provider)
            resp = llm.llm_call()
            ans = llm.extract_response(resp).strip()
            return "是" in ans
        except Exception as e:
            logging.error(f"LLM判断失败: {e}")
            return False
    
    def _load_json_terms(self, json_path: Optional[str | Path]) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        读取 JSON，返回 (数组, {term: item_dict})
        （从 TermDisambiguator 类复制的方法）
        """
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
    
    def _generate_def_cluster_triples(
        self,
        json_data: List[Dict],
        cluster_json_path: str | Path,
    ):
        """
        为合并后的术语库生成 def聚类 三元组
        
        Args:
            json_data: 合并后的术语数据
            cluster_json_path: 三元组JSON文件路径
        """
        cluster_path = Path(cluster_json_path)
        
        # 读取已有的三元组（增量更新）
        existing_triples = {}
        if cluster_path.exists():
            try:
                existing_triples = json.loads(cluster_path.read_text(encoding="utf-8"))
                if not isinstance(existing_triples, dict):
                    existing_triples = {}
            except:
                existing_triples = {}
        
        # 检查哪些术语需要更新
        terms_to_update = []
        new_terms_with_synonyms = 0
        
        for item in json_data:
            main_term = item["term"]
            synonyms = item.get("synonyms", [])
            
            # 只处理有同义词的术语
            if not synonyms:
                # 如果该术语之前在三元组中，但现在没有同义词了，删除它
                if main_term in existing_triples:
                    del existing_triples[main_term]
                    logging.info(f"  删除无同义词术语: {main_term}")
                continue
            
            # 检查是否有变化（简化逻辑：总是重新生成）
            terms_to_update.append(main_term)
            new_terms_with_synonyms += 1
            logging.info(f"  新增/更新术语: {main_term} (同义词: {len(synonyms)})")
        
        # 为需要更新的术语生成三元组
        if terms_to_update:
            logging.info(f"\n  需要更新的术语数: {len(terms_to_update)}")
            
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
        logging.info(f"\n[三元组保存] 已保存至 {cluster_json_path}")
        logging.info(f"  总术语数（有同义词）: {len(existing_triples)}")
        logging.info(f"  新增/更新带同义词术语: {new_terms_with_synonyms}")
    
    def update_article_triples(
        self,
        articles_dir: str | Path,
        entity_json_path: str | Path,
        relation_json_path: str | Path,
        backup: bool = True
    ):
        """
        使用合并后的术语库更新文章文件中的三元组
        
        Args:
            articles_dir: 文章文件目录（每文件多篇文章）
            entity_json_path: 合并后的 Entity.json 路径
            relation_json_path: 合并后的 Relation.json 路径
            backup: 是否备份原文件
        """
        articles_dir = Path(articles_dir)
        if not articles_dir.exists():
            logging.warning(f"文章目录不存在: {articles_dir}")
            return
        
        # 加载术语映射
        logging.info("\n" + "=" * 60)
        logging.info("加载术语映射")
        logging.info("=" * 60)
        
        _, entity_map = self._load_json_terms(entity_json_path)
        _, relation_map = self._load_json_terms(relation_json_path)
        
        logging.info(f"加载实体映射: {len(entity_map)} 个")
        logging.info(f"加载关系映射: {len(relation_map)} 个")
        
        # 扫描文章文件
        article_files = list(articles_dir.glob("*.json"))
        if not article_files:
            logging.warning(f"文章目录中没有找到JSON文件: {articles_dir}")
            return
        
        logging.info(f"\n找到 {len(article_files)} 个文章文件")
        
        updated_files = 0
        total_articles = 0
        total_triples = 0
        
        for article_file in article_files:
            logging.info(f"\n{'='*60}")
            logging.info(f"处理文件: {article_file.name}")
            logging.info(f"{'='*60}")
            
            try:
                # 读取文章数据
                with open(article_file, 'r', encoding='utf-8') as f:
                    articles_data = json.load(f)
                
                if not isinstance(articles_data, list):
                    logging.warning(f"  文件格式不是列表，跳过")
                    continue
                
                # 备份原文件
                if backup:
                    backup_file = article_file.with_suffix('.json.bak')
                    shutil.copy2(article_file, backup_file)
                    logging.info(f"  备份原文件: {backup_file.name}")
                
                # 逐篇文章处理
                file_updated = False
                for article_idx, article in enumerate(articles_data):
                    article_name = article.get('name', f'文章_{article_idx}')
                    
                    # 遍历所有结果和三元组
                    article_updated = False
                    for result in article.get('results', []):
                        if 'output' not in result or 'relations' not in result['output']:
                            continue
                        
                        for rel in result['output']['relations']:
                            if not isinstance(rel, dict) or 'triple' not in rel:
                                continue
                            
                            triple = rel['triple']
                            if len(triple) < 3:
                                continue
                            
                            head, relation, tail = triple[0], triple[1], triple[2]
                            updated = False
                            
                            # 更新 head（检查是否在实体映射中）
                            if head in entity_map:
                                main_head = entity_map[head]['term']
                                if main_head != head:
                                    rel['triple'][0] = main_head
                                    updated = True
                            
                            # 更新 relation（检查是否在关系映射中）
                            if relation in relation_map:
                                main_relation = relation_map[relation]['term']
                                if main_relation != relation:
                                    rel['triple'][1] = main_relation
                                    updated = True
                            
                            # 更新 tail（检查是否在实体映射中）
                            if tail in entity_map:
                                main_tail = entity_map[tail]['term']
                                if main_tail != tail:
                                    rel['triple'][2] = main_tail
                                    updated = True
                            
                            if updated:
                                article_updated = True
                                total_triples += 1
                    
                    if article_updated:
                        file_updated = True
                        total_articles += 1
                        logging.info(f"  更新文章: {article_name}")
                
                # 写回更新后的数据
                if file_updated:
                    with open(article_file, 'w', encoding='utf-8') as f:
                        json.dump(articles_data, f, ensure_ascii=False, indent=2)
                    logging.info(f"  ✓ 文件更新完成")
                    updated_files += 1
                else:
                    logging.info(f"  - 文件无需更新")
                
            except Exception as e:
                logging.error(f"  处理失败: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        logging.info("\n" + "=" * 60)
        logging.info("文章更新完成")
        logging.info(f"更新文件数: {updated_files}")
        logging.info(f"更新文章数: {total_articles}")
        logging.info(f"更新三元组数: {total_triples}")
        logging.info("=" * 60)
    
    def merge_all(
        self,
        input_dir: str | Path,
        output_file: str | Path,
        threshold_low: float = 0.70,
        threshold_high: float = 0.85,
        cleanup_source: bool = True
    ):
        """
        主合并流程：跨文件比较并合并术语
        
        Args:
            input_dir: 输入术语库目录
            output_file: 输出合并后的文件
            threshold_low: 低阈值（需LLM确认）
            threshold_high: 高阈值（直接合并）
            cleanup_source: 是否在完成后删除源文件
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)
        
        logging.info("=" * 60)
        logging.info("开始跨文件术语库合并")
        logging.info("=" * 60)
        
        # 1. 加载所有术语库
        term_bases = self.load_all_term_bases(input_dir)
        
        if not term_bases:
            logging.warning("没有加载到任何术语库文件")
            return
        
        # 2. 构建全局术语列表
        all_terms = []
        for filename, terms in term_bases.items():
            for idx, term in enumerate(terms):
                all_terms.append((filename, idx, term))
        
        logging.info(f"\n总共加载 {len(all_terms)} 个术语")
        
        # 3. 待删除的术语集合
        terms_to_delete: Set[Tuple[str, str]] = set()
        merged_terms: List[Dict] = []
        processed_terms: Set[str] = set()
        
        # 4. 两两比较（跨文件）
        logging.info("=" * 60)
        logging.info("开始跨文件术语比较")
        logging.info("=" * 60)
        
        merge_count = 0
        
        for i, (file_a, idx_a, term_a) in enumerate(all_terms):
            # 跳过已删除或已处理的 term_a
            if (file_a, term_a['term']) in terms_to_delete or term_a['term'] in processed_terms:
                continue
            
            for j, (file_b, idx_b, term_b) in enumerate(all_terms):
                if i <= j:
                    continue
                if (file_b, term_b['term']) in terms_to_delete:
                    continue
                
                # 计算相似度
                emb_a = term_a.get('embedding', [])
                emb_b = term_b.get('embedding', [])
                
                if not emb_a or not emb_b:
                    continue
                
                sim = self.disambiguator._cos(emb_a, emb_b)
                
                if sim < threshold_low:
                    continue
                
                logging.info(f"\n比较: '{term_a['term']}'({file_a}) vs '{term_b['term']}'({file_b})")
                logging.info(f"  相似度: {sim:.4f}")
                
                should_merge = False
                
                # 高阈值：直接合并
                if sim >= threshold_high:
                    logging.info(f"  高阈值直接合并")
                    should_merge = True
                # 中阈值：LLM确认
                elif sim >= threshold_low:
                    logging.info(f"  中阈值LLM确认...")
                    should_merge = self.llm_confirm_merge(term_a, term_b, sim)
                
                # 执行合并
                if should_merge:
                    self.merge_terms(term_a, term_b)
                    terms_to_delete.add((file_b, term_b['term']))
                    merge_count += 1
                    logging.info(f"  ✓ 合并成功")
            
            merged_terms.append(term_a)
            processed_terms.add(term_a['term'])
        
        # 5. 收集独立术语
        logging.info("\n" + "=" * 60)
        logging.info("收集独立术语")
        logging.info("=" * 60)
        
        independent_count = 0
        for filename, terms in term_bases.items():
            for term in terms:
                key = (filename, term['term'])
                if key not in terms_to_delete and term['term'] not in processed_terms:
                    merged_terms.append(term)
                    processed_terms.add(term['term'])
                    independent_count += 1
        
        logging.info(f"独立术语数: {independent_count}")
        
        # 6. 保存合并结果
        logging.info("\n" + "=" * 60)
        logging.info("保存合并结果")
        logging.info("=" * 60)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_terms, f, ensure_ascii=False, indent=2)
        
        logging.info(f"合并结果已保存: {output_file}")
        logging.info(f"总术语数: {len(merged_terms)} (合并: {merge_count}, 独立: {independent_count})")
        
        # 7. 新增：生成概念聚类三元组（仅 definitions.json）
        if "definitions" in str(output_file):
            logging.info("\n" + "=" * 60)
            logging.info("生成概念聚类三元组")
            logging.info("=" * 60)
            
            # 根据输出文件路径确定三元组文件路径
            base_dir = output_file.parent
            cluster_dir = base_dir / "entity_cluster_triples"
            cluster_file = cluster_dir / output_file.name.replace("definitions", "entity_cluster_triples")
            
            self._generate_def_cluster_triples(
                json_data=merged_terms,
                cluster_json_path=cluster_file
            )
        
        # 8. 新增：更新文章文件中的三元组
        if hasattr(self, '_articles_dir_for_update') and self._articles_dir_for_update:
            logging.info("\n" + "=" * 60)
            logging.info("更新文章文件中的三元组")
            logging.info("=" * 60)
            
            # 确定术语库路径
            base_terms_dir = output_file.parent
            entity_json_path = base_terms_dir / "Entity.json"
            relation_json_path = base_terms_dir / "Relation.json"
            
            self.update_article_triples(
                articles_dir=self._articles_dir_for_update,
                entity_json_path=entity_json_path,
                relation_json_path=relation_json_path,
                backup=True
            )
        
        # 9. 清理源文件
        if cleanup_source:
            logging.info("\n" + "=" * 60)
            logging.info("清理源文件")
            logging.info("=" * 60)
            
            files_deleted = 0
            
            for filename, terms in term_bases.items():
                file_path = input_dir / filename
                try:
                    file_path.unlink()
                    logging.info(f"  删除文件: {file_path}")
                    files_deleted += 1
                except Exception as e:
                    logging.error(f"  删除失败 {file_path}: {e}")
            
            logging.info(f"清理完成: 删除 {files_deleted} 个文件")


def batch_merge_directories(
    base_terms_dir: str | Path = r"E:\KKGG\output\terms",
    entity_threshold_low: float = 0.69,
    entity_threshold_high: float = 0.84,
    relation_threshold_low: float = 0.62,
    relation_threshold_high: float = 0.92,
    definitions_threshold_low: float = 0.69,
    definitions_threshold_high: float = 0.84,
    cleanup_source: bool = True,
    articles_dir: Optional[str | Path] = None
):
    """批处理函数：合并三个术语库目录"""
    base_terms_dir = Path(base_terms_dir)
    
    merge_configs = [
        {
            "name": "Entity",
            "input_dir": base_terms_dir / "Entity",
            "output_file": base_terms_dir / "Entity.json",
            "threshold_low": entity_threshold_low,
            "threshold_high": entity_threshold_high,
            "description": "实体术语库",
            "update_articles": True
        },
        {
            "name": "Relation", 
            "input_dir": base_terms_dir / "Relation",
            "output_file": base_terms_dir / "Relation.json",
            "threshold_low": relation_threshold_low,
            "threshold_high": relation_threshold_high,
            "description": "关系术语库",
            "update_articles": True
        },
        {
            "name": "definitions",
            "input_dir": base_terms_dir / "definitions",
            "output_file": base_terms_dir / "definitions.json",
            "threshold_low": definitions_threshold_low,
            "threshold_high": definitions_threshold_high,
            "description": "概念聚类库",
            "generate_triples": True,
            "update_articles": False
        }
    ]
    
    logging.info("=" * 70)
    logging.info("启动批处理：合并所有术语库目录")
    logging.info(f"根目录: {base_terms_dir}")
    if articles_dir:
        logging.info(f"文章目录: {articles_dir}")
    logging.info("=" * 70)
    
    merger = TermBaseMerger(api_provider="qianwen")
    
    # 如果指定了文章目录，保存到 merger 实例
    if articles_dir:
        merger._articles_dir_for_update = Path(articles_dir)
    
    for config in merge_configs:
        logging.info(f"\n{config['name']} 目录")
        logging.info("-" * 50)
        logging.info(f"输入: {config['input_dir']}")
        logging.info(f"输出: {config['output_file']}")
        logging.info(f"阈值: [{config['threshold_low']}, {config['threshold_high']}]")
        logging.info(f"清理源文件: {'是' if cleanup_source else '否'}")
        
        if not config['input_dir'].exists():
            logging.warning(f"  目录不存在，跳过")
            continue
        
        json_files = list(config['input_dir'].glob("*.json"))
        if not json_files:
            logging.warning(f"  目录中没有JSON文件，跳过")
            continue
        
        try:
            merger.merge_all(
                input_dir=config['input_dir'],
                output_file=config['output_file'],
                threshold_low=config['threshold_low'],
                threshold_high=config['threshold_high'],
                cleanup_source=cleanup_source
            )
            logging.info(f"  ✓ {config['description']} 合并完成")
            
            if config.get('generate_triples'):
                logging.info(f"  ✓ 概念聚类三元组已生成")
            
            if config.get('update_articles') and hasattr(merger, '_articles_dir_for_update'):
                logging.info(f"  ✓ 文章三元组已更新")
            
        except Exception as e:
            logging.error(f"  ✗ {config['description']} 合并失败: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    logging.info("\n" + "=" * 70)
    logging.info("所有术语库合并任务执行完毕")
    logging.info("=" * 70)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="跨文件术语库合并工具（自动清理版）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--batch', action='store_true',
                           help='批处理模式：自动合并三个目录')
    mode_group.add_argument('--input_dir', type=str,
                           help='输入术语库目录')
    
    parser.add_argument('--output_file', type=str,
                       help='输出文件路径（单个模式必需）')
    parser.add_argument('--threshold_low', type=float, default=0.70,
                       help='相似度低阈值，默认0.70')
    parser.add_argument('--threshold_high', type=float, default=0.85,
                       help='相似度高阈值，默认0.85')
    parser.add_argument('--no_cleanup', action='store_true',
                       help='保留源文件，不删除')
    
    parser.add_argument('--articles_dir', type=str,
                       help='文章文件目录（用于更新三元组）')
    
    parser.add_argument('--api_provider', type=str, default='qianwen',
                       help='LLM API提供商，默认qianwen')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别，默认INFO')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.batch:
        batch_merge_directories(
            cleanup_source=not args.no_cleanup,
            articles_dir=args.articles_dir
        )
    else:
        if not args.output_file:
            parser.error("单个模式需要提供 --output_file 参数")
        
        merger = TermBaseMerger(api_provider=args.api_provider)
        
        # 如果指定了文章目录，保存到 merger 实例
        if args.articles_dir:
            merger._articles_dir_for_update = Path(args.articles_dir)
        
        merger.merge_all(
            input_dir=args.input_dir,
            output_file=args.output_file,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
            cleanup_source=not args.no_cleanup
        )


if __name__ == "__main__":
    main()