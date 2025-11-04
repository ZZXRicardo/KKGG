#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：测试实体消歧和概念聚类功能
使用已有的输出.json文件，不需要创建新实体
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from disambiguator_clusterer import TermDisambiguator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def test_entity_disambiguation():
    """
    测试实体消歧功能
    
    从输出.json中收集label='a'的实体和关系词，进行消歧处理
    """
    print("\n" + "=" * 80)
    print("测试1: 实体消歧（Entity Disambiguation）")
    print("=" * 80)
    
    # 定义输入输出路径
    input_json_path = r"E:\KKGG\output\KG\输出.json"
    output_json_path = r"E:\KKGG\output\KG\输出_消歧后.json"
    
    # 读取JSON数据
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"✅ 成功读取输入文件: {input_json_path}")
        logging.info(f"   数据项数量: {len(data)}")
    except FileNotFoundError:
        logging.error(f"❌ 输入文件不存在: {input_json_path}")
        logging.info("请确保文件路径正确")
        return
    except Exception as e:
        logging.error(f"❌ 读取输入文件失败: {e}")
        return
    
    # 收集所有label=='a'的实体（不论位置）和所有关系词
    entity_a_terms = set()      # label=='a'的实体（head或tail）
    relation_terms = set()       # 所有关系词
    entity_context_parts = []    # 用于构建实体上下文
    relation_context_parts = []  # 用于构建关系上下文
    
    logging.info("\n开始收集实体和关系词...")
    
    for item in data:
        if 'results' not in item:
            continue
            
        for result in item['results']:
            if 'output' not in result:
                continue
            
            output = result['output']
            
            # 从relations数组中提取
            if 'relations' in output:
                for rel in output['relations']:
                    if not isinstance(rel, dict):
                        continue
                    
                    triple = rel.get('triple', [])
                    labels = rel.get('label', [])
                    
                    if len(triple) >= 3 and len(labels) >= 2:
                        head = triple[0]
                        relation = triple[1]
                        tail = triple[2]
                        head_label = labels[0]  # head的label
                        tail_label = labels[1]  # tail的label
                        
                        # 1. 检查head的label，如果是'a'则收集
                        if head_label and str(head_label).lower() == 'a' and head:
                            entity_a_terms.add(head)
                            entity_context_parts.append(f"{head} {relation} {tail}")
                        
                        # 2. 检查tail的label，如果是'a'则收集
                        if tail_label and str(tail_label).lower() == 'a' and tail:
                            entity_a_terms.add(tail)
                            entity_context_parts.append(f"{head} {relation} {tail}")
                        
                        # 3. 收集所有关系词
                        if relation:
                            relation_terms.add(relation)
                        
                        # 4. 构建关系上下文
                        relation_context_parts.append(f"{head} {relation} {tail}")
    
    # 转换为列表
    entity_a_list = sorted(list(entity_a_terms))
    relation_list = sorted(list(relation_terms))
    
    logging.info(f"\n✅ 数据收集完成:")
    logging.info(f"   收集到 {len(entity_a_list)} 个label='a'的实体")
    logging.info(f"   收集到 {len(relation_list)} 个关系词")
    logging.info(f"   实体样例（前10个）: {entity_a_list[:10]}")
    logging.info(f"   关系样例（前10个）: {relation_list[:10]}")
    
    if not entity_a_list and not relation_list:
        logging.warning("⚠️  未找到任何实体或关系词，跳过消歧")
        return
    
    # 构建共享上下文（限制长度避免过长）
    entity_shared_context = " ".join(entity_context_parts[:100])
    relation_shared_context = " ".join(relation_context_parts[:100])
    
    logging.info(f"\n构建共享上下文:")
    logging.info(f"   实体上下文长度: {len(entity_shared_context)} 字符")
    logging.info(f"   关系上下文长度: {len(relation_shared_context)} 字符")
    
    # 初始化消歧器并执行消歧
    logging.info("\n开始执行消歧...")
    disambiguator = TermDisambiguator(api_provider="qianwen")
    
    try:
        updated_entities, updated_relations = disambiguator.Disambiguate(
            entity_terms=entity_a_list,
            relation_terms=relation_list,
            entity_shared_context=entity_shared_context,
            relation_shared_context=relation_shared_context
        )
        
        logging.info(f"\n✅ 消歧完成:")
        logging.info(f"   更新后实体数: {len(updated_entities)}")
        logging.info(f"   更新后关系数: {len(updated_relations)}")
        
    except Exception as e:
        logging.error(f"❌ 消歧过程出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 创建映射字典：原术语 -> 更新后术语
    entity_mapping = dict(zip(entity_a_list, updated_entities))
    relation_mapping = dict(zip(relation_list, updated_relations))
    
    # 统计变化
    entity_changes = sum(1 for k, v in entity_mapping.items() if k != v)
    relation_changes = sum(1 for k, v in relation_mapping.items() if k != v)
    
    logging.info(f"\n映射统计:")
    logging.info(f"   实体映射变化数: {entity_changes}")
    logging.info(f"   关系映射变化数: {relation_changes}")
    logging.info(f"   实体映射示例（前3个变化）:")
    for original, updated in list(entity_mapping.items())[:10]:
        if original != updated:
            logging.info(f"      '{original}' -> '{updated}'")
    logging.info(f"   关系映射示例（前3个变化）:")
    for original, updated in list(relation_mapping.items())[:10]:
        if original != updated:
            logging.info(f"      '{original}' -> '{updated}'")
    
    # 更新原JSON数据中的三元组
    logging.info("\n开始更新三元组...")
    updated_count = 0
    
    for item in data:
        if 'results' not in item:
            continue
            
        for result in item['results']:
            if 'output' not in result or 'relations' not in result['output']:
                continue
            
            relations = result['output']['relations']
            for rel in relations:
                if not isinstance(rel, dict) or 'triple' not in rel:
                    continue
                
                triple = rel['triple']
                if len(triple) >= 3:
                    head, relation, tail = triple[0], triple[1], triple[2]
                    
                    # 更新head（如果在entity_mapping中）
                    if head in entity_mapping and entity_mapping[head] != head:
                        rel['triple'][0] = entity_mapping[head]
                        updated_count += 1
                    
                    # 更新relation（如果在relation_mapping中）
                    if relation in relation_mapping and relation_mapping[relation] != relation:
                        rel['triple'][1] = relation_mapping[relation]
                        updated_count += 1
                    
                    # 更新tail（如果在entity_mapping中）
                    if tail in entity_mapping and entity_mapping[tail] != tail:
                        rel['triple'][2] = entity_mapping[tail]
                        updated_count += 1
    
    # 保存更新后的JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"\n✅ 更新后的数据已保存至: {output_json_path}")
        logging.info(f"   共更新了 {updated_count} 个术语位置")
    except Exception as e:
        logging.error(f"❌ 保存更新后的文件失败: {e}")
    
    # 查看生成的JSON文件
    entity_json_path = r"E:\KKGG\output\terms\Entity.json"
    relation_json_path = r"E:\KKGG\output\terms\Relation.json"
    
    logging.info(f"\n生成的JSON文件:")
    try:
        with open(entity_json_path, 'r', encoding='utf-8') as f:
            entity_data = json.load(f)
        logging.info(f"   Entity.json: {len(entity_data)} 个实体")
        logging.info(f"   前3个实体:")
        for item in entity_data[:3]:
            logging.info(f"      术语: {item['term']}, 同义词: {item.get('synonyms', [])}")
    except Exception as e:
        logging.warning(f"   无法读取 Entity.json: {e}")
    
    try:
        with open(relation_json_path, 'r', encoding='utf-8') as f:
            relation_data = json.load(f)
        logging.info(f"   Relation.json: {len(relation_data)} 个关系")
        logging.info(f"   前3个关系:")
        for item in relation_data[:3]:
            logging.info(f"      术语: {item['term']}, 同义词: {item.get('synonyms', [])}")
    except Exception as e:
        logging.warning(f"   无法读取 Relation.json: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 测试1完成: 实体消歧")
    print("=" * 80)


def test_concept_clustering():
    """
    测试概念聚类功能
    
    从输出.json中收集label='b'的实体，进行聚类处理
    """
    print("\n" + "=" * 80)
    print("测试2: 概念聚类（Concept Clustering）")
    print("=" * 80)
    
    # 定义输入路径
    input_json_path = r"E:\KKGG\output\KG\输出.json"
    
    # 读取JSON数据
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"✅ 成功读取输入文件: {input_json_path}")
        logging.info(f"   数据项数量: {len(data)}")
    except FileNotFoundError:
        logging.error(f"❌ 输入文件不存在: {input_json_path}")
        logging.info("请确保文件路径正确")
        return
    except Exception as e:
        logging.error(f"❌ 读取输入文件失败: {e}")
        return
    
    # 收集所有label=='b'的实体（不论位置）
    entity_b_terms = set()   # label=='b'的实体（head或tail）
    context_parts = []        # 用于构建共享上下文
    
    logging.info("\n开始收集label='b'的实体...")
    
    for item in data:
        if 'results' not in item:
            continue
            
        for result in item['results']:
            if 'output' not in result:
                continue
            
            output = result['output']
            
            # 从relations数组中提取
            if 'relations' in output:
                for rel in output['relations']:
                    if not isinstance(rel, dict):
                        continue
                    
                    triple = rel.get('triple', [])
                    labels = rel.get('label', [])
                    
                    if len(triple) >= 3 and len(labels) >= 2:
                        head = triple[0]
                        relation = triple[1]
                        tail = triple[2]
                        head_label = labels[0]  # head的label
                        tail_label = labels[1]  # tail的label
                        
                        # 1. 检查head的label，如果是'b'则收集
                        if head_label and str(head_label).lower() == 'b' and head:
                            entity_b_terms.add(head)
                            context_parts.append(f"{head} {relation} {tail}")
                        
                        # 2. 检查tail的label，如果是'b'则收集
                        if tail_label and str(tail_label).lower() == 'b' and tail:
                            entity_b_terms.add(tail)
                            context_parts.append(f"{head} {relation} {tail}")
    
    # 转换为列表
    entity_b_list = sorted(list(entity_b_terms))
    
    logging.info(f"\n✅ 数据收集完成:")
    logging.info(f"   收集到 {len(entity_b_list)} 个label='b'的实体")
    logging.info(f"   实体样例（前10个）: {entity_b_list[:10]}")
    
    if not entity_b_list:
        logging.warning("⚠️  未找到任何label='b'的实体，跳过聚类")
        return
    
    # 构建共享上下文（限制长度）
    shared_context = " ".join(context_parts[:200])
    
    logging.info(f"\n构建共享上下文:")
    logging.info(f"   上下文长度: {len(shared_context)} 字符")
    logging.info(f"   上下文片段数: {len(context_parts[:200])}")
    
    # 初始化聚类器并执行聚类
    logging.info("\n开始执行聚类...")
    clusterer = TermDisambiguator(api_provider="qianwen")
    
    try:
        cluster_result = clusterer.clusterer(
            terms=entity_b_list,
            shared_context=shared_context
        )
        
        logging.info(f"\n✅ 聚类完成:")
        logging.info(f"   生成的聚类结果已保存")
        logging.info(f"   聚类结果包含 {len(cluster_result)} 个术语")
        
        # 打印部分聚类结果
        logging.info(f"\n聚类结果详情（前5个术语）:")
        for i, (term, triples) in enumerate(list(cluster_result.items())[:5]):
            logging.info(f"   术语 '{term}':")
            logging.info(f"      关联的同义词链数量: {len(triples)}")
            if triples:
                logging.info(f"      示例三元组: {triples[0]}")
        
    except Exception as e:
        logging.error(f"❌ 聚类过程出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 查看生成的JSON文件
    definitions_json_path = r"E:\KKGG\output\terms\definitions.json"
    cluster_json_path = r"E:\KKGG\output\KG\def_cluster.json"
    
    logging.info(f"\n生成的JSON文件:")
    try:
        with open(definitions_json_path, 'r', encoding='utf-8') as f:
            definitions_data = json.load(f)
        logging.info(f"   definitions.json: {len(definitions_data)} 个术语")
        logging.info(f"   前3个术语:")
        for item in definitions_data[:3]:
            logging.info(f"      术语: {item['term']}")
            logging.info(f"      解释: {item.get('explanation', '')[:60]}...")
            logging.info(f"      同义词: {item.get('synonyms', [])}")
    except Exception as e:
        logging.warning(f"   无法读取 definitions.json: {e}")
    
    try:
        with open(cluster_json_path, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        logging.info(f"\n   def_cluster.json: {len(cluster_data)} 个术语的聚类信息")
        logging.info(f"   前3个术语的聚类:")
        for term, triples in list(cluster_data.items())[:3]:
            logging.info(f"      术语: {term} (关联{len(triples)}个三元组)")
            if triples:
                for t in triples[:2]:
                    logging.info(f"         ({t['head']}, {t['relation']}, {t['tail']})")
    except Exception as e:
        logging.warning(f"   无法读取 def_cluster.json: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 测试2完成: 概念聚类")
    print("=" * 80)


def main():
    """
    主测试函数
    """
    print("\n" + "=" * 80)
    print("实体消歧与概念聚类功能测试")
    print("=" * 80)
    print("\n此测试脚本将:")
    print("1. 从已有的输出.json中读取数据")
    print("2. 测试实体消歧功能（处理label='a'的实体和关系）")
    print("3. 测试概念聚类功能（处理label='b'的实体）")
    print("4. 验证生成的JSON文件")
    print("\n注意: 确保以下文件存在:")
    print("   - E:\\KKGG\\output\\KG\\输出.json")
    print("\n" + "=" * 80)
    
    # 测试1: 实体消歧
    try:
        test_entity_disambiguation()
    except Exception as e:
        logging.error(f"测试1执行失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # 测试2: 概念聚类
    try:
        test_concept_clustering()
    except Exception as e:
        logging.error(f"测试2执行失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # 总结
    print("\n" + "=" * 80)
    print("测试完成总结")
    print("=" * 80)
    print("\n生成的文件:")
    print("1. 实体消歧相关:")
    print("   - E:\\KKGG\\output\\terms\\Entity.json (实体术语及同义词)")
    print("   - E:\\KKGG\\output\\terms\\Relation.json (关系术语及同义词)")
    print("   - E:\\KKGG\\output\\KG\\输出_消歧后.json (消歧后的三元组)")
    print("\n2. 概念聚类相关:")
    print("   - E:\\KKGG\\output\\terms\\definitions.json (术语定义)")
    print("   - E:\\KKGG\\output\\KG\\def_cluster.json (聚类三元组)")
    print("\n请检查上述文件确认结果!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()