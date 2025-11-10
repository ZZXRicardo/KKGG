#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高相似度词分析 - 简洁版本
只输出相似度矩阵和解释
"""

from disambiguator_clusterer import TermDisambiguator

def analyze_high_similarity_terms():
    """分析高相似度词"""
    
    disambiguator = TermDisambiguator(api_provider="qianwen")
    
    # 高相似度词列表（从你的结果中提取）
    high_sim_terms = ["陆军", "海军", "空军", "煎", "炒", "炸"]
    
    print("高相似度词分析")
    print("=" * 60)
    
    # 获取所有术语的解释和嵌入向量
    term_data = {}
    for term in high_sim_terms:
        explanation, embedding = disambiguator.generate_explanation_and_embedding(term)
        term_data[term] = {
            "explanation": explanation,
            "embedding": embedding
        }
        print(f"{term}: {explanation}")
    
    print("\n" + "=" * 60)
    print("相似度矩阵")
    print("=" * 60)
    
    # 生成相似度矩阵
    print(" " * 6, end="")
    for term in high_sim_terms:
        print(f"{term:8}", end="")
    print()
    
    for i, term1 in enumerate(high_sim_terms):
        print(f"{term1:6}", end="")
        for j, term2 in enumerate(high_sim_terms):
            if i == j:
                print("  1.000  ", end="")
            else:
                sim = disambiguator._cos(term_data[term1]["embedding"], term_data[term2]["embedding"])
                print(f"  {sim:.3f}  ", end="")
        print()

if __name__ == "__main__":
    analyze_high_similarity_terms()