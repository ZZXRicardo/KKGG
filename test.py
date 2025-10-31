# 调试脚本
import sys
import os

api_path = r"E:\KKGG\project\API调用"
print(f"检查路径: {api_path}")
print(f"路径存在: {os.path.exists(api_path)}")

if os.path.exists(api_path):
    print("目录内容:")
    for file in os.listdir(api_path):
        print(f"  - {file}")
    
    # 检查具体文件
    llm_path = os.path.join(api_path, "LLM.py")
    embedding_path = os.path.join(api_path, "Emdedding.py")
    
    print(f"\nLLM.py 存在: {os.path.exists(llm_path)}")
    print(f"Emdedding.py 存在: {os.path.exists(embedding_path)}")
    
    # 尝试导入
    sys.path.insert(0, api_path)
    try:
        from LLM import LLM
        print("✅ LLM 导入成功")
    except ImportError as e:
        print(f"❌ LLM 导入失败: {e}")
    
    try:
        from Emdedding import Embedding
        print("✅ Embedding 导入成功")
    except ImportError as e:
        print(f"❌ Embedding 导入失败: {e}")