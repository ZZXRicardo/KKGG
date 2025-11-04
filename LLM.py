import requests
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 已加载环境变量文件: {env_path}")

class LLM:
    API_CONFIGS = {
        "deepseek": {
            "url": "https://api.deepseek.com/chat/completions",
            "model": "deepseek-chat",
            "key_env": "DEEPSEEK_API_KEY"
        },
        "deepseek-reasoning": {
            "url": "https://api.deepseek.com/chat/completions",
            "model": "deepseek-reasoner",
            "key_env": "DEEPSEEK_API_KEY"
        },
        "qianwen": {
            "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            "model": "qwen3-max", 
            "key_env": "QIANWEN_API_KEY"
        },
        "doubao": {
            "url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            "model": "doubao-seed-1-6-250615",
            "key_env": "DOUBAO_API_KEY"
        }
    }
    
    def __init__(self, prompt="", api_provider="deepseek"):
        self.prompt = prompt
        self.api_provider = api_provider
        
        # 验证API提供商是否支持
        if api_provider not in self.API_CONFIGS:
            raise ValueError(f"不支持的API提供商: {api_provider}。支持的有: {list(self.API_CONFIGS.keys())}")

        # 验证环境变量是否存在
        config = self.API_CONFIGS[api_provider]
        api_key = os.getenv(config['key_env'])
        if not api_key:
            raise ValueError(f"请设置环境变量: {config['key_env']}")

    def _get_api_key(self):
        """从环境变量获取API密钥"""
        config = self.API_CONFIGS[self.api_provider]
        api_key = os.getenv(config['key_env'])
        if not api_key:
            raise ValueError(f"环境变量 {config['key_env']} 未设置")
        return api_key

    def llm_call(self):
        """调用API，兼容多个提供商，固定参数temperature=1, max_tokens=150"""
        if not self.prompt:
            raise ValueError("Prompt cannot be empty. Please set a prompt.")
        
        config = self.API_CONFIGS[self.api_provider]
        api_key = self._get_api_key()
        
        if self.api_provider == "deepseek":
            return self._call_deepseek(config, api_key)
        elif self.api_provider == "deepseek-reasoning":
            return self._call_deepseek_reasoning(config, api_key)
        elif self.api_provider == "qianwen":
            return self._call_qianwen(config, api_key)
        elif self.api_provider == "doubao":
            return self._call_doubao(config, api_key)
        else:
            raise ValueError(f"未实现的API提供商: {self.api_provider}")

    def _call_deepseek(self, config, api_key):
        """调用Deepseek API，固定参数temperature=1, max_tokens=150"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": self.prompt}],
            "temperature": 1,
            "max_tokens": 1000
        }
        response = requests.post(config["url"], headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Deepseek API 错误: {response.text}")
            return {"error": f"API 请求失败，状态码：{response.status_code}", "details": response.text}

    def _call_deepseek_reasoning(self, config, api_key):
        """调用Deepseek Reasoning API，固定参数temperature=1, max_tokens=150"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": self.prompt}],
            "temperature": 1,
            "max_tokens": 3000,
            "reasoning": True  # 启用思考模式
        }
        response = requests.post(config["url"], headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Deepseek Reasoning API 错误: {response.text}")
            return {"error": f"API 请求失败，状态码：{response.status_code}", "details": response.text}

    def _call_qianwen(self, config, api_key):
        """调用千问API，固定参数temperature=1, max_tokens=150"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config["model"],
            "input": {
                "messages": [
                    {"role": "user", "content": self.prompt}
                ]
            },
            "parameters": {
                "temperature": 1,
                "max_tokens": 1000
            }
        }
        response = requests.post(config["url"], headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"千问 API 错误: {response.text}")
            return {"error": f"API 请求失败，状态码：{response.status_code}", "details": response.text}

    def _call_doubao(self, config, api_key):
        """调用豆包(方舟 OpenAI 兼容) API，固定参数temperature=1, max_tokens=150"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": self.prompt}],
            "temperature": 1,
            "max_tokens": 1000
        }
        response = requests.post(config["url"], headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"豆包 API 错误: {response.text}")
            return {"error": f"API 请求失败，状态码：{response.status_code}", "details": response.text}

    def extract_response(self, api_response):
        """从API响应中提取文本内容，兼容不同提供商"""
        if "error" in api_response:
            return f"错误: {api_response['error']}"
        try:
            if self.api_provider == "deepseek":
                return api_response['choices'][0]['message']['content']
            elif self.api_provider == "deepseek-reasoning":
                # 对于思考模式，我们只返回最终的回答内容，不返回思考过程
                message = api_response['choices'][0]['message']
                # 如果有常规的content字段，使用它（这是最终回答）
                if 'content' in message and message['content']:
                    return message['content']
                # 如果没有常规content，但有思考内容，说明思考过程被截断了，没有生成最终回答
                elif 'reasoning_content' in message:
                    return "思考过程已完成，但未生成最终回答（可能因token限制被截断）"
                else:
                    return "未获取到有效回复"
            elif self.api_provider == "qianwen":
                return api_response['output']['choices'][0]['message']['content']
            elif self.api_provider == "doubao":
                return api_response['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as e:
            return f"解析响应失败: {str(e)}"

# 示例用法
if __name__ == "__main__":
    print("=== 使用 Deepseek API ===")
    llm_deepseek = LLM(prompt="你好，请介绍一下你自己", api_provider="deepseek")
    result_deepseek = llm_deepseek.llm_call()
    print(f"原始响应: {result_deepseek}")
    content_deepseek = llm_deepseek.extract_response(result_deepseek)
    print(f"提取内容: {content_deepseek}")

    print("\n=== 使用 Deepseek Reasoning API ===")
    llm_deepseek_reasoning = LLM(prompt="请解释一下量子计算的基本原理", api_provider="deepseek-reasoning")
    result_deepseek_reasoning = llm_deepseek_reasoning.llm_call()
    print(f"原始响应: {result_deepseek_reasoning}")
    content_deepseek_reasoning = llm_deepseek_reasoning.extract_response(result_deepseek_reasoning)
    print(f"提取内容: {content_deepseek_reasoning}")

    print("\n=== 使用千问 API ===")
    llm_qianwen = LLM(prompt="你好，请介绍一下你自己", api_provider="qianwen")
    result_qianwen = llm_qianwen.llm_call()
    print(f"原始响应: {result_qianwen}")
    content_qianwen = llm_qianwen.extract_response(result_qianwen)
    print(f"提取内容: {content_qianwen}")
    
    print("\n=== 使用豆包 API ===")
    llm_doubao = LLM(prompt="你好，请介绍一下你自己", api_provider="doubao")
    result_doubao = llm_doubao.llm_call()
    print(f"原始响应: {result_doubao}")
    content_doubao = llm_doubao.extract_response(result_doubao)
    print(f"提取内容: {content_doubao}")