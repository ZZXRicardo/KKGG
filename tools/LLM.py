import requests

class LLM:
    # 支持多个API提供商
    API_CONFIGS = {
        "deepseek": {
            "url": "https://api.deepseek.com/chat/completions",
            "model": "deepseek-chat",
            "key": "sk-d0f3da2caff640aab4da1cc25737849f"  # 测试密钥
        },
        "qianwen": {
            "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            "model": "qwen3-max",
            "key": "sk-ecf819b71fae427bb1ca8be81a257509"
        },
        # ✅ 新增：豆包（Doubao / ByteArk）
        "doubao": {
            # 方舟(Ark) OpenAI 兼容接口：chat.completions
            "url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            # 你示例里的模型 ID（可按需改成你的线上可用模型）
            "model": "doubao-seed-1-6-250615",  # ✅ 修正这里
            "key": "c9edb0ad-d9b8-4ed3-a5f6-69bfa1208dcc"
        }
    }
    
    def __init__(self, prompt="", api_provider="deepseek"):
        self.prompt = prompt
        self.api_provider = api_provider
        
        # 验证API提供商是否支持
        if api_provider not in self.API_CONFIGS:
            raise ValueError(f"不支持的API提供商: {api_provider}。支持的有: {list(self.API_CONFIGS.keys())}")

    def llm_call(self):
        """调用API，兼容多个提供商，固定参数temperature=1, max_tokens=150"""
        if not self.prompt:
            raise ValueError("Prompt cannot be empty. Please set a prompt.")
        
        config = self.API_CONFIGS[self.api_provider]
        
        if self.api_provider == "deepseek":
            return self._call_deepseek(config)
        elif self.api_provider == "qianwen":
            return self._call_qianwen(config)
        elif self.api_provider == "doubao":
            return self._call_doubao(config)
        else:
            raise ValueError(f"未实现的API提供商: {self.api_provider}")

    def _call_deepseek(self, config):
        """调用Deepseek API，固定参数temperature=1, max_tokens=150"""
        headers = {
            "Authorization": f"Bearer {config['key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": self.prompt}],
            "temperature": 1,
            "max_tokens": 150
        }
        response = requests.post(config["url"], headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Deepseek API 错误: {response.text}")
            return {"error": f"API 请求失败，状态码：{response.status_code}", "details": response.text}

    def _call_qianwen(self, config):
        """调用千问API，固定参数temperature=1, max_tokens=150"""
        headers = {
            "Authorization": f"Bearer {config['key']}",
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
                "max_tokens": 150
            }
        }
        response = requests.post(config["url"], headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"千问 API 错误: {response.text}")
            return {"error": f"API 请求失败，状态码：{response.status_code}", "details": response.text}

    def _call_doubao(self, config):
        """调用豆包(方舟 OpenAI 兼容) API，固定参数temperature=1, max_tokens=150"""
        headers = {
            "Authorization": f"Bearer {config['key']}",
            "Content-Type": "application/json"
        }
        # 方舟的 chat.completions 与 OpenAI 兼容：messages/choices[0].message.content
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": self.prompt}],
            "temperature": 1,
            "max_tokens": 150
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
            elif self.api_provider == "qianwen":
                return api_response['output']['choices'][0]['message']['content']
            elif self.api_provider == "doubao":
                # 与 OpenAI 对齐
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
