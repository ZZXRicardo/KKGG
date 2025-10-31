import os
from pathlib import Path
from dotenv import load_dotenv

class ProjectConfig:
    """项目配置类"""
    
    DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
    QIANWEN_API_KEY = "QIANWEN_API_KEY" 
    DOUBAO_API_KEY = "DOUBAO_API_KEY"
    
    _loaded = False
    
    @classmethod
    def init_project(cls):
        if cls._loaded:
            return True
            
        project_root = Path(__file__).parent.parent
        env_path = project_root / '.env'
        
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ 已加载环境变量文件: {env_path}")
            cls._loaded = True
            return True
        else:
            print(f"⚠️  未找到 .env 文件: {env_path}")
            cls._loaded = True
            return False
    
    @classmethod
    def get(cls, key, default=None):
        if not cls._loaded:
            cls.init_project()
        return os.getenv(key, default)
    
    @classmethod
    def require(cls, key):
        value = cls.get(key)
        if not value:
            raise ValueError(f"必需的环境变量 {key} 未设置")
        return value
    
    @classmethod
    def check_required_vars(cls):
        required_vars = [
            cls.DEEPSEEK_API_KEY,
            cls.QIANWEN_API_KEY,
            cls.DOUBAO_API_KEY
        ]
        
        missing = []
        for var in required_vars:
            if not cls.get(var):
                missing.append(var)
        
        if missing:
            print("❌ 以下必需环境变量未设置:")
            for var in missing:
                print(f"   - {var}")
            return False
        else:
            print("✅ 所有必需环境变量已设置")
            return True
