"""
配置管理模块
从 config.toml 读取配置
"""

import tomllib
from pathlib import Path


def load_config(config_path="config.toml"):
    """加载配置文件"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(path, "rb") as f:
        return tomllib.load(f)


# 全局配置
_config = None


def get_config():
    """获取配置（单例模式）"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_api_config():
    """获取 API 配置"""
    config = get_config()
    return config.get("api", {})


def get_model_config():
    """获取模型配置"""
    config = get_config()
    return config.get("model", {})


def get_database_config():
    """获取数据库配置"""
    config = get_config()
    return config.get("database", {})


def get_ingestion_config():
    """获取文档处理配置"""
    config = get_config()
    return config.get("ingestion", {})


def get_retrieval_config():
    """获取检索配置"""
    config = get_config()
    return config.get("retrieval", {})


def get_topics():
    """获取核心主题列表"""
    config = get_config()
    return config.get("topics", {}).get("core_topics", [])
