"""
LLM Configuration Module
为 RAG 系统提供 LLM 接口抽象层，支持 OpenAI 兼容接口和未来的 Ollama 集成
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()


def get_llm(temperature=0.3):
    """
    获取配置好的 LLM 实例
    
    Args:
        temperature: 温度参数，控制生成随机性 (0-1)
    
    Returns:
        ChatOpenAI: 配置好的 LLM 实例
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    if not api_key:
        raise ValueError("未找到 OPENAI_API_KEY，请检查 .env 文件")
    
    if not base_url:
        raise ValueError("未找到 OPENAI_API_BASE，请检查 .env 文件")
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    
    return llm


def get_ollama_llm(model_name="qwen2.5:7b", temperature=0.3):
    """
    获取 Ollama 本地 LLM 实例（预留接口）
    
    Args:
        model_name: Ollama 模型名称
        temperature: 温度参数
    
    Returns:
        Ollama: 配置好的 Ollama 实例
    """
    # 预留 Ollama 接口，暂不实现
    # from langchain_community.llms import Ollama
    # return Ollama(model=model_name, temperature=temperature)
    raise NotImplementedError("Ollama 接口暂未实现，请使用 get_llm()")
