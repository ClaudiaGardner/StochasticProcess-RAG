"""
随机过程 RAG 问答系统主程序
功能：基于向量数据库进行检索增强生成（RAG）问答
"""

import os
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from config_manager import (
    get_api_config, get_model_config, get_database_config, get_retrieval_config
)


def get_embeddings(offline=None):
    """获取 Embedding 模型（支持本地 HuggingFace 或 API）"""
    model_config = get_model_config()
    embedding_model = model_config.get("embedding_model", "local")
    
    # 检查离线模式
    if offline is None:
        offline = model_config.get("offline_mode", False)
    
    if embedding_model == "local":
        import os as _os
        
        # 离线模式：完全使用本地缓存，不联网
        if offline:
            _os.environ['HF_HUB_OFFLINE'] = '1'
            _os.environ['TRANSFORMERS_OFFLINE'] = '1'
            print("  🏠 离线模式：使用本地缓存的 Embedding 模型")
        else:
            # 设置 HuggingFace 镜像（在线模式）
            _os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # 自动检测 GPU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  📦 使用本地 HuggingFace Embedding 模型 (设备: {device})...")
        
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        # 使用 OpenAI 兼容接口
        api_config = get_api_config()
        return OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_config["api_key"],
            openai_api_base=api_config["base_url"],
        )


def get_llm(model_name=None, temperature=0.3, offline=None):
    """获取 LLM 实例，支持在线 API 和离线本地模型（Ollama）"""
    model_config = get_model_config()
    
    # 如果未指定，从配置读取离线模式
    if offline is None:
        offline = model_config.get("offline_mode", False)
    
    if offline:
        # 离线模式：使用 Ollama 本地模型
        local_url = model_config.get("local_llm_url", "http://localhost:11434")
        local_model = model_config.get("local_llm_model", "qwen2.5:7b")
        
        print(f"  🏠 使用本地 Ollama 模型: {local_model}")
        
        return ChatOpenAI(
            model=local_model,
            temperature=temperature,
            openai_api_key="ollama",  # Ollama 不需要真实 API key
            openai_api_base=f"{local_url}/v1",
        )
    else:
        # 在线模式：使用 API
        api_config = get_api_config()
        
        # 如果指定了模型名称就使用，否则使用配置中的第一个模型
        if model_name is None:
            chat_models = model_config.get("chat_models", ["gemini-3-pro-preview"])
            model_name = chat_models[0] if isinstance(chat_models, list) else chat_models
        
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_config["api_key"],
            openai_api_base=api_config["base_url"],
        )


def load_vectorstore(offline=None):
    """加载已有的向量数据库"""
    db_config = get_database_config()
    model_config = get_model_config()
    persist_directory = db_config.get("chroma_dir", "./chroma_db")
    
    # 检查离线模式
    if offline is None:
        offline = model_config.get("offline_mode", False)
    
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"向量数据库不存在: {persist_directory}\n"
            f"请先运行 'python ingest.py' 构建数据库"
        )
    
    print(f"📂 正在加载向量数据库: {persist_directory}")
    
    embeddings = get_embeddings(offline=offline)
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    print("✅ 向量数据库加载成功")
    return vectorstore


def create_qa_chain(vectorstore, llm):
    """创建简单的 QA 检索函数"""
    retrieval_config = get_retrieval_config()
    top_k = retrieval_config.get("top_k", 5)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    class SimpleQAChain:
        def __init__(self, retriever, llm):
            self.retriever = retriever
            self.llm = llm
        
        def invoke(self, inputs):
            question = inputs.get("input", "")
            
            # 检索相关文档
            docs = self.retriever.invoke(question)
            
            # 构建上下文
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            # 构建提示
            prompt = f"""你是一位概率论与随机过程领域的专家教授。请基于以下背景知识回答学生的问题。

**背景知识**:
{context}

---

**回答要求**：
1. 严格基于背景知识中的定义和定理进行回答
2. **数学公式格式要求（非常重要）**：
   - 行内公式**必须**使用美元符号格式：`$公式$`，例如 $P(X=k)$
   - 行间公式**必须**使用双美元符号格式：`$$公式$$`
   - **禁止**使用 \\( \\) 或 \\[ \\] 格式！
3. 使用标准概率论记号（$P$, $E$, $\\operatorname{{Var}}$, $\\sigma$ 等）
4. 如果是例题或习题，请给出详细的解题步骤
5. 如果背景知识不足以完整回答，请说明并提供你的专业见解
6. 回答时使用清晰的结构和条理

**学生问题**: {question}

**教授回答**:"""
            
            # 调用 LLM
            response = self.llm.invoke(prompt)
            
            # 处理不同类型的响应对象
            if isinstance(response, str):
                answer = response
            else:
                answer = response.content
            
            # 后处理：转换 LaTeX 公式格式，确保 Markdown 兼容
            answer = convert_latex_format(answer)
            
            return {
                "input": question,
                "context": docs,
                "answer": answer
            }
    
    return SimpleQAChain(retriever, llm)


def convert_latex_format(text):
    """
    将 LaTeX 公式格式从 \\(...\\) 和 \\[...\\] 转换为 $...$ 和 $$...$$ 格式
    这样可以确保在标准 Markdown 渲染器中正确显示数学公式
    """
    import re
    
    # 转换行间公式：\[...\] -> $$...$$
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # 转换行内公式：\(...\) -> $...$
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    return text


def save_answer_as_markdown(question, answer, context_docs, output_dir="./answers", mode="rag"):
    """将回答保存为格式化的Markdown文档
    
    Args:
        mode: "rag" - AI生成回答, "search" - 纯检索结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名（使用时间戳避免重复）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_question = "".join(c for c in question[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_question = safe_question.replace(' ', '_')
    prefix = "search_" if mode == "search" else ""
    filename = f"{prefix}{timestamp}_{safe_question}.md"
    filepath = os.path.join(output_dir, filename)
    
    # 根据模式选择标题
    if mode == "search":
        title = "# 随机过程 - 检索结果\n"
        section_title = "## 📚 检索到的相关内容"
    else:
        title = "# 随机过程 RAG 问答系统 - 回答记录\n"
        section_title = "## 🤖 AI 回答"
    
    # 构建Markdown内容
    markdown_content = f"""{title}
**查询时间**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}
**问题**: {question}

{section_title}

{answer}

---

## 📚 参考来源 ({len(context_docs)} 个)

"""
    
    # 添加参考来源
    for i, doc in enumerate(context_docs[:5], 1):  # 最多显示5个来源
        doc_type = doc.metadata.get('type', 'original')
        type_label = {
            'original': '📄 原文',
            'solved_problem': '✅ 已解答例题',
            'supplementary_knowledge': '📖 补充知识'
        }.get(doc_type, '📄')
        
        extra_info = ""
        if 'problem_id' in doc.metadata:
            extra_info = f" - 例题 {doc.metadata['problem_id']}"
        elif 'topic' in doc.metadata:
            extra_info = f" - {doc.metadata['topic']}"
        
        markdown_content += f"""
### [{i}] {type_label}{extra_info}

```
{doc.page_content[:500]}{"..." if len(doc.page_content) > 500 else ""}
```

---
"""
    
    markdown_content += f"""

## 📄 系统信息

- **向量数据库**: Chroma
- **嵌入模型**: 本地 HuggingFace
- **LLM**: {get_model_config().get('chat_models', ['默认'])[0]}
- **检索文档数**: {len(context_docs)}

---
*由随机过程 RAG 问答系统生成*
"""
    
    # 保存文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return filepath


def main():
    """主函数 - 交互式问答"""
    import sys
    
    # 检查命令行参数
    OFFLINE_MODE = '--offline' in sys.argv
    SEARCH_ONLY = '--search' in sys.argv  # 纯检索模式：只返回原文，不使用 LLM
    
    print("="*60)
    print(" 🎓 随机过程 RAG 问答系统")
    if OFFLINE_MODE:
        print(" 🏠 离线模式（使用本地 Ollama 模型）")
    if SEARCH_ONLY:
        print(" 🔍 纯检索模式（只返回教材原文，不使用 LLM）")
    print("="*60)
    
    try:
        # 命令行参数优先，否则使用配置文件
        model_config = get_model_config()
        offline = OFFLINE_MODE or model_config.get("offline_mode", False)
        
        vectorstore = load_vectorstore(offline=offline)
        retrieval_config = get_retrieval_config()
        top_k = retrieval_config.get("top_k", 5)
        
        # 纯检索模式：不需要 LLM
        if not SEARCH_ONLY:
            print("🤖 正在初始化大语言模型...")
            llm = get_llm(temperature=model_config.get("temperature", 0.3), offline=offline)
            print("✅ LLM 初始化成功")
            
            print("🔗 正在创建 QA 检索链...")
            qa_chain = create_qa_chain(vectorstore, llm)
            print("✅ QA 链创建成功\n")
        else:
            qa_chain = None
            print("✅ 纯检索模式就绪\n")
        
        print("💬 开始问答（输入 'quit' 或 'exit' 退出）")
        if SEARCH_ONLY:
            print("📖 纯检索模式：直接返回教材原文和已有解答")
        print("💡 示例问题：")
        print("   - 什么是马尔可夫链？")
        print("   - 泊松过程有什么性质？")
        print("   - 解释常返态和瞬时态的区别")
        print()
        
        while True:
            question = input("🙋 您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n👋 再见！")
                break
            
            try:
                print("\n🔍 正在检索相关知识...")
                
                # 纯检索模式：只返回检索结果，不使用 LLM
                if SEARCH_ONLY:
                    docs = vectorstore.similarity_search(question, k=top_k)
                    
                    print("\n" + "="*60)
                    print(f"📚 找到 {len(docs)} 个相关文档:")
                    print("="*60)
                    
                    # 构建结果文本
                    result_text = ""
                    for i, doc in enumerate(docs, 1):
                        doc_type = doc.metadata.get('type', 'original')
                        type_label = {
                            'original': '📄 教材原文',
                            'solved_problem': '✅ 已解答例题/习题',
                            'supplementary_knowledge': '📖 补充知识'
                        }.get(doc_type, '📄')
                        
                        # 获取来源信息
                        extra_info = ""
                        source_info = ""
                        
                        # 页码信息
                        if 'page' in doc.metadata:
                            source_info += f"第 {doc.metadata['page'] + 1} 页"
                        
                        # 来源文件
                        if 'source_file' in doc.metadata:
                            if source_info:
                                source_info += f" | {doc.metadata['source_file']}"
                            else:
                                source_info = doc.metadata['source_file']
                        elif 'source' in doc.metadata:
                            # PyPDFLoader 默认的 source 字段
                            import os
                            source_name = os.path.basename(doc.metadata['source'])
                            if source_info:
                                source_info += f" | {source_name}"
                            else:
                                source_info = source_name
                        
                        # 题目ID
                        if 'problem_id' in doc.metadata:
                            extra_info = f" - {doc.metadata['problem_id']}"
                        elif 'topic' in doc.metadata:
                            extra_info = f" - {doc.metadata['topic']}"
                        
                        # 构建标题
                        header = f"[{i}] {type_label}{extra_info}"
                        if source_info:
                            header += f"\n    📍 来源: {source_info}"
                        
                        print(f"\n{'='*60}")
                        print(header)
                        print("-"*60)
                        # 显示完整内容
                        print(doc.page_content)
                        
                        # 累积到结果文本（Markdown 格式）
                        result_text += f"\n## [{i}] {type_label}{extra_info}\n\n"
                        if source_info:
                            result_text += f"**📍 来源**: {source_info}\n\n"
                        result_text += doc.page_content + "\n\n---\n"
                    
                    print("\n" + "="*60)
                    
                    # 保存到 Markdown 文件
                    try:
                        filepath = save_answer_as_markdown(
                            question=question,
                            answer=result_text,
                            context_docs=docs,
                            mode="search"  # 标记为检索模式
                        )
                        print(f"💾 检索结果已保存到: {filepath}")
                    except Exception as e:
                        print(f"❌ 保存失败: {str(e)}")
                else:
                    # 正常 RAG 模式
                    result = qa_chain.invoke({"input": question})
                    
                    print("\n" + "="*60)
                    print("🤖 回答:")
                    print("-"*60)
                    print(result['answer'])
                    print("="*60)
                    
                    # 自动保存回答为Markdown文件
                    try:
                        filepath = save_answer_as_markdown(
                            question=result['input'],
                            answer=result['answer'],
                            context_docs=result['context']
                        )
                        print(f"💾 回答已自动保存到: {filepath}")
                        print("📄 可以使用 Markdown 查看器打开文件，数学公式将正确显示")
                    except Exception as e:
                        print(f"❌ 保存失败: {str(e)}")
                    
                    if result.get('context'):
                        print(f"\n📚 参考来源 ({len(result['context'])} 个):")
                        for i, doc in enumerate(result['context'][:3], 1):
                            doc_type = doc.metadata.get('type', 'original')
                            type_label = {
                                'original': '📄 原文',
                                'solved_problem': '✅ 已解答例题',
                                'supplementary_knowledge': '📖 补充知识'
                            }.get(doc_type, '📄')
                            
                            extra_info = ""
                            if 'problem_id' in doc.metadata:
                                extra_info = f" - 例题 {doc.metadata['problem_id']}"
                            elif 'topic' in doc.metadata:
                                extra_info = f" - {doc.metadata['topic']}"
                            
                            print(f"\n[{i}] {type_label}{extra_info}")
                            print("-"*40)
                            content = doc.page_content[:150]
                            print(content + "..." if len(doc.page_content) > 150 else content)
                
                print("\n")
                
            except Exception as e:
                print(f"\n❌ 问答出错: {str(e)}\n")
    
    except FileNotFoundError as e:
        print(f"\n❌ {str(e)}\n")
    except Exception as e:
        print(f"\n❌ 系统错误: {str(e)}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
