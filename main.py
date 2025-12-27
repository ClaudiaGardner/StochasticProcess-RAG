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


def get_embeddings():
    """获取 Embedding 模型（支持本地 HuggingFace 或 API）"""
    model_config = get_model_config()
    embedding_model = model_config.get("embedding_model", "local")
    
    if embedding_model == "local":
        # 设置 HuggingFace 镜像
        import os as _os
        _os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        from langchain_huggingface import HuggingFaceEmbeddings
        print("  📦 使用本地 HuggingFace Embedding 模型...")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
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


def get_llm(model_name=None, temperature=0.3):
    """获取 LLM 实例，支持指定模型名称"""
    api_config = get_api_config()
    model_config = get_model_config()
    
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


def load_vectorstore():
    """加载已有的向量数据库"""
    db_config = get_database_config()
    persist_directory = db_config.get("chroma_dir", "./chroma_db")
    
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"向量数据库不存在: {persist_directory}\n"
            f"请先运行 'python ingest.py' 构建数据库"
        )
    
    print(f"📂 正在加载向量数据库: {persist_directory}")
    
    embeddings = get_embeddings()
    
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
2. **所有数学公式必须使用 LaTeX 格式**：
   - 行内公式使用 `$...$`，如 $P(X=k)$
   - 行间公式使用 `$$...$$`
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
            
            return {
                "input": question,
                "context": docs,
                "answer": answer
            }
    
    return SimpleQAChain(retriever, llm)


def save_answer_as_markdown(question, answer, context_docs, output_dir="./answers"):
    """将回答保存为格式化的Markdown文档"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名（使用时间戳避免重复）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_question = "".join(c for c in question[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_question = safe_question.replace(' ', '_')
    filename = f"{timestamp}_{safe_question}.md"
    filepath = os.path.join(output_dir, filename)
    
    # 构建Markdown内容
    markdown_content = f"""# 随机过程 RAG 问答系统 - 回答记录

**提问时间**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}
**问题**: {question}

## 🤖 AI 回答

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
    print("="*60)
    print(" 🎓 随机过程 RAG 问答系统")
    print("="*60)
    
    try:
        vectorstore = load_vectorstore()
        
        print("🤖 正在初始化大语言模型...")
        model_config = get_model_config()
        llm = get_llm(temperature=model_config.get("temperature", 0.3))
        print("✅ LLM 初始化成功")
        
        print("🔗 正在创建 QA 检索链...")
        qa_chain = create_qa_chain(vectorstore, llm)
        print("✅ QA 链创建成功\n")
        
        print("💬 开始问答（输入 'quit' 或 'exit' 退出）")
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
                    print("� 可以使用 Markdown 查看器打开文件，数学公式将正确显示")
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
