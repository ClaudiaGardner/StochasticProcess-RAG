"""
文档摄入与向量化模块
功能：解析 PDF 文档、提取例题习题、调用 API 解答、向量化并存储到 Chroma 数据库
"""

import os
import re
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config_manager import (
    get_api_config, get_model_config, get_database_config,
    get_ingestion_config, get_topics
)


def get_embeddings():
    """获取 Embedding 模型（使用 OpenAI 兼容接口）"""
    api_config = get_api_config()
    model_config = get_model_config()
    
    return OpenAIEmbeddings(
        model=model_config.get("embedding_model", "text-embedding-3-small"),
        openai_api_key=api_config["api_key"],
        openai_api_base=api_config["base_url"],
    )


def get_llm():
    """获取 LLM 实例"""
    api_config = get_api_config()
    model_config = get_model_config()
    
    return ChatOpenAI(
        model=model_config.get("chat_model", "gemini-3-pro-preview"),
        temperature=model_config.get("temperature", 0.3),
        openai_api_key=api_config["api_key"],
        openai_api_base=api_config["base_url"],
    )


def load_and_split_pdf(pdf_path, chunk_size=800, chunk_overlap=150):
    """加载 PDF 并切分文档"""
    print(f"📖 正在加载 PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ 成功加载 {len(documents)} 页")
    
    separators = [
        "\n【例题",
        "\n【例",
        "\n例题",
        "\n习题",
        "\n§",
        "\n定义",
        "\n定理",
        "\n证明",
        "\n\n",
        "\n",
        "。",
        " ",
        ""
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    
    print(f"🔪 正在切分文档 (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    splits = text_splitter.split_documents(documents)
    print(f"✅ 成功切分为 {len(splits)} 个片段")
    
    return splits, documents


def extract_problems(documents):
    """从文档中提取例题和习题（课后作业）"""
    problems = []
    
    # 合并所有页面内容
    full_text = "\n".join([doc.page_content for doc in documents])
    
    # 例题匹配模式 - 更全面
    example_patterns = [
        # 【例题 X.Y】格式
        (r'【例题\s*([\d\.]+)】\s*(.+?)(?=【例题|【例|例题\s*[\d\.]|例\s*[\d\.]|§|$)', 'example'),
        # 【例 X.Y】格式  
        (r'【例\s*([\d\.]+)】\s*(.+?)(?=【例题|【例|例题\s*[\d\.]|例\s*[\d\.]|§|$)', 'example'),
        # 例题 X.Y 格式（无方括号，有空格）
        (r'(?<![【])例题\s+([\d\.]+)\s*(.+?)(?=例题\s*[\d\.]|例\s*[\d\.]|§|$)', 'example'),
        # 例题X.Y 格式（无空格）
        (r'(?<![【])例题([\d\.]+)\s*(.+?)(?=例题[\d\.]|例\s*[\d\.]|§|$)', 'example'),
        # 例 X.Y.Z 格式（如 例0.1.1）
        (r'(?<![【例题])例\s*([\d\.]+)\s*(.+?)(?=例题|例\s*[\d\.]|§|$)', 'example'),
    ]
    
    seen_ids = set()
    
    # 提取例题
    for pattern, prob_type in example_patterns:
        try:
            matches = re.findall(pattern, full_text, re.DOTALL)
            for match in matches:
                prob_id = match[0].strip()
                content = match[1].strip()
                
                # 标准化 ID（去除多余空格）
                prob_id = re.sub(r'\s+', '', prob_id)
                
                unique_key = f"example_{prob_id}"
                if unique_key in seen_ids:
                    continue
                seen_ids.add(unique_key)
                
                if len(content) > 30:
                    content = re.sub(r'\s+', ' ', content)[:2000]
                    problems.append({
                        'id': f"例题{prob_id}",
                        'content': content,
                        'type': 'example'
                    })
        except Exception as e:
            print(f"  ⚠️ 模式匹配错误: {str(e)}")
    
    # 提取课后作业
    homework_sections = re.findall(r'§\s*([\d\.]+)\s*课后作业\s*\n(.+?)(?=§|\Z)', full_text, re.DOTALL)
    for section_id, section_content in homework_sections:
        # 在每个作业章节中提取单独的题目
        hw_problems = re.findall(r'(\d+)\.\s*(.+?)(?=\n\d+\.|$)', section_content, re.DOTALL)
        for hw_num, hw_content in hw_problems:
            prob_id = f"{section_id}.{hw_num}"
            unique_key = f"hw_{prob_id}"
            
            if unique_key in seen_ids:
                continue
            seen_ids.add(unique_key)
            
            hw_content = hw_content.strip()
            if len(hw_content) > 20:
                hw_content = re.sub(r'\s+', ' ', hw_content)[:2000]
                problems.append({
                    'id': f"习题{prob_id}",
                    'content': hw_content,
                    'type': 'exercise'
                })
    
    # 按类型和 ID 排序
    def sort_key(p):
        # 提取数字进行排序
        nums = re.findall(r'[\d\.]+', p['id'])
        if nums:
            parts = nums[0].split('.')
            return (0 if p['type'] == 'example' else 1, 
                    [float(x) if x else 0 for x in parts])
        return (2, [999])
    
    problems.sort(key=sort_key)
    
    return problems


def solve_problem_with_api(llm, problem_id, problem_content, problem_type):
    """使用 API 解答例题或习题，正确处理数学公式"""
    type_name = '例题' if problem_type == 'example' else '习题'
    
    prompt = f"""你是一位概率论与随机过程领域的资深数学教授。请详细解答以下{type_name}，并且不要有客套话，直接完成要求即可。

**重要说明**：题目中可能包含数学公式，请仔细识别并正确理解。

---
## {type_name} {problem_id}

{problem_content}

---

请按以下格式严格回答：

### 题目分析
- 简洁明了，不用分点
- 识别题目中的数学符号和公式
- 分析本题考查的核心概念（如马尔可夫链、泊松过程、随机游走等）
- 明确需要求解的目标

### 解题过程
给出完整详细的解题步骤。**所有数学公式必须使用 LaTeX 格式**：
- 行内公式使用 `$...$`，例如：$P(X=k)$
- 行间公式使用 `$$...$$`，例如：
$$P_{{ij}} = P(X_{{n+1}}=j | X_n=i)$$

请确保：
1. 每一步推导都有清晰的解释
2. 公式书写规范，使用正确的 LaTeX 语法
3. 概率符号、期望、方差等使用标准记号

### 答案
给出最终答案，用数学公式表达

### 知识延伸
- 简洁明了
- 总结本题涉及的定理和性质
- 列出相关的重要公式
- 指出常见的解题技巧和易错点

请开始解答："""
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"  ⚠️ API 调用失败: {str(e)}")
        return None


def generate_supplementary_knowledge(llm, topic):
    """使用 API 生成补充知识，正确输出数学公式"""
    prompt = f"""你是一位概率论与随机过程领域的专家教授。请针对以下主题提供系统、详细的知识讲解。

**主题**：{topic}

**格式要求**：
- 所有数学公式必须使用 LaTeX 格式
- 行内公式使用 `$...$`
- 行间公式使用 `$$...$$`
- 使用标准的概率论记号（如 $P$, $E$, $\operatorname{{Var}}$, $\sigma$ 等）

---

请包含以下内容：

## 1. 基本定义
给出严格的数学定义。例如：
$$P(A|B) = \\frac{{P(A \\cap B)}}{{P(B)}}$$

## 2. 核心性质
列举重要的性质和定理，每个性质用公式表达

## 3. 关键公式
给出常用的计算公式，如概率计算、期望、方差等
用公式列表形式展示

## 4. 典型例子
用具体数值举例说明概念的应用
包含完整的计算过程

## 5. 与其他概念的联系
说明与相关概念（马尔可夫链、泊松过程、随机游走等）的关系

## 6. 常见考点与易错点
- 学习要点
- 常见错误
- 解题技巧

请使用严谨的数学语言进行阐述："""
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"  ⚠️ 知识生成失败: {str(e)}")
        return None


def create_vectorstore(documents, persist_directory):
    """创建向量存储"""
    print(f"🧠 正在初始化 Embedding 模型...")
    embeddings = get_embeddings()
    
    print(f"💾 正在创建向量数据库并持久化到: {persist_directory}")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✅ 向量数据库创建成功！")
    return vectorstore


def main():
    """主函数"""
    # 从配置读取参数
    db_config = get_database_config()
    ing_config = get_ingestion_config()
    
    PDF_PATH = ing_config.get("pdf_path", "data/SP-10-12.pdf")
    CHROMA_DIR = db_config.get("chroma_dir", "./chroma_db")
    SOLUTIONS_DIR = db_config.get("solutions_dir", "./solutions")
    CHUNK_SIZE = ing_config.get("chunk_size", 800)
    CHUNK_OVERLAP = ing_config.get("chunk_overlap", 150)
    MAX_PROBLEMS = ing_config.get("max_problems_to_solve", 10)
    
    # 检查 PDF 是否存在
    if not os.path.exists(PDF_PATH):
        print(f"❌ 错误: 未找到 PDF 文件 {PDF_PATH}")
        return
    
    # 如果向量数据库已存在，询问是否重建
    if os.path.exists(CHROMA_DIR):
        response = input(f"⚠️  向量数据库 {CHROMA_DIR} 已存在，是否重建？(y/n): ")
        if response.lower() != 'y':
            print("❌ 取消操作")
            return
        print("🗑️  删除旧数据库...")
        import shutil
        shutil.rmtree(CHROMA_DIR)
    
    # 创建目录
    Path(SOLUTIONS_DIR).mkdir(exist_ok=True)
    
    try:
        # 步骤 1: 加载和切分文档
        splits, raw_documents = load_and_split_pdf(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # 打印示例片段
        print("\n📄 示例文档片段:")
        print("-" * 60)
        print(splits[0].page_content[:300])
        print("-" * 60)
        
        # 步骤 2: 提取例题和习题
        print("\n🔍 正在提取例题和习题...")
        problems = extract_problems(raw_documents)
        print(f"✅ 找到 {len(problems)} 个例题/习题")
        
        # 显示找到的题目列表
        examples = [p for p in problems if p['type'] == 'example']
        exercises = [p for p in problems if p['type'] == 'exercise']
        print(f"\n📋 题目统计: {len(examples)} 个例题, {len(exercises)} 个习题")
        
        # 生成题目索引文档
        index_file = f"{SOLUTIONS_DIR}/题目索引.md"
        print(f"\n📝 正在生成题目索引文档: {index_file}")
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("# 随机过程 - 题目索引\n\n")
            f.write(f"**生成时间**: {Path(PDF_PATH).stem}\n\n")
            f.write(f"**总计**: {len(problems)} 道题目 ({len(examples)} 例题 + {len(exercises)} 习题)\n\n")
            f.write("---\n\n")
            
            # 写入例题
            if examples:
                f.write("## 例题列表\n\n")
                for i, p in enumerate(examples, 1):
                    f.write(f"### {p['id']}\n\n")
                    f.write(f"{p['content']}\n\n")
                    f.write("---\n\n")
            
            # 写入习题
            if exercises:
                f.write("## 习题列表（课后作业）\n\n")
                for i, p in enumerate(exercises, 1):
                    f.write(f"### {p['id']}\n\n")
                    f.write(f"{p['content']}\n\n")
                    f.write("---\n\n")
        
        print(f"✅ 题目索引已保存到: {index_file}")
        
        # 显示部分题目预览
        print("\n📋 题目预览 (前20个):")
        for i, p in enumerate(problems[:20], 1):
            print(f"  {i}. {p['id']}: {p['content'][:40]}...")
        if len(problems) > 20:
            print(f"  ... 还有 {len(problems) - 20} 个题目 (完整列表见 {index_file})")
        
        # 步骤 3: 按章节组织题目并使用 API 解答
        llm = get_llm()
        solved_docs = []
        
        # 按章节分组题目
        def get_chapter(prob_id):
            """从题目ID提取章节号"""
            nums = re.findall(r'[\d]+', prob_id)
            if nums:
                return nums[0]  # 第一个数字作为章节号
            return "其他"
        
        # 分组
        chapters = {}
        for prob in problems:
            chapter = get_chapter(prob['id'])
            if chapter not in chapters:
                chapters[chapter] = []
            chapters[chapter].append(prob)
        
        print(f"\n📚 按章节分组:")
        for ch, probs in sorted(chapters.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
            print(f"  第 {ch} 章: {len(probs)} 道题目")
        
        # 解答所有题目并按章节生成文档
        problems_to_solve = problems if MAX_PROBLEMS == 0 else problems[:MAX_PROBLEMS]
        
        if problems_to_solve:
            print(f"\n📝 正在使用 API 解答 {len(problems_to_solve)} 个题目...")
            
            # 用于收集每章的解答
            chapter_solutions = {}
            
            for i, prob in enumerate(problems_to_solve, 1):
                chapter = get_chapter(prob['id'])
                print(f"  [{i}/{len(problems_to_solve)}] 正在解答 {prob['id']}...")
                
                solution = solve_problem_with_api(llm, prob['id'], prob['content'], prob['type'])
                
                if solution:
                    # 收集到对应章节
                    if chapter not in chapter_solutions:
                        chapter_solutions[chapter] = []
                    chapter_solutions[chapter].append({
                        'id': prob['id'],
                        'content': prob['content'],
                        'solution': solution,
                        'type': prob['type']
                    })
                    
                    # 添加到向量库
                    solved_docs.append(Document(
                        page_content=f"{prob['id']}：{prob['content']}\n\n解答：{solution}",
                        metadata={
                            'type': 'solved_problem', 
                            'problem_id': prob['id'],
                            'problem_type': prob['type'],
                            'chapter': chapter
                        }
                    ))
                    print(f"    ✅ 已解答")
            
            # 按章节生成文档
            print(f"\n📄 正在生成章节解答文档...")
            for chapter, solutions in sorted(chapter_solutions.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                chapter_file = f"{SOLUTIONS_DIR}/第{chapter}章_题目与解答.md"
                with open(chapter_file, 'w', encoding='utf-8') as f:
                    f.write(f"# 第 {chapter} 章 - 题目与解答\n\n")
                    f.write(f"**本章共 {len(solutions)} 道题目**\n\n")
                    f.write("---\n\n")
                    
                    for sol in solutions:
                        type_label = "例题" if sol['type'] == 'example' else "习题"
                        f.write(f"## {sol['id']}\n\n")
                        f.write(f"### 题目\n\n{sol['content']}\n\n")
                        f.write(f"### 解答\n\n{sol['solution']}\n\n")
                        f.write("---\n\n")
                
                print(f"  ✅ 已生成: {chapter_file} ({len(solutions)} 道题)")
        
        # 生成补充知识
        core_topics = get_topics()
        if core_topics:
            print(f"\n📚 正在生成 {len(core_topics)} 个主题的补充知识...")
            for i, topic in enumerate(core_topics, 1):
                print(f"  [{i}/{len(core_topics)}] {topic}...")
                knowledge = generate_supplementary_knowledge(llm, topic)
                if knowledge:
                    # 保存到文件
                    filename = f"{SOLUTIONS_DIR}/知识点_{i}_{topic[:10]}.md"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"# {topic}\n\n{knowledge}\n")
                    
                    # 添加到向量库
                    solved_docs.append(Document(
                        page_content=f"{topic}\n\n{knowledge}",
                        metadata={'type': 'supplementary_knowledge', 'topic': topic}
                    ))
                    print(f"    ✅ 已生成并保存")
        
        # 步骤 4: 合并所有文档并创建向量存储
        print(f"\n📊 合并文档...")
        all_documents = splits + solved_docs
        print(f"✅ 共 {len(all_documents)} 个文档片段 (原文: {len(splits)}, 解答+知识: {len(solved_docs)})")
        
        vectorstore = create_vectorstore(all_documents, CHROMA_DIR)
        
        # 测试检索
        print("\n🔍 测试检索功能...")
        test_query = "什么是马尔可夫链"
        results = vectorstore.similarity_search(test_query, k=2)
        print(f"查询: '{test_query}'")
        print(f"找到 {len(results)} 个相关片段")
        
        print("\n" + "="*60)
        print("🎉 所有步骤完成！")
        print("="*60)
        print(f"   📁 向量数据库: {CHROMA_DIR}")
        print(f"   📁 解答文件: {SOLUTIONS_DIR}")
        print(f"   📊 总文档数: {len(all_documents)}")
        print("   🚀 运行 'python main.py' 开始问答")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
