"""
文档摄入与向量化模块
功能：解析 PDF 文档、提取例题习题、调用 API 解答、向量化并存储到 Chroma 数据库
"""

import os
import re
import json
from pathlib import Path

# PDF 解析库
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# OCR 库（支持数学公式）
try:
    from pix2text import Pix2Text
    HAS_PIX2TEXT = True
except ImportError:
    HAS_PIX2TEXT = False
    print("⚠️ Pix2Text 未安装，OCR 功能不可用")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config_manager import (
    get_api_config, get_model_config, get_database_config,
    get_ingestion_config, get_topics
)


def load_pdf_with_ocr(pdf_path, use_ocr=True):
    """使用 OCR 加载 PDF（支持数学公式识别），带缓存"""
    import pickle
    import hashlib
    
    # 生成缓存文件路径
    pdf_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()[:8]
    cache_file = f"./ocr_cache_{pdf_hash}.pkl"
    
    # 检查缓存
    if os.path.exists(cache_file):
        print(f"📦 发现 OCR 缓存文件: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            documents = [Document(page_content=d['content'], metadata=d['metadata']) 
                        for d in cached_data]
            print(f"✅ 从缓存加载 {len(documents)} 页")
            return documents
        except Exception as e:
            print(f"⚠️ 缓存读取失败: {e}，重新 OCR...")
    
    if not HAS_PYMUPDF:
        print("❌ PyMuPDF 未安装，无法进行 OCR")
        return None
    
    print(f"📖 使用 {'Pix2Text OCR' if use_ocr and HAS_PIX2TEXT else 'PyMuPDF'} 加载 PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    documents = []
    
    # 初始化 OCR
    p2t = None
    if use_ocr and HAS_PIX2TEXT:
        print("  🔄 初始化 Pix2Text OCR（首次运行需要下载模型）...")
        try:
            p2t = Pix2Text.from_config()
            print("  ✅ Pix2Text OCR 初始化成功")
        except Exception as e:
            print(f"  ⚠️ Pix2Text 初始化失败: {e}，回退到 PyMuPDF")
            p2t = None
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        if p2t:
            # 使用 OCR 提取（支持数学公式）
            try:
                # 将页面转换为图片
                pix = page.get_pixmap(dpi=200)
                img_path = f"temp_page_{page_num}.png"
                pix.save(img_path)
                
                # OCR 识别
                result = p2t.recognize(img_path, resized_shape=1200)
                text = result.to_markdown() if hasattr(result, 'to_markdown') else str(result)
                
                # 清理临时文件
                os.remove(img_path)
                
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num, "method": "ocr"}
                    ))
                    print(f"  📄 OCR 完成第 {page_num + 1} 页")
            except Exception as e:
                print(f"  ⚠️ 第 {page_num + 1} 页 OCR 失败: {e}")
                # 回退到 PyMuPDF
                text = page.get_text("text")
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num, "method": "pymupdf"}
                    ))
        else:
            # 使用 PyMuPDF 提取
            text = page.get_text("text")
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": page_num, "method": "pymupdf"}
                ))
    
    doc.close()
    
    # 保存缓存
    try:
        cache_data = [{'content': d.page_content, 'metadata': d.metadata} for d in documents]
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"💾 OCR 结果已缓存到: {cache_file}")
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")
    print(f"✅ 成功加载 {len(documents)} 页")
    return documents


def load_pdf_with_pymupdf(pdf_path):
    """使用 PyMuPDF 加载 PDF（对中文支持更好）"""
    if not HAS_PYMUPDF:
        return None
    
    print(f"📖 使用 PyMuPDF 加载 PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    documents = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # 提取纯文本
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num}
            ))
    
    doc.close()
    print(f"✅ PyMuPDF 成功加载 {len(documents)} 页")
    return documents


def get_embeddings():
    """获取 Embedding 模型（支持本地 HuggingFace 或 API）"""
    model_config = get_model_config()
    embedding_model = model_config.get("embedding_model", "local")
    
    if embedding_model == "local":
        # 设置 HuggingFace 镜像（解决国内网络问题）
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        from langchain_huggingface import HuggingFaceEmbeddings
        print("  📦 使用本地 HuggingFace Embedding 模型...")
        
        # 尝试多个模型，按顺序回退
        models_to_try = [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-zh-v1.5",
        ]
        
        for model_name in models_to_try:
            try:
                print(f"  🔄 尝试加载模型: {model_name}")
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                print(f"  ⚠️ 模型 {model_name} 加载失败: {str(e)[:80]}")
                continue
        
        raise RuntimeError("所有 Embedding 模型加载失败，请检查网络连接")
    else:
        # 使用 OpenAI 兼容接口
        api_config = get_api_config()
        return OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_config["api_key"],
            openai_api_base=api_config["base_url"],
        )


def get_llm(model_name=None):
    """获取 LLM 实例，支持指定模型名称"""
    api_config = get_api_config()
    model_config = get_model_config()
    
    # 如果指定了模型名称就使用，否则使用配置中的第一个模型
    if model_name is None:
        chat_models = model_config.get("chat_models", ["gemini-3-pro-preview"])
        model_name = chat_models[0] if isinstance(chat_models, list) else chat_models
    
    return ChatOpenAI(
        model=model_name,
        temperature=model_config.get("temperature", 0.3),
        openai_api_key=api_config["api_key"],
        openai_api_base=api_config["base_url"],
    )


def load_and_split_pdf(pdf_path, chunk_size=800, chunk_overlap=150, use_ocr=False):
    """加载 PDF 并切分文档（支持 OCR 模式）"""
    
    # 根据参数选择加载方式
    if use_ocr:
        documents = load_pdf_with_ocr(pdf_path, use_ocr=True)
    else:
        documents = load_pdf_with_pymupdf(pdf_path)
    
    if documents is None:
        # 回退到 PyPDFLoader
        print(f"📖 使用 PyPDFLoader 加载 PDF: {pdf_path}")
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
    """从文档中提取例题和习题（课后作业）- 改进版"""
    problems = []
    
    # 合并所有页面内容
    full_text = "\n".join([doc.page_content for doc in documents])
    
    # 例题匹配模式 - 使用更灵活的模式
    example_ids = set()
    # 匹配 【例题X.X】 格式
    for m in re.finditer(r'【例题[\s]*([0-9]+\.[0-9]+)】', full_text):
        example_ids.add(m.group(1))
    # 也尝试匹配 "例题 X.X" 格式（无方括号）
    for m in re.finditer(r'例题\s*([0-9]+\.[0-9]+)', full_text):
        example_ids.add(m.group(1))
    
    print(f"  📊 PDF中检测到 {len(example_ids)} 个例题编号")
    
    # 对每个例题编号提取内容
    for eid in sorted(example_ids, key=lambda x: [float(n) for n in x.split('.')]):
        # 查找该例题的内容（到下一个例题或章节结束）
        pattern = rf'【?例题[\s]*{re.escape(eid)}】?\s*(.+?)(?=【?例题|§\s*[0-9]+\.[0-9]+|$)'
        match = re.search(pattern, full_text, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            content = re.sub(r'\s+', ' ', content)[:2000]
            
            if len(content) > 10:
                problems.append({
                    'id': f"例题{eid}",
                    'content': content,
                    'type': 'example'
                })
            else:
                problems.append({
                    'id': f"例题{eid}",
                    'content': f"例题{eid}（PDF解析不完整，请参考原文）",
                    'type': 'example'
                })
        else:
            problems.append({
                'id': f"例题{eid}",
                'content': f"例题{eid}（PDF解析不完整，请参考原文）",
                'type': 'example'
            })
    
    seen_ids = set()
    
    # 提取课后作业/习题 - 使用改进的多模式匹配
    homework_structure = [
        ('0', 6, ['课后作业', '作业'], 8),
        ('1', 4, ['课后作业', '作业'], 3),
        ('2', 4, ['课后习题', '习题'], 7),
        ('3', 6, ['课后习题', '习题'], 14),
        ('4', 5, ['课后作业', '作业'], 10),
    ]
    
    for chapter, section, hw_names, num_problems in homework_structure:
        chapter_exercises_found = []
        
        # 尝试多种模式匹配课后作业部分
        section_content = None
        for hw_name in hw_names:
            # 模式1: "X.X 课后作业" 或 "X.X. 课后作业"
            patterns = [
                rf'{chapter}\.{section}\.?\s*{hw_name}\s*\n(.+?)(?=\n[0-9]+\.[0-9]+\s|\nChapter|\Z)',
                rf'{chapter}\.{section}\.?\s*{hw_name}(.+?)(?=\n[0-9]+\.[0-9]+\s|\Z)',
                rf'{hw_name}\s*\n(.+?)(?=\n[0-9]+\.[0-9]+\s|\Z)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
                if match and len(match.group(1)) > 50:
                    section_content = match.group(1)
                    break
            if section_content:
                break
        
        if section_content:
            # 尝试多种习题编号格式
            # 格式1: "1. 题目内容" 或 "1、题目内容"
            hw_problems = re.findall(r'(\d+)[\.、\s]\s*(.+?)(?=\n\s*\d+[\.、\s]|\Z)', section_content, re.DOTALL)
            
            for hw_num, hw_content in hw_problems:
                hw_num = int(hw_num)
                if hw_num > num_problems:  # 跳过超出范围的题号
                    continue
                    
                prob_id = f"{chapter}.{hw_num}"
                unique_key = f"hw_{prob_id}"
                
                if unique_key in seen_ids:
                    continue
                seen_ids.add(unique_key)
                
                hw_content = hw_content.strip()
                if len(hw_content) > 15:
                    hw_content = re.sub(r'\s+', ' ', hw_content)[:2000]
                    problems.append({
                        'id': f"习题{prob_id}",
                        'content': hw_content,
                        'type': 'exercise'
                    })
                    chapter_exercises_found.append(hw_num)
        
        # 补充缺失的题号
        found_count = len([p for p in problems if p['id'].startswith(f"习题{chapter}.")])
        if found_count < num_problems:
            print(f"  ⚠️ 第{chapter}章只找到 {found_count}/{num_problems} 道习题")
            for i in range(1, num_problems + 1):
                prob_id = f"{chapter}.{i}"
                unique_key = f"hw_{prob_id}"
                if unique_key not in seen_ids:
                    seen_ids.add(unique_key)
                    problems.append({
                        'id': f"习题{prob_id}",
                        'content': f"第{chapter}章第{i}题（PDF解析不完整，请参考原文）",
                        'type': 'exercise'
                    })
    
    # 按类型和 ID 排序
    def sort_key(p):
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
    
    prompt = f"""你是一位概率论与随机过程领域的资深数学教授。请详细解答以下{type_name}，并且不要有客套话，直接完成要求。

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
    
    # 获取模型优先级列表
    model_config = get_model_config()
    chat_models = model_config.get("chat_models", ["gemini-3-pro-preview"])
    if not isinstance(chat_models, list):
        chat_models = [chat_models]
    
    # 对每个模型尝试，失败则切换下一个
    for model_idx, model_name in enumerate(chat_models):
        llm = get_llm(model_name)
        
        # 每个模型重试2次
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = llm.invoke(prompt)
                if model_idx > 0:
                    print(f"    ✅ 使用备选模型 {model_name} 成功")
                return response.content
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"  ⚠️ 模型 {model_name} 失败 (重试 {attempt + 1}/{max_retries}): {error_msg[:80]}")
                    import time
                    time.sleep(wait_time)
                else:
                    if model_idx < len(chat_models) - 1:
                        print(f"  🔄 模型 {model_name} 失败，切换到 {chat_models[model_idx + 1]}...")
                    else:
                        print(f"  ❌ 所有模型均失败: {error_msg[:100]}")
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
    
    # 获取模型优先级列表
    model_config = get_model_config()
    chat_models = model_config.get("chat_models", ["gemini-3-pro-preview"])
    if not isinstance(chat_models, list):
        chat_models = [chat_models]
    
    # 对每个模型尝试，失败则切换下一个
    for model_idx, model_name in enumerate(chat_models):
        llm = get_llm(model_name)
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = llm.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"  ⚠️ 模型 {model_name} 失败 (重试 {attempt + 1}/{max_retries}): {error_msg[:80]}")
                    import time
                    time.sleep(wait_time)
                else:
                    if model_idx < len(chat_models) - 1:
                        print(f"  🔄 切换到 {chat_models[model_idx + 1]}...")
                    else:
                        print(f"  ❌ 知识生成最终失败: {error_msg[:100]}")
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
    import sys
    
    # 检查命令行参数
    USE_OCR = '--ocr' in sys.argv
    if USE_OCR:
        print("🔬 已启用 OCR 模式（支持数学公式识别）")
    
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
        # 步骤 1: 加载和切分文档（使用 OCR 或 PyMuPDF）
        splits, raw_documents = load_and_split_pdf(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, use_ocr=USE_OCR)
        
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
        
        # 读取现有题目索引文件中的手动补充内容
        index_file = f"{SOLUTIONS_DIR}/题目索引.md"
        existing_problems = {}
        if os.path.exists(index_file):
            print(f"\n📖 读取现有题目索引，保留手动补充内容...")
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # 解析现有题目内容
            import re as re_existing
            for match in re_existing.finditer(r'### (例题|习题)(\d+\.\d+)\n\n(.+?)(?=\n---|\Z)', content, re_existing.DOTALL):
                prob_type = match.group(1)
                prob_id = f"{prob_type}{match.group(2)}"
                prob_content = match.group(3).strip()
                # 只保留非"PDF解析不完整"的内容
                if 'PDF解析不完整' not in prob_content and '请参考原文' not in prob_content:
                    existing_problems[prob_id] = prob_content
            print(f"   ✅ 发现 {len(existing_problems)} 个手动补充的题目")
        
        # 合并：优先使用手动补充的内容
        for p in problems:
            if p['id'] in existing_problems:
                # 如果手动补充了完整内容，使用手动版本
                if len(existing_problems[p['id']]) > len(p['content']) or 'PDF解析不完整' in p['content']:
                    p['content'] = existing_problems[p['id']]
        
        # 生成题目索引文档
        print(f"\n📝 正在生成题目索引文档: {index_file}")
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("# 随机过程 - 题目索引\n\n")
            f.write(f"**生成时间**: {Path(PDF_PATH).stem}\n\n")
            f.write(f"**总计**: {len(problems)} 道题目 ({len(examples)} 例题 + {len(exercises)} 习题)\n\n")
            f.write("---\n\n")
            
            # 去重函数：保留内容最完整的版本
            def deduplicate_problems(prob_list):
                seen = {}
                for p in prob_list:
                    pid = p['id']
                    content = p['content']
                    # 如果已存在，比较哪个更完整
                    if pid in seen:
                        old_content = seen[pid]['content']
                        # 优先选择不包含"PDF解析不完整"的版本
                        if 'PDF解析不完整' in old_content and 'PDF解析不完整' not in content:
                            seen[pid] = p
                        # 或者选择更长的版本
                        elif 'PDF解析不完整' not in old_content and 'PDF解析不完整' not in content:
                            if len(content) > len(old_content):
                                seen[pid] = p
                    else:
                        seen[pid] = p
                return list(seen.values())
            
            # 去重
            examples_dedup = deduplicate_problems(examples)
            exercises_dedup = deduplicate_problems(exercises)
            
            # 写入例题（限制每题内容长度为300字符）
            if examples_dedup:
                f.write("## 例题列表\n\n")
                for p in sorted(examples_dedup, key=lambda x: [float(n) for n in re.findall(r'[\d.]+', x['id'])]):
                    content_preview = p['content'][:300] + "..." if len(p['content']) > 300 else p['content']
                    f.write(f"### {p['id']}\n\n")
                    f.write(f"{content_preview}\n\n")
                    f.write("---\n\n")
            
            # 写入习题
            if exercises_dedup:
                f.write("## 习题列表（课后作业）\n\n")
                for p in sorted(exercises_dedup, key=lambda x: [float(n) for n in re.findall(r'[\d.]+', x['id'])]):
                    content_preview = p['content'][:300] + "..." if len(p['content']) > 300 else p['content']
                    f.write(f"### {p['id']}\n\n")
                    f.write(f"{content_preview}\n\n")
                    f.write("---\n\n")
        
        print(f"✅ 题目索引已保存到: {index_file}")
        print(f"   (去重后: {len(examples_dedup)} 例题, {len(exercises_dedup)} 习题)")
        
        # 显示部分题目预览
        print("\n📋 题目预览 (前20个):")
        all_dedup = examples_dedup + exercises_dedup
        for i, p in enumerate(all_dedup[:20], 1):
            print(f"  {i}. {p['id']}: {p['content'][:40]}...")
        if len(all_dedup) > 20:
            print(f"  ... 还有 {len(all_dedup) - 20} 个题目 (完整列表见 {index_file})")
        
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
            # 检查已解答的题目
            solved_count = 0
            skipped_count = 0
            
            print(f"\n📝 正在使用 API 解答 {len(problems_to_solve)} 个题目...")
            
            # 用于收集每章的解答
            chapter_solutions = {}
            
            for i, prob in enumerate(problems_to_solve, 1):
                chapter = get_chapter(prob['id'])
                
                # 检查是否已有解答文件
                safe_id = prob['id'].replace('.', '_')
                solution_file = f"{SOLUTIONS_DIR}/{safe_id}.md"
                
                need_resolve = False
                existing_solution = None
                
                # 检查是否已有解答文件（例题和习题使用相同逻辑）
                if os.path.exists(solution_file):
                    with open(solution_file, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    
                    # 检查现有文件中的题目内容是否包含"PDF解析不完整"
                    if 'PDF解析不完整' in existing_content or '请参考原文' in existing_content:
                        # 检查新提取的题目内容是否已经完整了
                        if 'PDF解析不完整' not in prob['content'] and '请参考原文' not in prob['content']:
                            # 新内容完整，需要重新解答
                            need_resolve = True
                            print(f"  [{i}/{len(problems_to_solve)}] 🔄 重新解答 {prob['id']} (之前解析不完整，现已修复)")
                        else:
                            # 仍然解析不完整，跳过
                            skipped_count += 1
                            print(f"  [{i}/{len(problems_to_solve)}] ⚠️ 跳过 {prob['id']} (PDF解析仍不完整)")
                            
                            # 提取解答部分
                            solution_match = re.search(r'## 解答\n\n(.+?)(?=\n---|$)', existing_content, re.DOTALL)
                            if solution_match:
                                existing_solution = solution_match.group(1).strip()
                            else:
                                existing_solution = existing_content
                    else:
                        # 已有完整解答，跳过
                        skipped_count += 1
                        print(f"  [{i}/{len(problems_to_solve)}] ⏭️ 跳过 {prob['id']} (已有解答)")
                        
                        # 提取解答部分
                        solution_match = re.search(r'## 解答\n\n(.+?)(?=\n---|$)', existing_content, re.DOTALL)
                        if solution_match:
                            existing_solution = solution_match.group(1).strip()
                        else:
                            existing_solution = existing_content
                else:
                    # 没有解答文件，需要解答
                    need_resolve = True
                    print(f"  [{i}/{len(problems_to_solve)}] 🆕 正在解答 {prob['id']}...")
                
                if not need_resolve and existing_solution:
                    # 使用现有解答
                    if chapter not in chapter_solutions:
                        chapter_solutions[chapter] = []
                    chapter_solutions[chapter].append({
                        'id': prob['id'],
                        'content': prob['content'],
                        'solution': existing_solution,
                        'type': prob['type']
                    })
                    
                    # 添加到向量库
                    solved_docs.append(Document(
                        page_content=f"{prob['id']}：{prob['content']}\n\n解答：{existing_solution}",
                        metadata={
                            'type': 'solved_problem', 
                            'problem_id': prob['id'],
                            'problem_type': prob['type'],
                            'chapter': chapter
                        }
                    ))
                    continue
                
                # 需要调用 API 解答
                solution = None
                if need_resolve:
                    if not os.path.exists(solution_file):
                        print(f"  [{i}/{len(problems_to_solve)}] 🆕 正在解答 {prob['id']}...")
                    
                    solution = solve_problem_with_api(llm, prob['id'], prob['content'], prob['type'])
                else:
                    continue  # 应该不会到达这里，但作为保护
                
                if solution:
                    solved_count += 1
                    
                    # 立即保存到单独文件
                    with open(solution_file, 'w', encoding='utf-8') as f:
                        f.write(f"# {prob['id']}\n\n")
                        f.write(f"## 题目\n\n{prob['content']}\n\n")
                        f.write(f"## 解答\n\n{solution}\n\n")
                        f.write("---\n")
                    
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
                    print(f"    ✅ 已解答并保存到 {solution_file}")
            
            print(f"\n📊 解答统计: 新解答 {solved_count} 道, 跳过已有 {skipped_count} 道")
            
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
