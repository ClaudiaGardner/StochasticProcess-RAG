# 📚 StochasticProcess-RAG

English | [简体中文](README_CN.md)

> 基于 RAG（检索增强生成）技术的随机过程智能问答系统

一个专为概率论与随机过程课程设计的 AI 学习助手，通过向量数据库检索和大语言模型生成，为学生提供精准、详细的数学问题解答。

## ✨ 核心特性

- 🔍 **智能检索**：基于 Chroma 向量数据库，快速定位相关知识点
- 🤖 **AI 解答**：支持多种大语言模型（Claude、GPT 等），提供详细的解题步骤
- 📖 **知识库构建**：自动解析 PDF 教材，提取例题和习题
- 📐 **LaTeX 公式支持**：完美渲染数学公式，支持 Markdown 输出
- 💾 **答案归档**：自动保存问答记录，方便复习回顾
- 🔄 **OCR 支持**：可选的数学公式 OCR 识别（Pix2Text）

## 🚀 快速开始

### 环境要求

- Python 3.11+ (推荐)
- NVIDIA GPU (可选，用于加速 embedding)

### 安装步骤

1. **克隆项目并创建 Conda 环境**

```bash
git clone https://github.com/ClaudiaGardner/StochasticProcess-RAG.git
cd StochasticProcess-RAG

# 创建 Conda 环境（推荐，避免依赖冲突）
conda create -n stochastic-rag python=3.11 -y
conda activate stochastic-rag
```

2. **安装 PyTorch（如有 GPU）**

```bash
# GPU 版本（CUDA 11.8，加速 5-10 倍）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

> 💡 无 GPU 可跳过此步，系统会自动使用 CPU

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **配置 API**

复制配置模板并填写你的 API 信息：

```bash
cp config-template.toml config.toml
```

编辑 `config.toml`，填写以下信息：

```toml
[api]
provider = "your-provider"  # 如 "openai", "anthropic" 等
base_url = "https://api.example.com/v1"
api_key = "your-api-key-here"

[model]
chat_models = ["claude-sonnet-4-5-20250929"]  # 你的模型列表
embedding_model = "local"  # 使用本地 embedding 模型
temperature = 0.3

[database]
chroma_dir = "./chroma_db"
solutions_dir = "./solutions"

[ingestion]
pdf_path = "data/SP-10-12.pdf"  # 你的教材 PDF 路径
chunk_size = 800
chunk_overlap = 150
max_problems_to_solve = 0  # 0 表示解答所有题目
```

> ⚠️ **注意**：`config.toml` 包含敏感信息，已被 `.gitignore` 排除，不会上传到 GitHub

5. **准备数据**

将你的随机过程教材 PDF 放入 `data/` 目录，并在 `config.toml` 中指定路径。

6. **构建知识库**

```bash
# 基础模式（使用 PyMuPDF）
python ingest.py

# OCR 模式（支持数学公式识别，需要额外安装 pix2text）
python ingest.py --ocr
```

这一步会：
- 解析 PDF 文档
- 提取例题和习题
- 使用 AI 生成详细解答
- 构建向量数据库
- 生成补充知识点

7. **开始问答**

```bash
python main.py
```

## 📖 使用示例

启动系统后，你可以提出各种问题：

```
🙋 您的问题: 什么是马尔可夫链？

🔍 正在检索相关知识...

============================================================
🤖 回答:
------------------------------------------------------------
马尔可夫链是一种特殊的随机过程，具有无记忆性（Markov性质）...

[详细解答内容]
============================================================

💾 回答已自动保存到: ./answers/20250101_120000_什么是马尔可夫链.md
📄 可以使用 Markdown 查看器打开文件，数学公式将正确显示
```

### 示例问题

- 什么是马尔可夫链？
- 泊松过程有什么性质？
- 解释常返态和瞬时态的区别
- 如何计算转移概率矩阵？
- 随机游走的期望值是多少？

## 📁 项目结构

```
StochasticProcess-RAG/
├── main.py                 # 主程序 - 交互式问答
├── ingest.py              # 数据摄入 - PDF 解析与向量化
├── config_manager.py      # 配置管理
├── llm_config.py          # LLM 配置辅助
├── config.toml            # 配置文件（不上传到 Git）
├── config-template.toml   # 配置模板
├── requirements.txt       # Python 依赖
├── .gitignore            # Git 忽略规则
├── .env.example          # 环境变量示例
├── data/                 # 教材 PDF 存放目录
├── chroma_db/            # 向量数据库（自动生成）
├── solutions/            # AI 生成的题目解答
└── answers/              # 问答记录归档
```

## 🛠️ 技术栈

- **LangChain**：LLM 应用框架
- **Chroma**：向量数据库
- **HuggingFace**：本地 Embedding 模型
- **PyMuPDF / Pix2Text**：PDF 解析与 OCR
- **OpenAI API**：兼容多种 LLM 提供商

## 🔧 高级配置

### 使用 OCR 模式

如果你的 PDF 包含复杂的数学公式，推荐使用 OCR 模式：

```bash
# 安装 OCR 依赖
pip install pix2text

# 使用 OCR 模式构建知识库
python ingest.py --ocr
```

### 🏠 离线模式 (Offline Mode)

支持完全离线运行，无需云端 API：

**方式一：命令行参数**
```bash
# 离线构建知识库（使用已有解答）
python ingest.py --offline

# 离线问答（使用本地 Ollama）
python main.py --offline
```

**方式二：配置文件**
```toml
[model]
offline_mode = true
local_llm_url = "http://localhost:11434"
local_llm_model = "qwen2.5:7b"
```

**前提条件**：
1. 安装 [Ollama](https://ollama.ai/)：`ollama pull qwen2.5:7b`
2. 启动服务：`ollama serve`


### 自定义检索参数

在 `config.toml` 中调整检索参数：

```toml
[retrieval]
top_k = 5  # 每次检索返回的文档数量（增加可获得更多上下文）
```

### 添加补充知识点

在 `config.toml` 中添加你想要系统生成的知识点：

```toml
[topics]
core_topics = [
    "马尔可夫链的定义和转移概率矩阵",
    "泊松过程及其无记忆性",
    "常返态与瞬时态的定义和判定",
    # 添加更多主题...
]
```

## 📝 答案格式

系统生成的答案包含：

- **题目分析**：识别核心概念和求解目标
- **解题过程**：详细的推导步骤（LaTeX 格式）
- **最终答案**：数学公式表达
- **知识延伸**：相关定理、公式和解题技巧
- **参考来源**：检索到的原文片段

所有答案自动保存为 Markdown 文件，支持任何 Markdown 阅读器查看。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## ⚠️ 免责声明

- 本项目仅供学习交流使用
- AI 生成的答案仅供参考，请结合教材和课堂内容学习
- 使用本系统时请遵守相关 API 服务商的使用条款

## 🙏 致谢

- LangChain 团队提供的优秀框架
- Chroma 向量数据库
- HuggingFace 社区的开源模型
- 所有贡献者和使用者

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 [GitHub Issue](https://github.com/ClaudiaGardner/StochasticProcess-RAG/issues)
- 发起 [Discussion](https://github.com/ClaudiaGardner/StochasticProcess-RAG/discussions)

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
