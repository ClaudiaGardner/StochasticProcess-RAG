# 📚 随机过程 RAG 问答系统

[English](README.md) | 简体中文

> 一个基于检索增强生成（RAG）技术的智能学习助手，专为随机过程课程设计

## 🎯 项目简介

本项目是一个专门针对概率论与随机过程课程的 AI 问答系统。它能够：

- 📖 自动解析教材 PDF，提取例题和习题
- 🤖 使用大语言模型生成详细的解题步骤
- 🔍 通过向量检索快速定位相关知识点
- 💾 保存所有问答记录，方便复习

## ✨ 主要功能

### 1. 智能问答
- 输入任何关于随机过程的问题
- 系统自动检索相关知识
- AI 生成详细解答，包含完整的数学推导

### 2. 题目解答
- 自动识别教材中的例题和习题
- 生成规范的解题步骤
- 支持 LaTeX 数学公式

### 3. 知识库管理
- 向量化存储教材内容
- 支持增量更新
- 高效的相似度检索

## 🚀 快速开始

### 第一步：创建环境并安装依赖

**推荐使用 Conda 环境**（避免依赖冲突）：

```bash
# 克隆项目
git clone https://github.com/ClaudiaGardner/StochasticProcess-RAG.git
cd StochasticProcess-RAG

# 创建 Conda 环境
conda create -n stochastic-rag python=3.11 -y
conda activate stochastic-rag

# 安装 PyTorch GPU 版本（推荐，加速 5-10 倍）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt

# 验证 GPU 是否可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

> 💡 如果没有 NVIDIA GPU，可以跳过 PyTorch GPU 安装步骤，系统会自动使用 CPU

### 第二步：配置 API

1. 复制配置模板：
```bash
cp config-template.toml config.toml
```

2. 编辑 `config.toml`，填写你的 API 信息：

```toml
[api]
provider = "你的提供商"
base_url = "https://api.example.com/v1"
api_key = "你的API密钥"

[model]
chat_models = ["claude-sonnet-4-5-20250929"]
embedding_model = "local"  # 使用本地模型，无需额外费用
```

> 💡 **提示**：`config.toml` 已被 `.gitignore` 排除，不会上传到 GitHub，你的 API 密钥是安全的

### 第三步：准备数据

将你的随机过程教材 PDF 放入 `data/` 目录。

### 第四步：构建知识库

```bash
# 基础模式（推荐）
python ingest.py

# OCR 模式（如果 PDF 中数学公式较多）
python ingest.py --ocr
```

这个过程会：
1. 解析 PDF 文档
2. 提取所有例题和习题
3. 使用 AI 生成详细解答
4. 构建向量数据库
5. 生成补充知识点

### 第五步：开始使用

```bash
python main.py
```

然后就可以开始提问了！

## 💡 使用示例

```
🙋 您的问题: 什么是马尔可夫链的转移概率矩阵？

🔍 正在检索相关知识...

============================================================
🤖 回答:
------------------------------------------------------------
马尔可夫链的转移概率矩阵是描述状态转移规律的核心工具...

转移概率矩阵 $P$ 的定义为：
$$P = (p_{ij})_{i,j \in S}$$

其中 $p_{ij} = P(X_{n+1}=j | X_n=i)$ 表示从状态 $i$ 转移到状态 $j$ 的概率。

[详细解答...]
============================================================

💾 回答已保存到: ./answers/20250101_120000_什么是马尔可夫链的转移概率矩阵.md
```

## 📁 项目结构说明

```
StochasticProcess-RAG/
├── main.py                 # 主程序：交互式问答界面
├── ingest.py              # 数据处理：PDF 解析和向量化
├── config_manager.py      # 配置管理工具
├── config.toml            # 配置文件（包含 API 密钥，不上传）
├── config-template.toml   # 配置模板（可以上传）
├── requirements.txt       # Python 依赖列表
├── data/                  # 存放教材 PDF
├── chroma_db/            # 向量数据库（自动生成）
├── solutions/            # AI 生成的题目解答
└── answers/              # 问答记录存档
```

## 🔧 高级配置

### 调整检索数量

在 `config.toml` 中修改：

```toml
[retrieval]
top_k = 5  # 增加这个值可以获得更多上下文
```

### 添加自定义知识点

```toml
[topics]
core_topics = [
    "马尔可夫链的定义和转移概率矩阵",
    "泊松过程及其无记忆性",
    "你想要添加的知识点...",
]
```

### 使用 OCR 模式

如果教材中的数学公式较多或扫描质量不佳：

```bash
# 安装 OCR 依赖
pip install pix2text

# 使用 OCR 模式
python ingest.py --ocr
```

### 🏠 离线模式（不需要联网 API）

项目支持完全离线运行，无需调用云端 API：

**方式一：使用命令行参数**
```bash
# 离线构建知识库（使用已有解答，跳过 API 生成）
python ingest.py --offline

# 离线问答（使用本地 Ollama 模型）
python main.py --offline
```

**方式二：修改配置文件**
```toml
[model]
offline_mode = true  # 启用离线模式
local_llm_url = "http://localhost:11434"  # Ollama 服务地址
local_llm_model = "qwen2.5:7b"  # 本地模型名称
```

**离线模式前提**：
1. 安装 [Ollama](https://ollama.ai/) 并下载模型：`ollama pull qwen2.5:7b`
2. 确保 Ollama 服务运行：`ollama serve`
3. 首次运行 `ingest.py` 时需要联网下载 embedding 模型（之后会缓存）


## 🛡️ 隐私与安全

- ✅ `config.toml` 已被 `.gitignore` 排除，不会上传到 GitHub
- ✅ 你的 API 密钥完全保存在本地
- ✅ 所有数据处理都在本地进行
- ⚠️ 请不要将 `config.toml` 分享给他人

## 📝 常见问题

### Q: 如何更换 LLM 模型？

A: 在 `config.toml` 中修改 `chat_models` 列表：

```toml
[model]
chat_models = ["claude-sonnet-4-5-20250929", "gpt-4"]  # 按优先级排列
```

### Q: 向量数据库占用空间太大怎么办？

A: 可以调整分块大小：

```toml
[ingestion]
chunk_size = 600  # 减小这个值
chunk_overlap = 100
```

### Q: 如何重建知识库？

A: 删除 `chroma_db/` 目录后重新运行 `python ingest.py`

### Q: 支持哪些 PDF 格式？

A: 支持文本型 PDF 和扫描型 PDF（需要 OCR 模式）

### Q: 本地 embedding 模型会使用 GPU 吗？

A: **会自动检测并使用 GPU**（如果可用）。系统会自动检测你的 GPU：
- ✅ 如果检测到 NVIDIA GPU（CUDA），会自动使用 GPU 加速
- ✅ 如果没有 GPU，会自动回退到 CPU

运行时会显示使用的设备：
```
📦 使用本地 HuggingFace Embedding 模型 (设备: cuda)...
```

**GPU 加速的优势**：
- 🚀 向量化速度提升 5-10 倍
- ⚡ 问答响应更快
- 💪 适合处理大量文档

**确保 GPU 可用**：
```bash
# 检查是否安装了 PyTorch GPU 版本
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 如果显示 False，需要安装 GPU 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

如果你有好的想法或发现了 bug，请：
1. Fork 本项目
2. 创建特性分支
3. 提交你的更改
4. 发起 Pull Request

## 📄 开源协议

本项目采用 MIT 协议开源 - 详见 [LICENSE](LICENSE) 文件

## ⚠️ 免责声明

- 本项目仅供学习交流使用
- AI 生成的答案仅供参考，请以教材和老师讲解为准
- 请遵守你使用的 API 服务的使用条款

## 🙏 鸣谢

感谢以下开源项目：
- [LangChain](https://github.com/langchain-ai/langchain)
- [Chroma](https://github.com/chroma-core/chroma)
- [HuggingFace](https://huggingface.co/)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)

## 📮 联系方式

- 提交 [GitHub Issue](https://github.com/ClaudiaGardner/StochasticProcess-RAG/issues)
- 参与 [讨论区](https://github.com/ClaudiaGardner/StochasticProcess-RAG/discussions)

---

**⭐ 觉得有用？给个 Star 吧！**
