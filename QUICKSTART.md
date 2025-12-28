# 🚀 快速开始指南

本指南将帮助你在 5 分钟内启动 StochasticProcess-RAG 系统。

## 📋 前置要求

- Python 3.8 或更高版本
- 一个支持 OpenAI API 格式的 LLM 服务（如 OpenAI、Anthropic、或其他兼容服务）

## 🔧 安装步骤

### 1. 获取代码

```bash
git clone https://github.com/ClaudiaGardner/StochasticProcess-RAG.git
cd StochasticProcess-RAG
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 💡 建议使用虚拟环境：
> ```bash
> python -m venv venv
> # Windows
> venv\Scripts\activate
> # Linux/Mac
> source venv/bin/activate
> ```

### 3. 配置 API

```bash
# 复制配置模板
cp config-template.toml config.toml
```

然后编辑 `config.toml`：

```toml
[api]
provider = "openai"  # 或 "anthropic" 等
base_url = "https://api.openai.com/v1"  # 你的 API 地址
api_key = "sk-..."  # 你的 API 密钥

[model]
chat_models = ["gpt-4"]  # 你要使用的模型
embedding_model = "local"  # 保持 "local" 即可，无需额外费用
```

### 4. 准备数据

将你的随机过程教材 PDF 文件放入 `data/` 目录：

```bash
# 创建 data 目录（如果不存在）
mkdir data

# 将 PDF 复制到 data 目录
# 例如：copy your-textbook.pdf data/SP-10-12.pdf
```

然后在 `config.toml` 中指定 PDF 路径：

```toml
[ingestion]
pdf_path = "data/SP-10-12.pdf"  # 改为你的 PDF 文件名
```

### 5. 构建知识库

```bash
python ingest.py
```

这个过程会：
- ✅ 解析 PDF 文档
- ✅ 提取例题和习题
- ✅ 生成 AI 解答
- ✅ 构建向量数据库

> ⏱️ 首次运行可能需要 10-30 分钟，取决于 PDF 大小和题目数量

### 6. 开始使用！

```bash
python main.py
```

现在你可以开始提问了！

## 💬 使用示例

```
🙋 您的问题: 什么是马尔可夫链？

🔍 正在检索相关知识...

============================================================
🤖 回答:
------------------------------------------------------------
马尔可夫链是一种特殊的随机过程...
[详细解答]
============================================================
```

## 🎯 常用命令

```bash
# 启动问答系统
python main.py

# 重建知识库
python ingest.py

# 使用 OCR 模式（如果 PDF 质量不佳）
python ingest.py --ocr
```

## ⚠️ 常见问题

### Q: 提示 "向量数据库不存在"

**A:** 你需要先运行 `python ingest.py` 构建知识库

### Q: API 调用失败

**A:** 检查 `config.toml` 中的 API 配置是否正确：
- `base_url` 是否正确
- `api_key` 是否有效
- 模型名称是否正确

### Q: 内存不足

**A:** 在 `config.toml` 中减小 chunk_size：

```toml
[ingestion]
chunk_size = 600  # 从 800 减小到 600
```

### Q: PDF 解析效果不好

**A:** 尝试使用 OCR 模式：

```bash
pip install pix2text
python ingest.py --ocr
```

## 📚 下一步

- 阅读 [完整文档](README_CN.md)
- 查看 [配置说明](config-template.toml)
- 参与 [讨论](https://github.com/ClaudiaGardner/StochasticProcess-RAG/discussions)

## 🆘 需要帮助？

- 查看 [Issues](https://github.com/ClaudiaGardner/StochasticProcess-RAG/issues)
- 在 [Discussions](https://github.com/ClaudiaGardner/StochasticProcess-RAG/discussions) 提问

---

祝学习愉快！🎓
