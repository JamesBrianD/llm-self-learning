# 入门指南

欢迎开始使用我们的文档系统！

## 安装要求

在开始之前，请确保您的系统满足以下要求：

- Python 3.7+
- pip 包管理器

## 安装步骤

### 1. 安装 MkDocs Material

```bash
pip install mkdocs-material
```

### 2. 创建新项目

```bash
mkdocs new my-project
cd my-project
```

### 3. 配置主题

编辑 `mkdocs.yml` 文件：

```yaml
site_name: 我的项目
theme:
  name: material
```

### 4. 启动开发服务器

```bash
mkdocs serve
```

现在您可以在浏览器中访问 `http://127.0.0.1:8000` 查看您的文档了！

## 下一步

- [编写第一篇文档](../tutorials/first-document.md)
- [了解 Markdown 语法](../reference/markdown.md)