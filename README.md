# ComfyUI-Gemini-Kontext

[English](#english) | [中文](#中文)

## 中文

### 简介

ComfyUI-Gemini-Kontext 是一个专为 ComfyUI 设计的 Google Gemini API 集成插件，主要提供 **Kontext 提示词优化** 和 **多语言翻译** 功能。

### 主要功能

#### 🎯 Kontext 提示词优化器
- **智能提示词优化**：基于 Gemini 模型和官方 Kontext 文档，自动优化您的图像生成提示词
- **图像分析增强**：结合图像内容分析，生成更精准的 Kontext 提示词
- **官方文档集成**：自动读取 Black Forest Labs 官方 Kontext 提示词指南
- **多种优化级别**：支持基础、标准、高级三种优化强度
- **灵活配置**：可自定义用户描述、附加上下文和执行后操作

#### 🌐 多语言翻译功能
- **文本翻译**：支持 50+ 种语言的高质量翻译
- **批量翻译**：一次性翻译多段文本，提高工作效率
- **图像分析**：智能分析图像内容并生成多语言描述
- **简单易用**：即使是简单的 Kontext 项目也可以直接使用 API 翻译功能

### 模型建议

- **推荐使用**：Gemini 2.0 Flash 模型（最新且性能优异）
- **备选方案**：Gemini 1.5 Pro/Flash 模型
- **注意事项**：Gemini 2.5 为实验性模型，请根据需要自行选择使用

### 安装方法

1. 将插件下载到 ComfyUI 的 `custom_nodes` 目录
2. 安装依赖：
   ```bash
   pip install google-generativeai>=0.3.0```