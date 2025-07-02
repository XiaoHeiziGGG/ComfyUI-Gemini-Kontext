"""
ComfyUI Gemini Translator Node Package

一个专门用于文本翻译的ComfyUI节点包，基于Google Gemini API。
支持多种语言互译，包括单文本翻译和批量翻译功能。
"""

from .gemini_translator import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.0.0"
__author__ = "ComfyUI Community"
__description__ = "Google Gemini API powered translation nodes for ComfyUI"

# 导出节点映射
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 版本信息
WEB_DIRECTORY = "./web"