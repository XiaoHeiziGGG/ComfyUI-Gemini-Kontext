import io
import json
import re
import os
import io
from typing import Optional, List, Tuple
import base64
from PIL import Image

# 导入ComfyUI相关模块
try:
    import folder_paths
    from comfy.utils import ProgressBar
except ImportError:
    pass

# 导入Google Gemini API
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("[Gemini-Kontext] 警告: 未找到google-generativeai库，请运行: pip install google-generativeai")

def log(message: str, message_type: str = 'info'):
    """日志函数"""
    print(f"[Gemini-Kontext] {message}")

def get_api_key(api_name: str) -> str:
    """获取API密钥"""
    # 查找api_key.ini文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_key_file = None
    
    # 在多个可能的位置查找api_key.ini
    search_paths = [
        os.path.join(current_dir, 'api_key.ini'),
        os.path.join(os.path.dirname(current_dir), 'api_key.ini'),
        os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'ComfyUI_LayerStyle_Advance', 'api_key.ini'),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            api_key_file = path
            break
    
    if not api_key_file:
        raise FileNotFoundError("找不到api_key.ini文件，请确保已正确配置API密钥")
    
    try:
        with open(api_key_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith(api_name + '='):
                    api_key = line.split('=', 1)[1].strip()
                    if api_key:
                        return api_key
        raise ValueError(f"在api_key.ini中找不到{api_name}的配置")
    except Exception as e:
        raise Exception(f"读取API密钥时出错: {str(e)}")

# Gemini安全设置
# 更宽松的安全设置（谨慎使用）
gemini_safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH", 
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH"
    }
]

class LS_GeminiTranslator:
    """
    专门用于翻译的Gemini节点，支持多种语言互译
    """
    
    CATEGORY = '🌐Gemini-Kontext'
    FUNCTION = "translate_text"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    OUTPUT_IS_LIST = (False,)

    def __init__(self):
        self.NODE_NAME = 'GeminiTranslator'

    @classmethod
    def INPUT_TYPES(cls):
        # Gemini模型列表
        gemini_model_list = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite", 
            "gemini-2.0-pro",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "learnlm-1.5-pro-experimental"
        ]
        
        # 支持的语言列表
        language_list = [
            "English",
            "中文 (Chinese)",
            "日本語 (Japanese)", 
            "한국어 (Korean)",
            "Français (French)",
            "Deutsch (German)",
            "Español (Spanish)",
            "Italiano (Italian)",
            "Português (Portuguese)",
            "Русский (Russian)",
            "العربية (Arabic)",
            "हिन्दी (Hindi)",
            "Auto Detect (自动检测)"
        ]
        
        target_language_list = [
            "English",
            "中文 (Chinese)",
            "日本語 (Japanese)", 
            "한국어 (Korean)",
            "Français (French)",
            "Deutsch (German)",
            "Español (Spanish)",
            "Italiano (Italian)",
            "Português (Portuguese)",
            "Русский (Russian)",
            "العربية (Arabic)",
            "हिन्दी (Hindi)"
        ]
        
        return {
            "required": {
                "model": (gemini_model_list, {"default": "gemini-2.5-flash"}),
                "source_language": (language_list, {"default": "Auto Detect (自动检测)"}),
                "target_language": (target_language_list, {"default": "English"}),
                "text_to_translate": ("STRING", {
                    "default": "请输入要翻译的文本",
                    "multiline": True
                }),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.1}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 100, "max": 8192, "step": 100}),
                "preserve_formatting": ("BOOLEAN", {"default": True}),
                "add_context": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "context_info": ("STRING", {
                    "default": "翻译上下文或专业领域信息（可选）",
                    "multiline": True
                }),
            }
        }

    def translate_text(self, model, source_language, target_language, text_to_translate, 
                      temperature, max_output_tokens, preserve_formatting, add_context, 
                      context_info=""):
        
        try:
            import google.generativeai as genai
        except ImportError:
            error_msg = "请安装google-generativeai库: pip install google-generativeai"
            log(error_msg, message_type='error')
            return (error_msg,)
        
        try:
            # 配置API
            api_key = get_api_key('google_api_key')
            genai.configure(api_key=api_key, transport='rest')
            
            # 创建模型实例
            g_model = genai.GenerativeModel(
                model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": 0.95,
                    "top_k": 40
                },
                safety_settings=gemini_safety_settings
            )
            
            # 构建翻译提示词
            source_lang = source_language if source_language != "Auto Detect (自动检测)" else "automatically detected language"
            
            system_prompt = f"""
你是一个专业的翻译专家。请将以下文本从{source_lang}翻译成{target_language}。

翻译要求：
1. 保持原文的意思和语调
2. 使用自然、流畅的表达
3. 如果是专业术语，请保持准确性
4. {'保持原文的格式和结构' if preserve_formatting else '可以调整格式以适应目标语言'}
5. 只返回翻译结果，不要包含任何解释或额外内容
"""
            
            if add_context and context_info.strip():
                system_prompt += f"\n\n上下文信息：{context_info}"
            
            # 构建完整的提示
            full_prompt = f"""{system_prompt}

要翻译的文本：
{text_to_translate}

翻译结果："""
            
            log(f"{self.NODE_NAME}: 正在使用 {model} 进行翻译...")
            log(f"{self.NODE_NAME}: 源语言: {source_language} -> 目标语言: {target_language}")
            
            # 发送请求
            response = g_model.generate_content(full_prompt)
            
            if response and response.text:
                translated_text = response.text.strip()
                log(f"{self.NODE_NAME}: 翻译完成")
                log(f"{self.NODE_NAME}: 原文: {text_to_translate[:100]}{'...' if len(text_to_translate) > 100 else ''}")
                log(f"{self.NODE_NAME}: 译文: {translated_text[:100]}{'...' if len(translated_text) > 100 else ''}")
                return (translated_text,)
            else:
                error_msg = "翻译失败：未收到有效响应"
                log(f"{self.NODE_NAME}: {error_msg}", message_type='error')
                return (error_msg,)
                
        except Exception as e:
            error_msg = f"翻译出错: {str(e)}"
            log(f"{self.NODE_NAME}: {error_msg}", message_type='error')
            return (error_msg,)

class LS_GeminiBatchTranslator:
    """
    批量翻译节点，支持多段文本同时翻译
    """
    
    CATEGORY = '🌐Gemini-Kontext'
    FUNCTION = "batch_translate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_texts",)
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.NODE_NAME = 'GeminiBatchTranslator'

    @classmethod
    def INPUT_TYPES(cls):
        # 在LS_GeminiBatchTranslator类中 (第255行)
        gemini_model_list = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-pro",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "learnlm-1.5-pro-experimental"
        ]
        
        language_list = [
            "English",
            "中文 (Chinese)",
            "日本語 (Japanese)", 
            "한국어 (Korean)",
            "Français (French)",
            "Deutsch (German)",
            "Español (Spanish)",
            "Auto Detect (自动检测)"
        ]
        
        return {
            "required": {
                "model": (gemini_model_list, {"default": "gemini-2.5-flash"}),
                "source_language": (language_list, {"default": "Auto Detect (自动检测)"}),
                "target_language": (language_list[:-1], {"default": "English"}),
                "texts_to_translate": ("STRING", {
                    "default": "文本1\n---\n文本2\n---\n文本3",
                    "multiline": True
                }),
                "separator": ("STRING", {"default": "---"}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.1}),
            }
        }

    def batch_translate(self, model, source_language, target_language, texts_to_translate, 
                       separator, temperature):
        
        try:
            import google.generativeai as genai
        except ImportError:
            error_msg = "请安装google-generativeai库: pip install google-generativeai"
            log(error_msg, message_type='error')
            return ([error_msg],)
        
        # 分割文本
        text_list = [text.strip() for text in texts_to_translate.split(separator) if text.strip()]
        
        if not text_list:
            return (["没有找到要翻译的文本"],)
        
        try:
            # 配置API
            api_key = get_api_key('google_api_key')
            genai.configure(api_key=api_key, transport='rest')
            
            g_model = genai.GenerativeModel(
                model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 4096,
                    "top_p": 0.95,
                    "top_k": 40
                },
                safety_settings=gemini_safety_settings
            )
            
            translated_texts = []
            source_lang = source_language if source_language != "Auto Detect (自动检测)" else "automatically detected language"
            
            log(f"{self.NODE_NAME}: 开始批量翻译 {len(text_list)} 段文本")
            
            for i, text in enumerate(text_list):
                try:
                    prompt = f"""
请将以下文本从{source_lang}翻译成{target_language}。
只返回翻译结果，不要包含任何解释：

{text}
"""
                    
                    response = g_model.generate_content(prompt)
                    
                    if response and response.text:
                        translated_text = response.text.strip()
                        translated_texts.append(translated_text)
                        log(f"{self.NODE_NAME}: 完成第 {i+1}/{len(text_list)} 段翻译")
                    else:
                        translated_texts.append(f"翻译失败: 第{i+1}段")
                        
                except Exception as e:
                    error_msg = f"第{i+1}段翻译出错: {str(e)}"
                    translated_texts.append(error_msg)
                    log(f"{self.NODE_NAME}: {error_msg}", message_type='error')
            
            log(f"{self.NODE_NAME}: 批量翻译完成")
            return (translated_texts,)
            
        except Exception as e:
            error_msg = f"批量翻译出错: {str(e)}"
            log(f"{self.NODE_NAME}: {error_msg}", message_type='error')
            return ([error_msg],)

class LS_GeminiImageAnalyzer:
    """
    Gemini图片识别和分析节点
    """
    
    CATEGORY = '🌐Gemini-Kontext'
    FUNCTION = "analyze_image"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_result",)
    OUTPUT_IS_LIST = (False,)

    def __init__(self):
        self.NODE_NAME = 'GeminiImageAnalyzer'

    @classmethod
    def INPUT_TYPES(cls):
        # 支持最新的Gemini模型
        gemini_model_list = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b"
        ]
        
        # 分析类型选项
        analysis_types = [
            "详细描述 (Detailed Description)",
            "简短描述 (Brief Description)", 
            "物体识别 (Object Detection)",
            "场景分析 (Scene Analysis)",
            "文字识别 (OCR)",
            "情感分析 (Emotion Analysis)",
            "艺术风格分析 (Art Style Analysis)",
            "自定义提示 (Custom Prompt)"
        ]
        
        # 输出语言选项
        output_languages = [
            "中文 (Chinese)",
            "English",
            "日本語 (Japanese)", 
            "한국어 (Korean)",
            "Français (French)",
            "Deutsch (German)",
            "Español (Spanish)"
        ]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (gemini_model_list, {"default": "gemini-2.5-flash"}),
                "analysis_type": (analysis_types, {"default": "详细描述 (Detailed Description)"}),
                "output_language": (output_languages, {"default": "中文 (Chinese)"}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.1}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 100, "max": 8192, "step": 100}),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "请描述这张图片",
                    "multiline": True
                }),
                "additional_instructions": ("STRING", {
                    "default": "额外的分析要求（可选）",
                    "multiline": True
                }),
            }
        }

    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        import torch
        import numpy as np
        
        # 确保tensor在CPU上
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        
        # 转换为numpy数组
        if isinstance(tensor, torch.Tensor):
            array = tensor.numpy()
        else:
            array = tensor
        
        # 处理批次维度
        if len(array.shape) == 4:
            array = array[0]  # 取第一张图片
        
        # 确保值在0-255范围内
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        
        # 转换为PIL图像
        if len(array.shape) == 3:
            if array.shape[2] == 3:  # RGB
                return Image.fromarray(array, 'RGB')
            elif array.shape[2] == 4:  # RGBA
                return Image.fromarray(array, 'RGBA')
        elif len(array.shape) == 2:  # 灰度图
            return Image.fromarray(array, 'L')
        
        raise ValueError(f"不支持的图像格式，shape: {array.shape}")

    def image_to_base64(self, pil_image):
        """将PIL图像转换为base64编码"""
        # 如果是RGBA，转换为RGB
        if pil_image.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[-1])  # 使用alpha通道作为mask
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 压缩图像以减少API调用大小
        max_size = (1024, 1024)
        if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 转换为base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_base64

    def get_analysis_prompt(self, analysis_type, output_language, custom_prompt="", additional_instructions=""):
        """根据分析类型生成提示词"""
        
        # 语言映射
        lang_map = {
            "中文 (Chinese)": "中文",
            "English": "English",
            "日本語 (Japanese)": "日本語", 
            "한국어 (Korean)": "한국어",
            "Français (French)": "Français",
            "Deutsch (German)": "Deutsch",
            "Español (Spanish)": "Español"
        }
        
        output_lang = lang_map.get(output_language, "中文")
        
        # 基础提示词模板
        base_prompts = {
            "详细描述 (Detailed Description)": f"请用{output_lang}详细描述这张图片，包括场景、物体、颜色、构图、氛围等各个方面。",
            "简短描述 (Brief Description)": f"请用{output_lang}简洁地描述这张图片的主要内容。",
            "物体识别 (Object Detection)": f"请用{output_lang}识别并列出图片中的所有物体和元素。",
            "场景分析 (Scene Analysis)": f"请用{output_lang}分析图片的场景类型、环境特征和整体氛围。",
            "文字识别 (OCR)": f"请用{output_lang}识别并提取图片中的所有文字内容。",
            "情感分析 (Emotion Analysis)": f"请用{output_lang}分析图片传达的情感、氛围和感受。",
            "艺术风格分析 (Art Style Analysis)": f"请用{output_lang}分析图片的艺术风格、技法和美学特征。",
            "自定义提示 (Custom Prompt)": custom_prompt if custom_prompt.strip() else f"请用{output_lang}描述这张图片。"
        }
        
        prompt = base_prompts.get(analysis_type, base_prompts["详细描述 (Detailed Description)"])
        
        if additional_instructions.strip():
            prompt += f"\n\n额外要求：{additional_instructions}"
        
        return prompt

    def analyze_image(self, image, model, analysis_type, output_language, temperature, 
                     max_output_tokens, custom_prompt="", additional_instructions=""):
        
        try:
            import google.generativeai as genai
        except ImportError:
            error_msg = "请安装google-generativeai库: pip install google-generativeai"
            log(error_msg, message_type='error')
            return (error_msg,)
        
        try:
            # 配置API
            api__key = get_api_key('google_api_key')
            genai.configure(api_key=api_key, transport='rest')
            
            # 创建模型实例
            g_model = genai.GenerativeModel(
                model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": 0.95,
                    "top_k": 40
                },
                safety_settings=gemini_safety_settings
            )
            
            # 转换图像
            pil_image = self.tensor_to_pil(image)
            
            # 生成分析提示词
            prompt = self.get_analysis_prompt(analysis_type, output_language, custom_prompt, additional_instructions)
            
            log(f"{self.NODE_NAME}: 正在使用 {model} 分析图片...")
            log(f"{self.NODE_NAME}: 分析类型: {analysis_type}")
            log(f"{self.NODE_NAME}: 输出语言: {output_language}")
            
            # 发送请求
            response = g_model.generate_content([prompt, pil_image])
            
            if response and response.text:
                analysis_result = response.text.strip()
                log(f"{self.NODE_NAME}: 图片分析完成")
                log(f"{self.NODE_NAME}: 分析结果: {analysis_result[:100]}{'...' if len(analysis_result) > 100 else ''}")
                return (analysis_result,)
            else:
                error_msg = "图片分析失败：未收到有效响应"
                log(f"{self.NODE_NAME}: {error_msg}", message_type='error')
                return (error_msg,)
                
        except Exception as e:
            error_msg = f"图片分析出错: {str(e)}"
            log(f"{self.NODE_NAME}: {error_msg}", message_type='error')
            return (error_msg,)

class LS_GeminiKontextOptimizer:
    """
    支持多语言输入，输出符合Kontext格式的英文提示词
    """
    
    CATEGORY = '🌐Gemini-Kontext'  # 修改这一行
    FUNCTION = "optimize_kontext_prompt"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("optimized_prompt", "analysis_details")
    OUTPUT_IS_LIST = (False, False)

    def __init__(self):
        self.NODE_NAME = 'GeminiKontextOptimizer'

    @classmethod
    def INPUT_TYPES(cls):
        # Gemini模型列表
        gemini_model_list = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-pro",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "learnlm-1.5-pro-experimental"
        ]
        
        # 优化级别选项
        optimization_levels = [
            "Basic (基础优化)",
            "Advanced (高级优化)", 
            "Professional (专业优化)"
        ]
        
        # 编辑类型
        edit_types = [
            "Object Modification (物体修改)",
            "Background Change (背景更换)",
            "Style Transfer (风格转换)",
            "Character Consistency (角色一致性)",
            "Text Editing (文本编辑)",
            "Complex Transformation (复杂变换)",
            "Auto Detect (自动检测)"
        ]
        
        # 执行后操作选项
        post_execution_options = [
            "None (无操作)",
            "Fixed Seed (固定种子)",
            "Random Seed (随机种子)"
        ]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "user_description": ("STRING", {
                    "default": "我希望该人物不做任何变化，背景变为沙滩",
                    "multiline": True
                }),
                "model": (gemini_model_list, {"default": "gemini-2.5-flash"}),
                "edit_type": (edit_types, {"default": "Auto Detect (自动检测)"}),
                "optimization_level": (optimization_levels, {"default": "Advanced (高级优化)"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.1}),
                "max_output_tokens": ("INT", {"default": 1024, "min": 100, "max": 4096, "step": 100}),
                "post_execution": (post_execution_options, {"default": "None (无操作)"}),
                "stop_on_failure": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "additional_context": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "可选：提供额外的上下文信息"
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }

    def get_kontext_system_prompt(self, optimization_level: str, edit_type: str) -> str:
        """
        生成Kontext优化的系统提示词，让Gemini自己读取官方文档
        """
        base_prompt = """
You are an expert Kontext image editing prompt optimizer. 

IMPORTANT: First, please read and understand the official Kontext documentation at this link:
https://docs.bfl.ai/guides/prompting_guide_kontext_i2i#prompt-precision%3A-from-basic-to-comprehensive

After reading the official documentation, follow these requirements:

1. **Analyze the provided image** to understand:
   - Main subjects, characters, objects
   - Background and environment
   - Style, lighting, colors
   - Any text elements

2. **Transform user descriptions** into optimal Kontext prompts based on the official guidelines you just read

3. **Output requirements**:
   - Generate ONLY the optimized prompt in English
   - NO explanations, NO additional text
   - Follow the exact patterns from the official documentation
   - Be specific and detailed but concise
   - Focus on the exact change requested

4. **Apply official best practices** from the documentation for:
   - Object modifications
   - Style transfers
   - Character consistency
   - Background changes
   - Text editing
   - Complex transformations

Remember: Read the official documentation first, then apply those guidelines to optimize the user's request.
"""
        
        return base_prompt

    def optimize_kontext_prompt(self, image, user_description, model, edit_type, optimization_level, temperature, max_output_tokens, post_execution, stop_on_failure, additional_context="", seed=-1):
        """
        优化Kontext提示词的主要方法
        """
        import random
        
        try:
            # 处理种子设置
            if post_execution == "Random Seed (随机种子)":
                current_seed = random.randint(0, 0xffffffffffffffff)
            elif post_execution == "Fixed Seed (固定种子)" and seed != -1:
                current_seed = seed
            else:
                current_seed = -1  # 默认值
            
            # 获取API密钥并配置Gemini
            api_key = get_api_key('google_api_key')
            genai.configure(api_key=api_key)
            
            # 创建模型实例
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_output_tokens,
            }
            
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=gemini_safety_settings
            )
            
            # 转换图像
            pil_image = self.tensor_to_pil(image)
            image_base64 = self.image_to_base64(pil_image)
            
            # 分析图像
            log("正在分析图像内容...")
            image_analysis = self.analyze_image_for_kontext(image_base64, model_instance)
            
            # 构建优化提示词
            system_prompt = self.get_kontext_system_prompt(optimization_level, edit_type)
            
            optimization_prompt = f"""
{system_prompt}

IMAGE ANALYSIS:
{image_analysis}

USER REQUEST: {user_description}

ADDITIONAL CONTEXT: {additional_context if additional_context else "None"}

TASK: 
1. First read the official Kontext documentation at: https://docs.bfl.ai/guides/prompting_guide_kontext_i2i#prompt-precision%3A-from-basic-to-comprehensive
2. Based on the documentation and image analysis, generate the optimal Kontext prompt
3. Output ONLY the prompt, no explanations

OUTPUT:
"""
            
            # 发送请求
            log("正在让Gemini读取官方文档并优化提示词...")
            response = model_instance.generate_content([
                optimization_prompt,
                {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            ])
            
            # 处理结果
            try:
                result_text = response.text.strip()
                
                # 清理输出，只保留提示词
                if "OUTPUT:" in result_text:
                    result_text = result_text.split("OUTPUT:")[-1].strip()
                
                # 移除多余的格式化
                lines = result_text.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('Optimized Prompt:') and not line.startswith('**'):
                        clean_lines.append(line)
                
                result_text = ' '.join(clean_lines).strip()
                
                log("Kontext提示词优化完成")
                return (result_text, f"优化完成 - 模型: {model} (已读取官方文档)")
                
            except Exception as e:
                error_msg = f"Kontext提示词优化失败: {str(e)}"
                log(error_msg, "error")
                if stop_on_failure:
                    return (f"错误: {str(e)}", error_msg)
                else:
                    return (f"优化失败: {str(e)}", error_msg)
        
        except Exception as e:
            error_msg = f"Kontext提示词优化出错: {str(e)}"
            log(error_msg, "error")
            return (f"错误: {str(e)}", error_msg)

    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        import torch
        import numpy as np
        
        # 确保tensor在CPU上
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        
        # 转换为numpy数组
        if len(tensor.shape) == 4:
            # 批次维度，取第一个
            tensor = tensor[0]
        
        # 确保值在0-1范围内
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        
        # 转换为uint8
        tensor = tensor.clamp(0, 255).byte()
        
        # 转换为numpy数组
        if hasattr(tensor, 'numpy'):
            array = tensor.numpy()
        else:
            array = np.array(tensor)
        
        # 确保是HWC格式
        if len(array.shape) == 3 and array.shape[0] in [1, 3, 4]:
            array = np.transpose(array, (1, 2, 0))
        
        # 转换为PIL图像
        if array.shape[2] == 1:
            array = array.squeeze(2)
            pil_image = Image.fromarray(array, mode='L')
        elif array.shape[2] == 3:
            pil_image = Image.fromarray(array, mode='RGB')
        elif array.shape[2] == 4:
            pil_image = Image.fromarray(array, mode='RGBA')
        else:
            raise ValueError(f"不支持的图像通道数: {array.shape[2]}")
        
        return pil_image
    
    def image_to_base64(self, pil_image):
        """将PIL图像转换为base64编码"""
        import io
        
        # 转换为RGB模式（如果不是的话）
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 保存到字节流
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        
        # 转换为base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_base64

    def analyze_image_for_kontext(self, image_base64, model_instance):
        """为Kontext优化专门分析图像内容"""
        try:
            analysis_prompt = """
请分析这张图片，重点关注以下方面：
1. 主要对象和人物
2. 背景环境和场景
3. 艺术风格和视觉特征
4. 颜色、光照和构图
5. 任何文字元素

请用简洁的英文描述，为后续的Kontext编辑提供准确的上下文信息。
"""
            
            response = model_instance.generate_content([
                analysis_prompt,
                {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            ])
            
            if response and response.text:
                return response.text.strip()
            else:
                return "Unable to analyze image content"
                
        except Exception as e:
            log(f"图像分析失败: {str(e)}", "error")
            return "Image analysis failed"

# 节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiTranslator": LS_GeminiTranslator,
    "GeminiBatchTranslator": LS_GeminiBatchTranslator,
    "GeminiImageAnalyzer": LS_GeminiImageAnalyzer,
    "GeminiKontextOptimizer": LS_GeminiKontextOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiTranslator": "🌐 Gemini Translator",
    "GeminiBatchTranslator": "🌐 Gemini Batch Translator", 
    "GeminiImageAnalyzer": "🖼️ Gemini Image Analyzer",
    "GeminiKontextOptimizer": "✨ Gemini Kontext Optimizer",
}
