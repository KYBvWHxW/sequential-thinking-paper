from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai import AsyncOpenAI, OpenAIError
import os
import time
import uuid
import json
import asyncio
from dotenv import load_dotenv
from .utils.logger import logger, with_logging

load_dotenv()

from pydantic import Field, validator

class AnalysisRequest(BaseModel):
    content: str = Field(..., min_length=1)
    max_segments: int = Field(5, gt=0, le=10)
    analysis_type: str = Field("article", pattern='^(article|code|conversation)$')
    additional_instructions: Optional[str] = None

    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = {'article', 'code', 'conversation'}
        if v not in valid_types:
            raise ValueError(f'Analysis type must be one of: {valid_types}')
        return v

class AnalysisResponse(BaseModel):
    segments: List[Dict]
    metadata: Dict

app = FastAPI()

class SequentialThinkingServer:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = AsyncOpenAI(api_key=self.api_key)

    @with_logging
    async def analyze_content(self, request: AnalysisRequest) -> AnalysisResponse:
        try:
            # 构建提示词
            system_prompt = self._get_system_prompt(request.analysis_type)
            user_prompt = self._build_user_prompt(request)

            logger.debug(
                "Prompts generated",
                event="prompts_generated",
                details={
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt
                }
            )

            # 调用 OpenAI API
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            logger.debug(
                "OpenAI response received",
                event="openai_response_received",
                details={
                    "model": response.model,
                    "total_tokens": response.usage.total_tokens,
                    "response_content": str(response)
                }
            )

            # 解析响应
            analysis_result = self._parse_response(response.choices[0].message)
            logger.debug(
                "Analysis result processed",
                event="analysis_result",
                details={"result": analysis_result}
            )

            # 构建响应
            response_obj = AnalysisResponse(
                segments=analysis_result["segments"],
                metadata={
                    "analysis_type": request.analysis_type,
                    "total_segments": len(analysis_result["segments"]),
                    "model": response.model,
                    "total_tokens": response.usage.total_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens
                }
            )
            logger.info(
                "Request processed successfully",
                event="request_success",
                details={
                    "segments_count": len(analysis_result["segments"]),
                    "total_tokens": response.usage.total_tokens
                }
            )
            return response_obj

        except OpenAIError as api_error:
            logger.error(
                "OpenAI API error occurred",
                event="openai_api_error",
                details={"error_message": str(api_error)}
            )
            # 判断是否是速率限制错误
            if "rate limit" in str(api_error).lower():
                raise HTTPException(
                    status_code=429,  # Too Many Requests
                    detail=f"Rate limit exceeded: {str(api_error)}"
                )
            raise HTTPException(
                status_code=503,  # Service Unavailable
                detail=f"OpenAI API error: {str(api_error)}"
            )
        except asyncio.TimeoutError as timeout_error:
            logger.error(
                "Request timed out",
                event="request_timeout",
                details={"error_message": str(timeout_error)}
            )
            raise HTTPException(
                status_code=503,  # Service Unavailable
                detail=f"Request timed out: {str(timeout_error)}"
            )
        except Exception as e:
            logger.error(
                "Unexpected error occurred",
                event="unexpected_error",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            # 判断是否是速率限制错误
            if "rate limit" in str(e).lower():
                raise HTTPException(
                    status_code=429,  # Too Many Requests
                    detail=f"Rate limit exceeded: {str(e)}"
                )
            # 判断是否是 OpenAI 相关错误
            if isinstance(e, OpenAIError):
                raise HTTPException(
                    status_code=503,  # Service Unavailable
                    detail=f"OpenAI API error: {str(e)}"
                )
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

            # 解析响应
            try:
                if not response.choices or not response.choices[0].message:
                    raise ValueError("Invalid response format: missing choices or message")
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response content")
                
                logger.debug(
                    event="parsing_response",
                    function="analyze_content",
                    details={"content": content}
                )
                analysis_result = self._parse_response(response.choices[0].message)
                logger.debug(
                    event="analysis_result",
                    function="analyze_content",
                    details={"result": analysis_result}
                )
                
                if not analysis_result or "segments" not in analysis_result:
                    raise ValueError("Invalid analysis result format")
            except Exception as parse_error:
                logger.error(
                    event="response_parsing_error",
                    function="analyze_content",
                    details={"error_message": str(parse_error)}
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Response parsing error: {str(parse_error)}"
                )
            
            # 构建响应
            try:
                segments = analysis_result.get("segments", [])
                if not isinstance(segments, list):
                    raise ValueError("Segments must be a list")
                
                response_obj = AnalysisResponse(
                    segments=segments,
                    metadata={
                        "analysis_type": request.analysis_type,
                        "total_segments": len(segments),
                        "model": response.model,
                        "total_tokens": response.usage.total_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens
                    }
                )
                logger.info(
                    event="request_success",
                    function="analyze_content",
                    details={
                        "segments_count": len(segments),
                        "total_tokens": response.usage.total_tokens
                    }
                )
                return response_obj
            except Exception as e:
                logger.error(
                    event="response_construction_error",
                    function="analyze_content",
                    details={"error_message": str(e)}
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Error constructing response: {str(e)}"
                )
        except HTTPException:
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )

    def _get_system_prompt(self, analysis_type: str) -> str:
        prompts = {
            "article": """你是一个专业的文章分析专家。你需要：
1. 将文章分解为有意义的段落
2. 提取每个段落的关键主题和情感
3. 识别适合的可视化类型
4. 生成图片生成提示词""",
            
            "code": """你是一个专业的代码分析专家。你需要：
1. 分析代码的主要组件和功能
2. 识别关键算法和数据结构
3. 提出适当的可视化方式
4. 生成架构图提示词""",
            
            "conversation": """你是一个专业的对话分析专家。你需要：
1. 识别对话的主要主题和转折点
2. 分析说话者的情感变化
3. 提取关键的互动模式
4. 生成对话流程可视化建议"""
        }
        return prompts.get(analysis_type, prompts["article"])

    def _build_user_prompt(self, request: AnalysisRequest) -> str:
        base_prompt = f"请分析以下内容，最多分为{request.max_segments}个段落：\n\n{request.content}"
        if request.additional_instructions:
            base_prompt += f"\n\n额外说明：{request.additional_instructions}"
        return base_prompt

    def _parse_response(self, message) -> Dict:
        try:
            content = message.content if hasattr(message, 'content') else str(message)
            print(f"[DEBUG] Response content: {content}")

            # 尝试解析 JSON
            try:
                result = json.loads(content)
                if not isinstance(result, dict) or 'segments' not in result:
                    raise ValueError("Response must be a dictionary with 'segments' key")
                
                # 验证每个段落的字段
                for segment in result['segments']:
                    # 确保所有必需字段都存在
                    if not all(key in segment for key in ['title', 'content', 'keywords', 'emotion']):
                        raise ValueError("Missing required fields in segment")
                    
                    # 设置默认值
                    segment['visualization_type'] = segment.get('visualization_type', 'data_visualization')
                    segment['image_prompt'] = segment.get('image_prompt', '生成一张简单的插图')

                    # 确保 keywords 是列表
                    if isinstance(segment['keywords'], str):
                        # 移除可能的外层括号和引号
                        keywords_text = segment['keywords'].strip('[]()（）"\'')
                        # 尝试使用各种分隔符分割
                        for separator in [',', '，', ';', '；', ' and ', '、']:
                            if separator in keywords_text:
                                segment['keywords'] = [
                                    k.strip().strip('.,。，[]()（）"\'')
                                    for k in keywords_text.split(separator)
                                    if k.strip() and not k.strip().isspace()
                                ]
                                break
                        else:
                            # 如果没有分隔符，将整个字符串作为一个关键词
                            segment['keywords'] = [keywords_text.strip()]
                    
                    # 移除空关键词并确保唯一性
                    segment['keywords'] = list(dict.fromkeys(
                        [k for k in segment['keywords'] if k and not k.isspace()]
                    ))

                return result

            except (json.JSONDecodeError, ValueError):
                # 如果不是 JSON，尝试解析结构化文本
                lines = content.split('\n')
                current_segment = {
                    'title': '',
                    'content': '',
                    'keywords': [],
                    'emotion': '',
                    'visualization_type': 'data_visualization',
                    'image_prompt': ''
                }

                # 解析每一行
                content_found = False
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # 处理段落内容
                    if '段落' in line:
                        colon_index = line.find('：')
                        if colon_index > 0:
                            current_segment['content'] = line[colon_index + 1:].strip()
                            # 尝试提取标题（取第一句话）
                            title_end = current_segment['content'].find('。')
                            if title_end > 0:
                                current_segment['title'] = current_segment['content'][:title_end]
                            content_found = True
                    
                    # 处理关键词
                    elif '主题' in line:
                        colon_index = line.find('：')
                        if colon_index > 0:
                            keywords_text = line[colon_index + 1:].strip('。')
                            current_segment['keywords'] = [
                                k.strip().strip('.,。，[](）（"\'')
                                for k in keywords_text.split('，')
                                if k.strip() and not k.strip().isspace()
                            ]
                            content_found = True
                    
                    # 处理情感
                    elif '情感' in line:
                        colon_index = line.find('：')
                        if colon_index > 0:
                            current_segment['emotion'] = line[colon_index + 1:].strip('。')
                            content_found = True
                    
                    # 处理可视化类型
                    elif '可视化' in line:
                        colon_index = line.find('：')
                        if colon_index > 0:
                            current_segment['visualization_type'] = line[colon_index + 1:].strip('。')
                            content_found = True
                    
                    # 处理图片提示词
                    elif '图片' in line:
                        colon_index = line.find('：')
                        if colon_index > 0:
                            current_segment['image_prompt'] = line[colon_index + 1:].strip('。')
                            content_found = True

                # 如果没有找到任何有效内容，抛出错误
                if not content_found:
                    raise ValueError("Error constructing response: no valid content found")

                # 确保所有必需字段都存在
                if not current_segment['title']:
                    current_segment['title'] = '未命名段落'
                if not current_segment['content']:
                    current_segment['content'] = '无内容'
                if not current_segment['keywords']:
                    current_segment['keywords'] = ['无关键词']
                if not current_segment['emotion']:
                    current_segment['emotion'] = '中性'
                
                return {'segments': [current_segment]}
        except Exception as e:
            logger.error(
                "Error parsing response",
                event="response_parsing_error",
                details={"error_message": str(e)}
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing response: {str(e)}"
            )

    def _get_visualization_type(self, content: str, keywords: List[str]) -> str:
        """根据内容和关键词确定最适合的可视化类型"""
        content_lower = content.lower()
        keywords_lower = [k.lower() for k in keywords]

        # 数据相关内容
        if any(word in content_lower or word in keywords_lower 
               for word in ['数据', '统计', '比例', '增长', 'data', 'statistics']):
            return 'data_visualization'

        # 流程相关内容
        if any(word in content_lower or word in keywords_lower 
               for word in ['步骤', '流程', '过程', 'process', 'workflow']):
            return 'flowchart'

        # 对比相关内容
        if any(word in content_lower or word in keywords_lower 
               for word in ['对比', '比较', '区别', 'comparison', 'versus']):
            return 'comparison'

        # 概念相关内容
        if any(word in content_lower or word in keywords_lower 
               for word in ['概念', '理论', '原理', 'concept', 'theory']):
            return 'concept_map'

        # 时间相关内容
        if any(word in content_lower or word in keywords_lower 
               for word in ['时间', '历史', '发展', 'timeline', 'history']):
            return 'timeline'

        # 默认使用图示
        return 'illustration'

    def _generate_image_prompt(self, segment: Dict) -> str:
        """根据段落内容生成详细的图片生成提示词"""
        visualization_type = segment['visualization_type']
        title = segment['title']
        keywords = segment['keywords']
        emotion = segment['emotion']

        prompt_templates = {
            'data_visualization': "Create a professional {emotion} infographic showing {title}. Include elements: {keywords}",
            'flowchart': "Design a clear {emotion} flowchart illustrating {title}. Steps include: {keywords}",
            'comparison': "Create a {emotion} comparison diagram for {title}. Compare these aspects: {keywords}",
            'concept_map': "Generate a {emotion} concept map explaining {title}. Key concepts: {keywords}",
            'timeline': "Design a {emotion} timeline visualization for {title}. Key events: {keywords}",
            'illustration': "Create a conceptual {emotion} illustration representing {title}. Key elements: {keywords}"
        }

        template = prompt_templates.get(visualization_type, prompt_templates['illustration'])
        return template.format(
            title=title,
            keywords=', '.join(keywords),
            emotion=emotion
        )

server = SequentialThinkingServer()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(request: AnalysisRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(
        f"Request started: {request_id}",
        event="request_start",
        details={
            "request_id": request_id,
            "content_length": len(request.content),
            "max_segments": request.max_segments,
            "analysis_type": request.analysis_type
        }
    )
    
    try:
        result = await server.analyze_content(request)
        logger.info(
            "Request completed successfully",
            event="request_end",
            details={
                "processing_time_ms": (time.time() - start_time) * 1000,
                "status": "success"
            }
        )
        return result
    except HTTPException as http_error:
        logger.error(
            f"HTTP error occurred: {http_error.detail}",
            event="request_error",
            details={
                "error_type": "HTTPException",
                "error_message": http_error.detail,
                "status_code": http_error.status_code,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        )
        raise  # 直接重新抛出原始异常
    except Exception as e:
        logger.error(
            f"Error in analyze_content: {str(e)}",
            event="request_error",
            details={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        )
        # 将所有未处理的异常转换为 500 错误
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) from e
