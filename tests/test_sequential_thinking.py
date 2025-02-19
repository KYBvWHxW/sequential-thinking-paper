import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from fastapi import HTTPException
import sys
import os
import logging
from typing import Dict, Any
from openai import OpenAIError

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from mcp_servers.sequential_thinking.server import app, SequentialThinkingServer, AnalysisRequest
from unittest.mock import AsyncMock, patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 测试数据
TEST_ARTICLE = """
翻译模型上新 | 情感洞察能力升级！AI解说大师翻译服务全球化覆盖

字幕总透着"塑料感"？译文像念说明书，潜台词被一键抹除，让人一头雾水……
直译的情感缺失，让剧情张力大打折扣。
"""

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_openai():
    with patch('mcp_servers.sequential_thinking.server.AsyncOpenAI') as mock:
        yield mock

@pytest.mark.asyncio
async def test_analyze_content_success(test_client, mock_openai):
    # 模拟 OpenAI 响应
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(
            message={
                "content": "段落1：本文主要介绍了AI解说大师的翻译服务升级，并指出了传统翻译模型的问题，如字幕总透着“塑料感”，译文像念说明书，潜台词被一键抹除，让人一头雾水，以及直译的情感缺失，使得剧情张力大打折扣。\n\n关键主题：AI解说大师的翻译服务全球化覆盖和升级，以及传统翻译模型的问题。\n\n情感：此段有一种积极的改变信号，表示AI解说大师正在改进他们的服务以解决传统翻译模型的问题。\n\n可视化类型：此段可以使用比较图来显示AI解说大师翻译服务升级前后的变化，或者用饼图来显示全球覆盖率。\n\n图片生成提示词：AI翻译服务，全球覆盖，服务升级，传统与现代翻译模型的比较。"
            }
        )
    ]
    mock_response.model = "gpt-4"
    mock_response.usage.total_tokens = 150
    mock_response.usage.completion_tokens = 50
    mock_response.usage.prompt_tokens = 100
    
    mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

    response = test_client.post(
        "/analyze",
        json={
            "content": TEST_ARTICLE,
            "max_segments": 1,
            "analysis_type": "article"
        }
    )

    assert response.status_code == 200
    result = response.json()
    assert "segments" in result
    assert len(result["segments"]) == 1
    assert "metadata" in result
    assert result["metadata"]["total_segments"] == 1

@pytest.mark.asyncio
async def test_analyze_content_invalid_request(test_client):
    response = test_client.post(
        "/analyze",
        json={
            "max_segments": 1,
            "analysis_type": "article"
        }
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_analyze_content_openai_error(test_client, mock_openai):
    # 创建一个模拟的 AsyncOpenAI 实例
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=OpenAIError("OpenAI API Error")
    )
    mock_openai.return_value = mock_client

    with pytest.raises(HTTPException) as exc_info:
        await SequentialThinkingServer().analyze_content(
            AnalysisRequest(
                content=TEST_ARTICLE,
                max_segments=1,
                analysis_type="article"
            )
        )
    assert exc_info.value.status_code == 503
    assert "OpenAI API error" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_analyze_content_invalid_response(test_client, mock_openai):
    # 创建一个模拟的 AsyncOpenAI 实例
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(
            message={
                "content": "这不是一个有效的 JSON 格式"
            }
        )
    ]
    mock_response.model = "gpt-4"
    mock_response.usage = AsyncMock(
        total_tokens=100,
        completion_tokens=50,
        prompt_tokens=50
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_openai.return_value = mock_client

    with pytest.raises(HTTPException) as exc_info:
        await SequentialThinkingServer().analyze_content(
            AnalysisRequest(
                content=TEST_ARTICLE,
                max_segments=1,
                analysis_type="article"
            )
        )
    assert exc_info.value.status_code == 500
    assert "Error constructing response" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_analyze_content_timeout(test_client, mock_openai):
    # 创建一个模拟的 AsyncOpenAI 实例
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=asyncio.TimeoutError("Request timed out")
    )
    mock_openai.return_value = mock_client

    with pytest.raises(HTTPException) as exc_info:
        await SequentialThinkingServer().analyze_content(
            AnalysisRequest(
                content=TEST_ARTICLE,
                max_segments=1,
                analysis_type="article"
            )
        )
    assert exc_info.value.status_code == 503
    assert "timed out" in str(exc_info.value.detail).lower()

@pytest.mark.asyncio
async def test_analyze_content_rate_limit(test_client, mock_openai):
    # 创建一个模拟的 AsyncOpenAI 实例
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=OpenAIError("Rate limit exceeded")
    )
    mock_openai.return_value = mock_client

    with pytest.raises(HTTPException) as exc_info:
        await SequentialThinkingServer().analyze_content(
            AnalysisRequest(
                content=TEST_ARTICLE,
                max_segments=1,
                analysis_type="article"
            )
        )
    assert exc_info.value.status_code == 429
    assert "Rate limit" in str(exc_info.value.detail)
