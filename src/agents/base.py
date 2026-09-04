"""
Base LLM Agent Framework with strict Pydantic JSON schema enforcement.
Supports Gemini, OpenAI, Anthropic, and Mock providers.
"""
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Dict, Any, Optional
import json
import logging
from pydantic import BaseModel, ValidationError
import httpx

from src.config.settings import settings
from src.core.enums import AgentRole

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    def __init__(self, role: AgentRole, schema_class: Type[T]):
        self.role = role
        self.schema_class = schema_class

    @abstractmethod
    def build_prompt(self, context: Dict[str, Any]) -> str:
        """Construct the prompt with data and instructions."""
        pass

    async def execute(self, context: Dict[str, Any]) -> T:
        """
        Executes the agent: builds prompt, calls LLM, validates output against schema.
        Falls back to safe default mock logic on parsing error, network timeout, or if provider is 'mock'.
        """
        prompt = self.build_prompt(context)
        raw_response = ""

        try:
            if settings.llm_provider == "gemini" and settings.gemini_api_key:
                raw_response = await self._call_gemini(prompt)
            elif settings.llm_provider == "groq" and settings.groq_api_key:
                raw_response = await self._call_groq(prompt)
            elif settings.llm_provider == "huggingface" and settings.huggingface_api_key:
                raw_response = await self._call_huggingface(prompt)
            elif settings.llm_provider == "openrouter" and settings.openrouter_api_key:
                raw_response = await self._call_openrouter(prompt)
            elif settings.llm_provider == "openai" and settings.openai_api_key:
                raw_response = await self._call_openai(prompt)
            elif settings.llm_provider == "anthropic" and settings.anthropic_api_key:
                raw_response = await self._call_anthropic(prompt)
            else:
                return self.get_mock_response(context)
        except Exception as e:
            logger.warning(f"LLM API call failed for {self.role.value}: {e}. Falling back to default mock logic.")
            return self.get_mock_response(context)

        # Parse & Validate JSON
        try:
            cleaned = self._clean_json(raw_response)
            parsed_dict = json.loads(cleaned)
            return self.schema_class.model_validate(parsed_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Agent {self.role.value} schema parsing failed: {e}. Falling back to default.")
            return self.get_mock_response(context)

    def _clean_json(self, text: str) -> str:
        """Strips markdown code fences ```json ... ``` if present."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @abstractmethod
    def get_mock_response(self, context: Dict[str, Any]) -> T:
        """Deterministic offline fallback matching the schema."""
        pass

    async def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model}:generateContent?key={settings.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_openai(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_groq(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
        payload = {
            "model": settings.llm_model if "llama" in settings.llm_model or "mixtral" in settings.llm_model else "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_huggingface(self, prompt: str) -> str:
        model_name = settings.llm_model if "/" in settings.llm_model else "meta-llama/Llama-3.3-70B-Instruct"
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {settings.huggingface_api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1024, "return_full_text": False}
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
            elif isinstance(data, dict):
                return data.get("generated_text", str(data))
            return str(data)

    async def _call_openrouter(self, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Auto Trading Multi-Agent"
        }
        payload = {
            "model": settings.llm_model if "/" in settings.llm_model else "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_anthropic(self, prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

