import logging
import os
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# 使用环境变量中的模型配置
_openai_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
_openai_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
_openai_model = os.environ.get("LLM_MODEL") or "gpt-5-mini"

openai_client = OpenAI(
    api_key=_openai_api_key,
    base_url=_openai_base_url if _openai_base_url else None
)


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        # Let OpenAI handle the pydantic parsing directly
        # gpt-5-mini 不支持 temperature=0，使用默认值 1
        completion = openai_client.beta.chat.completions.parse(
            model=_openai_model,
            messages=messages,
            response_format=MemoryCategories
        )

        parsed: MemoryCategories = completion.choices[0].message.parsed
        return [cat.strip().lower() for cat in parsed.categories]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        try:
            logging.debug(f"[DEBUG] Raw response: {completion.choices[0].message.content}")
        except Exception as debug_e:
            logging.debug(f"[DEBUG] Could not extract raw response: {debug_e}")
        raise
