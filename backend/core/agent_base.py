from langchain_groq import ChatGroq
from abc import ABC, abstractmethod
import instructor
from pydantic import BaseModel
from loguru import logger
from types import SimpleNamespace
import json
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()


PROVIDER = os.getenv("COMMON_MODEL")

class AgentBase(ABC):
    def __init__(self, name, max_retries=2, verbose=True, provider: str = None, api_key=None):
        """
        provider: e.g. 'openai/gpt-4o-mini' or 'groq/llama-3.1-8b-instant'
        """
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose
        if provider is None:
            self.provider = PROVIDER
        else:
            self.provider = provider.lower()
        self.api_key = api_key

        # Build instructor client from provider string
        self.client = instructor.from_provider(self.provider, api_key=api_key, mode=instructor.Mode.JSON, async_client=True)

    @abstractmethod
    async def execute(self, *args, **kwargs):
        pass

    async def get_completion(
        self, 
        chat_history, 
        response_model: BaseModel = None, 
        temperature=0.0, 
        max_tokens=1000
    ):
        """
        chat_history: list of {role: 'user'|'system'|'assistant', content: str}
        response_model: optional Pydantic model for structured output
        """

        if self.verbose:
            logger.info(f"[{self.name}] Sending messages to {self.provider.upper()}:")
            for msg in chat_history:
                logger.debug(f"  {msg['role']}: {msg['content']}")

        # -------------------------------
        # Case 1: Structured output mode
        # -------------------------------
        if response_model is not None:
            print("Pydantic Schema Provided")
            completion, completion_raw = await self.client.chat.completions.create_with_completion(
                messages=chat_history,
                response_model=response_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            usage = getattr(completion_raw, "usage", None) or {}
            completion_dict = completion.model_dump()
            content_str = json.dumps(completion_dict, ensure_ascii=False)
            usage_metadata = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

            return SimpleNamespace(
                content=content_str,
                parsed=completion,
                usage_metadata=usage_metadata,
            )

        # -------------------------------
        # Case 2: No Pydantic model : normal chat completion
        # -------------------------------
        provider_str = self.provider.lower().strip()     
        parts = provider_str.split("/", 1)
        model = parts[1] if len(parts) > 1 else None

        if self.provider.lower().startswith("groq"):
            llm = ChatGroq(
                model=model or "meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
            )
        elif self.provider.lower().startswith("openai"):
            llm = ChatOpenAI(
                model=model or "gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        completion = await llm.ainvoke(chat_history)

        # Normalize token usage
        if hasattr(completion, "usage_metadata"):
            usage = completion.usage_metadata
        elif hasattr(completion, "response_metadata"):
            # fallback for OpenAI depending on how LangChain returns metadata
            usage = completion.response_metadata.get("token_usage", {})
        else:
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Ensure consistent keys
        completion.usage_metadata = {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
            "total_tokens": usage.get("total_tokens", usage.get("total_tokens", 0))
        }

        return completion

    async def call_model(
        self, 
        chat_history, 
        response_model: BaseModel = None, 
        temperature=0.0, 
        max_tokens=1500
    ):
        global Total_LLM_Calls, Total_token_usage

        retries = 0
        wait_time = 1
        max_wait = 65

        while retries < self.max_retries:
            try:
                reply = await self.get_completion(
                    chat_history,
                    response_model=response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )


                if self.verbose:
                    logger.info(f"[{self.name}] Received structured response: {reply}")

                return reply

            except Exception as e:
                retries += 1
                logger.error(
                    f"[{self.name}] Error during {self.provider.upper()} call: {e}. "
                    f"Retry {retries}/{self.max_retries}. Sleeping {wait_time}s before retry."
                )
                await asyncio.sleep(wait_time)
                wait_time = min(wait_time * 2, max_wait)

        raise Exception(f"[{self.name}] Failed to get response after {self.max_retries} retries.")