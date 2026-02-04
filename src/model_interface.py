from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load .env from the project root:
# F:\university\milan\NLP\project\.env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass
class ModelConfig:
    name: str


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class DummyEchoModel(BaseModel):
    """Use this to test the pipeline without spending API tokens."""
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        preview = user_prompt[:200].replace("\n", " ")
        return (
            "NO CHANGE: (dummy)\n"
            f"USER_PROMPT_PREVIEW: {preview} ..."
        )



class ClaudeChatModel(BaseModel):
    """
    Claude via Anthropic Messages API.
    Reads ANTHROPIC_API_KEY from .env (loaded above).
    """
    def __init__(self, config: ModelConfig, max_tokens: int = 512, temperature: float = 0.2):
        super().__init__(config)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.config.name,
            system=system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        # msg.content is a list of blocks; we join text blocks
        parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)

        return "\n".join(parts).strip()
