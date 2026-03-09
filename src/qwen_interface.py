from __future__ import annotations

from dataclasses import dataclass
import os
from huggingface_hub import InferenceClient


@dataclass
class HFConfig:
    name: str                      # model id "Qwen/Qwen2.5-7B-Instruct"
    token_env: str = "HF_TOKEN"
    temperature: float = 0.2
    max_tokens: int = 512


class HFChatModel:
    """
    Drop-in replacement for ClaudeChatModel:
    exposes:
      - self.config (with .name)
      - generate(system_prompt, user_prompt) -> str
    """
    def __init__(self, config: HFConfig):
        token = os.getenv(config.token_env)
        if not token:
            raise RuntimeError(f"Missing {config.token_env} in environment (.env not loaded?)")
        self.config = config
        self.client = InferenceClient(api_key=token)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.config.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
