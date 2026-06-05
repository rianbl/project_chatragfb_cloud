from __future__ import annotations

import re
from pathlib import Path

import yaml


class FilePromptStore:
    def __init__(self, prompt_file: str | Path) -> None:
        self._prompt_file = Path(prompt_file)
        self._cache: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._cache is None:
            with self._prompt_file.open("r", encoding="utf-8") as source:
                data = yaml.safe_load(source) or {}
            if not isinstance(data, dict):
                raise ValueError("Prompt store must be a mapping.")
            self._cache = {str(key): str(value) for key, value in data.items()}
        return self._cache

    def get_prompt(self, prompt_name: str) -> str:
        prompts = self._load()
        if prompt_name not in prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found in store.")
        return prompts[prompt_name]

    def render(self, prompt_name: str, **variables) -> str:
        template = self.get_prompt(prompt_name)
        required = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", template))
        missing = sorted(required - set(variables.keys()))
        if missing:
            raise KeyError(f"Missing prompt variables: {', '.join(missing)}")

        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        return rendered
