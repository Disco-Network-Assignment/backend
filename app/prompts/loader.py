"""Loads every prompt file once and renders it on demand.

A prompt file is frontmatter (name, version) plus `# System` and `# User` sections; fragments
have only a `# User` section. Variables are `{{name}}`; objects render as pretty JSON. Rendering
with an unknown or missing variable raises, so a typo in a template fails at startup or in the
test suite, never silently in production."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_SECTION = re.compile(r"^# (System|User)\s*$", re.M)
_VARIABLE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptError(Exception):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str | None
    user: str
    path: Path

    @property
    def variables(self) -> frozenset[str]:
        text = f"{self.system or ''}\n{self.user}"
        return frozenset(_VARIABLE.findall(text))


@dataclass(frozen=True)
class RenderedPrompt:
    name: str
    version: str
    instructions: str         # the agent's system prompt; empty for fragments
    input: str                # the user message


class PromptLoader:
    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir
        self._templates: dict[str, PromptTemplate] = {}
        for path in sorted(prompts_dir.rglob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            template = self._parse(path)
            if template.name in self._templates:
                raise PromptError(f"duplicate prompt name '{template.name}' in {path}")
            self._templates[template.name] = template
        if not self._templates:
            raise PromptError(f"no prompt files found in {prompts_dir}")

    @property
    def names(self) -> list[str]:
        return sorted(self._templates)

    def get(self, name: str) -> PromptTemplate:
        try:
            return self._templates[name]
        except KeyError:
            raise PromptError(f"unknown prompt '{name}'; known: {self.names}") from None

    def version(self, name: str) -> str:
        return self.get(name).version

    def render(self, name: str, **variables: Any) -> RenderedPrompt:
        template = self.get(name)
        missing = template.variables - variables.keys()
        unknown = variables.keys() - template.variables
        if missing or unknown:
            raise PromptError(f"prompt '{name}': missing {sorted(missing)}, unknown {sorted(unknown)}")
        return RenderedPrompt(
            name=template.name,
            version=template.version,
            instructions=self._substitute(template.system or "", variables),
            input=self._substitute(template.user, variables),
        )

    def render_fragment(self, name: str, **variables: Any) -> str:
        """Fragments are user-message snippets appended on retries."""
        return self.render(name, **variables).input

    # ---- parsing ----
    @staticmethod
    def _parse(path: Path) -> PromptTemplate:
        text = path.read_text(encoding="utf-8")
        match = _FRONTMATTER.match(text)
        if match is None:
            raise PromptError(f"{path}: missing frontmatter")
        meta = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        if "name" not in meta or "version" not in meta:
            raise PromptError(f"{path}: frontmatter needs 'name' and 'version'")
        body = text[match.end():]
        sections: dict[str, str] = {}
        parts = _SECTION.split(body)
        # parts = [preamble, "System", text, "User", text, ...]
        for heading, content in zip(parts[1::2], parts[2::2], strict=True):
            sections[heading] = content.strip()
        if "User" not in sections:
            raise PromptError(f"{path}: needs a '# User' section")
        return PromptTemplate(name=meta["name"], version=meta["version"],
                              system=sections.get("System"), user=sections["User"], path=path)

    @staticmethod
    def _substitute(text: str, variables: dict[str, Any]) -> str:
        def replace(match: re.Match[str]) -> str:
            value = variables[match.group(1)]
            if isinstance(value, str):
                return value
            return json.dumps(value, indent=1, ensure_ascii=False, default=str)

        return _VARIABLE.sub(replace, text)
