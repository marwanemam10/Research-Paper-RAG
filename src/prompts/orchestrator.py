from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).resolve().parent


class PromptOrchestrator:
    """
    إدارة واستدعاء قوالب الـ Prompts بصيغة Markdown مع دعم الـ Versioning (@v1, @v2).
    """

    @staticmethod
    def load_prompt_text(prompt_name: str, version: str = "v1") -> str:
        
        if "@" in prompt_name:
            file_name = f"{prompt_name}.md" if not prompt_name.endswith(".md") else prompt_name
        else:
            file_name = f"{prompt_name}@{version}.md"

        prompt_path = PROMPTS_DIR / file_name

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @classmethod
    def get_prompt(cls, prompt_name: str = "rag", version: str = "v1") -> ChatPromptTemplate:
        template_str = cls.load_prompt_text(prompt_name, version)
        return ChatPromptTemplate.from_template(template_str)