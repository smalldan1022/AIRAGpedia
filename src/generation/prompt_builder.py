from abc import ABC, abstractmethod

from generation.templates.wikipedia_prompt import WikipediaPrompt


class BasePromptBuilder(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build(self, content: str, query: str) -> str: ...


class WikipediaPromptBuilder(BasePromptBuilder):
    DELIMITER = "\n\n---\n\n"

    def __init__(self, prompt_template: WikipediaPrompt):
        self.template = prompt_template

    def build(self, content: str, query: str) -> str:
        sections = [
            # 1. Role & Task
            self.template.system_prompt.strip(),
            # 2. Output format (schema)
            f"Output Schema:\n```\n{self.template.pydantic_schema}\n```",
            # 3. Examples (few-shot)
            f"Examples:\n{self.template.example.strip()}",
            # 4. Actual content
            f'Content:\n"""\n{content}\n"""',
            # 5. Rules
            f"Instructions:\n{self.template.instructions.strip()}",
            # 6. Question
            f"Question: {query}",
        ]
        return self.DELIMITER.join(sections)
