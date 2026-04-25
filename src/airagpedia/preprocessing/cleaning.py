import re
from abc import ABC, abstractmethod
from ingestion.loaders.base import RawDocument


class BaseCleaner(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def clean(self): ...


class TextCleaner(BaseCleaner):
    def __init__(self, language: str = "en"):
        super().__init__()
        self.language = language

    def _remove_wikipedia_templates(self, text: str) -> str:
        # Remove {{ }} template syntax, like {{cite web|...}}
        text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
        # Remove [[ ]] internal link and keep text like [[Machine learning|ML]] → ML
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
        # Remove == title syntax like == History ==
        text = re.sub(r"={2,}.*?={2,}", "", text)
        return text

    def _remove_special_chars(self, text: str) -> str:
        text = re.sub(r"[^\w\s.,!?;:()\-']", " ", text)
        return text

    def _remove_extra_whitespace(self, text: str) -> str:
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean(self, doc: RawDocument) -> RawDocument:
        text = doc.text
        text = self._remove_wikipedia_templates(text)
        text = self._remove_special_chars(text)
        text = self._remove_extra_whitespace(text)
        return doc.model_copy(update={"text": text})
