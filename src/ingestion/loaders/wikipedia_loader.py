import wikipediaapi
from pydantic import field_validator

from ingestion.loaders.base import BaseLoader, LoaderConfig, RawDocument, SourceType
from utils.logger import LoggerFactory

logger = LoggerFactory(level="INFO").get_logger(name=__name__)


class WikipediaConfig(LoaderConfig):
    source: str = "Wikipedia"
    language: str = "en"
    document_type: str = "html"
    max_pages: int = 50

    min_text_length: int = 1000
    min_year: int | None = None
    min_categories: int = 3

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        allowed = ["en", "zh"]
        if v not in allowed:
            raise ValueError(f"language must be one of {allowed}")
        return v


class WikipediaLoader(BaseLoader):
    def __init__(self, config: WikipediaConfig) -> None:
        super().__init__(config)
        self.wiki = wikipediaapi.Wikipedia(
            language=config.language,
            user_agent="AIRAGpedia/0.1.0",
        )

    def _is_valid_page(self, page: wikipediaapi.WikipediaPage) -> bool:
        if not page.exists() or not page.text:
            return False
        if len(page.text) < self.config.min_text_length:
            return False
        # TODO: finish this feature
        if self.config.min_year is not None:
            pass
        if len(page.categories) < self.config.min_categories:
            return False
        return True

    def load(self, category: str) -> list[RawDocument]:
        cat_page = self.wiki.page(f"Category:{category}")

        if not cat_page.exists():
            raise ValueError(
                f"Category '{category}' not found. "
                f"Check https://en.wikipedia.org/wiki/Category:{category.replace(' ', '_')}"
            )

        docs = []
        for _, page in cat_page.categorymembers.items():
            if len(docs) >= self.config.max_pages:
                break
            if not self._is_valid_page(page):
                continue

            docs.append(
                RawDocument(
                    title=page.title,
                    text=page.text,
                    category=category,
                    source_type=SourceType.WIKIPEDIA,
                    metadata={
                        "url": page.fullurl,
                        "pageid": page.pageid,
                        "category": category,
                    },
                )
            )
            logger.info(f"[{len(docs)}/{self.config.max_pages}] Loaded: {page.title}")

        return docs
