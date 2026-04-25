import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from generation.base import BaseGenerator, BaseGeneratorConfig, GenerationResult
from utils.logger import LoggerFactory


class GeminiConfig(BaseGeneratorConfig):
    max_tokens: int = 2500
    output_type: str = "stream"
    temperature: float = 0.4
    model: str = "gemini-2.5-flash"


class GeminiGenerationResult(GenerationResult):
    text: str = ""
    model: str = ""
    input_tokens: int = 2000
    output_tokens: int = 2500
    output_type: str = "json"


class GeminiGenerator(BaseGenerator):
    DEFAULT_PROMPT = ""

    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt="") -> GeminiGenerationResult:
        prompt = prompt or self.DEFAULT_PROMPT
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            ),
        )

        return GenerationResult(
            text=response.text,
            model=self.config.model,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )


if __name__ == "__main__":
    logger = LoggerFactory(level="INFO").get_logger(name=__name__)
    generator = GeminiGenerator(GeminiConfig())
    result = generator.generate("What is machine learning in one sentence?")
    logger.info(result.text)
    logger.info(
        f"tokens — input: {result.input_tokens}, output: {result.output_tokens}"
    )
