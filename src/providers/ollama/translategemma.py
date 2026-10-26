import json
import os
from functools import lru_cache

from langchain_ollama import ChatOllama


@lru_cache(maxsize=1)
def _load_translategemma_data() -> dict:
    """Loads and caches the Translategemma.json data."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "translategemma.json")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def translategemma_languages() -> list[str]:
    """Returns a list of languages supported by Translategemma."""

    data = _load_translategemma_data()
    return [key for key in data["supported"].keys()]


def get_language_code(lang_code: str) -> str:
    """Returns the appropriate language code to be used with Translategemma.
    This method is used to comply with the Translategemma documentation, but to keep
    the general language code set-up the same across translation models."""

    data = _load_translategemma_data()
    return data["short_code_remap"].get(lang_code, lang_code)


def get_language(lang_code: str) -> str:
    """Returns the language in accordance with the provided language code for Translategemma."""

    data = _load_translategemma_data()
    return data["supported"][lang_code]


def get_translation_translategemma(
    from_lang: str, to_lang: str, prompt: str, temperature: float, base_url: str
) -> dict:
    """Invokes a translation from translategemma 4B"""

    llm = ChatOllama(
        model="translategemma:latest",
        temperature=temperature,
        base_url=base_url,
        disable_streaming=True,
    )

    from_language = get_language(from_lang)
    to_language = get_language(to_lang)
    from_lang_code = get_language_code(from_lang)
    to_lang_code = get_language_code(to_lang)

    instruction = f"""
    You are a professional {from_language} ({from_lang_code}) to {to_language} ({to_lang_code}) translator.
    Your goal is to accurately convey the meaning and nuances of the original {from_language} text while adhering to {to_language} grammar, vocabulary, and cultural sensitivities.
    Produce only the {to_language} translation, without any additional explanations or commentary.
    Please translate the following {from_language} text into {to_language}:\n\n{prompt}"""

    translation = llm.invoke(instruction)

    return {
        "prompt": prompt,
        "translation": translation.content,
        "from_lang_code": from_lang,
        "to_lang_code": to_lang,
    }
