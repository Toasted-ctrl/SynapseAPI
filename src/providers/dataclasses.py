from enum import Enum


class LangChainCon(str, Enum):
    ANTHROPIC = "ChatAnthropic"
    OLLAMA = "ChatOllama"
    OPENAI = "ChatOpenAI"
    GOOGLE = "ChatGoogleGenerativeAI"
