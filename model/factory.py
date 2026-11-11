from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=rag_conf["chat_api_key"],
            base_url=rag_conf["chat_base_url"]
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
       return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            openai_api_key=rag_conf["embedding_api_key"],
            openai_api_base=rag_conf["embedding_base_url"]
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
