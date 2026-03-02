from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from .config import RagConfig

def setup_models(cfg: RagConfig) -> None:
    params = cfg.params_rag
    llm_cfg = cfg.llm_rag

    Settings.llm = Ollama(
        model=llm_cfg["model_name"],
        temperature=params["temperature"],
        request_timeout=600,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=llm_cfg["embedding_model_name"])
    Settings.node_parser = SentenceSplitter(
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
    )