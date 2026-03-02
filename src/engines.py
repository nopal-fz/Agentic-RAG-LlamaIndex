from typing import List, Tuple

from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from .config import RagConfig


def build_engines_and_tools(
    cfg: RagConfig,
    vector_index: VectorStoreIndex,
    summary_index: SummaryIndex,
):
    params = cfg.params_rag

    vector_engine = vector_index.as_query_engine(similarity_top_k=params["top_k"])
    summary_engine = summary_index.as_query_engine(response_mode="tree_summarize")

    # bilingual descriptions help router for Indonesian queries
    tools: List[QueryEngineTool] = [
        QueryEngineTool(
            query_engine=summary_engine,
            metadata=ToolMetadata(
                name="summarize",
                description=(
                    "EN: Use for high-level overviews, executive summaries, key takeaways, "
                    "or when the user asks to summarize / overview / main points.\n"
                    "ID: Pakai untuk ringkasan/gambaran umum/poin penting/kesimpulan."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=vector_engine,
            metadata=ToolMetadata(
                name="vector_qa",
                description=(
                    "EN: Use for technical, specific questions: methods, experiments, metrics, "
                    "implementation details, comparisons, how/why questions requiring deep retrieval.\n"
                    "ID: Pakai untuk pertanyaan teknis spesifik: metodologi, eksperimen, metrik, "
                    "detail implementasi, perbandingan, bagaimana/kenapa."
                ),
            ),
        ),
    ]

    selector = LLMSingleSelector.from_defaults()
    return vector_engine, summary_engine, tools, selector