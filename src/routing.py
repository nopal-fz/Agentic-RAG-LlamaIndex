from typing import List

from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool


def route_tool_name(selector: LLMSingleSelector, tools: List[QueryEngineTool], query: str) -> str:
    """
    Return tool name chosen by the selector: 'summarize' or 'vector_qa'.
    Handles minor API differences across llama_index versions.
    """
    # Try common signatures
    try:
        sel = selector.select(query_engine_tools=tools, query=query)  # type: ignore
        idx = getattr(sel, "ind", None)
        if idx is None:
            idx = getattr(sel, "index", None)
        if isinstance(idx, int):
            return tools[idx].metadata.name
    except Exception:
        pass

    try:
        sel = selector.select(tools=tools, query=query)  # type: ignore
        idx = getattr(sel, "ind", None)
        if idx is None:
            idx = getattr(sel, "index", None)
        if isinstance(idx, int):
            return tools[idx].metadata.name
    except Exception:
        pass

    # last resort heuristic fallback
    q = query.lower()
    if any(k in q for k in ["summary", "summarize", "overview", "key takeaways", "ringkas", "ringkasan", "tldr", "tl;dr"]):
        return "summarize"
    return "vector_qa"