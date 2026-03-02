import os
import re
from typing import List, Set, Tuple

from llama_index.core import Settings
from llama_index.core.postprocessor import SentenceTransformerRerank

from .config import RagConfig


def generate_plan(query: str) -> List[str]:
    prompt = f"""
            Break this research question into 3 to 5 focused subtopics.
            Use the SAME language as the question.

            Question:
            {query}

            Output ONLY a numbered list.
            """
    resp = Settings.llm.complete(prompt).text

    subtopics: List[str] = []
    for line in resp.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if m:
            subtopics.append(m.group(1).strip())

    if not subtopics:
        subtopics = [l.strip("-• ").strip() for l in resp.splitlines() if l.strip()]

    return subtopics[:5]


def retrieve_evidence(vector_engine, reranker, subtopic: str) -> Tuple[List[str], Set[str]]:
    resp = vector_engine.query(subtopic)
    nodes = getattr(resp, "source_nodes", []) or []

    reranked = reranker.postprocess_nodes(nodes, query_str=subtopic)

    contexts: List[str] = []
    sources: Set[str] = set()

    for n in reranked:
        try:
            contexts.append(n.node.get_content())
        except Exception:
            continue
        meta = getattr(n.node, "metadata", {}) or {}
        src = meta.get("file_name") or meta.get("file_path") or "unknown_source"
        sources.add(os.path.basename(str(src)))

    return contexts, sources


def rewrite_subtopic(main_query: str, subtopic: str) -> str:
    prompt = f"""
            Rewrite the subtopic into a better retrieval query.
            Keep it short, specific, and include key terms from the paper domain (fine-tuning, RAG).
            Use the SAME language as the question/subtopic.
            Return ONLY the rewritten query.

            Main question: {main_query}
            Subtopic: {subtopic}
            """
    return Settings.llm.complete(prompt).text.strip()


def retrieve_with_retry(
    vector_engine,
    reranker,
    main_query: str,
    subtopic: str,
    max_retry: int = 1,
) -> Tuple[List[str], Set[str]]:
    ctxs, srcs = retrieve_evidence(vector_engine, reranker, subtopic)
    if ctxs:
        return ctxs, srcs

    for _ in range(max_retry):
        new_sub = rewrite_subtopic(main_query, subtopic)
        ctxs, srcs = retrieve_evidence(vector_engine, reranker, new_sub)
        if ctxs:
            return ctxs, srcs

    return [], set()


def synthesize_final_answer(query: str, plan: List[str], evidence_blocks: List[str]) -> str:
    evidence_text = "\n\n---\n\n".join(evidence_blocks)

    prompt = f"""
            You are a research assistant. Use ONLY the evidence provided.

            Question:
            {query}

            Subtopics:
            {chr(10).join([f"- {p}" for p in plan])}

            Evidence:
            {evidence_text}

            Write a structured answer with headings and a comparison where relevant.
            If evidence is insufficient for a claim, say: "Not found in the provided papers".
            """
    return Settings.llm.complete(prompt).text


def run_agentic_detail(cfg: RagConfig, vector_engine, query: str) -> Tuple[List[str], str, Set[str]]:
    params = cfg.params_rag
    llm_cfg = cfg.llm_rag

    reranker = SentenceTransformerRerank(
        model=llm_cfg["rerank_model_name"],
        top_n=params["rerank_top_k"],
    )

    plan = generate_plan(query)

    all_evidence: List[str] = []
    all_sources: Set[str] = set()

    for sub in plan:
        ctxs, srcs = retrieve_with_retry(
            vector_engine,
            reranker,
            query,
            sub,
            max_retry=params.get("rewrite_max_retry", 1),
        )
        all_evidence.extend(ctxs)
        all_sources |= srcs

    answer = synthesize_final_answer(query, plan, all_evidence)
    return plan, answer, all_sources