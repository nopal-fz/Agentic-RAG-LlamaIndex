import time

from src.config import load_config
from src.models import setup_models
from src.indexing import build_or_load_indexes
from src.engines import build_engines_and_tools
from src.routing import route_tool_name
from src.agentic import run_agentic_detail

def main():
    cfg = load_config("config.yaml")
    setup_models(cfg)

    vector_index, summary_index = build_or_load_indexes(cfg)
    vector_engine, summary_engine, tools, selector = build_engines_and_tools(cfg, vector_index, summary_index)

    q = input("Ask: ").strip()
    if not q:
        raise SystemExit("Empty query.")

    total_start = time.time()
    tool_name = route_tool_name(selector, tools, q)
    print(f"[TIMER] Router select -> {tool_name}")

    if tool_name == "summarize":
        t_sum = time.time()
        resp = summary_engine.query(q)
        summary_text = getattr(resp, "response", str(resp))
        print(f"\nSUMMARY:\n{summary_text.strip()}")
        print(f"[TIMER] Summary engine: {time.time() - t_sum:.2f}s")
        print(f"[TIMER] TOTAL: {time.time() - total_start:.2f}s")
        return

    # vector_qa selected => agentic detail pipeline
    t_det = time.time()
    plan, answer, sources = run_agentic_detail(cfg, vector_engine, q)
    print(f"[TIMER] Detail pipeline: {time.time() - t_det:.2f}s")

    if plan:
        print("\nPLAN:")
        for i, p in enumerate(plan, 1):
            print(f"{i}. {p}")

    if answer.strip():
        print("\nANSWER:\n")
        print(answer.strip())

    if sources:
        print("\nSOURCES:")
        for s in sorted(sources):
            print("-", s)

    print(f"[TIMER] TOTAL: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()