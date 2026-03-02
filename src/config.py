import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RagConfig:
    data_rag: Dict[str, Any]
    llm_rag: Dict[str, Any]
    params_rag: Dict[str, Any]


def load_config(path: str = "config.yaml") -> RagConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return RagConfig(
        data_rag=cfg["data-rag"],
        llm_rag=cfg["llm-rag"],
        params_rag=cfg["params"],
    )