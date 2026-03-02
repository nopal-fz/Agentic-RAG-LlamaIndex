# src/indexing.py

import os
from typing import Tuple

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    SummaryIndex,
)
from llama_index.core.readers import SimpleDirectoryReader

def _dir_has_files(p: str) -> bool:
    return os.path.exists(p) and os.path.isdir(p) and len(os.listdir(p)) > 0

def build_or_load_indexes(cfg) -> Tuple[VectorStoreIndex, SummaryIndex]:
    data = cfg.data_rag

    vector_dir = data["vector_index_dir"]
    summary_dir = data["summary_index_dir"]
    data_dir = data["data_dir"]

    # Load if exists
    if _dir_has_files(vector_dir) and _dir_has_files(summary_dir):
        v_sc = StorageContext.from_defaults(persist_dir=vector_dir)
        s_sc = StorageContext.from_defaults(persist_dir=summary_dir)
        vector_index = load_index_from_storage(v_sc)
        summary_index = load_index_from_storage(s_sc)
        return vector_index, summary_index

    # Collect PDFs
    pdf_files = [
        os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir)
        if fn.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")

    docs = SimpleDirectoryReader(input_files=pdf_files).load_data()

    vector_index = VectorStoreIndex.from_documents(docs)
    summary_index = SummaryIndex.from_documents(docs)

    os.makedirs(vector_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    vector_index.storage_context.persist(persist_dir=vector_dir)
    summary_index.storage_context.persist(persist_dir=summary_dir)

    return vector_index, summary_index

def build_or_load_indexes_from_dir(
    data_dir: str,
    vector_index_dir: str,
    summary_index_dir: str,
) -> Tuple[VectorStoreIndex, SummaryIndex]:

    if _dir_has_files(vector_index_dir) and _dir_has_files(summary_index_dir):
        v_sc = StorageContext.from_defaults(persist_dir=vector_index_dir)
        s_sc = StorageContext.from_defaults(persist_dir=summary_index_dir)
        vector_index = load_index_from_storage(v_sc)
        summary_index = load_index_from_storage(s_sc)
        return vector_index, summary_index

    pdf_files = [
        os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir)
        if fn.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")

    docs = SimpleDirectoryReader(input_files=pdf_files).load_data()

    vector_index = VectorStoreIndex.from_documents(docs)
    summary_index = SummaryIndex.from_documents(docs)

    os.makedirs(vector_index_dir, exist_ok=True)
    os.makedirs(summary_index_dir, exist_ok=True)

    vector_index.storage_context.persist(persist_dir=vector_index_dir)
    summary_index.storage_context.persist(persist_dir=summary_index_dir)

    return vector_index, summary_index