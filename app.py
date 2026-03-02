# import standard libraries
import os
import sys
import hashlib
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio


# Import project modules
from src.config import load_config
from src.models import setup_models
from src.indexing import build_or_load_indexes_from_dir
from src.engines import build_engines_and_tools
from src.routing import route_tool_name
from src.agentic import run_agentic_detail

# Fix Python path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

cache_dir = os.path.join(root_dir, ".cache")

# App setup
app = FastAPI(title="Agentic RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load config once at startup
cfg = load_config("config.yaml")
setup_models(cfg)

# In-memory session store  { session_id -> engines/tools }
sessions: dict[str, dict] = {}


# Helpers
def hash_files(file_bytes_list: list[bytes]) -> str:
    h = hashlib.sha256()
    for b in file_bytes_list:
        h.update(b)
    return h.hexdigest()[:16]


def save_uploads(file_list: list[tuple[str, bytes]], target_dir: str) -> list[str]:
    os.makedirs(target_dir, exist_ok=True)
    saved = []
    for name, data in file_list:
        out_path = os.path.join(target_dir, name)
        with open(out_path, "wb") as w:
            w.write(data)
        saved.append(out_path)
    return saved


# Routes
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    Accept one or more PDF uploads, build (or load cached) indexes,
    return a session_id to be used for subsequent queries.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Read all bytes first
    file_data: list[tuple[str, bytes]] = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF.")
        content = await f.read()
        file_data.append((f.filename, content))

    # Deterministic hash so same file set reuses cache
    docset_id = hash_files([data for _, data in file_data])

    uploads_dir = os.path.join(cache_dir, "uploads", docset_id)
    vector_dir  = os.path.join(cache_dir, "indexes", docset_id, "vector")
    summary_dir = os.path.join(cache_dir, "indexes", docset_id, "summary")

    # Fresh upload folder
    if os.path.exists(uploads_dir):
        shutil.rmtree(uploads_dir)

    saved = save_uploads(file_data, uploads_dir)
    pdfs  = [p for p in saved if p.lower().endswith(".pdf")]

    if not pdfs:
        raise HTTPException(status_code=500, detail="No PDFs were saved correctly.")

    # Build / load indexes (blocking — wrap in thread for real prod)
    vector_index, summary_index = build_or_load_indexes_from_dir(
        data_dir=uploads_dir,
        vector_index_dir=vector_dir,
        summary_index_dir=summary_dir,
    )

    vector_engine, summary_engine, tools, selector = build_engines_and_tools(
        cfg, vector_index, summary_index
    )

    # Store in session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "vector_engine":  vector_engine,
        "summary_engine": summary_engine,
        "tools":          tools,
        "selector":       selector,
        "filenames":      [f.filename for f in files],
    }

    return {
        "session_id": session_id,
        "files":      [os.path.basename(p) for p in pdfs],
        "message":    "Index ready.",
    }


class QueryRequest(BaseModel):
    session_id: str
    query: str


@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    """
    Route the query and return the answer.
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload files first.")

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty.")

    sess = sessions[req.session_id]
    vector_engine  = sess["vector_engine"]
    summary_engine = sess["summary_engine"]
    tools          = sess["tools"]
    selector       = sess["selector"]

    tool_name = route_tool_name(selector, tools, req.query)

    if tool_name == "summarize":
        resp   = summary_engine.query(req.query)
        result = getattr(resp, "response", str(resp))
        return {
            "type":   "summary",
            "router": tool_name,
            "answer": result,
            "plan":   [],
            "sources": [],
        }
    else:
        plan, answer, sources = run_agentic_detail(cfg, vector_engine, req.query)
        return {
            "type":    "detail",
            "router":  tool_name,
            "answer":  answer,
            "plan":    plan,
            "sources": sorted(sources),
        }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": "Session cleared."}


@app.get("/health")
async def health():
    return {"status": "ok"}