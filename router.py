import os
from rag_pipeline import run_rag
from structured_pipeline import run_structured

def route_document(file, question):
    file_type = file.name.split(".")[-1]

    if file_type in ["pdf", "txt", "docx"]:
        return run_rag(file, question)

    elif file_type in ["csv", "xlsx"]:
        return run_structured(file, question)

    else:
        return "Unsupported file type"