
import os
import pdfplumber

def load_text_from_pdf(path: str, max_pages: int = 40) -> str:
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                t = page.extract_text() or ""
                text_parts.append(t)
    except Exception as e:
        print(f"[WARN] PDF falló: {path} -> {e}")
        return ""
    return "\n".join(text_parts)

def load_text_from_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] TXT falló: {path} -> {e}")
        return ""

def load_text_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_text_from_pdf(path)
    if ext == ".txt":
        return load_text_from_txt(path)
    # Nota: .docx/.doc no se incluyen para simplificar deploy en Streamlit Cloud
    return ""
