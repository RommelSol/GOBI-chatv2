import os
from typing import Optional

def _read_pdf(path: str) -> str:
    """
    Extrae texto de PDF intentando en este orden:
      1) pdfplumber (PDFs digitales)
      2) PyMuPDF/fitz (si está instalado)
      3) (opcional) OCR con Tesseract si todo lo anterior dio vacío
    Devuelve cadena (puede ser "").
    """
    text = ""

    # 1) Intento con pdfplumber
    try:
        import pdfplumber  # pip install pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"[WARN] pdfplumber falló en {path}: {e}")

    # 2) Si aún no hay texto, intento con fitz (si está disponible)
    if not text.strip():
        try:
            import fitz  # PyMuPDF (pip install pymupdf)
            with fitz.open(path) as doc:
                for page in doc:
                    # 'text' suele ir bien, 'blocks' a veces capta mejor tablas
                    text += (page.get_text("text") or "") + "\n"
        except ImportError:
            print(f"[INFO] PyMuPDF (fitz) no instalado; omito fallback para {path}")
        except Exception as e:
            print(f"[WARN] PyMuPDF falló en {path}: {e}")

    # 3) (Opcional) OCR si sigue vacío y quieres habilitarlo
    #    Actívalo poniendo la variable de entorno GOBI_ENABLE_OCR=1
    if not text.strip() and os.environ.get("GOBI_ENABLE_OCR") == "1":
        try:
            from pdf2image import convert_from_path   # pip install pdf2image pillow
            import pytesseract                        # pip install pytesseract
            # Nota: en Windows debes instalar Tesseract en el sistema:
            # https://github.com/UB-Mannheim/tesseract/wiki
            images = convert_from_path(path, dpi=200)
            ocr_texts = []
            for img in images:
                ocr_texts.append(pytesseract.image_to_string(img, lang="spa"))
            text = "\n".join(ocr_texts)
            if not text.strip():
                print(f"[INFO] OCR no obtuvo texto útil en {path}")
        except Exception as e:
            print(f"[WARN] OCR no disponible o falló en {path}: {e}")

    return text or ""

def load_text_from_path(path: str) -> Optional[str]:
    """Lee texto de PDF, DOCX, DOC y TXT. Devuelve None si falla."""
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".pdf":
            text = _read_pdf(path)
            return text if text.strip() else ""  # devuelve "" si no hubo texto

        elif ext in (".docx", ".doc"):
            import docx  # pip install python-docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)

        else:
            print(f"[WARN] Formato no soportado: {path}")
            return None
    except Exception as e:
        print(f"[ERROR] No se pudo leer {path}: {e}")
        return None