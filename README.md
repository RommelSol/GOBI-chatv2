
# GOBI · Chatbot Documental (Streamlit)

Demo pública pensada desplegado en **share.streamlit.io**. Interfaz estilo chat, respuestas limitadas a 300 palabras y enlaces a las fuentes cuando existan.

## Estructura
```
streamlit_app.py          # punto de entrada
app/
  ├─ config.py            # parámetros (límite de palabras, rutas)
  └─ retrieval.py         # índice TF-IDF sobre CSV/PDF/TXT
data/
  ├─ docs/                # PDFs (con texto) o TXT
  └─ knowledge/           # CSV (chatbot_dato1.csv)
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Esto lógicamente debe ser dentro de un venv (virtual enviroment).

## Despliegue en Streamlit Cloud
1. Sube este repo a GitHub.
2. En Streamlit Cloud, elige este repo y **Main file**: `streamlit_app.py`.
3. (Opcional) Define `PUBLIC_DOC_BASE_URL` dentro de `app/config.py` para que los enlaces a PDFs apunten a tu repo público.

## Notas
- Esta demo no realiza OCR. Para PDFs escaneados, sube TXT por ahora.
- Si no hay documentos o KB, GOBI mostrará mensajes genéricos.
- Aún no cuenta con botones seleccionables en donde se pueda añadir opciones predefinidas de respuestas.
