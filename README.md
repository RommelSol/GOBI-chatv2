
# GOBI – Streamlit Demo (ZIP listo para Share)

Esta demo permite desplegar un chatbot documental en Streamlit Cloud o Hugging Face Spaces.
Luego podrás subir manualmente tus **PDFs/TXT** a `data/docs/` y tu **CSV** a `data/knowledge/chatbot_dato1.csv`.

## Ejecutar en local
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy en Streamlit Cloud
- Sube este ZIP a un repo público.
- En share.streamlit.io selecciona el repo y el archivo principal `streamlit_app.py`.
- (Opcional) Define `PUBLIC_DOC_BASE_URL` en `app/config.py` si subes PDFs al repo para links clicables.

## Estructura
```
streamlit_app.py          # punto de entrada
app/
  config.py               # rutas y parámetros
  document_reader.py      # lectura PDF/TXT segura
  gobi_core.py            # indexado TF-IDF y respuesta con límite de 300
  knowledge.py            # CSV de KB (pregunta, respuesta, link)
  emotion.py              # tono empático básico
data/
  docs/                   # coloca aquí tus PDFs/TXT
  knowledge/
    chatbot_dato1.csv     # coloca aquí tu CSV (opcional)
```
