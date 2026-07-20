from fastapi import FastAPI, Query
from transformers import pipeline

app = FastAPI(title="Mi API de Practica - FastAPI + Hugging Face")

#Cargamos los pipelines de Hugging Face UNA sola vez al arrancar el servidor (así no se recargan en cada petición, lo cual sería muy lento)
sentiment_pipeline = pipeline("sentiment-analysis")
summarization_pipeline = pipeline("summarization")


@app.get("/")
def read_root():
    """Endpoint de bienvenida."""
    return {"mensaje": "Bienvenido a mi API de practica con FastAPI y Hugging Face"}


@app.get("/health")
def health_check():
    """Endpoint para comprobar que el servicio esta activo."""
    return {"status": "ok"}


@app.get("/sentiment")
def analyze_sentiment(text: str = Query(..., description="Texto a analizar")):
    """Analiza el sentimiento (positivo/negativo) de un texto usando un pipeline de HF."""
    result = sentiment_pipeline(text)
    return {"input": text, "result": result}


@app.get("/summarize")
def summarize_text(text: str = Query(..., description="Texto largo a resumir")):
    """Genera un resumen de un texto largo usando un pipeline de HF."""
    result = summarization_pipeline(text, max_length=60, min_length=10, do_sample=False)
    return {"input": text, "summary": result}


@app.get("/info")
def get_info():
    """Informacion general sobre la API."""
    return {
        "nombre": "Practica FastAPI + HuggingFace",
        "endpoints": ["/", "/health", "/sentiment", "/summarize", "/info"]
    }
