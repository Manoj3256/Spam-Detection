from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import joblib


models={}
@asynccontextmanager
async def lifespan(app:FastAPI):
    #loading the models
    models["vectorizer"]=joblib.load("models/vectorizer_file.pkl")
    models["predictor"]=joblib.load("models/model_file.pkl")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan,
              title="Spam detection API",
              version="1.0",
              description="Detecting spam messages using SVC And TfidfVectorizer")

class PredictResponse(BaseModel):
    text_input:str
    prediction:str
    spam_probability:str

class Input(BaseModel):
    text:str
@app.get("/health")
async def health():
    return {
        "status":"ready",
        "model_loaded":"predictor" in models
    }
@app.post("/predict",response_model=PredictResponse)
async def prediction(message:Input):
    if not message.text.strip():
        raise HTTPException(status_code=422, detail="Input text is **empty**")
    vectorized_message=models['vectorizer'].transform([message.text])
    final_result=models["predictor"].predict(vectorized_message)[0].item() 
    result_percentage=models["predictor"].predict_proba(vectorized_message)[0]

    return {"text_input":message.text,
            "prediction": "Ham" if final_result==0 else "Spam",
            "spam_probability": f"{round(result_percentage[1]*100,2)}%"}
    