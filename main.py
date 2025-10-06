from fastapi import FastAPI, UploadFile, File
from pipeline import run_pipeline


app = FastAPI(title="AutoSentinel", version="0.1")


@app.post("/recognize_plate")
async def recognize_plate(file: UploadFile = File(...)):
    data = await file.read()
    result = run_pipeline(data)
    return result.model_dump()
