from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from predict import predict_image
from result_analysis import analyze_results
import os

app = FastAPI()

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Save the uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    result = predict_image(filepath)

    return {"result": result, "image_path": filepath}

@app.get("/analysis")
def analysis():
    analysis_results = analyze_results()
    return JSONResponse({
        "accuracy": analysis_results["accuracy"],
        "confusion_matrix": "/analysis/confusion_matrix",
        "loss_curve": "/analysis/loss_curve",
        "confusion_matrix_explanation": analysis_results["confusion_matrix_explanation"],
        "loss_curve_explanation": analysis_results["loss_curve_explanation"]
    })

@app.get("/analysis/confusion_matrix")
def get_confusion_matrix():
    return FileResponse("static/analysis/confusion_matrix.png")

@app.get("/analysis/loss_curve")
def get_loss_curve():
    return FileResponse("static/analysis/loss_curve.png")

@app.get("/analysis/explanations")
def get_explanations():
    analysis_results = analyze_results()
    return JSONResponse({
        "confusion_matrix": analysis_results.get("confusion_matrix_explanation", "No explanation available."),
        "loss_curve": analysis_results.get("loss_curve_explanation", "No explanation available.")
    })