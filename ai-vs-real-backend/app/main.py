
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from .model import ClassificationModel, CaptioningModel
from .utils import preprocess_for_classification

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classification_model = ClassificationModel("best_model.pth")
captioning_model = CaptioningModel()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {str(e)}"})

    tensor = preprocess_for_classification(image)
    is_real, confidence = classification_model.predict(tensor)
    description = captioning_model.generate_caption(image)

    return {"is_real": is_real, "confidence": confidence, "description": description}