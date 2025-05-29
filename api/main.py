from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import numpy as np
import torch
from project_name.data.preprocess import preprocess

from project_name.models.CNN import AudioCNN
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Bat Audio Classification API",
    description="""
    This API classifies bat audio recordings. Upload a .wav file and receive the predicted bat species.
    The API handles all preprocessing (normalization, resizing, grayscale conversion) internally.
    
    Error Handling:
    - Returns HTTP 400 for invalid files or unsupported formats.
    - Returns HTTP 500 for internal errors.
    """
)

# Allow CORS for local development (Swagger UI, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Pipistrellus pipistrellus", "Myotis daubentonii"]

# Load model (for now)
def load_model():
    num_classes = len(CLASS_NAMES)
    learning_rate = 0.001
    model = AudioCNN(num_classes=num_classes, learning_rate=learning_rate, number_of_epochs=5)
    # model.load_state_dict(torch.load('path_to_model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float

@app.post("/predict/", response_model=PredictionResponse, summary="Classify a bat audio file", tags=["Prediction"])
async def predict(
    file: UploadFile = File(
        default=None,
        description="Optional .wav file. If not provided, the built-in sample will be used."
    )
):
    """
    Upload a .wav audio file. If no file is provided, the built-in sample will be used.
    Only .wav files are accepted.
    
    **Request:**
    - file: .wav audio file (multipart/form-data, optional)
    
    **Response:**
    - class_name: Predicted bat species (string)
    - confidence: Model confidence (float, 0-1)
    """
    import tempfile
    sample_path = os.path.join("data", "sample", "XC912090 - Gewone dwergvleermuis - Pipistrellus pipistrellus.wav")
    use_temp = False
    audio_path = None
    try:
        # Treat empty file or empty filename as no file provided
        if file is not None and getattr(file, 'filename', None) and file.filename.strip():
            if not file.filename.lower().endswith('.wav'):
                raise HTTPException(status_code=400, detail="Only .wav files are supported.")
            contents = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(contents)
                audio_path = tmp.name
            use_temp = True
        else:
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail="Sample file not found.")
            audio_path = sample_path
            use_temp = False
        try:
            # Inline prediction logic (was predict_from_path)
            slices, sr = preprocess(audio_path)
            if not slices or len(slices) == 0:
                raise ValueError("Audio could not be processed.")
            all_probs = []
            for x in slices:
                x = (x - np.mean(x)) / (np.std(x) + 1e-8)
                x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    all_probs.append(probs)
            all_probs = np.array(all_probs)
            avg_probs = np.mean(all_probs, axis=0)
            pred_idx = int(np.argmax(avg_probs))
            confidence = float(avg_probs[pred_idx])
            class_name = CLASS_NAMES[pred_idx]
        except ValueError as ve:
            if use_temp and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            raise HTTPException(status_code=400, detail=str(ve))
        if use_temp and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        return PredictionResponse(class_name=class_name, confidence=confidence)
    except HTTPException:
        raise
    except Exception as e:
        if use_temp and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/", summary="API Home", tags=["Info"])
def root():
    """
    Welcome, test test test
    """
    return {"message": "Welcome to the Bat Audio Classification API. Use /predict/ to classify bat audio files."}
