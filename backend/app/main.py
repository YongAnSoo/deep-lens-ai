import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.inference import predict_video_file
from app.visual_utils import generate_fft_spectrum_image


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


app = FastAPI(title="DeepLens AI Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/")
def root():
    return {
        "message": "DeepLens AI backend is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Deepfake Detection API is running",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only mp4, mov, avi, mkv video files are supported",
        )

    task_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{task_id}{file_ext}"

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_result = predict_video_file(str(video_path))

        task_output_dir = OUTPUT_DIR / task_id
        fft_image_path = generate_fft_spectrum_image(
            video_path=str(video_path),
            output_dir=task_output_dir,
        )

        fft_spectrum_url = None
        if fft_image_path:
            fft_spectrum_url = f"/outputs/{task_id}/fft_spectrum.png"

        module_b = raw_result.get("module_b") or {}
        module_b_details = raw_result.get("module_b_details") or {}

        if fft_spectrum_url:
            module_b_details.setdefault("fft", {})
            module_b_details["fft"]["spectrum_url"] = fft_spectrum_url
            raw_result["module_b_details"] = module_b_details

        llm_analysis = (
            module_b.get("llm_analysis")
            or raw_result.get("llm_analysis")
            or raw_result.get("llm_explanation")
            or {}
        )

        if raw_result.get("error"):
            return {
                "task_id": task_id,
                "filename": file.filename,
                "label": raw_result.get("video_label", "unknown"),
                "fake_probability": raw_result.get("video_fake_prob", 0.5),
                "real_probability": raw_result.get("video_real_prob", 0.5),
                "risk_score": raw_result.get("risk_score"),
                "severity": raw_result.get("severity"),
                "frames_analyzed": raw_result.get("num_frames", 0),
                "num_faces": raw_result.get("num_faces", 0),
                "suspicious_frames": [],
                "gradcam_paths": raw_result.get("gradcam_paths", {}),
                "module_b": module_b,
                "module_b_details": module_b_details,
                "llm_analysis": llm_analysis,
                "explanation": {
                    "summary": "Detection process was not completed.",
                    "main_reasons": [
                        f"Error type: {raw_result.get('error')}",
                        "Possible reasons: failed frame extraction, no face detected, or low video quality.",
                    ],
                },
                "raw_result": raw_result,
            }

        fake_probability = float(raw_result.get("video_fake_prob", 0.5))
        real_probability = float(raw_result.get("video_real_prob", 1 - fake_probability))

        return {
            "task_id": task_id,
            "filename": file.filename,

            "label": raw_result.get("video_label", "unknown"),
            "fake_probability": round(fake_probability, 4),
            "real_probability": round(real_probability, 4),
            "risk_score": raw_result.get("risk_score"),
            "severity": raw_result.get("severity"),
            "confidence": raw_result.get("confidence"),
            "consistency": raw_result.get("consistency"),
            "frames_analyzed": raw_result.get("num_frames", 0),
            "num_faces": raw_result.get("num_faces", 0),
            "method": raw_result.get("method"),

            "suspicious_frames": raw_result.get("suspicious_frame_indices", []),
            "gradcam_paths": raw_result.get("gradcam_paths", {}),
            "module_b": module_b,
            "module_b_details": module_b_details,
            "llm_analysis": llm_analysis,

            "explanation": {
                "summary": llm_analysis.get(
                    "text",
                    "Deepfake detection completed with basic explainability output.",
                ),
                "main_reasons": [
                    f"Visual fake probability: {round(fake_probability, 4)}",
                    f"Risk score: {raw_result.get('risk_score')}",
                    f"Severity: {raw_result.get('severity')}",
                    f"Detected faces: {raw_result.get('num_faces', 0)}",
                    f"Fusion method: {raw_result.get('method')}",
                    f"LLM provider: {llm_analysis.get('provider', 'local_template')}",
                ],
            },

            "model_result": {
                "face_stats": raw_result.get("face_stats"),
                "module_b": module_b,
                "module_b_details": module_b_details,
                "llm_analysis": llm_analysis,
            },

            "raw_result": raw_result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
