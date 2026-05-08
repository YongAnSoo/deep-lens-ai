# DeepLens AI

Explainable Deepfake Detection WebApp.

This repository contains:

- `backend/`: FastAPI backend for video upload and model inference.
- `backend/deepfake_core/`: the integrated Deepfake model core from `deepfake_project_without_dataset/src`.
- `backend/models/`: trained model weights, fusion calibrators, and MediaPipe model file.
- `frontend/`: React + Vite frontend for upload, result display, and explainability pages.

## Backend local run

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## Frontend local run

```powershell
cd frontend
npm install
npm run dev
```

Create `frontend/.env` if needed:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Render deploy

Use either the included `render.yaml` as a Blueprint, or create a Web Service manually:

```text
Root Directory: backend
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Health check:

```text
https://your-render-service.onrender.com/health
```

## Vercel deploy

```text
Root Directory: frontend
Framework: Vite
Build Command: npm run build
Output Directory: dist
```

Environment variable:

```text
VITE_API_BASE_URL=https://your-render-service.onrender.com
```

## LLM configuration

The backend works without API keys by using the local template fallback in `deepfake_core/llm.py`.

For OpenAI:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
```

For DeepSeek:

```text
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Do not commit API keys into source code.

## Important notes

- `backend/models/best_model.pth` is included because the backend needs it during Render deployment.
- `backend/uploads/` and `backend/outputs/` are runtime directories and are ignored by Git.
- Render free instances may cold start slowly and model inference may be slow on CPU.
