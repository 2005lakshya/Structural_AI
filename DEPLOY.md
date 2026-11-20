# Hosting StructuralAI

The app is two separately hosted pieces:

| Piece | Host | Why |
|---|---|---|
| FastAPI backend + models (`Dockerfile`) | Hugging Face Spaces, Docker SDK (free CPU: 2 vCPU, 16 GB RAM) | PyTorch + the Swin model need a few GB of RAM; serverless hosts can't run it |
| React frontend (`frontend-react/`) | Vercel (free) | Static build, reads the backend URL from `REACT_APP_API_URL` |

Both are free. A free Space sleeps after ~48 h without traffic and takes a minute or two to wake;
inference runs on CPU, so it is slower than on a GPU laptop.

## 1. Backend on Hugging Face Spaces

1. Create an account at https://huggingface.co and an access token with **write** scope
   (Settings → Access Tokens).
2. New Space → name it (e.g. `structuralai-api`) → SDK **Docker** → template **Blank** → Create.
3. Optional: Space → Settings → Variables and secrets → add secret `GEMINI_API_KEY`
   (and variable `GEMINI_MODEL`). Without it the report/chat use the rule-based fallback.
4. From the repo root, push the committed code (Git LFS must be installed):

   ```powershell
   .\deploy\push_to_hf.ps1 -Space <your-hf-username>/structuralai-api
   ```

   When git asks for a password, paste the access token.
5. Watch the build in the Space's **Logs** tab (first build ~5–10 min). When it says *Running*, check
   `https://<your-hf-username>-structuralai-api.hf.space/health` and `/docs`.

Re-run step 4 after each change you want deployed.

## 2. Frontend on Vercel

1. https://vercel.com → Add New → Project → import `2005lakshya/Structural_AI` from GitHub.
2. **Root Directory:** `frontend-react` (framework preset: Create React App is detected).
3. **Environment Variables:** `REACT_APP_API_URL` = `https://<your-hf-username>-structuralai-api.hf.space`
   (no trailing slash).
4. Deploy. Every push to `main` redeploys automatically. If you change `REACT_APP_API_URL`, redeploy,
   since it is baked in at build time.

## 3. Optional: Streamlit app (`app.py`)

Streamlit Community Cloud (https://share.streamlit.io) can deploy `app.py` straight from the GitHub
repo, but its ~1 GB RAM limit is tight for the PyTorch models. A second Docker Space is more reliable.

## Test the backend image locally

```powershell
docker build -t structuralai-backend .
docker run --rm -p 7860:7860 --env-file .env structuralai-backend
```

Then open http://localhost:7860/docs.

## Notes

- CORS allows all origins (`backend/main.py`). Once the Vercel URL is fixed you can restrict
  `allow_origins` to it.
- `requirements-server.txt` is the slim runtime set used by the image (CPU torch is installed
  separately). `scikit-learn` and `xgboost` are pinned to the versions the `models/ml/*.pkl` files
  were saved with; change them only together with re-saving those models.
- The backend writes generated files to `reports/` inside the container; they are lost on restart.
