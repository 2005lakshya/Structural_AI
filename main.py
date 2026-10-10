import os

from backend.main import app


if __name__ == "__main__":
    import uvicorn

    # Port is configurable because 8000 is often taken by other local services.
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get("STRUCTURALAI_PORT", "8000")), reload=True)
