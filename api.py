from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
import uuid
import json
import asyncio
from typing import Optional
import pandas as pd
import gradio as gr
from main import run_pipeline, create_gradio_app

app = FastAPI(title="Data Pipeline Agent API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store job status
job_status = {}

@app.get("/")
async def root():
    return RedirectResponse(url="/gradio")  # Redirect root to Gradio UI

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and return a file ID"""
    # Save uploaded file to temp location
    file_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Store file info
    job_status[file_id] = {
        "status": "uploaded",
        "file_path": file_path,
        "temp_dir": temp_dir,
        "filename": file.filename
    }
    
    return {"file_id": file_id, "message": "File uploaded successfully"}

@app.post("/run_pipeline/{file_id}")
async def run_pipeline_api(file_id: str, target_column: str, background_tasks: BackgroundTasks):
    """Start the pipeline processing"""
    if file_id not in job_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Update job status
    job_status[file_id]["status"] = "processing"
    job_status[file_id]["target_column"] = target_column
    
    # Run pipeline in background
    background_tasks.add_task(process_pipeline, file_id)
    
    return {"file_id": file_id, "message": "Pipeline started"}

async def process_pipeline(file_id: str):
    """Process the pipeline in the background"""
    try:
        # Get file info
        file_info = job_status[file_id]
        file_path = file_info["file_path"]
        target_col = file_info["target_column"]
        
        # Update status
        job_status[file_id]["status"] = "processing"
        job_status[file_id]["step"] = "Reading file"
        
        # Run pipeline
        job_status[file_id]["step"] = "Running pipeline"
        html_report, model_path = run_pipeline(file_path, target_col)
        
        # Update status
        job_status[file_id]["status"] = "completed"
        job_status[file_id]["report"] = html_report
        job_status[file_id]["model_path"] = model_path
        
    except Exception as e:
        job_status[file_id]["status"] = "error"
        job_status[file_id]["error"] = str(e)

@app.get("/status/{file_id}")
async def get_status(file_id: str):
    """Get the status of a pipeline job"""
    if file_id not in job_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    return job_status[file_id]

@app.get("/report/{file_id}")
async def get_report(file_id: str):
    """Get the HTML report for a completed job"""
    if file_id not in job_status or job_status[file_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Report not found")
    
    return HTMLResponse(content=job_status[file_id]["report"])

@app.get("/download/{file_id}")
async def download_model(file_id: str):
    """Download the trained model"""
    if file_id not in job_status or job_status[file_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_path = job_status[file_id]["model_path"]
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        path=model_path,
        filename=os.path.basename(model_path),
        media_type="application/octet-stream"
    )

# Create static directory for web UI
if not os.path.exists("static"):
    os.makedirs("static")

    app.mount("/static", StaticFiles(directory="static"), name="static")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add these routes after your existing routes
@app.get("/favicon.ico")
async def favicon():
    """Handle favicon request"""
    return RedirectResponse(url="https://fastapi.tiangolo.com/img/favicon.png")

@app.get("/manifest.json")
async def get_manifest():
    """Serve full PWA manifest"""
    manifest_content = {
        "name": "Data Pipeline Agent",
        "short_name": "Data Pipeline",
        "description": "AI-powered data cleaning, visualization, and model training",
        "start_url": "/gradio",  # Point to Gradio path
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#3070f0",
        # Add icons if you have them (e.g., save PNGs to static/ and reference)
        # "icons": [{"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"}]
    }
    return Response(content=json.dumps(manifest_content), media_type="application/json")


@app.get("/sw.js")
async def get_service_worker():
    """Basic service worker to satisfy PWA fetches"""
    sw_content = """
    // Minimal service worker for PWA caching
    self.addEventListener('fetch', event => {
      event.respondWith(fetch(event.request));
    });
    """
    return Response(content=sw_content, media_type="application/javascript")


@app.get("/gradio/gradio_api/upload_progress")
async def upload_progress():
    """Handle upload progress requests"""
    return {"status": "completed", "progress": 100}

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Handle Chrome DevTools request"""
    return {"status": "ok"}

@app.get("/gradio/gradio_api/app_id")
async def app_id():
    """Handle app ID request"""
    return {"app_id": "data-pipeline-agent"}

# Get Gradio app and mount it with proper configuration
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(
    app, 
    gradio_app,  # Your Blocks instance
    path="/gradio",
    root_path="/gradio"  # Critical: Tells Gradio the base path for WS/APIs
)



# Set required attributes to prevent errors
gradio_app.max_file_size = 50 * 1024 * 1024  # 50MB
gradio_app.enable_queue = True

# Mount Gradio app with proper path
app.mount("/gradio", gradio_app.app, name="gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)