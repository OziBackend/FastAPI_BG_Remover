from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException, status
import os
import io
import asyncio
from PIL import Image, ImageOps
import uuid
from datetime import datetime
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import asyncio

from controller.controller import (
    remove_Background,
    readFile,
    clear_Folder,
    getTemplate,

    detect_face_and_crop_image
)

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

# Mount static files directory if needed
app.mount("/static", StaticFiles(directory="static"), name="static")

#Load RetinaFace model from InsightFace
model = FaceAnalysis(name="buffalo_l", root='C:/Users/muhammadannasasif/.insightface')
model.prepare(ctx_id=0) #set to -1 for automatic device (CPU/GPU)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

executor = ThreadPoolExecutor(max_workers=2)  # Create a thread pool

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI"}

@app.post("/remove_bg")
@app.post("/remove_bg/")
async def upload_image(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")  # Log the filename
    print(f"Content type: {file.content_type}")  # Log the content type
    try:
        # Create the uploads/outputs directory if it doesn't exist
        os.makedirs("uploads/outputs", exist_ok=True)

        # Read the uploaded file
        contents = await file.read()
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Convert the file to .webp format
        image = Image.open(io.BytesIO(contents))
        new_filename = f"{file.filename.rsplit('.', 1)[0]}_{unique_id}_{timestamp}.webp"
        webp_filename = f"{new_filename}"

        try:
            future = executor.submit(remove_Background, image, webp_filename)
            image_path = await asyncio.wrap_future(future)
            if not image_path:
                raise HTTPException(
                    status_code=500,
                    detail="Background removal failed: no image path returned"
                )
            return {"filename": image_path}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Processing failed: {e}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.get("/read_file")
async def read_file(request: Request):  # Accept the request parameter
    return readFile("uploads/outputs", request)

@app.delete("/clear_uploads")
async def clear_uploads():
    return clear_Folder("uploads/outputs")

#=========================================================

@app.post("/detect_face_and_crop_image")
@app.post("/detect_face_and_crop_image/")
async def crop_image(
    file: UploadFile = File(...),
    width: float = Form(None),
    height: float = Form(None),
    unit: str = Form(None),  # 'px', 'inch', or 'mm',
    dpi: float = Form(None),  
):
    print(f"Received file: {file.filename}")  # Log the filename
    print(f"Content type: {file.content_type}")  # Log the content type
    print(f"Width: {width}, Height: {height}, Unit: {unit}")
    if unit is not None:
        unit = unit.strip("'").strip('"')
    print(f"DPI: {dpi}")
    try:
        # Create the uploads/cropped_output directory if it doesn't exist
        os.makedirs("uploads/cropped_output", exist_ok=True)
        os.makedirs("uploads/resized_output", exist_ok=True)

        # Read the uploaded file
        contents = await file.read()
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Prepare file paths
        base_name = f"{file.filename.rsplit('.', 1)[0]}_{unique_id}_{timestamp}"
        cropped_filename = f"{base_name}.webp"
        cropped_filepath = f"uploads/cropped_output/{cropped_filename}"

        # Load image as numpy array (OpenCV expects BGR)
        image_stream = io.BytesIO(contents)
        pil_image = Image.open(image_stream).convert('RGB')
        pil_image = ImageOps.exif_transpose(pil_image)
        image_np = np.array(pil_image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            detect_face_and_crop_image,
            image_cv, width, height, unit, dpi, cropped_filepath, model, base_name
            )
        return result

    except HTTPException:
        raise
    except Exception as e:
        print("error", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.get("/read_cropped_file")
async def read_cropped_file(request: Request):
    return readFile("uploads/cropped_output", request)

@app.delete("/clear_cropped_uploads")
async def clear_cropped_uploads():
    return clear_Folder("uploads/cropped_output")


#=========================================================
#=========================================================

@app.get("/template/{template_name}", response_class=HTMLResponse)
async def get_template(request: Request, template_name: str):
    return getTemplate(template_name)