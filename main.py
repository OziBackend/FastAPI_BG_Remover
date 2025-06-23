from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException, status, Form
import os
import io
from PIL import Image
import uuid
from datetime import datetime
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import rembg
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
import jwt  # Import the jwt library

import cv2
import numpy as np
from insightface.app import FaceAnalysis

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

def detect_faces(image, ratio=None):
    # image is already loaded (numpy array)
    if image is None:
        raise ValueError(f"Image data is None")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # If ratio is specified and faces are detected, create ratio-adjusted image
    ratio_image = None
    ratio_output_path = None
    
    if ratio is not None and len(faces) > 0:
        # Get the first detected face (assuming it's the main subject)
        (x, y, w, h) = faces[0]
        
        # Calculate center of the face
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Calculate the desired dimensions based on ratio
        # ratio = width / height
        if ratio > 1:  # Landscape
            # Use face height as reference
            target_height = int(h * 2)  # Make it 2x the face height
            target_width = int(target_height * ratio)
        else:  # Portrait or square
            # Use face width as reference
            target_width = int(w * 2)  # Make it 2x the face width
            target_height = int(target_width / ratio)
        
        # Calculate crop boundaries
        crop_x1 = max(0, face_center_x - target_width // 2)
        crop_y1 = max(0, face_center_y - target_height // 2)
        crop_x2 = min(image.shape[1], crop_x1 + target_width)
        crop_y2 = min(image.shape[0], crop_y1 + target_height)
        
        # Adjust if we hit image boundaries
        if crop_x2 - crop_x1 < target_width:
            if crop_x1 == 0:
                crop_x2 = min(image.shape[1], target_width)
            else:
                crop_x1 = max(0, image.shape[1] - target_width)
        
        if crop_y2 - crop_y1 < target_height:
            if crop_y1 == 0:
                crop_y2 = min(image.shape[0], target_height)
            else:
                crop_y1 = max(0, image.shape[0] - target_height)
        
        # Crop the image
        ratio_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return ratio_image

def detect_faces_retinaface(image, ratio=None):
    
    if image is None:
        print(f"Failed to load image")
        return None, []

    # Detect faces
    faces = model.get(image)

    if not faces:
        print("No faces detected.")
        return None, []

    # Draw bounding boxes and landmarks
    for face in faces:
        # box = face.bbox.astype(int)
        # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        if face.landmark is not None:
            for point in face.landmark:
                cv2.circle(image, tuple(point.astype(int)), 2, (0, 0, 255), -1)

    # Cropping based on the first face and specified ratio
    cropped_image = None
    if ratio is not None and ratio > 0:
        (x1, y1, x2, y2) = faces[0].bbox.astype(int)
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
        face_width = x2 - x1
        face_height = y2 - y1

        if ratio > 1:
            target_height = int(face_height * 2)
            target_width = int(target_height * ratio)
        else:
            target_width = int(face_width * 2)
            target_height = int(target_width / ratio)

        crop_x1 = max(0, face_center_x - target_width // 2)
        crop_y1 = max(0, face_center_y - target_height // 2)
        crop_x2 = min(image.shape[1], crop_x1 + target_width)
        crop_y2 = min(image.shape[0], crop_y1 + target_height)

        # Adjust bounds if too close to edge
        if crop_x2 - crop_x1 < target_width:
            if crop_x1 == 0:
                crop_x2 = min(image.shape[1], target_width)
            else:
                crop_x1 = max(0, image.shape[1] - target_width)

        if crop_y2 - crop_y1 < target_height:
            if crop_y1 == 0:
                crop_y2 = min(image.shape[0], target_height)
            else:
                crop_y1 = max(0, image.shape[0] - target_height)

        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped_image


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
        webp_filename = f"uploads/outputs/{new_filename}"

        # Use threading to remove background
        future = executor.submit(rembg.remove, image)
        output_image = future.result()  # Wait for the result

        output_image.save(webp_filename, 'WEBP')

        image_path = f"http://172.16.0.94:9000/read_file?filename={new_filename}"

        return {"filename": image_path}
    except Exception as e:
        return {"error": str(e)}  # Return the error message

@app.get("/read_file")
async def read_file(request: Request):  # Accept the request parameter
    file_name = request.query_params["filename"]  # Use the instance to read filename
    file_path = f"uploads/outputs/{file_name}"
    print(file_path)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="image/jpeg"
        )  # Serve the file if it exists
    else:
        return {"error": "File not found"}  # Return an error if the file does not exist

@app.delete("/clear_uploads")
async def clear_uploads():
    try:
        # Remove all files in the uploads/outputs directory
        folder_path = "uploads/outputs"
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
        return {"message": "Uploads folder cleared successfully."}
    except Exception as e:
        return {"error": str(e)}  # Return the error message

#=========================================================

@app.post("/detect_face_and_crop_image")
@app.post("/detect_face_and_crop_image/")
async def crop_image(
    file: UploadFile = File(...),
    ratio: float = Form(1.0)
):
    print(f"Received file: {file.filename}")  # Log the filename
    print(f"Content type: {file.content_type}")  # Log the content type
    print(f"Ratio: {ratio}")
    try:
        # Create the uploads/cropped_output directory if it doesn't exist
        os.makedirs("uploads/cropped_output", exist_ok=True)

        # Read the uploaded file
        contents = await file.read()
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Prepare file paths
        new_filename = f"{file.filename.rsplit('.', 1)[0]}_{unique_id}_{timestamp}.jpg"
        cropped_filepath = f"uploads/cropped_output/{new_filename}"

        # Load image as numpy array (OpenCV expects BGR)
        image_stream = io.BytesIO(contents)
        pil_image = Image.open(image_stream).convert('RGB')
        image_np = np.array(pil_image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Use detect_faces to crop based on face and ratio
        # cropped_image = detect_faces(image_cv, ratio)
        cropped_image = detect_faces_retinaface(image_cv, ratio)

        # Save the cropped image (detect_faces already saves, but ensure here for clarity)
        if cropped_image is not None:
            cv2.imwrite(cropped_filepath, cropped_image)
            image_url = f"http://172.16.0.94:9000/read_cropped_file?filename={new_filename}"
            return {"filename": image_url}
        else:
            return {"error": "No face detected or cropping failed."}
    except Exception as e:
        return {"error": str(e)}  # Return the error message

@app.get("/read_cropped_file")
async def read_cropped_file(request: Request):  # Accept the request parameter
    file_name = request.query_params["filename"]  # Use the instance to read filename
    file_path = f"uploads/cropped_output/{file_name}"
    print(file_path)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="image/jpeg"
        )  # Serve the file if it exists
    else:
        return {"error": "File not found"}  # Return an error if the file does not exist

@app.delete("/clear_cropped_uploads")
async def clear_cropped_uploads():
    try:
        # Remove all files in the uploads/outputs directory
        folder_path = "uploads/cropped_output"
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
        return {"message": "Cropped Uploads folder cleared successfully."}
    except Exception as e:
        return {"error": str(e)}  # Return the error message

#=========================================================

# Function to verify the token
def verify_token(token: str):
    if token != "your_secret_token":  # Replace with your actual token verification logic
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/authorize")
async def authorize_user(token: str = Depends(verify_token)):  # Use dependency injection
    return {"message": "User authorized successfully."}  # Return success message

@app.get("/check_headers")
def check_headers(request: Request):  # Accept the request parameter
    bearer_token = request.headers.get("Authorization")  # Get the Authorization header
    print(bearer_token)  # Log the Bearer token

    if bearer_token and bearer_token.startswith("Bearer "):
        token = bearer_token.split(" ")[1]  # Extract the token part
        print("TOKEN:= ", token)
        try:
            # Decode the token (replace 'your_secret_key' with your actual secret key)
            decoded_token = jwt.decode(token, 'OZI_Backend_BG_Removal', algorithms=["HS256"])
            print(decoded_token)  # Print the decoded token
        except jwt.ExpiredSignatureError:
            print("Token has expired")
            return {"error": "Token has expired"}
        except jwt.InvalidTokenError:
            print("Invalid token")
            return {"error": "Invalid token"}
    else:
        return {"error": "No Bearer token found"}

    return {"headers": "returned"}

@app.post("/create_token")
def create_token():  # Accept a username as input
    # Here you would typically validate the username and possibly check a password
    # For simplicity, we are just creating a token for any username provided
    token = jwt.encode({"sub": 'bg_app'}, 'OZI_Backend_BG_Removal', algorithm="HS256")  # Create the token
    return {"token": token}  # Return the generated token

@app.get("/template/{template_name}", response_class=HTMLResponse)
async def get_template(request: Request, template_name: str):
    try:
        # Return the template with the given name
        return templates.TemplateResponse(f"{template_name}.html", {"request": request})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template {template_name}.html not found"
        )