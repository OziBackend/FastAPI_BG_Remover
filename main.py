from fastapi import FastAPI, UploadFile, File, Request
import os
import io
from PIL import Image
import uuid
from datetime import datetime
from fastapi.responses import FileResponse
import rembg
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

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
        return FileResponse(file_path)  # Serve the file if it exists
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