import cv2
import os
from fastapi import HTTPException, status

from fastapi.responses import FileResponse

import rembg
from environment import index

def remove_Background(image, filename):
    bg_removed = rembg.remove(image)
    bg_removed.save(filename, 'WEBP')

    image_path = f"http://{index.IP}:{index.PORT}/read_file?filename={filename}"
    return image_path

#===========================================================

def readFile(foldername, request):
    file_name = request.query_params["filename"]
    file_path = f"{foldername}/{file_name}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="image/jpeg"
        )
    else:
        return {"error": "File not found"}
    
#===========================================================


def clear_Folder(foldername):
    try:
        for filename in os.listdir(foldername):
            file_path = os.path.join(foldername, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return {"message": "Folder cleared successfully."}
    except Exception as e:
        return {"error": str(e)}
    

#===========================================================

    
def getTemplate(template_name):
    try:
        return FileResponse(
            f"templates/{template_name}.html",
            media_type="text/html"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template {template_name}.html not found"
        )

#===========================================================

def detect_face_and_crop_image(image, width, height, unit, dpi, filePath, model, base_name, ):
    try:
        faces = model.get(image)
        if not faces:
            return {"error": "No face detected."}
        
        (x1, y1, x2, y2) = faces[0].bbox.astype(int)
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
        
        crop_w = crop_h = None
        if width is not None and height is not None and unit is not None:
            if unit == 'px':
                crop_w = int(width)
                crop_h = int(height)
            elif unit == 'inch':
                crop_w = int(width * dpi)
                print(crop_w)
                crop_h = int(height * dpi)
                print(crop_h)
            elif unit == 'mm':
                crop_w = int((width / 25.4) * dpi)
                crop_h = int((height / 25.4) * dpi)
            else:
                return {"error": "Invalid unit. Use 'px', 'inch', or 'mm'."}
            
            crop_x1 = max(0, face_center_x - crop_w // 2)
            crop_y1 = max(0, face_center_y - crop_h // 2)
            crop_x2 = min(image.shape[1], crop_x1 + crop_w)
            crop_y2 = min(image.shape[0], crop_y1 + crop_h)
            
            # Adjust bounds if too close to edge
            if crop_x2 - crop_x1 < crop_w:
                if crop_x1 == 0:
                    crop_x2 = min(image.shape[1], crop_w)
                else:
                    crop_x1 = max(0, image.shape[1] - crop_w)
            if crop_y2 - crop_y1 < crop_h:
                if crop_y1 == 0:
                    crop_y2 = min(image.shape[0], crop_h)
                else:
                    crop_y1 = max(0, image.shape[0] - crop_h)

            cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

            if cropped_image is not None:
                bg_removed = rembg.remove(cropped_image)
                # Save the cropped image
                cv2.imwrite(filePath, bg_removed)

                
                filename = filePath.split("/")[-1]
                image_url = f"http://{index.IP}:{index.PORT}/read_cropped_file?filename={filename}"
                return {"filename": image_url}
            else:
                return {"error": "Cropping failed."}
    except Exception as e:
        return {"error": str(e)}
        

#===========================================================

def resize_and_save_image(image, output_dir, base_name, resolutions):
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape[:2]
    target_w, target_h = resolutions
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    output_path = os.path.join(output_dir, f"{base_name}.jpg")
    cv2.imwrite(output_path, resized_image)
    return output_path




