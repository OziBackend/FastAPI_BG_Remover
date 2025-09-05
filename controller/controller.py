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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
#===========================================================


def clear_Folder(foldername):
    try:
        for filename in os.listdir(foldername):
            file_path = os.path.join(foldername, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return {"message": "Folder cleared successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    

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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected."
            )
        
        (x1, y1, x2, y2) = faces[0].bbox.astype(int)
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
        
        # Step 1: First crop with face-centered padding
        face_w = x2 - x1
        face_h = y2 - y1
        padding_ratio = 0.2  # 40% of face size as padding on each side
        pad_w = int(face_w * padding_ratio)
        pad_h = int(face_h * padding_ratio)

        # Calculate initial crop bounds with padding
        crop_x1 = max(0, x1 - pad_w)
        crop_y1 = max(0, y1 - pad_h)
        crop_x2 = min(image.shape[1], x2 + pad_w)
        crop_y2 = min(image.shape[0], y2 + pad_h)

        # Ensure the crop is at least as big as the face box
        if crop_x2 - crop_x1 < face_w:
            crop_x2 = min(image.shape[1], crop_x1 + face_w)
        if crop_y2 - crop_y1 < face_h:
            crop_y2 = min(image.shape[0], crop_y1 + face_h)

        # Apply the initial face-centered crop
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Step 2: If user provided dimensions, apply additional cropping
        if width is not None and height is not None and unit is not None:
            # Calculate target dimensions based on unit and DPI
            if unit == 'px':
                target_w = int(width)
                target_h = int(height)
            elif unit == 'inch':
                target_w = int(width * dpi)
                target_h = int(height * dpi)
            elif unit == 'mm':
                target_w = int((width / 25.4) * dpi)
                target_h = int((height / 25.4) * dpi)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid unit. Use 'px', 'inch', or 'mm'."
                )
            
            # Get dimensions of the face-centered cropped image
            h, w = cropped_image.shape[:2]
            
            # Calculate the face center position relative to the cropped image
            face_center_in_crop_x = face_center_x - crop_x1
            face_center_in_crop_y = face_center_y - crop_y1
            
            # Check if the face-centered crop is large enough for target dimensions
            if w < target_w or h < target_h:
                # Need to expand the initial crop to accommodate target dimensions
                # Calculate how much more padding we need
                extra_pad_w = max(0, target_w - w)
                extra_pad_h = max(0, target_h - h)
                
                # Store the old crop bounds to update face center position
                old_crop_x1 = crop_x1
                old_crop_y1 = crop_y1
                
                # Expand the crop bounds in the original image
                crop_x1 = max(0, crop_x1 - extra_pad_w // 2)
                crop_y1 = max(0, crop_y1 - extra_pad_h // 2)
                crop_x2 = min(image.shape[1], crop_x2 + extra_pad_w // 2)
                crop_y2 = min(image.shape[0], crop_y1 + target_h)
                
                # Re-apply the expanded crop
                cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
                h, w = cropped_image.shape[:2]
                
                # Update face center position for the new expanded crop
                face_center_in_crop_x = face_center_x - crop_x1
                face_center_in_crop_y = face_center_y - crop_y1
            
            # Use the actual face center position instead of geometric center
            center_x = face_center_in_crop_x
            center_y = face_center_in_crop_y
            
            # Ensure the face center is within bounds
            center_x = max(0, min(w - 1, center_x))
            center_y = max(0, min(h - 1, center_y))
            
            print(f"Original face center: ({face_center_x}, {face_center_y})")
            print(f"Face center in crop: ({center_x}, {center_y})")
            print(f"Crop dimensions: {w} x {h}")
            
            # Calculate target aspect ratio
            target_aspect_ratio = target_w / target_h
            current_aspect_ratio = w / h
            
            # Determine the best crop size that maintains aspect ratio
            if current_aspect_ratio > target_aspect_ratio:
                # Image is wider than target - crop width to match target aspect ratio
                crop_w = int(h * target_aspect_ratio)
                crop_h = h
            else:
                # Image is taller than target - crop height to match target aspect ratio
                crop_w = w
                crop_h = int(w / target_aspect_ratio)
            
            # Make sure the crop dimensions don't exceed available dimensions
            crop_w = min(crop_w, w)
            crop_h = min(crop_h, h)
            
            # Calculate new crop bounds centered on the face
            new_crop_x1 = center_x - crop_w // 2
            new_crop_y1 = center_y - crop_h // 2
            new_crop_x2 = new_crop_x1 + crop_w
            new_crop_y2 = new_crop_y1 + crop_h
            
            # Ensure the crop bounds are within the image
            if new_crop_x1 < 0:
                new_crop_x1 = 0
                new_crop_x2 = crop_w
            elif new_crop_x2 > w:
                new_crop_x2 = w
                new_crop_x1 = w - crop_w
                
            if new_crop_y1 < 0:
                new_crop_y1 = 0
                new_crop_y2 = crop_h
            elif new_crop_y2 > h:
                new_crop_y2 = h
                new_crop_y1 = h - crop_h
            
            # Apply the final dimension-based crop
            cropped_image = cropped_image[new_crop_y1:new_crop_y2, new_crop_x1:new_crop_x2]
            
            # Now resize to exact target dimensions (this won't distort since aspect ratio matches)
            final_h, final_w = cropped_image.shape[:2]
            print(f"Requested dimensions: {target_w} x {target_h}")
            print(f"Cropped dimensions: {final_w} x {final_h}")
            print(f"Target aspect ratio: {target_aspect_ratio:.3f}, Crop aspect ratio: {final_w/final_h:.3f}")
            
            # Resize to exact target dimensions (no distortion since aspect ratios match)
            cropped_image = cv2.resize(cropped_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            print(f"Final dimensions after resize: {target_w} x {target_h}")

        if cropped_image is not None:
            # Save the cropped image
            cv2.imwrite(filePath, cropped_image)
            filename = filePath.split("/")[-1]
            image_url = f"http://{index.IP}:{index.PORT}/read_cropped_file?filename={filename}"
            return {"filename": image_url}
        else:
            print("Cropping Failed Error")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cropping failed."
            )
    except Exception as e:
        print("Error in Function", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        

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




