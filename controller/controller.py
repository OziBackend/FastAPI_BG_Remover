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

#===========================================================

def detect_face_and_crop_image(image, width, height, unit, dpi, filePath, model, base_name, ):
    try:
        faces = model.get(image)
        if not faces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected."
            )
        # Extract bounding box coordinates
        (x1, y1, x2, y2) = faces[0].bbox.astype(int)

        # Create a copy of the original image to draw the bounding box on
        image_with_bbox = image.copy()

        # Draw the bounding box (green color, 2 pixels thick)
        cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Define the output path for the image with bounding box
        bbox_filename = f"{base_name}_bbox.jpg"
        bbox_filepath = f"uploads/outputs/{bbox_filename}"

        # Save the image with the bounding box
        cv2.imwrite(bbox_filepath, image_with_bbox)
        
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
        
        # Step 1: First crop with face-centered padding
        face_w = x2 - x1
        face_h = y2 - y1
        padding_ratio = 0.6  # 40% of face size as padding on each side
        pad_w = int(face_w * padding_ratio)
        pad_h = int(face_h * padding_ratio)
        print("pad_w", pad_w)
        print("pad_h", pad_h)

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
        
        if width is not None and height is not None and unit is not None:
            # Convert user dimensions to pixels
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

            # Step 1: get original dimensions
            h, w = cropped_image.shape[:2]
            aspect_ratio_input = w / h
            aspect_ratio_target = target_w / target_h

            # Step 2: scale so smaller side fits, larger side will overflow
            if aspect_ratio_input > aspect_ratio_target:
                # Wider → fit height, overflow width
                new_h = target_h
                new_w = int(aspect_ratio_input * new_h)
            else:
                # Taller → fit width, overflow height
                new_w = target_w
                new_h = int(new_w / aspect_ratio_input)

            resized = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Step 3: center crop to target size
            x_start = (new_w - target_w) // 2
            y_start = (new_h - target_h) // 2
            cropped_image = resized[y_start:y_start + target_h, x_start:x_start + target_w]

            print(f"Final cropped+resized dimensions: {cropped_image.shape[1]} x {cropped_image.shape[0]}")
            print(f"Final resized dimensions: {target_w} x {target_h}")


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




