import cv2
import numpy as np
import pytesseract as pt
import os
import re

# Configure Tesseract path (update with your Tesseract installation path)
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# YOLO Model Configuration
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
MODEL_PATH = './static/models/best.onnx'
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_detections(img):
    # Convert image to YOLO input format
    image = img.copy()
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Create blob and run inference
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_suppression(input_image, detections):
    # Filter detections based on confidence
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])
                confidences.append(confidence)

    # Convert to numpy arrays
    boxes_np = np.array(boxes, dtype=np.int32)  # Changed to int32
    confidences_np = np.array(confidences, dtype=np.float32)

    # Handle empty detections
    if len(boxes_np) == 0:
        return boxes_np, confidences_np, np.array([])

    # Handle NMSBoxes return type differences
    try:
        index = cv2.dnn.NMSBoxes(boxes_np.tolist(), confidences_np.tolist(), 0.25, 0.45)
        
        # For OpenCV versions that return tuple
        if isinstance(index, tuple):
            index = index[0]
            
        # Flatten the array
        index = index.flatten()
    except Exception as e:
        print(f"NMS Error: {str(e)}")
        index = np.array([])

    return boxes_np, confidences_np, index


def extract_text(image, bbox):
    x, y, w, h = map(int, bbox)
    roi = image[y:y+h, x:x+w]
    if roi.size == 0: return ''
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    magic_color = apply_brightness_contrast(gray, 40, 70)
    text = pt.image_to_string(magic_color, lang='eng', config='--psm 6')
    
    # 🧼 Clean the text to remove unwanted special characters
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())  # Keep only A-Z and 0-9

    return cleaned_text.strip()


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    # Adjust brightness and contrast
    if brightness != 0:
        shadow = brightness
        highlight = 255
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def draw_results(image, boxes_np, confidences_np, index):
    text_list = []
    for ind in index:
        # Convert coordinates to integers
        x = int(boxes_np[ind][0])
        y = int(boxes_np[ind][1])
        w = int(boxes_np[ind][2])
        h = int(boxes_np[ind][3])
        
        # Draw rectangles with integer coordinates
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y-30), (x+w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y+h), (x+w, y+h+30), (0, 0, 0), -1)

        license_text = extract_text(image, (x, y, w, h))
        cv2.putText(image, license_text, (x, y+h+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        text_list.append(license_text)

    return image, text_list

def yolo_predictions(img):
    input_image, detections = get_detections(img)
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)
    
    if len(index) == 0:
        return img, []  # Return original image if no detections
    
    return draw_results(img, boxes_np, confidences_np, index)

# Main function for Flask integration
def object_detection(image_path, filename):
    # Read and process image
    image = cv2.imread(image_path)
    if image is None:
        return []
        
    # Run detection pipeline
    result_img, text_list = yolo_predictions(image)
    
    # Save result image
    output_path = os.path.join('./static/predict', filename)
    cv2.imwrite(output_path, result_img)
    
    return text_list