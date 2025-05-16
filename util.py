import easyocr
import re
import cv2
import numpy as np
from PIL import Image

# Initialize the reader with multiple languages to improve detection capability
reader = easyocr.Reader(['en', 'fr'], gpu=False)  # Add more languages if needed

# Helper functions
def correct_character(c, expected_type):
    """Correct a character based on the expected type (digit, letter, or any)."""
    index_map = {
        'O': '0', 'o': '0', 'Q': '0', 'D': '0',
        'I': '1', 'i': '1', 'L': '1', 'l': '1', '|': '1',
        'Z': '2', 'z': '2',
        'A': '4', 'a': '4',
        'S': '5', 's': '5',
        'G': '6', 'g': '6',
        'T': '7',
        'B': '8',
        'g': '9', 'q': '9',
        '0': 'O' if expected_type == 'letter' else '0',
        '1': 'I' if expected_type == 'letter' else '1',
        '2': 'Z' if expected_type == 'letter' else '2',
        '5': 'S' if expected_type == 'letter' else '5',
        '8': 'B' if expected_type == 'letter' else '8'
    }
    if expected_type == 'digit':
        return index_map.get(c, c) if not c.isdigit() else c
    elif expected_type == 'letter':
        corrected = index_map.get(c, c)
        return corrected.upper() if corrected.isalpha() else c
    else:
        # For 'any', just return corrected if possible
        return index_map.get(c, c)

def expected_plate_structure(text):
    """
    Determine expected structure based on common license plate patterns.
    This function can be customized for specific regions/countries.
    """
    # Default to any type
    structure = ['any'] * len(text)
    
    # Common patterns (customize based on your region)
    # Example: AA-1111 pattern (2 letters followed by 4 digits)
    if len(text) >= 6 and text[:2].isalpha() and text[2:].isdigit():
        structure = ['letter', 'letter'] + ['digit'] * (len(text) - 2)
    
    # Example: 111-AAA pattern (3 digits followed by 3 letters)
    elif len(text) >= 6 and text[:3].isdigit() and text[3:].isalpha():
        structure = ['digit'] * 3 + ['letter'] * (len(text) - 3)
    
    # Example: AA-111-BB pattern (alternating letters and digits)
    elif len(text) >= 7 and text[:2].isalpha() and text[2:5].isdigit() and text[5:].isalpha():
        structure = ['letter', 'letter'] + ['digit'] * 3 + ['letter'] * (len(text) - 5)
        
    return structure

def preprocess_plate_image(plate_crop):
    """
    Preprocess the license plate image to improve OCR accuracy.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
        
    # Convert to grayscale if the image is in color
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()
    
    # Apply adaptive thresholding to handle various lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Noise removal
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try different thresholds
    _, binary1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Increase contrast
    alpha = 1.5  # Contrast control
    beta = 10    # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Return multiple processed images for the OCR to try
    return [gray, thresh, binary1, adjusted, plate_crop]

def format_license(text):
    """Format the detected license plate text."""
    if text is None:
        return None

    # Remove non-alphanumeric characters
    text = re.sub(r'[^A-Za-z0-9]', '', text)
    
    # If text is too short, likely incorrect detection
    if len(text) < 2:
        return None

    expected = expected_plate_structure(text)
    corrected = ""

    for c, exp in zip(text, expected):
        corrected += correct_character(c, exp)

    return corrected

def read_license_plate(plate_crop):
    """Reads text from the license plate crop with improved processing."""
    if plate_crop is None or plate_crop.size == 0:
        return None, None, None
        
    try:
        # Preprocess the image with multiple methods
        processed_images = preprocess_plate_image(plate_crop)
        
        best_text = None
        best_bbox = None
        best_conf = -1
        
        # Try different preprocessing methods
        for img in processed_images:
            if img is None:
                continue
                
            # Try different OCR configurations
            # Using simpler parameters to avoid PIL.Image.ANTIALIAS issue
            detections = reader.readtext(
                img,
                paragraph=False,
                decoder='greedy'
            )
            
            if not detections:
                continue
                
            # Sort by confidence (higher is better)
            detections.sort(key=lambda x: x[2], reverse=True)
            
            for detection in detections:
                if len(detection) >= 3:  # Make sure we have bbox, text, and confidence
                    text = detection[1]
                    bbox = detection[0]
                    conf = detection[2]
                    
                    # Only consider if confidence is high enough
                    if conf > 0.3:  # Adjust this threshold as needed
                        formatted = format_license(text)
                        
                        # Keep the detection with highest confidence that has a valid format
                        if formatted and (conf > best_conf):
                            best_text = formatted
                            best_bbox = bbox
                            best_conf = conf
        
        if best_text:
            print(f"Successfully detected license plate: {best_text} with confidence {best_conf:.2f}")
        else:
            print("No license plate text detected with sufficient confidence")
            
        return best_text, best_bbox, best_conf
                
    except Exception as e:
        print(f"OCR reading error: {str(e)}")
        return None, None, None

def get_car(license_plate, track_ids):
    """Find which car the license plate belongs to based on position."""
    if license_plate is None or len(license_plate) < 4:
        return [-1, -1, -1, -1, -1]
        
    x1, y1, x2, y2 = license_plate
    plate_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    closest_track = None
    min_distance = float('inf')
    
    for track in track_ids:
        if len(track) < 5:
            continue
            
        x1v, y1v, x2v, y2v, track_id = track
        
        # Check if plate center is within vehicle bounding box
        if x1v <= plate_center[0] <= x2v and y1v <= plate_center[1] <= y2v:
            # Calculate center of vehicle
            vehicle_center = ((x1v + x2v) // 2, (y1v + y2v) // 2)
            
            # Calculate distance from plate to vehicle center
            distance = ((plate_center[0] - vehicle_center[0])**2 + 
                        (plate_center[1] - vehicle_center[1])**2)**0.5
                        
            if distance < min_distance:
                min_distance = distance
                closest_track = track
    
    return closest_track if closest_track is not None else [-1, -1, -1, -1, -1]

# Example usage function
def process_license_plate_image(image_path):
    """Process a license plate image and return the recognized text."""
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Could not load image"
            
        # Detect and read the license plate
        text, bbox, conf = read_license_plate(img)
        
        if text:
            return f"License plate: {text} (confidence: {conf:.2f})"
        else:
            return "No license plate detected"
            
    except Exception as e:
        return f"Error processing image: {str(e)}"