from ultralytics import YOLO
import cv2
import numpy as np
import time
import re
import os
from collections import Counter
from paddleocr import PaddleOCR

# Initialize PaddleOCR
# lang='en' for English text
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# ==================== PLATE READING FUNCTIONS ====================

def preprocess_plate_image(plate_crop):
    """
    Preprocess the license plate image to improve PaddleOCR accuracy.
    Creates multiple versions of the preprocessed image with different techniques.
    Returns an array of processed images to try with OCR.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None

    processed_images = []
    
    # 1. Basic resize - PaddleOCR works better with properly sized images
    height, width = plate_crop.shape[:2]
    
    # Scale factor calculation based on image size
    # For smaller plates, we scale more aggressively
    if width < 100:
        scale_factor = 3.0  # More aggressive scaling for distant plates
    else:
        scale_factor = 2.0  # Standard scaling for closer plates
        
    # Apply resize
    resized = enlarge_and_center_plate(plate_crop, target_height=64)
    
    # Convert to RGB if needed (PaddleOCR works better with RGB)
    if len(resized.shape) == 2:  # If grayscale
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:  # If already color but might be BGR
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Add basic RGB to our processed images
    processed_images.append(rgb)
    
    # 2. Apply contrast enhancement - PaddleOCR benefits from good contrast
    # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    processed_images.append(enhanced_rgb)
    
    # 3. Create a sharpened version - helps with blurry plates
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(rgb, -1, kernel)
    processed_images.append(sharpened)
    
    # 4. Create a version with adjusted brightness and contrast
    # This helps with poorly lit license plates
    alpha = 1.3  # Contrast control (1.0-3.0)
    beta = 10    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
    processed_images.append(adjusted)
    
    # 5. Create a binarized version - can help with certain plates
    # But convert it back to RGB for PaddleOCR
    if len(resized.shape) == 3:
        gray_for_bin = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray_for_bin = resized
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray_for_bin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    processed_images.append(binary_rgb)
    
    return processed_images

def enlarge_and_center_plate(plate_img, target_height=64, target_width=None):
    """
    Enlarges the license plate image so that the height of characters meets a minimum size
    and centers it on a white background. This improves OCR reliability with PaddleOCR.
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img

    # Get original dimensions
    if len(plate_img.shape) == 3:
        original_height, original_width = plate_img.shape[:2]
    else:
        original_height, original_width = plate_img.shape

    # Compute scale factor based on height
    scale_factor = target_height / float(original_height)
    new_width = int(original_width * scale_factor)
    
    # Use proper interpolation method for upscaling
    resized_img = cv2.resize(plate_img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

    # Determine canvas size
    canvas_width = target_width if target_width else max(new_width + 20, 160)
    
    # Create white background canvas matching input image type
    if len(resized_img.shape) == 3:
        canvas = np.ones((target_height, canvas_width, 3), dtype=np.uint8) * 255  # RGB white
    else:
        canvas = np.ones((target_height, canvas_width), dtype=np.uint8) * 255  # Grayscale white

    # Center the image on the canvas
    start_x = (canvas_width - new_width) // 2
    if len(resized_img.shape) == 3:
        canvas[:, start_x:start_x + new_width] = resized_img
    else:
        canvas[:, start_x:start_x + new_width] = resized_img

    return canvas

def read_license_plate(plate_crop):
    """
    Reads text from the license plate crop with PaddleOCR.
    Uses multiple preprocessed versions to maximize chances of detection.
    Returns the detected text, bbox, and confidence.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None, None, None

    try:
        processed_images = preprocess_plate_image(plate_crop)
        if not processed_images:
            return None, None, None

        best_text = None
        best_conf = -1
        best_bbox = None

        for img in processed_images:
            if img is None:
                continue

            # Run OCR on the processed image using PaddleOCR
            # PaddleOCR returns a list of results in a different format than RapidOCR
            result = ocr.ocr(img, cls=True)
            
            if not result or len(result) == 0 or result[0] is None:
                continue

            for line in result[0]:
                # PaddleOCR format: [[[x1,y1],[x2,y1],[x3,y3],[x4,y4]], (text, score)]
                bbox, (text, score) = line
                
                # Clean the text - keep only alphanumeric characters
                text_clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                
                # Filter out nonsensical results (too short)
                if text_clean and len(text_clean) >= 5 and score > best_conf:
                    best_text = text_clean
                    best_conf = score
                    best_bbox = bbox

        if best_text:
            print(f"Detected license plate: {best_text} with confidence {best_conf:.2f}")
        else:
            print("No license plate text detected with sufficient confidence")

        return best_text, best_bbox, best_conf

    except Exception as e:
        print(f"OCR reading error: {str(e)}")
        return None, None, None

# ==================== CONSENSUS FUNCTION ====================

class PlateConsensusTracker:
    """Tracks license plate readings across frames to find consensus."""
    
    def __init__(self, frames_required=3, max_queue_size=10):
        """
        Initialize the consensus tracker.
        frames_required: Number of matching readings required for consensus
        max_queue_size: Maximum number of readings to store before flushing
        """
        self.frames_required = frames_required
        self.max_queue_size = max_queue_size
        self.plate_readings = []
        self.current_consensus = None
    
    def add_reading(self, plate_text, confidence):
        """Add a new plate reading to the tracker."""
        if not plate_text:
            return None
            
        # Add the new reading
        self.plate_readings.append(plate_text)
        
        # Keep the queue at a reasonable size
        if len(self.plate_readings) > self.max_queue_size:
            self.plate_readings.pop(0)
        
        # Check for consensus
        return self.check_consensus()
    
    def check_consensus(self):
        """
        Check if we have reached consensus on a plate reading.
        Returns the consensus text if found, None otherwise.
        """
        if len(self.plate_readings) < self.frames_required:
            return None
            
        # Count occurrences of each plate reading
        counter = Counter(self.plate_readings)
        
        # Find the most common reading
        most_common = counter.most_common(1)
        
        # If the most common reading appears at least frames_required times, we have consensus
        if most_common and most_common[0][1] >= self.frames_required:
            consensus_text = most_common[0][0]
            self.current_consensus = consensus_text
            return consensus_text
            
        return None
    
    def get_current_consensus(self):
        """Get the current consensus reading."""
        return self.current_consensus
    
    def reset(self):
        """Reset the tracker."""
        self.plate_readings = []
        self.current_consensus = None

# ==================== MAIN CODE ====================

# Load license plate detector model
try:
    # Use relative path for better portability
    license_plate_detector = YOLO('license_plate_detector.pt')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Create a consensus tracker
consensus_tracker = PlateConsensusTracker(frames_required=3)

# Open video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source. Trying backup source...")
    exit()

# Set lower resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps_list = []
last_time = time.time()
sample_rate = 3  # Take 3 frames per second for plate reading

print("Starting video processing loop...")
frame_count = 0
last_sample_time = time.time()
current_consensus = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    frame_count += 1
    
    # Calculate and update FPS
    current_time = time.time()
    fps = 1 / (current_time - last_time)
    last_time = current_time
    fps_list.append(fps)
    if len(fps_list) > 30:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)

    # Sample frames at the specified rate for plate reading
    sample_frame = False
    if current_time - last_sample_time >= 1.0/sample_rate:
        sample_frame = True
        last_sample_time = current_time
    
    # Store original frame for display
    original_frame = frame.copy()
    
    # Convert to RGB for YOLO model (it expects color images)
    # IMPORTANT FIX: Always ensure we're passing RGB images to YOLO
    if len(frame.shape) == 2:  # If grayscale
        # Convert grayscale to RGB
        processing_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:  # If already color
        # Convert BGR to RGB
        processing_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get grayscale version for OCR preparation
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    # Detect license plates - now using RGB input
    license_plates = license_plate_detector(processing_frame, conf=0.5)[0]
    
    # Process each detected license plate
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        
        # Ensure coordinates are within frame boundaries
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(frame.shape[1], x2)), int(min(frame.shape[0], y2))
        
        # Skip if plate is too small
        if (x2-x1) < 20 or (y2-y1) < 10:
            continue

        # Draw license plate bounding box
        cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(original_frame, f"Plate: {score:.2f}", (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        try:
            if sample_frame:
                # Crop the license plate - For PaddleOCR we'll use color image
                license_plate_crop = frame[y1:y2, x1:x2]
                license_plate_crop = cv2.resize(license_plate_crop, (0, 0), fx=2, fy=2)
                
                # Process the license plate with OCR
                license_plate_text, _, license_plate_text_score = read_license_plate(license_plate_crop)
                
                # Update consensus tracker if we have a reading
                if license_plate_text:
                    consensus = consensus_tracker.add_reading(license_plate_text, license_plate_text_score)
                    if consensus:
                        current_consensus = consensus
                        print(f"🎯 CONSENSUS REACHED: {consensus}")

            # Display current state on frame
            if current_consensus:
                cv2.putText(original_frame, f"VERIFIED: {current_consensus}", (x1, y2 + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif sample_frame and license_plate_text:
                cv2.putText(original_frame, f"READING: {license_plate_text}", (x1, y2 + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(original_frame, "DETECTING...", (x1, y2 + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        except Exception as e:
            print(f"Error processing license plate: {e}")

    # Display information on frame
    cv2.putText(original_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if current_consensus:
        cv2.putText(original_frame, f"PLATE: {current_consensus}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(original_frame, "Q to quit, R to reset", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display frame
    display_frame = cv2.resize(original_frame, (0, 0), fx=0.8, fy=0.8)
    cv2.imshow('test', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset consensus tracking
        consensus_tracker.reset()
        current_consensus = None
        print("Consensus tracking reset")

# Clean up resources
cap.release()
cv2.destroyAllWindows()
print("Processing complete")
