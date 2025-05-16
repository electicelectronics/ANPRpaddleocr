from ultralytics import YOLO
import cv2
import numpy as np
import time
import re
from collections import Counter
from paddleocr import PaddleOCR

# Add timing functions
class TimingAnalyzer:
    """
    A class to track and analyze the timing of different steps in the license plate 
    detection and OCR pipeline.
    """
    
    def __init__(self):
        self.detection_times = []
        self.detection_to_reading_delays = []
        self.reading_times = []
        self.consensus_times = []
        
        self.detection_start_time = None
        self.detection_end_time = None
        self.reading_start_time = None
        self.reading_end_time = None
        
        self.avg_detection_time = 0
        self.avg_detection_to_reading_delay = 0
        self.avg_reading_time = 0
        self.avg_consensus_time = 0
    
    def start_detection_timer(self):
        self.detection_start_time = time.time()
    
    def stop_detection_timer(self):
        if self.detection_start_time is None:
            return 0
            
        self.detection_end_time = time.time()
        detection_time = self.detection_end_time - self.detection_start_time
        self.detection_times.append(detection_time)
        
        self.avg_detection_time = sum(self.detection_times) / len(self.detection_times)
        print(f"YOLO detection time: {detection_time:.4f} seconds")
        
        return detection_time
    
    def start_reading_timer(self):
        self.reading_start_time = time.time()
        
        if self.detection_end_time is not None:
            delay = self.reading_start_time - self.detection_end_time
            self.detection_to_reading_delays.append(delay)
            self.avg_detection_to_reading_delay = sum(self.detection_to_reading_delays) / len(self.detection_to_reading_delays)
            print(f"Delay between detection and reading: {delay:.4f} seconds")
    
    def stop_reading_timer(self):
        if self.reading_start_time is None:
            return 0
            
        self.reading_end_time = time.time()
        reading_time = self.reading_end_time - self.reading_start_time
        self.reading_times.append(reading_time)
        
        self.avg_reading_time = sum(self.reading_times) / len(self.reading_times)
        print(f"OCR reading time: {reading_time:.4f} seconds")
        
        return reading_time
        
    def add_consensus_time(self, consensus_time):
        """Record a time taken to reach consensus"""
        if consensus_time is not None:
            self.consensus_times.append(consensus_time)
            self.avg_consensus_time = sum(self.consensus_times) / len(self.consensus_times)
    
    def print_timing_summary(self):
        print("\n===== TIMING SUMMARY =====")
        print(f"Average YOLO detection time: {self.avg_detection_time:.4f} seconds")
        print(f"Average delay between detection and reading: {self.avg_detection_to_reading_delay:.4f} seconds")
        print(f"Average OCR reading time: {self.avg_reading_time:.4f} seconds")
        print(f"Total average processing time: {(self.avg_detection_time + self.avg_detection_to_reading_delay + self.avg_reading_time):.4f} seconds")
        
        if self.consensus_times:
            print(f"Average time to consensus: {self.avg_consensus_time:.4f} seconds")
            
        print(f"Maximum theoretical FPS: {1/(self.avg_detection_time + self.avg_reading_time):.2f}")
        print("=========================\n")
    
    def reset_timers(self):
        self.detection_start_time = None
        self.detection_end_time = None
        self.reading_start_time = None
        self.reading_end_time = None
        self.detection_times = []
        self.detection_to_reading_delays = []
        self.reading_times = []
        self.consensus_times = []
        self.avg_detection_time = 0
        self.avg_detection_to_reading_delay = 0
        self.avg_reading_time = 0
        self.avg_consensus_time = 0

# Create global timing analyzer instance
timing_analyzer = TimingAnalyzer()

# Initialize PaddleOCR - using only essential parameters
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Optimized plate reading functions
def preprocess_plate_image(plate_crop):
    """Preprocess the license plate image for OCR"""
    if plate_crop is None or plate_crop.size == 0:
        return None

    processed_images = []
    
    # Resize and center the plate
    resized = enlarge_and_center_plate(plate_crop, target_height=64)
    
    # Convert to RGB if needed
    if len(resized.shape) == 2:  # If grayscale
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:  # If already color but might be BGR
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    processed_images.append(rgb)
    
    # Create a binarized version
    if len(resized.shape) == 3:
        gray_for_bin = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray_for_bin = resized
    
    binary = cv2.adaptiveThreshold(
        gray_for_bin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    processed_images.append(binary_rgb)
    
    return processed_images

def enlarge_and_center_plate(plate_img, target_height=64, target_width=None):
    """Enlarge and center the plate image on a white background"""
    if plate_img is None or plate_img.size == 0:
        return plate_img

    # Get original dimensions
    original_height, original_width = plate_img.shape[:2] if len(plate_img.shape) == 3 else plate_img.shape

    # Compute scale factor based on height
    scale_factor = target_height / float(original_height)
    new_width = int(original_width * scale_factor)
    
    # Resize
    resized_img = cv2.resize(plate_img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

    # Create canvas with correct dimensions
    canvas_width = target_width if target_width else max(new_width + 20, 160)
    
    # Create white background canvas
    if len(resized_img.shape) == 3:
        canvas = np.ones((target_height, canvas_width, 3), dtype=np.uint8) * 255
    else:
        canvas = np.ones((target_height, canvas_width), dtype=np.uint8) * 255

    # Center the image on the canvas
    start_x = (canvas_width - new_width) // 2
    if len(resized_img.shape) == 3:
        canvas[:, start_x:start_x + new_width] = resized_img
    else:
        canvas[:, start_x:start_x + new_width] = resized_img

    return canvas

def read_license_plate(plate_crop):
    """Read text from license plate using OCR"""
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

            # Run OCR 
            result = ocr.ocr(img, cls=True)
            
            if not result or len(result) == 0 or result[0] is None:
                continue

            for line in result[0]:
                bbox, (text, score) = line
                
                # Clean the text
                text_clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                
                # Filter results
                if text_clean and len(text_clean) >= 5 and score > best_conf:
                    best_text = text_clean
                    best_conf = score
                    best_bbox = bbox

        if best_text:
            print(f"Detected license plate: {best_text} with confidence {best_conf:.2f}")
        
        return best_text, best_bbox, best_conf

    except Exception as e:
        print(f"OCR reading error: {str(e)}")
        return None, None, None

# Consensus tracking
class PlateConsensusTracker:
    """Tracks license plate readings across frames to find consensus."""
    
    def __init__(self, frames_required=3, max_queue_size=10):
        self.frames_required = frames_required
        self.max_queue_size = max_queue_size
        self.plate_readings = []
        self.current_consensus = None
        self.first_detection_time = None
        self.consensus_time = None
        self.consensus_duration = None
    
    def add_reading(self, plate_text, confidence):
        if not plate_text:
            return None
        
        # Record first detection time if this is the first valid reading
        if not self.first_detection_time and len(self.plate_readings) == 0:
            self.first_detection_time = time.time()
            
        self.plate_readings.append(plate_text)
        
        if len(self.plate_readings) > self.max_queue_size:
            self.plate_readings.pop(0)
        
        return self.check_consensus()
    
    def check_consensus(self):
        if len(self.plate_readings) < self.frames_required:
            return None
            
        counter = Counter(self.plate_readings)
        most_common = counter.most_common(1)
        
        if most_common and most_common[0][1] >= self.frames_required:
            consensus_text = most_common[0][0]
            
            # Only record consensus time if this is a new consensus
            if self.current_consensus != consensus_text:
                self.current_consensus = consensus_text
                self.consensus_time = time.time()
                
                # Calculate duration if we have both start and end times
                if self.first_detection_time is not None:
                    self.consensus_duration = self.consensus_time - self.first_detection_time
                    print(f"⏱️ Time to consensus: {self.consensus_duration:.2f} seconds")
                
            return consensus_text
            
        return None
    
    def get_current_consensus(self):
        return self.current_consensus
    
    def get_consensus_time(self):
        """
        Returns the time taken to reach consensus (from first detection to consensus).
        Returns None if consensus hasn't been reached yet.
        """
        return self.consensus_duration
    
    def reset(self):
        self.plate_readings = []
        self.current_consensus = None
        self.first_detection_time = None
        self.consensus_time = None
        self.consensus_duration = None

# Main execution code
def main():
    # Load license plate detector model
    try:
        license_plate_detector = YOLO('license_plate_detector.pt')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create a consensus tracker
    consensus_tracker = PlateConsensusTracker(frames_required=3)

    # Open video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_list = []
    last_time = time.time()
    sample_rate = 3 # Take 3 frames per second for plate reading

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
        
        # Convert to RGB for YOLO model
        processing_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Start timing the detection
        timing_analyzer.start_detection_timer()
        
        # Detect license plates
        license_plates = license_plate_detector(processing_frame, conf=0.5)[0]
        
        # Stop timing the detection
        timing_analyzer.stop_detection_timer()
        
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
                    # Crop the license plate
                    license_plate_crop = frame[y1:y2, x1:x2]
                    license_plate_crop = cv2.resize(license_plate_crop, (0, 0), fx=2, fy=2)
                    
                    # Start timing the reading process
                    timing_analyzer.start_reading_timer()
                    
                    # Process the license plate with OCR
                    license_plate_text, _, license_plate_text_score = read_license_plate(license_plate_crop)
                    
                    # Stop timing the reading process
                    timing_analyzer.stop_reading_timer()
                    
                    # Update consensus tracker if we have a reading
                    if license_plate_text:
                        consensus = consensus_tracker.add_reading(license_plate_text, license_plate_text_score)
                        if consensus:
                            current_consensus = consensus
                            consensus_time = consensus_tracker.get_consensus_time()
                            if consensus_time:
                                print(f"CONSENSUS REACHED: {consensus} (Time to consensus: {consensus_time:.2f}s)")
                                # Add the consensus time to the timing analyzer
                                timing_analyzer.add_consensus_time(consensus_time)
                            else:
                                print(f"CONSENSUS REACHED: {consensus}")

                # Display current state on frame
                if current_consensus:
                    cv2.putText(original_frame, f"VERIFIED: {current_consensus}", (x1, y2 + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif sample_frame and 'license_plate_text' in locals() and license_plate_text:
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
        
        cv2.putText(original_frame, "Q: quit | R: reset", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display frame
        display_frame = cv2.resize(original_frame, (0, 0), fx=0.8, fy=0.8)
        cv2.imshow('License Plate Detection', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Print timing summary before quitting
            timing_analyzer.print_timing_summary()
            break
        elif key == ord('r'):
            # Reset consensus tracking
            consensus_tracker.reset()
            timing_analyzer.reset_timers()
            current_consensus = None
            print("Consensus tracking and timers reset")
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete")

if __name__ == "__main__":
    main()