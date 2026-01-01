import cv2
import numpy as np
import pyautogui
import time

# Initialize webcam with DirectShow backend (Windows)
print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera 0 failed, trying camera 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera!")
    print("Please check:")
    print("1. Close any apps using camera (Teams, Zoom, Skype)")
    print("2. Check camera permissions in Windows Settings")
    print("3. Make sure camera is connected")
    exit()

print("✅ Camera opened successfully!")
cap.set(3, 640)
cap.set(4, 480)

# State tracking
current_action = None
gesture_delay = 0.15
last_gesture_time = 0

print("=" * 50)
print("Hill Climb Racing Gesture Controller")
print("=" * 50)
print("✊ FIST (closed hand) = BRAKE (LEFT arrow)")
print("✋ OPEN HAND (spread fingers) = ACCELERATE (RIGHT arrow)")
print("Press 'Q' to quit")
print("=" * 50)
print("\nTip: Use good lighting and keep hand in center of frame")

# Background subtraction for better hand detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Calibration frames
print("Calibrating... Keep hand out of frame for 2 seconds")
for i in range(60):
    ret, frame = cap.read()
    if ret:
        bg_subtractor.apply(frame)
    time.sleep(0.033)
print("Calibration done! You can now use your hand.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Define ROI (Region of Interest) - center area
    roi_y1, roi_y2 = 100, 400
    roi_x1, roi_x2 = 150, 490
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    cv2.putText(frame, "Place hand here", (roi_x1, roi_y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Convert ROI to HSV for skin detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Skin color range (works for most skin tones)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask for skin
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    current_time = time.time()
    
    if contours and (current_time - last_gesture_time > gesture_delay):
        # Get largest contour (hand)
        hand_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand_contour)
        
        # Only process if contour is large enough
        if area > 5000:
            # Draw contour on ROI
            cv2.drawContours(roi, [hand_contour], -1, (0, 255, 0), 2)
            
            # Calculate convex hull and defects
            hull = cv2.convexHull(hand_contour, returnPoints=False)
            
            if len(hull) > 3 and len(hand_contour) > 3:
                try:
                    defects = cv2.convexityDefects(hand_contour, hull)
                    
                    # Count defects (spaces between fingers)
                    defect_count = 0
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(hand_contour[s][0])
                            end = tuple(hand_contour[e][0])
                            far = tuple(hand_contour[f][0])
                            
                            # Calculate angle
                            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                            angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                            
                            # Count defects with angle less than 90 degrees
                            if angle <= np.pi/2 and d > 10000:
                                defect_count += 1
                                cv2.circle(roi, far, 5, (0, 0, 255), -1)
                    
                    # FIST: 0-1 defects (closed hand)
                    if defect_count <= 1:
                        if current_action != "BRAKE":
                            pyautogui.keyUp('right')
                            pyautogui.keyDown('left')
                            current_action = "BRAKE"
                            last_gesture_time = current_time
                        
                        cv2.putText(frame, "BRAKE", (250, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.circle(frame, (50, 50), 30, (0, 0, 255), -1)
                    
                    # OPEN HAND: 3+ defects (spread fingers)
                    elif defect_count >= 3:
                        if current_action != "ACCELERATE":
                            pyautogui.keyUp('left')
                            pyautogui.keyDown('right')
                            current_action = "ACCELERATE"
                            last_gesture_time = current_time
                        
                        cv2.putText(frame, "ACCELERATE", (200, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.circle(frame, (50, 50), 30, (0, 255, 0), -1)
                    
                    # PARTIAL: 2 defects (neutral)
                    else:
                        if current_action is not None:
                            pyautogui.keyUp('left')
                            pyautogui.keyUp('right')
                            current_action = None
                            last_gesture_time = current_time
                        
                        cv2.putText(frame, "NEUTRAL", (250, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                        cv2.circle(frame, (50, 50), 30, (255, 255, 0), -1)
                    
                    # Display defect count for debugging
                    cv2.putText(frame, f"Defects: {defect_count}", (10, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                except:
                    pass
    
    else:
        # No hand detected
        if current_action is not None and current_time - last_gesture_time > 0.5:
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')
            current_action = None
        
        cv2.putText(frame, "NO HAND", (250, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
    
    # Show mask in corner for debugging
    mask_small = cv2.resize(mask, (160, 120))
    frame[10:130, 480:640] = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    
    cv2.imshow("Hill Climb Racing Controller", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pyautogui.keyUp('left')
pyautogui.keyUp('right')
cap.release()
cv2.destroyAllWindows()
print("\nController stopped. Keys released.")
