import cv2
import numpy as np
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Configuration
MODEL_PATH = "arabic_sign_lang_model.keras"  # Path to your Keras model
INPUT_SIZE = (200, 200)  # Standard input size for most CNN models, adjust if needed
HAND_DETECTION_CONFIDENCE = 0.7  # Minimum confidence for MediaPipe hand detection
HAND_TRACKING_CONFIDENCE = 0.5  # Minimum confidence for MediaPipe hand tracking
PADDING_FACTOR = 0.5  # Additional padding around the hand
PREDICTION_THRESHOLD = 0.6  # Minimum confidence score for prediction

# Arabic sign language class names (update with your actual class names)
class_names = ['Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh', 
               'Jeem', 'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad', 
               'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta', 'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain']

def preprocess_image(image):
    """Preprocess the image for the Keras model"""
    # Resize the image
    img = cv2.resize(image, INPUT_SIZE)
    # Convert to RGB if the model was trained on RGB images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to float32 and normalize to [0,1]
    img = img_to_array(img) / 255.0
    # Expand dimensions for batch size
    img = np.expand_dims(img, axis=0)
    return img

def run_real_time_detection():
    # Load the trained Keras model
    try:
        model = load_model(MODEL_PATH)
        print("Keras model loaded successfully")
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE
    )
    print("MediaPipe hands initialized")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting real-time detection. Press 'q' to quit.")
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    last_prediction = "No hand detected"
    prediction_confidence = 0.0
    
    # For prediction smoothing
    prediction_history = []
    max_history = 5
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Resize frame for display
        display_frame = frame.copy()
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on display frame
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get bounding box of hand
                h, w, _ = frame.shape
                x_min, x_max, y_min, y_max = w, 0, h, 0
                
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)
                
                # Make the box square (for better model input)
                square_size = max(x_max - x_min, y_max - y_min)
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                # Add padding with the PADDING_FACTOR
                square_size_with_padding = int(square_size * (1 + PADDING_FACTOR))
                
                # Calculate new square coordinates
                x_min_square = max(0, center_x - square_size_with_padding // 2)
                y_min_square = max(0, center_y - square_size_with_padding // 2)
                x_max_square = min(w, center_x + square_size_with_padding // 2)
                y_max_square = min(h, center_y + square_size_with_padding // 2)
                
                # Draw the padded bounding box
                cv2.rectangle(display_frame, (x_min_square, y_min_square), (x_max_square, y_max_square), (255, 0, 0), 2)
                
                # Extract hand image
                hand_img = frame[y_min_square:y_max_square, x_min_square:x_max_square]
                
                if hand_img.size != 0:  # Make sure the slice is not empty
                    # Preprocess the hand image for the model
                    processed_img = preprocess_image(hand_img)
                    
                    # Make prediction with Keras model
                    predictions = model.predict(processed_img, verbose=0)[0]
                    
                    # Get the predicted class index and confidence
                    class_idx = np.argmax(predictions)
                    confidence = predictions[class_idx]
                    
                    if confidence >= PREDICTION_THRESHOLD:
                        # Get the class name
                        if class_idx < len(class_names):
                            predicted_class = class_names[class_idx]
                        else:
                            predicted_class = f"Unknown ({class_idx})"
                        
                        # Add to prediction history for smoothing
                        prediction_history.append((predicted_class, confidence))
                        if len(prediction_history) > max_history:
                            prediction_history.pop(0)
                        
                        # Get most common prediction from history
                        if prediction_history:
                            pred_counts = {}
                            for pred, conf in prediction_history:
                                if pred in pred_counts:
                                    pred_counts[pred] = (pred_counts[pred][0] + 1, 
                                                        pred_counts[pred][1] + conf)
                                else:
                                    pred_counts[pred] = (1, conf)
                            
                            # Find prediction with highest count
                            max_count = 0
                            max_conf = 0
                            for pred, (count, conf) in pred_counts.items():
                                avg_conf = conf / count
                                if count > max_count or (count == max_count and avg_conf > max_conf):
                                    max_count = count
                                    max_conf = avg_conf
                                    last_prediction = pred
                                    prediction_confidence = avg_conf
                
                # Display prediction
                cv2.putText(display_frame, f"Sign: {last_prediction} ({prediction_confidence:.2f})", 
                            (x_min_square, y_min_square - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            # If no hands detected
            cv2.putText(display_frame, "No hand detected", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            prediction_history = []  # Clear history when no hand is detected
        
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - start_time >= 1.0:  # Update FPS every second
            fps = frame_count / (current_time - start_time)
            frame_count = 0
            start_time = current_time
        
        # Display the frame
        cv2.imshow("Arabic Sign Language Detection", display_frame)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    run_real_time_detection()