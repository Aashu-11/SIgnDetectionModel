import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
import mediapipe as mp
import time
import base64
from flask import Flask, render_template, Response, jsonify
import threading
from queue import Queue
import json

# Import the necessary classes from your original script
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]

        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)

        return torch.cat(features, 1)

class AdvancedSignLanguageModel(nn.Module):
    def __init__(self, num_classes=24):
        super(AdvancedSignLanguageModel, self).__init__()

        # Load pre-trained ResNet backbone but modify for grayscale input
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)

        # Modify first layer to accept 1 channel instead of 3
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final FC layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove classification layer

        # Custom classification head with attention
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

        # Multiple projection heads for ensemble-like behavior
        self.classifier1 = nn.Linear(in_features, num_classes)
        self.classifier2 = nn.Linear(in_features, num_classes)
        self.classifier3 = nn.Linear(in_features, 512)
        self.classifier3_bn = nn.BatchNorm1d(512)
        self.classifier3_out = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Extract features
        features = self.resnet(x)

        # Attention mechanism
        attention_weights = F.softmax(self.attention(features), dim=1)
        attention_features = features * attention_weights

        # Multiple classifiers for ensemble-like behavior
        out1 = self.classifier1(self.dropout(features))
        out2 = self.classifier2(self.dropout(attention_features))

        # Deep classifier path
        out3 = self.classifier3(self.dropout(features))
        out3 = F.relu(self.classifier3_bn(out3))
        out3 = self.classifier3_out(self.dropout(out3))

        # Combine outputs
        return (out1 + out2 + out3) / 3

class CustomSignNet(nn.Module):
    def __init__(self, num_classes=24, growth_rate=24):
        super(CustomSignNet, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks
        self.dense1 = DenseBlock(64, growth_rate, num_layers=4)
        in_channels = 64 + 4 * growth_rate

        # Transition layer 1
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        in_channels = in_channels // 2

        # Dense block 2
        self.dense2 = DenseBlock(in_channels, growth_rate, num_layers=6)
        in_channels = in_channels + 6 * growth_rate

        # Transition layer 2
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        in_channels = in_channels // 2

        # Dense block 3
        self.dense3 = DenseBlock(in_channels, growth_rate, num_layers=8)
        in_channels = in_channels + 8 * growth_rate

        # Final batch norm
        self.bn_final = nn.BatchNorm2d(in_channels)

        # Spatial and Channel-wise Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Dense blocks with transitions
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn_final(x)
        x = self.relu(x)

        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_in = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_in)
        x = x * sa

        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class EnsembleModel(nn.Module):
    def __init__(self, num_classes=24):
        super(EnsembleModel, self).__init__()
        self.model1 = AdvancedSignLanguageModel(num_classes)
        self.model2 = CustomSignNet(num_classes)
        self.weights = nn.Parameter(torch.ones(2))

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)

        # Weighted average
        w = F.softmax(self.weights, dim=0)
        return w[0] * out1 + w[1] * out2

class SignLanguageDetector:
    def __init__(self, model_path, num_classes=24):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe Hands for hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create model instance
        self.model = EnsembleModel(num_classes=num_classes)
        
        # Try to load the model using different approaches
        try:
            # First try loading as PKL file
            print(f"Attempting to load model from {model_path}...")
            
            # Load as dictionary 
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check what type of saved model it is
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    print("Loading model from state_dict key...")
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'accuracy' in checkpoint:
                        print(f"Loaded model with reported accuracy: {checkpoint['accuracy']:.4f}")
                else:
                    print("Loading full state dict...")
                    self.model.load_state_dict(checkpoint)
            else:
                print("Loading direct model instance...")
                self.model = checkpoint
                
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"First loading attempt failed: {e}")
            try:
                # Try again with pickle module
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model.to(self.device)
                self.model.eval()
                print("Model loaded successfully via pickle!")
            except Exception as e2:
                print(f"Second loading attempt failed: {e2}")
                raise Exception(f"Could not load model using any method. Original error: {e}\nSecond error: {e2}")
        
        # Define transforms for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Define class labels (ASL alphabet)
        # ASL doesn't include J and Z in static form as they require motion
        self.labels = {}
        letter_idx = 0
        for i in range(26):
            if i != 9 and i != 25:  # Skip J (9) and Z (25)
                self.labels[letter_idx] = chr(ord('A') + i)
                letter_idx += 1
                
        # For tracking predictions to smooth output
        self.recent_predictions = []
        self.prediction_history_size = 5
        
        # For FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
    
    def preprocess_hand_image(self, hand_img):
        """Convert hand region to proper format for the model"""
        # Convert to grayscale if it's a color image
        if len(hand_img.shape) == 3 and hand_img.shape[2] == 3:
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            
        # Resize to 28x28 (model input size)
        resized = cv2.resize(hand_img, (28, 28))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(resized)
        
        # Apply transformations
        tensor_image = self.transform(pil_image)
        
        return tensor_image.unsqueeze(0)  # Add batch dimension
    
    def get_smooth_prediction(self, prediction_idx):
        """Apply smoothing to predictions to reduce flickering"""
        # Add current prediction to history
        self.recent_predictions.append(prediction_idx)
        
        # Keep only recent predictions
        if len(self.recent_predictions) > self.prediction_history_size:
            self.recent_predictions.pop(0)
            
        # Return most common prediction
        from collections import Counter
        prediction_counts = Counter(self.recent_predictions)
        return prediction_counts.most_common(1)[0][0]
    
    def extract_hand_from_landmarks(self, frame, landmarks):
        """Extract hand ROI based on hand landmarks"""
        h, w, _ = frame.shape
        
        # Get bounding box of hand
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
                
        # Add padding around hand
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Make square by taking the larger dimension
        width = x_max - x_min
        height = y_max - y_min
        
        if width > height:
            # Adjust height to make it square
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - width // 2)
            y_max = min(h, center_y + width // 2)
        else:
            # Adjust width to make it square
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - height // 2)
            x_max = min(w, center_x + height // 2)
        
        # Extract hand region
        hand_roi = frame[y_min:y_max, x_min:x_max]
        
        # Return both the ROI and the coordinates
        return hand_roi, (x_min, y_min, x_max, y_max)
    
    def predict(self, frame):
        """Detect hands and predict sign language"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        # Default values if no hand is detected
        hand_roi = None
        roi_coords = None
        prediction_idx = None
        prediction_letter = None
        confidence = None
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract hand region using landmarks
            hand_roi, roi_coords = self.extract_hand_from_landmarks(frame, hand_landmarks)
            
            if hand_roi is not None and hand_roi.size > 0:
                # Preprocess hand image
                input_tensor = self.preprocess_hand_image(hand_roi)
                input_tensor = input_tensor.to(self.device)
                
                # Make prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = F.softmax(output, dim=1)
                    confidence_tensor, predicted = torch.max(probabilities, 1)
                    confidence = confidence_tensor.item()
                    raw_prediction_idx = predicted.item()
                
                # Apply smoothing
                prediction_idx = self.get_smooth_prediction(raw_prediction_idx)
                prediction_letter = self.labels.get(prediction_idx, "Unknown")
            
            # Draw hand landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return prediction_idx, prediction_letter, confidence, roi_coords

# Flask Application for Web Deployment
app = Flask(__name__)

# Global variables for sharing data between threads
latest_frame = None
latest_prediction = None
latest_confidence = None
processing = False
frame_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)

def find_model_file():
    """Find available model files in current directory"""
    model_extensions = ['.pth', '.pt', '.pkl']
    potential_models = []
    
    # Look for model files
    for file in os.listdir():
        file_lower = file.lower()
        if any(file_lower.endswith(ext) for ext in model_extensions):
            potential_models.append(file)
    
    # Prioritize certain filenames
    priority_names = [
        'sign_language_final_model.pth',
        'best_model.pth',
        'model.pth',
        'sign_language_model.pkl'
    ]
    
    for name in priority_names:
        for model in potential_models:
            if name in model:
                return model
    
    # Return first found if no priority match
    if potential_models:
        return potential_models[0]
    
    return None

# Initialize detector globally
model_path = find_model_file()
if not model_path:
    print("Error: No model file found. Placeholder model will be used for demonstration.")
    # Create a directory for model if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    # Use dummy path, the app will handle the error gracefully
    model_path = "models/placeholder_model.pth"

try:
    detector = SignLanguageDetector(model_path, num_classes=24)
    detector_initialized = True
except Exception as e:
    print(f"Error initializing detector: {e}")
    detector_initialized = False

def process_frames():
    """Background thread to process video frames"""
    global latest_frame, latest_prediction, latest_confidence
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                continue
                
            try:
                # Process frame with detector
                prediction_idx, prediction_letter, confidence, roi_coords = detector.predict(frame)
                
                # Draw bounding box and landmarks if hand detected
                if roi_coords:
                    x_min, y_min, x_max, y_max = roi_coords
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Add text to the frame
                if prediction_letter:
                    prediction_text = f"Letter: {prediction_letter}"
                    confidence_text = f"Confidence: {confidence:.2f}" if confidence else ""
                    
                    cv2.putText(frame, prediction_text, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
                    cv2.putText(frame, prediction_text, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 2)
                    
                    if confidence:
                        cv2.putText(frame, confidence_text, (10, 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                        cv2.putText(frame, confidence_text, (10, 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 1)
                
                # Update latest results for frontend
                latest_frame = frame
                latest_prediction = prediction_letter
                latest_confidence = confidence
                
                # Put results in queue for API endpoint
                if not result_queue.full():
                    result_queue.put({
                        'prediction': prediction_letter,
                        'confidence': confidence if confidence else 0.0
                    })
                
            except Exception as e:
                print(f"Error processing frame: {e}")

# Create routes for web application
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

def generate_frames():
    """Generator for video streaming"""
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Flip the frame for more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Add to queue for processing
        if not frame_queue.full():
            frame_queue.put(frame)
        
        # Use the latest processed frame if available
        output_frame = latest_frame if latest_frame is not None else frame
        
        # Add instructions
        cv2.putText(output_frame, "Show hand sign in camera view", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the output to the web interface
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    """API endpoint to get the latest prediction"""
    if not result_queue.empty():
        result = result_queue.get()
    else:
        result = {'prediction': None, 'confidence': 0.0}
    
    return jsonify(result)

# Create the HTML template directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Create the index.html file
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Language Recognition</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
            color: #333;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 1200px;
            width: 100%;
        }
        .video-container {
            width: 100%;
            max-width: 800px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.19), 0 6px 6px rgba(0, 0, 0, 0.23);
            margin-bottom: 20px;
        }
        #video {
            width: 100%;
            height: auto;
            display: block;
        }
        .result-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            width: 80%;
            max-width: 600px;
        }
        .letter-display {
            font-size: 72px;
            font-weight: bold;
            margin: 20px 0;
            color: #3498db;
            height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .confidence-bar-container {
            width: 100%;
            height: 30px;
            background-color: #ecf0f1;
            border-radius: 15px;
            margin: 20px 0;
            overflow: hidden;
        }
        .confidence-bar {
            height: 100%;
            background-color: #2ecc71;
            border-radius: 15px;
            width: 0%;
            transition: width 0.3s ease;
        }
        .confidence-text {
            font-size: 18px;
            color: #7f8c8d;
        }
        .instruction {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 5px solid #3498db;
            color: #2c3e50;
            width: 80%;
            max-width: 600px;
            border-radius: 5px;
        }
        .history-container {
            display: flex;
            margin-top: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            width: 80%;
            max-width: 600px;
            overflow-x: auto;
        }
        .history-letter {
            font-size: 24px;
            margin: 0 10px;
            padding: 10px 15px;
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            min-width: 25px;
            text-align: center;
        }
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        .word-display {
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
            color: #2c3e50;
            min-height: 40px;
        }
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            .letter-display {
                font-size: 48px;
                height: 70px;
            }
            .history-letter {
                font-size: 18px;
                margin: 0 5px;
                padding: 8px 12px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sign Language Recognition System</h1>
        
        <div class="video-container">
            <img id="video" src="/video_feed" alt="Video Stream">
        </div>
        
        <div class="instruction">
            <p>Show your hand sign in the camera view. The system will detect American Sign Language letters.</p>
            <p>Note: This system works best with a well-lit environment and clear hand gestures.</p>
        </div>
        
        <div class="result-container">
            <h2>Detected Letter</h2>
            <div class="letter-display" id="letter-display">-</div>
            <div class="confidence-bar-container">
                <div class="confidence-bar" id="confidence-bar"></div>
            </div>
            <div class="confidence-text" id="confidence-text">Confidence: 0%</div>
        </div>
        
        <div class="history-container" id="letter-history">
            <!-- Letter history will be displayed here -->
        </div>
        
        <div class="result-container">
            <h2>Formed Word</h2>
            <div class="word-display" id="word-display"></div>
            <div class="controls">
                <button id="add-letter">Add Letter</button>
                <button id="clear-word">Clear Word</button>
                <button id="space">Add Space</button>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let currentLetter = null;
        let letterHistory = [];
        let currentWord = '';
        
        // Update prediction
        function updatePrediction() {
            fetch('/get_prediction')
                .then(response => response.json())
                .then(data => {
                    if (data.prediction) {
                        // Update displayed letter
                        document.getElementById('letter-display').textContent = data.prediction;
                        currentLetter = data.prediction;
                        
                        // Update confidence bar
                        const confidencePercentage = (data.confidence * 100).toFixed(1);
                        document.getElementById('confidence-bar').style.width = `${confidencePercentage}%`;
                        document.getElementById('confidence-text').textContent = `Confidence: ${confidencePercentage}%`;
                        
                        // Color the confidence bar based on confidence level
                        const confidenceBar = document.getElementById('confidence-bar');
                        if (data.confidence < 0.5) {
                            confidenceBar.style.backgroundColor = '#e74c3c'; // Red for low confidence
                        } else if (data.confidence < 0.75) {
                            confidenceBar.style.backgroundColor = '#f39c12'; // Orange for medium confidence
                        } else {
                            confidenceBar.style.backgroundColor = '#2ecc71'; // Green for high confidence
                        }
                    } else {
                        document.getElementById('letter-display').textContent = '-';
                        document.getElementById('confidence-bar').style.width = '0%';
                        document.getElementById('confidence-text').textContent = 'Confidence: 0%';
                    }
                })
                .catch(error => {
                    console.error('Error fetching prediction:', error);
                });
        }
        
        // Add letter to formed word
        document.getElementById('add-letter').addEventListener('click', () => {
            if (currentLetter) {
                // Add letter to word
                currentWord += currentLetter;
                document.getElementById('word-display').textContent = currentWord;
                
                // Add to history
                addLetterToHistory(currentLetter);
            }
        });
        
        // Clear word
        document.getElementById('clear-word').addEventListener('click', () => {
            currentWord = '';
            document.getElementById('word-display').textContent = '';
            
            // Clear history as well
            letterHistory = [];
            document.getElementById('letter-history').innerHTML = '';
        });
        
        // Add space
        document.getElementById('space').addEventListener('click', () => {
            currentWord += ' ';
            document.getElementById('word-display').textContent = currentWord;
            
            // Add space indicator to history
            addLetterToHistory('␣');
        });
        
        // Add letter to history display
        function addLetterToHistory(letter) {
            letterHistory.push(letter);
            
            // Only keep last 10 letters in history
            if (letterHistory.length > 10) {
                letterHistory.shift();
            }
            
            // Update history display
            const historyContainer = document.getElementById('letter-history');
            historyContainer.innerHTML = '';
            
            letterHistory.forEach(ltr => {
                const letterElement = document.createElement('div');
                letterElement.className = 'history-letter';
                letterElement.textContent = ltr;
                historyContainer.appendChild(letterElement);
            });
        }
        
        // Update prediction every 100ms
        setInterval(updatePrediction, 100);
    </script>
</body>
</html>
    ''')

def main():
    # Start the background thread for processing frames
    if detector_initialized:
        processing_thread = threading.Thread(target=process_frames, daemon=True)
        processing_thread.start()
    
    # Run the Flask application
    print("="*50)
    print("Sign Language Recognition Web Server Starting")
    print("="*50)
    print("Access the application by opening a web browser and navigating to:")
    print("http://localhost:5000")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()