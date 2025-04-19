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
    
    def run_webcam(self):
        """Run the sign language detector with webcam input"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting webcam detection. Press 'q' to quit.")
        
        # Configure window for better viewing
        cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sign Language Recognition", 1280, 720)
        
        while cap.isOpened():
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam")
                break
            
            # Calculate FPS
            self.new_frame_time = time.time()
            fps = 1/(self.new_frame_time - self.prev_frame_time) if (self.new_frame_time - self.prev_frame_time) > 0 else 0
            self.prev_frame_time = self.new_frame_time
            
            # Flip the frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process frame and get prediction
            try:
                prediction_idx, prediction_letter, confidence, roi_coords = self.predict(frame)
                
                # Draw bounding box around hand if detected
                if roi_coords:
                    x_min, y_min, x_max, y_max = roi_coords
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Display prediction if available
                if prediction_letter:
                    prediction_text = f"Letter: {prediction_letter}"
                    confidence_text = f"Confidence: {confidence:.2f}" if confidence else ""
                    
                    # Display prediction with high visibility
                    cv2.putText(frame, prediction_text, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
                    cv2.putText(frame, prediction_text, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 2)
                    
                    if confidence:
                        cv2.putText(frame, confidence_text, (10, 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                        cv2.putText(frame, confidence_text, (10, 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 1)
            except Exception as e:
                print(f"Error during prediction: {e}")
            
            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, "Show hand sign in camera view", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow("Sign Language Recognition", frame)
            
            # Check for exit command (q key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.hands.close()
        cap.release()
        cv2.destroyAllWindows()

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

def main():
    print("="*50)
    print("Sign Language Recognition System")
    print("="*50)
    
    # Find model file
    model_path = find_model_file()
    
    if not model_path:
        print("Error: No model file found in the current directory.")
        print("Please ensure you have a trained model file (.pth, .pt, or .pkl) in this folder.")
        return
    
    print(f"Found model file: {model_path}")
    
    # Number of classes in your model - fixed at 24 for ASL static signs
    num_classes = 24
    
    try:
        # Initialize detector
        detector = SignLanguageDetector(model_path, num_classes)
        
        # Start webcam detection
        detector.run_webcam()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()