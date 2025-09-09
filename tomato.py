import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

"""
Tomato Disease Classification Model
(using MobileNetV2 .h5 file)
"""

# Configuration
MODEL_PATH = "tomato_disease_detector_mobilenetv2.h5"

# Default placeholder class names (will override if found in model metadata)
CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "YellowLeaf_Curl_Virus",
    "Mosaic_virus",
    "Healthy"
]

IMG_SIZE = (224, 224)
MIN_CONFIDENCE = 0.7  # Threshold for reliable predictions


def load_model():
    """Load and cache the TensorFlow model"""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully: {MODEL_PATH}")

        # Try to extract class names if available
        if hasattr(model, "class_names"):
            print("📌 Found class names in model metadata")
            global CLASS_NAMES
            CLASS_NAMES = model.class_names
        else:
            print("⚠️ Using default CLASS_NAMES (check training dataset for accuracy)")

        print(f"📋 Class Names: {CLASS_NAMES}")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Model loading failed: {str(e)}")


def predict(image_path):
    """
    Predict tomato disease from image
    
    Args:
        image_path: Path to image file (JPG/PNG)
    
    Returns:
        Dictionary with:
        - disease: Predicted class name
        - confidence: Prediction probability (0-1)
        - all_probabilities: Dict of all class probabilities
        - is_reliable: Bool if confidence > threshold
        - timestamp: ISO format timestamp
    """
    try:
        # Preprocess
        img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        probs = model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        
        return {
            "disease": CLASS_NAMES[class_idx],
            "confidence": confidence,
            "all_probabilities": {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs)},
            "is_reliable": confidence >= MIN_CONFIDENCE,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def save_prediction(result, output_file="predictions.json"):
    """Save prediction results to JSON file"""
    with open(output_file, 'a') as f:
        json.dump(result, f)
        f.write('\n')  # Newline for multiple entries


if __name__ == "__main__":
    # Initialize
    model = load_model()
    
    # Example usage
    test_image = "images/leafmold_tomato.png"  # 🔹 Your test image
    if os.path.exists(test_image):
        result = predict(test_image)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            # Print human-readable summary
            print(f"\n🍅 Prediction Result:")
            print(f"- Disease: {result['disease']}")
            print(f"- Confidence: {result['confidence']:.1%}")
            print(f"- Reliable: {'Yes' if result['is_reliable'] else 'No (low confidence)'}")
            
            # Save full results
            save_prediction(result)
            print(f"\n📂 Full results saved to predictions.json")
    else:
        print(f"❌ Error: Image not found at {test_image}")
