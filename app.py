"""
Flask Backend — Pneumonia Detection (model v4: EfficientNetB0, Kermany + RSNA)
Preprocessing must match train_colab.ipynb exactly (see preprocess_image_for_prediction).
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from typing import Optional
import json
from datetime import datetime
import numpy as np
import keras
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/v4/pneumonia_model_v4.keras'
MODEL_CONFIG_PATH = 'models/v4/model_config.json'

# Training-time settings exported by train_colab.ipynb (threshold chosen on validation, never on test)
IMG_SIZE = 224
THRESHOLD = 0.51
MODEL_CONFIG = {}
try:
    with open(MODEL_CONFIG_PATH, encoding='utf-8') as f:
        MODEL_CONFIG = json.load(f)
    IMG_SIZE = int(MODEL_CONFIG['input']['size'][0])
    THRESHOLD = float(MODEL_CONFIG['threshold'])
except Exception as e:
    print(f"⚠️  Could not read {MODEL_CONFIG_PATH} ({e}); using defaults size={IMG_SIZE}, threshold={THRESHOLD}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
print("=" * 70)
print("🚀 Loading AI Model...")
print("=" * 70)
model: Optional[keras.Model] = None
try:
    loaded = keras.models.load_model(MODEL_PATH)
    if not isinstance(loaded, keras.Model):
        raise TypeError(f"Unexpected model type: {type(loaded)}")
    model = loaded
    print(f"✅ Model loaded successfully!")
    print(f"   Layers: {len(model.layers)}")
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def make_upload_name(original, prefix):
    """Timestamped, filesystem-safe name that always keeps the image extension
    (secure_filename turns e.g. an Arabic name 'أمير.jpeg' into just 'jpeg')."""
    ext = original.rsplit('.', 1)[1].lower()
    stem = secure_filename(original.rsplit('.', 1)[0]) or 'image'
    return f"{prefix}_{stem}.{ext}"

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_prediction(img_path):
    """
    CRITICAL: Preprocess image EXACTLY as done during training (train_colab.ipynb, prep_pil):
    - PIL open -> grayscale ('L') -> resize IMG_SIZE x IMG_SIZE with BILINEAR
    - repeat the gray channel to 3 channels
    - keep pixels in 0..255 (EfficientNet rescales internally — do NOT divide by 255)
    """
    try:
        img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_array = np.asarray(img, dtype=np.float32)
        img_array = np.repeat(img_array[..., None], 3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        assert img_array.shape == (1, IMG_SIZE, IMG_SIZE, 3), f"Wrong shape: {img_array.shape}"
        return img_array

    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

def enhance_image_quality(img_path):
    """
    Optional: Apply image enhancement (use cautiously)
    Only use if images are low quality
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return img_path
        
        # Convert to grayscale for CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Save enhanced image
        root, ext = os.path.splitext(img_path)
        enhanced_path = f"{root}_enhanced{ext or '.png'}"
        cv2.imwrite(enhanced_path, enhanced_rgb)
        
        return enhanced_path
        
    except:
        return img_path

def predict_single_image(img_path, enhance=False):
    """
    Predict on a single image with proper preprocessing
    
    Args:
        img_path: Path to image file
        enhance: Whether to apply image enhancement (default: False)
    
    Returns:
        dict: Prediction results
    """
    try:
        # Optional enhancement (usually not needed for X-rays)
        if enhance:
            img_path = enhance_image_quality(img_path)
        
        # Preprocess image (MATCHES TRAINING EXACTLY)
        img_array = preprocess_image_for_prediction(img_path)
        
        # Predict using the model
        if model is None:
            raise Exception('Model not loaded')
        prediction = model.predict_on_batch(img_array)
        prob = float(prediction[0][0])
        
        # Interpret results
        # Model outputs sigmoid P(pneumonia): > THRESHOLD = Pneumonia (class 1), otherwise Normal
        if prob > THRESHOLD:
            diagnosis = 'Pneumonia'
            confidence = prob
            # Severity based on probability
            if prob > 0.9:
                severity = 'Very High'
            elif prob > 0.75:
                severity = 'High'
            elif prob > 0.6:
                severity = 'Medium'
            else:
                severity = 'Low'
        else:
            diagnosis = 'Normal'
            confidence = 1 - prob
            severity = 'N/A'
        
        # Confidence interpretation
        if confidence > 0.9:
            confidence_level = 'Very Confident'
        elif confidence > 0.75:
            confidence_level = 'Confident'
        elif confidence > 0.6:
            confidence_level = 'Moderate'
        else:
            confidence_level = 'Low Confidence'
        
        return {
            'success': True,
            'diagnosis': diagnosis,
            'confidence': round(confidence * 100, 2),
            'confidence_level': confidence_level,
            'severity': severity,
            'probabilities': {
                'pneumonia': round(prob * 100, 2),
                'normal': round((1 - prob) * 100, 2)
            },
            'raw_output': round(prob, 6),  # For debugging
            'threshold': THRESHOLD,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/api/predict-single', methods=['POST'])
def predict_single():
    """API: Predict on single image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file
        filename = make_upload_name(file.filename, datetime.now().strftime('%Y%m%d_%H%M%S'))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get enhancement preference (optional)
        enhance = request.form.get('enhance', 'false').lower() == 'true'
        
        # Predict
        result = predict_single_image(filepath, enhance=enhance)
        result['filename'] = filename
        result['filepath'] = f'/static/uploads/{filename}'
        
        # Log prediction for monitoring
        print(f"[PREDICTION] {result['diagnosis']} - Confidence: {result['confidence']}% - File: {filename}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-multiple', methods=['POST'])
def predict_multiple():
    """API: Predict on multiple images"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        
        if len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        # Process all files
        results = []
        normal_count = 0
        pneumonia_count = 0
        confidences = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save file
                saved_filename = make_upload_name(file.filename, datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
                file.save(filepath)
                
                # Predict
                prediction = predict_single_image(filepath)
                prediction['filename'] = file.filename
                prediction['saved_filename'] = saved_filename
                prediction['filepath'] = f'/static/uploads/{saved_filename}'
                
                results.append(prediction)
                
                # Statistics
                if prediction['success']:
                    if prediction['diagnosis'] == 'Normal':
                        normal_count += 1
                    else:
                        pneumonia_count += 1
                    confidences.append(prediction['confidence'])
        
        # Calculate statistics
        total_images = len(results)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        statistics = {
            'total_images': total_images,
            'normal_count': normal_count,
            'pneumonia_count': pneumonia_count,
            'normal_percentage': round((normal_count / total_images * 100), 2) if total_images > 0 else 0,
            'pneumonia_percentage': round((pneumonia_count / total_images * 100), 2) if total_images > 0 else 0,
            'avg_confidence': round(avg_confidence, 2),
            'min_confidence': round(min(confidences), 2) if confidences else 0,
            'max_confidence': round(max(confidences), 2) if confidences else 0,
            'std_confidence': round(np.std(confidences), 2) if len(confidences) > 1 else 0
        }
        
        print(f"[BATCH] Processed {total_images} images - Normal: {normal_count}, Pneumonia: {pneumonia_count}")
        
        return jsonify({
            'success': True,
            'total_processed': total_images,
            'results': results,
            'statistics': statistics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_images():
    """API: Compare two images"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'Two files required'}), 400
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'Invalid file types'}), 400
        
        # Process file 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename1 = make_upload_name(file1.filename, f"{timestamp}_1")
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename1)
        file1.save(filepath1)
        result1 = predict_single_image(filepath1)
        result1['filename'] = file1.filename
        result1['filepath'] = f'/static/uploads/{saved_filename1}'
        
        # Process file 2
        saved_filename2 = make_upload_name(file2.filename, f"{timestamp}_2")
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename2)
        file2.save(filepath2)
        result2 = predict_single_image(filepath2)
        result2['filename'] = file2.filename
        result2['filepath'] = f'/static/uploads/{saved_filename2}'
        
        # Comparison analysis
        same_diagnosis = result1['diagnosis'] == result2['diagnosis']
        confidence_diff = abs(result1['confidence'] - result2['confidence'])
        
        # Agreement level
        if same_diagnosis:
            if confidence_diff < 10:
                agreement = 'Strong Agreement'
            elif confidence_diff < 20:
                agreement = 'Moderate Agreement'
            else:
                agreement = 'Weak Agreement'
        else:
            agreement = 'Disagreement'
        
        comparison = {
            'same_diagnosis': same_diagnosis,
            'agreement_level': agreement,
            'confidence_difference': round(confidence_diff, 2),
            'more_confident': 'Image 1' if result1['confidence'] > result2['confidence'] else 'Image 2',
            'summary': f"Both images show {result1['diagnosis']}" if same_diagnosis else f"Different diagnoses detected"
        }
        
        print(f"[COMPARE] Image1: {result1['diagnosis']}, Image2: {result2['diagnosis']} - {agreement}")
        
        return jsonify({
            'success': True,
            'image1': result1,
            'image2': result2,
            'comparison': comparison,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Held-out test results from training (model_config.json); overall = both test sets combined
    tests = MODEL_CONFIG.get('test_results', {})
    kermany = tests.get('kermany_test_624', {}).get('at_threshold', {})
    rsna = tests.get('rsna_test', {}).get('at_threshold', {})
    parts = [m for m in (kermany, rsna) if m]
    tp, tn, fp, fn = (sum(m.get(k, 0) for m in parts) for k in ('tp', 'tn', 'fp', 'fn'))
    total = tp + tn + fp + fn
    pct = lambda a, b: f"{a / b * 100:.1f}%" if b else None
    # per-set metrics from raw counts (the stored values are pre-rounded, e.g. 93.95 would show as 94.0)
    def breakdown(m):
        n = sum(m.get(k, 0) for k in ('tp', 'tn', 'fp', 'fn'))
        return {'images': n,
                'accuracy': round((m.get('tp', 0) + m.get('tn', 0)) / n * 100, 1) if n else None,
                'recall': round(m['tp'] / (m['tp'] + m['fn']) * 100, 1) if n and m['tp'] + m['fn'] else None,
                'auc': m.get('auc')}

    return jsonify({
        'status': 'active',
        'model_name': 'Pneumonia Detection — EfficientNetB0',
        'version': '4.0',
        'architecture': MODEL_CONFIG.get('model', 'EfficientNetB0'),
        'layers': len(model.layers),
        'parameters': int(model.count_params()),
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'input_size': f'{IMG_SIZE}x{IMG_SIZE} grayscale→RGB',
        'preprocessing': 'Grayscale, bilinear resize, pixels 0..255',
        'threshold': THRESHOLD,
        'model_file': MODEL_PATH,
        'accuracy': pct(tp + tn, total),
        'recall': pct(tp, tp + fn),
        'precision': pct(tp, tp + fp),
        'specificity': pct(tn, tn + fp),
        'test_images': total,
        'train_images': MODEL_CONFIG.get('trained_on', {}).get('train'),
        'test_breakdown': {'kermany': breakdown(kermany), 'rsna': breakdown(rsna)},
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 Pneumonia Detection System - Starting Server...")
    print("=" * 70)
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Warning: Model file not found: {MODEL_PATH}")
        print("   Please ensure the model file is in models/v4/")
    
    if model is not None:
        print("\n✅ Model Status: READY")
        print(f"   Model: {MODEL_PATH}")
        print(f"   Input: {IMG_SIZE}x{IMG_SIZE} grayscale→RGB, threshold {THRESHOLD}")
    
    print(f"\n📍 Server: http://localhost:5000")
    print(f"📍 API: http://localhost:5000/api")
    print("\n📋 Endpoints:")
    print("   POST /api/predict-single")
    print("   POST /api/predict-multiple")
    print("   POST /api/compare")
    print("   GET  /api/model-info")
    print("   GET  /api/health")
    print("\n" + "=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)