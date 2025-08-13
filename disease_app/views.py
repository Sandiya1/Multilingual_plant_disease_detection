import os
import cv2
import base64
import tempfile
import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import requests

# ==================== MODEL SETUP ====================
MODEL_PATH = os.path.join("disease_app", "model", "model.h5")
model = load_model(MODEL_PATH)
img_height, img_width = model.input_shape[1:3]

class_labels = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Apple Rust', 'Apple Healthy',
    'Blueberry Healthy', 'Cherry Powdery Mildew', 'Cherry Healthy',
    'Corn Cercospora Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn Healthy',
    'Grape Black Rot', 'Grape Esca (Black Measles)', 'Grape Leaf Blight (Isariopsis Leaf Spot)', 'Grape Healthy',
    'Orange Huanglongbing (Citrus Greening)',
    'Peach Bacterial Spot', 'Peach Healthy',
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
    'Raspberry Healthy',
    'Soybean Healthy',
    'Squash Powdery Mildew',
    'Strawberry Leaf Scorch', 'Strawberry Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
    'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites (Two-spotted)', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Tomato Healthy'
]

prescriptions = {
    'Apple Scab': "Remove infected leaves, prune tree for airflow, spray with fungicide.",
    'Apple Black Rot': "Remove and destroy infected fruit and branches. Apply copper-based fungicide.",
    'Apple Cedar Apple Rust': "Remove nearby juniper hosts. Apply preventive fungicide in spring.",
    'Apple Healthy': "No action needed. Maintain regular watering and fertilization.",
    'Blueberry Healthy': "No action needed. Keep soil acidic and well-drained.",
    'Cherry Powdery Mildew': "Apply sulfur-based fungicide, prune for better airflow.",
    'Cherry Healthy': "No action needed. Maintain regular care.",
    'Corn Cercospora Leaf Spot': "Rotate crops, use resistant varieties, apply fungicide.",
    'Corn Common Rust': "Plant resistant varieties, apply fungicide if severe.",
    'Corn Northern Leaf Blight': "Use resistant seeds, rotate crops, apply fungicide.",
    'Corn Healthy': "No action needed. Maintain regular fertilization.",
    'Grape Black Rot': "Remove infected leaves, use fungicide during growing season.",
    'Grape Esca (Black Measles)': "Prune infected vines, avoid wounds, apply fungicide.",
    'Grape Leaf Blight (Isariopsis Leaf Spot)': "Remove infected leaves, use fungicide.",
    'Grape Healthy': "No action needed.",
    'Orange Huanglongbing (Citrus Greening)': "Remove infected trees, control psyllids with insecticide.",
    'Peach Bacterial Spot': "Remove infected fruit, apply copper fungicide.",
    'Peach Healthy': "No action needed.",
    'Pepper Bell Bacterial Spot': "Remove infected plants, apply copper spray.",
    'Pepper Bell Healthy': "No action needed.",
    'Potato Early Blight': "Use resistant varieties, apply fungicide, rotate crops.",
    'Potato Late Blight': "Destroy infected plants, apply copper fungicide.",
    'Potato Healthy': "No action needed.",
    'Raspberry Healthy': "No action needed.",
    'Soybean Healthy': "No action needed.",
    'Squash Powdery Mildew': "Apply sulfur fungicide, improve air circulation.",
    'Strawberry Leaf Scorch': "Remove infected leaves, use fungicide.",
    'Strawberry Healthy': "No action needed.",
    'Tomato Bacterial Spot': "Remove infected plants, apply copper spray.",
    'Tomato Early Blight': "Use resistant varieties, apply fungicide.",
    'Tomato Late Blight': "Destroy infected plants, apply copper fungicide.",
    'Tomato Leaf Mold': "Increase ventilation, apply fungicide.",
    'Tomato Septoria Leaf Spot': "Remove infected leaves, apply fungicide.",
    'Tomato Spider Mites (Two-spotted)': "Spray with miticide or insecticidal soap.",
    'Tomato Target Spot': "Apply fungicide, avoid overhead watering.",
    'Tomato Yellow Leaf Curl Virus': "Control whiteflies, remove infected plants.",
    'Tomato Mosaic Virus': "Remove infected plants, disinfect tools.",
    'Tomato Healthy': "No action needed."
}

# ==================== HELPER FUNCTION ====================
def process_leaf_image(image, is_bytes=False):
    """
    Takes uploaded file or bytes from webcam, preprocesses, predicts, and returns
    prediction, care instructions, and Base64 image for display.
    """
    if is_bytes:
        # Convert bytes to OpenCV image
        np_arr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        # Uploaded file
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Resize
    img_resized = cv2.resize(img, (img_width, img_height))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    
    # Prediction
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    label = class_labels[class_idx]
    care = prescriptions.get(label, "No instructions available.")
    
    # Encode image back to base64 for display
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return label, care, img_base64

# ==================== VIEWS ====================
def predict_disease(request):
    if request.method == "POST":
        action = request.POST.get("action")

        if action == "upload":
            leaf_image = request.FILES.get("leaf_image")
            if not leaf_image:
                return render(request, "upload.html", {"error": "Please upload an image."})
            prediction, care, image_data = process_leaf_image(leaf_image)

        elif action == "webcam":
            image_data_url = request.POST.get("captured_image")
            if not image_data_url:
                return render(request, "upload.html", {"error": "No photo captured."})

            header, encoded = image_data_url.split(",", 1)
            decoded_image = base64.b64decode(encoded)
            prediction, care, image_data = process_leaf_image(decoded_image, is_bytes=True)

        return render(request, "upload.html", {
            "prediction": prediction,
            "care": care,
            "image_data": image_data
        })

    return render(request, "upload.html")

# ==================== FERTILIZER SHOPS MAP ====================
@require_GET
def fertilizer_shops_mapbox(request):
    lat = request.GET.get('lat')
    lng = request.GET.get('lng')
    if not lat or not lng:
        return JsonResponse({'error': 'Missing lat or lng parameters'}, status=400)

    try:
        lat = float(lat)
        lng = float(lng)
    except ValueError:
        return JsonResponse({'error': 'Invalid lat or lng'}, status=400)

    # Overpass API query
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["shop"="fertilizer"](around:2000,{lat},{lng});
      way["shop"="fertilizer"](around:2000,{lat},{lng});
      relation["shop"="fertilizer"](around:2000,{lat},{lng});
    );
    out center;
    """
    response = requests.post(overpass_url, data={'data': query})
    data = response.json()

    shops = []
    for element in data.get('elements', []):
        name = element['tags'].get('name', 'Unnamed Shop')
        address_parts = []
        for key in ['addr:street', 'addr:housenumber', 'addr:city']:
            if element['tags'].get(key):
                address_parts.append(element['tags'][key])
        address = ", ".join(address_parts) if address_parts else ""

        if element['type'] == 'node':
            el_lat = element['lat']
            el_lng = element['lon']
        else:
            el_lat = element['center']['lat']
            el_lng = element['center']['lon']

        shops.append({
            'name': name,
            'address': address,
            'lat': el_lat,
            'lng': el_lng,
        })

    return JsonResponse({'shops': shops})