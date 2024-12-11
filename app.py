import cv2
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from flask_cors import CORS
from flask import Flask, request, jsonify

CLASS_MAPPING = {
    0: "Avulsion Fracture",
    1: "Comminuted Fracture",
    2: "Fracture Dislocation",
    3: "Greenstick Fracture",
    4: "Hairline Fracture"
}
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
CORS(app)

def preprocess(image_list, edge_detection_type='canny'):
    blurred_image = [cv2.medianBlur(img, 5) for img in image_list]
    if edge_detection_type == "sobel":
        sobel_edges = [
            cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            for img in blurred_image
        ]
        edge_detected_images = [cv2.convertScaleAbs(edge) for edge in sobel_edges]
    elif edge_detection_type == "canny":
        edge_detected_images = [
            cv2.Canny(img, 50, 100) for img in blurred_image
        ]
    else:
        raise ValueError("Invalid edge_detection_type. Choose 'sobel' or 'canny'.")
    return np.array(edge_detected_images)

def extract_descriptors(image_list, descriptor_type="SIFT", max_descriptors=None):
    if descriptor_type == "SIFT":
        extractor = cv2.SIFT_create(nfeatures=max_descriptors if max_descriptors else 0)
        descriptor_dim = 128
    elif descriptor_type == "AKAZE":
        extractor = cv2.AKAZE_create()
        descriptor_dim = 64
    else:
        raise ValueError("Invalid descriptor_type")

    all_descriptors = []
    descriptor_counts = []

    for img in image_list:
        _, descriptors = extractor.detectAndCompute(img, None)
        if descriptors is None:
            descriptor_count = 0
            descriptors = np.zeros((0, descriptor_dim), dtype=np.float32)
        else:
            descriptor_count = descriptors.shape[0]
            if max_descriptors and descriptor_count > max_descriptors:
                descriptors = descriptors[:max_descriptors]
            descriptors = normalize(descriptors, norm='l2')
        descriptor_counts.append(min(max_descriptors, descriptor_count) if max_descriptors else descriptor_count)
        all_descriptors.append(descriptors)

    return all_descriptors, descriptor_counts

def create_bovw(descriptors_list, kmeans):
    bovw_features = []
    for descriptor in descriptors_list:
        histogram = np.zeros(kmeans.n_clusters)
        for desc in descriptor:
            if len(desc) == kmeans.n_features_in_:
                cluster_idx = kmeans.predict(desc.reshape(1, -1))[0]
                histogram[cluster_idx] += 1
        bovw_features.append(histogram)
    return np.array(bovw_features)

def load_model():
    with open("./models/kmeans_sift.pkl", 'rb') as f:
        kmeans_sift_tester = pickle.load(f)

    with open("./models/kmeans_akaze.pkl", 'rb') as f:
        kmeans_akaze_tester = pickle.load(f)

    with open("./models/rf_model.pkl", 'rb') as f:
        rf_tester = pickle.load(f)

def validate_file_format(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict-image", methods=["POST"])
def predict_image():
    if 'file' not in request.files:
        return jsonify({
            "message": "file not found",
        }, 400)
    file = request.files['file']
    if file.filename == '':
        return jsonify({
        'message' : 'file not provided'
        }, 400)
    if not validate_file_format(file.filename):
        return jsonify({"message": "Invalid file format"}), 400
    try:
        print("File not read")
        file_stream = file.read()
        print("File read")
        np_image = np.frombuffer(file_stream, np.uint8)
        print("np_image")
        image = cv2.imdecode(np_image, cv2.IMREAD_GRAYSCALE) 
        image = cv2.resize(image, (256, 256))
        print("File decoded ", image.shape)
        preprocessed_images = preprocess([image], edge_detection_type="sobel")
        print("Image preprocessed", preprocessed_images.shape)

        preprocessed_images_sift, _ = extract_descriptors(preprocessed_images, descriptor_type='SIFT')
        preprocessed_images_akaze, _ = extract_descriptors(preprocessed_images, descriptor_type='AKAZE')
        print("Descriptors done.")

        print("Model loading...")
        with open("./models/kmeans_sift.pkl", 'rb') as f:
            kmeans_sift = pickle.load(f)
        with open("./models/kmeans_akaze.pkl", 'rb') as f:
            kmeans_akaze = pickle.load(f)
        with open("./models/rf_model.pkl", 'rb') as f:
            rf_model = pickle.load(f)
        print("Model loaded")
        print(rf_model)
        preprocessed_bovw_sift = create_bovw(preprocessed_images_sift, kmeans_sift)
        preprocessed_bovw_akaze = create_bovw(preprocessed_images_akaze, kmeans_akaze)

        preprocessed_bovw = np.hstack([preprocessed_bovw_sift, preprocessed_bovw_akaze])

        prediction = rf_model.predict(preprocessed_bovw)
        predicted_class = CLASS_MAPPING[prediction[0]]
        print("Prediction successful:", predicted_class)
        return jsonify({
            "predicted_class": predicted_class,
        }), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True, threaded=True)