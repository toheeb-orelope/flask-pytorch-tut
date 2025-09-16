from flask import Flask, request, jsonify
from app.torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# check if the file is an allowed type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def predict():
    # error handling
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})
        if not allowed_file(file.filename):
            return jsonify({"error": "file type not supported"})
        try:
            img_bytes = file.read()
            image_tensor = transform_image(img_bytes)
            prediction = get_prediction(image_tensor)
            data = {
                "prediction": prediction,
                "class_name": str(prediction),
            }
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    # 1 Load image
    # 2 image --> tensor
    # 3 prediction
    # 4 return json data
    # return jsonify({"message": "This is a placeholder for prediction results."})
