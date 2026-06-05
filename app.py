from flask import Flask, render_template, request
from predictor import Predictor
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

predictor = Predictor()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    image_file = request.files.get("image")
    csv_file = request.files.get("csv")

    image_path = None
    csv_path = None

    if image_file and image_file.filename:
        image_path = os.path.join(
            UPLOAD_FOLDER,
            image_file.filename
        )
        image_file.save(image_path)

    if csv_file and csv_file.filename:
        csv_path = os.path.join(
            UPLOAD_FOLDER,
            csv_file.filename
        )
        csv_file.save(csv_path)

    result = predictor.predict(
        image_path=image_path,
        csv_path=csv_path
    )

    return render_template(
        "index.html",
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)