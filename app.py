from flask import Flask, render_template, request
from predictor import Predictor
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
CSV_EXTENSIONS = {".csv"}

predictor = Predictor()

def _extension(filename):
    return os.path.splitext(filename)[1].lower()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    image_file = request.files.get("image")
    csv_file = request.files.get("csv")

    has_image = bool(image_file and image_file.filename)
    has_csv = bool(csv_file and csv_file.filename)

    errors = []

    if has_image and _extension(image_file.filename) not in IMAGE_EXTENSIONS:
        errors.append(
            f"Handwriting Image: \"{image_file.filename}\" is not a supported image "
            f"file. Please upload a PNG, JPG, JPEG, BMP, GIF, TIFF, or WEBP image."
        )

    if has_csv and _extension(csv_file.filename) not in CSV_EXTENSIONS:
        errors.append(
            f"Kinematic CSV: \"{csv_file.filename}\" is not a CSV file. "
            f"Please upload a file with a .csv extension."
        )

    if not has_image and not has_csv:
        errors.append(
            "Please upload a handwriting image, a kinematic CSV, or both "
            "before running a prediction."
        )

    if errors:
        return render_template("index.html", errors=errors)

    image_path = None
    csv_path = None

    if has_image:
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

    if has_csv:
        csv_path = os.path.join(UPLOAD_FOLDER, csv_file.filename)
        csv_file.save(csv_path)

    try:
        result = predictor.predict(image_path=image_path, csv_path=csv_path)
    except ValueError as e:
        return render_template("index.html", errors=[str(e)])
    except Exception:
        return render_template(
            "index.html",
            errors=[
                "An unexpected error occurred while processing your input. "
                "Please check your files and try again."
            ],
        )

    if result.get("error"):
        return render_template("index.html", errors=[result["error"]])

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)