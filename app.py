from flask import Flask, render_template, request
import cv2
import joblib
import os

app = Flask(__name__)

model = joblib.load("model.pkl")

def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, (100, 50))
    img = img.flatten()
    return img.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = "upload.jpg"
            filepath = os.path.join("static", filename)
            file.save(filepath)

            data = process_image(filepath)

            if data is None:
                result = "Image not readable ❌"
            else:
                pred = model.predict(data)[0]

                if pred == 1:
                    result = "Genuine Signature ✅"
                else:
                    result = "Fake Signature ❌"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)