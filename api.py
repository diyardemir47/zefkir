from flask import Flask, request, jsonify
import traceback
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"max_upload_mb": 2, "status": "ok"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = "upload.jpg"
        file.save(filename)
        size = os.path.getsize(filename)

        return jsonify({
            "status": "ok",
            "message": f"File received: {file.filename}",
            "saved_as": filename,
            "size_bytes": size
        })
    except Exception as e:
        print("ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
