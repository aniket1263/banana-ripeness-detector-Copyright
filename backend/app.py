from flask import Flask, jsonify, render_template
from flask_cors import CORS
from routes.predict import predict_bp
import os

# Use relative paths — works inside Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR, "frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "frontend", "static")
)

CORS(app, resources={r"/*": {"origins": "*"}})
app.register_blueprint(predict_bp)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"health": "ok"}), 200

if __name__ == '__main__':
    print("=" * 40)
    print("🍌  Banana Ripeness API")
    print("🌐  http://127.0.0.1:5000")
    print("=" * 40)
    app.run(host="0.0.0.0", port=5000, debug=False)