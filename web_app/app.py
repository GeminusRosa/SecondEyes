import torch
import torch.nn.functional as F
from PIL import Image
from typing import cast, Callable
from flask import Flask
from torch import Tensor
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL.Image import Image as PILImage
from flask import Flask, request, jsonify, send_from_directory

from src.models.fusion import ForensicNet

# Custom transform to handle images of any size
class SmartCropPad(object):
    """
    If image is smaller than 256x256: pad to 256x256
    If image is larger than 256x256: center crop to 256x256
    """
    def __init__(self, size=256):
        self.size = size
    
    def __call__(self, img):
        w, h = tuple(img.size)
        
        # If image is too small, pad it to 256x256
        if w < self.size or h < self.size:
            pad_w = max(0, self.size - w)
            pad_h = max(0, self.size - h)
            
            # Distribute padding evenly on both sides
            left = pad_w // 2
            top = pad_h // 2
            right = pad_w - left
            bottom = pad_h - top
            
            img = TF.pad(img, [left, top, right, bottom], fill=0)
        
        # Center crop to 256x256 if needed
        if w > self.size or h > self.size:
            img = transforms.CenterCrop(self.size)(img)
        
        return img

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    "best_model.pth",
    map_location=device
)

classes = checkpoint["classes"]

model = ForensicNet(len(classes))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

transform = cast(
    Callable[[PILImage], Tensor],
    transforms.Compose([
        SmartCropPad(size=256),
        transforms.ToTensor()
    ])
)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR / "static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # File validation
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        
        if file.filename == "" or file.filename is None:
            return jsonify({"error": "No file selected"}), 400
        
        # Check file size (max 10MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({"error": "File size exceeds 10MB limit"}), 400
        
        # Check file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({"error": f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"}), 400
        
        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            return jsonify({"error": "Invalid image file or corrupted image"}), 400

        tensor = transform(image)

        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():

            binary_logits, class_logits, feats = model(tensor)

            binary_probs = F.softmax(binary_logits, dim=1)[0]
            class_probs = F.softmax(class_logits, dim=1)[0]

        real_prob = float(binary_probs[0].item())
        fake_prob = float(binary_probs[1].item())

        class_idx = int(class_probs.argmax().item())
        
        # Get top 3 classes and their probabilities (including all classes)
        top_3_probs, top_3_indices = torch.topk(class_probs, k=min(3, len(classes)))
        top_classes = [
            {
                "class": classes[int(idx.item())],
                "probability": round(float(prob.item()) * 100, 2)
            }
            for prob, idx in zip(top_3_probs, top_3_indices)
        ]

        return jsonify({
            "real_probability": round(real_prob * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "predicted_class": classes[class_idx],
            "class_confidence": round(
                float(class_probs[class_idx].item()) * 100,
                2
            ),
            "all_classes": top_classes
        })
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)