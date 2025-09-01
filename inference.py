import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# Path to your saved model
MODEL_DIR = "./models/synthetic_derm_classifier"  
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model
image_processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR).to(device)

# Optional: confirm labels
id2label = model.config.id2label if hasattr(model.config, "id2label") else {
    0: "Healthy",
    1: "Disease"
}

def predict_image(image_path: str):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(-1).item()

    label = id2label[pred]
    return label

if __name__ == "__main__":
    test_image = "test_image.jpg"  # replace with your test file
    prediction = predict_image(test_image)
    print("Predicted disease:", prediction)
