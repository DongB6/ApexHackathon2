import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# ------------------------
# Load Trained Model
# ------------------------
@st.cache_resource
def load_model():
    try:
        model = torch.load("model.pt", map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

model = load_model()

# ------------------------
# Labeling
# ------------------------
id2label = {
    0: "Healthy",
    1: "Disease"
}

# ------------------------
# Preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Skin Disease Detection", page_icon=":guardsman:", layout="centered")

st.title("🩺 AI-Powered Skin Disease Detection")
st.markdown("Upload a skin image and let the AI predict the possible conditions:")

uploaded_file = st.file_uploader("Upload a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    if model is not None:
        # ------------------------
        # Inference with Progress
        # ------------------------
        with st.spinner("Analyzing the image..."):
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                topk = torch.topk(probs, k=min(3, len(id2label)))

        # ------------------------
        # Results
        # ------------------------
        st.subheader("Prediction Results:")

        for i in range(topk.indices.size(0)):
            label = id2label[topk.indices[i].item()]
            confidence = topk.values[i].item() * 100
            st.write(f"{i + 1}. {label}: {confidence:.2f}%")

        # Highlight the most likely condition
        best_label = id2label[topk.indices[0].item()]
        best_confidence = topk.values[0].item() * 100
        st.markdown(f"**✅ Most Likely Condition:** {best_label} ({best_confidence:.2f}%)")
    else:
        st.warning("No trained model found. Please train the model and save it as `model.pt` first.")
