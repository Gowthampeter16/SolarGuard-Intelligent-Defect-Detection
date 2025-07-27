import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import io

# -------------------------------
# Load Class Names
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
num_classes = len(class_names)

# -------------------------------
# Load Trained Model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load("mobilenet_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Title and Upload
st.title("🔍 Solar Panel Condition Classifier (MobileNetV2)")
st.write("Upload a solar panel image and classify its condition.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # add batch dim

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx].item()

    # Display Result
    st.markdown(f"### 🧠 Predicted Class: **{pred_class}**")
    st.markdown(f"### 🔢 Confidence: `{confidence:.2f}`")

    # Show probabilities
    st.subheader("📊 Class Probabilities")
    fig, ax = plt.subplots()
    ax.barh(class_names, probs.numpy(), color='skyblue')
    ax.set_xlabel("Probability")
    ax.set_title("Class Distribution")
    st.pyplot(fig)

# -------------------------------
# Show Model Performance Metrics
st.divider()
st.subheader("📈 Model Performance (Validation Set)")

if st.button("📥 Show Validation Report"):
    # Replace with real val_loader or pre-saved outputs
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    val_dir = "val_data"  # path to validation data directory
    val_dataset = ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.numpy())
            pred_labels.extend(preds.numpy())

    report = classification_report(true_labels, pred_labels, target_names=class_names)
    st.text_area("📋 Classification Report", report, height=300)

    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)