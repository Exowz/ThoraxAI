"""Model loading, preprocessing, prediction, and Grad-CAM generation."""

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import cv2

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, CHECKPOINT_DIR, get_device
from src.model import create_model
from src.eval import GradCAM


@st.cache_resource
def load_model(model_name: str):
    device = get_device()
    fine_tune = model_name != "cnn_baseline"
    model = create_model(model_name, fine_tune=fine_tune, device=device)
    try:
        ckpt = torch.load(f"{CHECKPOINT_DIR}/best_{model_name}.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, device, True, ckpt.get("epoch", "?")
    except FileNotFoundError:
        return model, device, False, None


def preprocess(image: Image.Image) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return t(image).unsqueeze(0)


def predict(model, image, device):
    with torch.no_grad():
        return torch.sigmoid(model(preprocess(image).to(device))).item()


def make_gradcam(model, image, device):
    gc = GradCAM(model, model.last_conv_layer)
    inp = preprocess(image).to(device).requires_grad_(True)
    cam = gc.generate(inp)
    cam_r = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    hm = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB) / 255.0
    return img_np, hm, np.clip(0.55 * img_np + 0.45 * hm, 0, 1)
