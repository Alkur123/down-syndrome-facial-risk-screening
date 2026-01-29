import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import gradio as gr
import cv2

from huggingface_hub import hf_hub_download

# ---------------- CONFIG ----------------

MODEL_REPO_ID = "jash-ai/ds-ensemble-models"  # HF model repo
EFF_FILENAME = "effb0_best.pth"
RES18_FILENAME = "resnet18_best.pth"

IMG_SIZE = 224
THRESHOLD = 0.80   # ensemble decision threshold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# ---------------- DOWNLOAD WEIGHTS FROM HUB ----------------

def get_weights_from_hub():
    """
    Downloads model weights from the Hugging Face Hub (cached after first use).
    """
    print(f"Downloading weights from repo: {MODEL_REPO_ID}")
    eff_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=EFF_FILENAME)
    res_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=RES18_FILENAME)
    print("Downloaded weight paths:")
    print("  EfficientNet-B0:", eff_path)
    print("  ResNet18:", res_path)
    return eff_path, res_path


EFF_WEIGHTS, RES18_WEIGHTS = get_weights_from_hub()


# ---------------- MODEL BUILDERS ----------------

def build_effb0(num_classes: int = 1):
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.get_classifier().in_features
    model.reset_classifier(0)
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def build_resnet18(num_classes: int = 1):
    model = timm.create_model("resnet18", pretrained=False)
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        in_features = model.get_classifier().in_features
        model.reset_classifier(num_classes)
    return model


# ---------------- LOAD MODELS ----------------

def load_models():
    effb0 = build_effb0().to(DEVICE)
    eff_state = torch.load(EFF_WEIGHTS, map_location=DEVICE)
    effb0.load_state_dict(eff_state)
    effb0.eval()

    res18 = build_resnet18().to(DEVICE)
    res_state = torch.load(RES18_WEIGHTS, map_location=DEVICE)
    res18.load_state_dict(res_state)
    res18.eval()

    print("Loaded EfficientNet-B0 and ResNet18 weights.")
    return effb0, res18


effb0, res18 = load_models()


# ---------------- SIMPLE GRAD-CAM IMPLEMENTATION ----------------

class SimpleGradCAM:
    """
    Very small Grad-CAM implementation for a single target layer.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_backward_hook(bwd_hook)

    def __call__(self, input_tensor):
        """
        input_tensor: (1, C, H, W)
        returns: numpy array (H, W) in [0,1]
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)  # (1,1) for binary
        score = logits.squeeze()           # scalar
        score.backward(retain_graph=True)

        grads = self.gradients        # (B, C, H, W)
        acts = self.activations       # (B, C, H, W)

        weights = grads.mean(dim=(2, 3), keepdim=True)      # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)     # (B,1,H,W)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam  # (H,W) in [0,1]


def overlay_cam_on_image(rgb_img, mask, alpha=0.4):
    """
    rgb_img: HxWx3 float32 in [0,1]
    mask:    HxW float32 in [0,1]
    returns: uint8 HxWx3 overlay
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = heatmap * alpha + rgb_img * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


# Grad-CAM with EfficientNet-B0 conv_head
target_layer = effb0.conv_head
cam_explainer = SimpleGradCAM(model=effb0, target_layer=target_layer)


# ---------------- PREPROCESSING ----------------

def preprocess_np_for_model(img_np: np.ndarray):
    """
    img_np: HxWxC uint8 from Gradio
    returns:
        tensor: (1, C, H, W) normalized
        rgb_vis: HxWxC float32 in [0,1] for overlay
    """
    pil = Image.fromarray(img_np).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(pil).astype(np.float32) / 255.0
    rgb_vis = img_np.copy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_np - mean) / std
    img_norm = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW

    tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0)
    return tensor.to(DEVICE), rgb_vis


# ---------------- ENSEMBLE PREDICT + EXPLAIN ----------------

def predict_and_explain(image_np, confirm_flag: bool):
    if image_np is None:
        return "No image", 0.0, None, "Upload an image to begin."

    if not confirm_flag:
        return (
            "Confirmation needed",
            0.0,
            Image.fromarray(image_np),
            "Tick the confirmation checkbox to acknowledge this is a research prototype, not medical advice.",
        )

    tensor, rgb_vis = preprocess_np_for_model(image_np)

    with torch.no_grad():
        logit_eff = effb0(tensor).squeeze(1)
        logit_res = res18(tensor).squeeze(1)
        avg_logits = (logit_eff + logit_res) / 2.0
        prob = torch.sigmoid(avg_logits).item()

    is_ds = prob >= THRESHOLD
    pred_label = "downSyndrome" if is_ds else "healthy"

    # CASE 1: healthy → no GradCAM, return original image
    if not is_ds:
        explanation = (
            f"Ensemble prediction: healthy (confidence {prob:.3f}, "
            f"threshold {THRESHOLD:.2f}).\n"
            "No high-risk facial patterns detected.\n"
            "This is a research prototype, NOT a medical or diagnostic tool."
        )
        overlay = Image.fromarray(image_np)
        return pred_label, prob, overlay, explanation

    # CASE 2: downSyndrome → GradCAM overlay using EfficientNet-B0
    mask = cam_explainer(tensor)              # (H,W) in [0,1]
    overlay_np = overlay_cam_on_image(rgb_vis, mask)
    overlay = Image.fromarray(overlay_np)

    explanation = (
        f"Ensemble prediction: downSyndrome (confidence {prob:.3f}, "
        f"threshold {THRESHOLD:.2f}).\n"
        "Highlighted regions show where the model focused (Grad-CAM on EfficientNet-B0).\n"
        "This is a research prototype and MUST NOT be used as medical advice or diagnosis."
    )

    return pred_label, prob, overlay, explanation


# ---------------- GRADIO UI ----------------

title = "Down Syndrome Research Prototype (Ensemble + Grad-CAM)"
description = (
    "Upload a face image, confirm you understand this is NOT a medical tool, "
    "and see an ensemble prediction (EfficientNet-B0 + ResNet18) with Grad-CAM "
    "explanations for positive cases. Decision threshold: "
    f"{THRESHOLD:.2f}. Research & educational use only."
)

demo = gr.Interface(
    fn=predict_and_explain,
    inputs=[
        gr.Image(type="numpy", label="Upload face image"),
        gr.Checkbox(label="I understand this is NOT medical advice."),
    ],
    outputs=[
        gr.Textbox(label="Predicted label"),
        gr.Number(label="Confidence (probability)"),
        gr.Image(label="Image / Grad-CAM"),
        gr.Textbox(label="Explanation"),
    ],
    title=title,
    description=description,
)

if __name__ == "__main__":
    demo.launch()