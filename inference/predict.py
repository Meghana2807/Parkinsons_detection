import torch
import torch.nn.functional as F
from Training.model import CNNTransformer


# -----------------------------
# LOAD MODEL
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNTransformer(num_classes=2)

model.load_state_dict(
    torch.load("models/transformer_model.pth", map_location=device)

)

model.to(device)
model.eval()


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_with_severity(image_tensor):
    """
    image_tensor: torch tensor [1, 3, 224, 224]
    """

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

    # Extract probabilities
    healthy_prob = probs[0][0].item()
    pd_prob = probs[0][1].item()

    healthy_percent = round(healthy_prob * 100, 2)
    pd_percent = round(pd_prob * 100, 2)

    # -----------------------------
    # CLASSIFICATION (FIXED)
    # -----------------------------
    healthy_prob = probs[0][0].item()
    pd_prob = probs[0][1].item()

# -----------------------------
# CALIBRATION (IMPORTANT)
# -----------------------------
    pd_prob = pd_prob + 0.15   # boost Parkinson sensitivity

# -----------------------------
# CLASS DECISION
# -----------------------------
    if pd_prob > healthy_prob:
        predicted_class = "Parkinson’s Disease"
        confidence = round(pd_prob * 100, 2)
    else:
        predicted_class = "Healthy Control"
        confidence = round(healthy_prob * 100, 2)

    # -----------------------------
    # SEVERITY CLASSIFICATION
    # -----------------------------
    if predicted_class == "Healthy Control":
        return {
            "class": predicted_class,
            "confidence": confidence,
            "severity": "None",
            "alert": "No Parkinson’s disease detected."
        }

    # Only for Parkinson cases
    if 0.5 <= pd_prob < 0.65:
        severity = "Early Stage"
        alert = "Early signs detected. Regular monitoring recommended."

    elif 0.65 <= pd_prob < 0.80:
        severity = "Moderate Stage"
        alert = "Moderate risk detected. Consult a neurologist."

    else:
        severity = "Severe Stage"
        alert = "High risk detected. Immediate medical attention advised."

    return {
        "class": predicted_class,
        "confidence": confidence,
        "severity": severity,
        "alert": alert
    }