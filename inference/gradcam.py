import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(model, input_tensor):

    model.eval()

    # Target layer (last CNN layer)
    target_layer = model.cnn[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(1)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam[0]