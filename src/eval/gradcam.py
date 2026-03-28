"""
gradcam.py -- Classes d'interpretabilite (Grad-CAM, Grad-CAM++, Integrated Gradients).

Contient :
- GradCAM : implementation from scratch (Selvaraju et al., ICCV 2017)
- GradCAMPlusPlus : meilleure localisation (Chattopadhay et al., WACV 2018)
- IntegratedGradients : attributions pixel par pixel (Sundararajan et al., ICML 2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Implementation de Grad-CAM.

    Ref: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

    Usage:
        grad_cam = GradCAM(model, model.last_conv_layer)
        heatmap = grad_cam.generate(input_tensor)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image: torch.Tensor) -> np.ndarray:
        """Genere la carte de chaleur Grad-CAM.

        Args:
            input_image: Tensor [1, 3, H, W] avec requires_grad=True

        Returns:
            Heatmap normalisee [0, 1] de shape (h, w) -- taille de la feature map
        """
        self.model.eval()
        output = self.model(input_image)
        self.model.zero_grad()
        output.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class GradCAMPlusPlus:
    """Implementation de Grad-CAM++.

    Ref: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based Visual
    Explanations for Deep Convolutional Networks", WACV 2018.

    Utilise les derivees secondes et troisiemes (gradients au carre et au cube)
    pour calculer des poids alpha plus precis, ce qui donne une meilleure
    localisation que Grad-CAM classique.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image: torch.Tensor) -> np.ndarray:
        """Genere la carte de chaleur Grad-CAM++.

        Args:
            input_image: Tensor [1, 3, H, W] avec requires_grad=True

        Returns:
            Heatmap normalisee [0, 1] de shape (h, w)
        """
        self.model.eval()
        output = self.model(input_image)
        self.model.zero_grad()
        output.backward()

        grads = self.gradients  # [1, C, H, W]
        acts = self.activations  # [1, C, H, W]

        # Derivees d'ordre superieur
        grads_2 = grads ** 2
        grads_3 = grads ** 3

        # Poids alpha (Eq. 9 du papier)
        denominator = 2.0 * grads_2 + (grads_3 * acts).sum(dim=[2, 3], keepdim=True)
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        alpha = grads_2 / denominator

        # Poids par canal
        weights = (alpha * F.relu(grads)).sum(dim=[2, 3], keepdim=True)

        # Carte de chaleur
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class IntegratedGradients:
    """Implementation d'Integrated Gradients.

    Ref: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.

    Calcule les attributions pixel par pixel en interpolant entre une baseline
    (image noire, tensor de zeros) et l'image d'entree, sur n_steps etapes.

    IG(x) = (x - baseline) * mean(gradients le long du chemin)
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute(
        self,
        input_image: torch.Tensor,
        n_steps: int = 50,
        baseline: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Calcule les attributions Integrated Gradients.

        Args:
            input_image: Tensor [1, 3, H, W]
            n_steps: Nombre d'etapes d'interpolation
            baseline: Image de reference (zeros par defaut)

        Returns:
            Attributions de shape (H, W) normalisees [0, 1]
        """
        self.model.eval()
        device = input_image.device

        if baseline is None:
            baseline = torch.zeros_like(input_image).to(device)

        # Interpolation lineaire entre baseline et input
        alphas = torch.linspace(0, 1, n_steps + 1, device=device).view(-1, 1, 1, 1)
        interpolated = baseline + alphas * (input_image - baseline)

        # Calculer les gradients a chaque etape
        all_gradients = []
        for i in range(n_steps + 1):
            step_input = interpolated[i].unsqueeze(0).requires_grad_(True)
            output = self.model(step_input)
            self.model.zero_grad()
            output.backward()
            all_gradients.append(step_input.grad.detach())

        # Moyenne des gradients (regle du trapeze)
        grads = torch.cat(all_gradients, dim=0)
        avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2.0

        # Attributions = (input - baseline) * avg_grads
        attributions = (input_image - baseline).squeeze(0) * avg_grads

        # Agreger sur les canaux (valeur absolue)
        attr_map = attributions.abs().sum(dim=0).cpu().numpy()
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

        return attr_map
