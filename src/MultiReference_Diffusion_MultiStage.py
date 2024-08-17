"""
Diffusion.py

This file contains the implementation of the Noise Prediction function to generate Latent Representations,
the Denoising Step function, and the whole DDIM Process.
Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 6, 2024
"""



from __future__ import annotations
import torch

from tqdm import tqdm
from typing import Callable
from diffusers import StableDiffusionXLPipeline

from .Tokenization_and_Embedding import embeddings_ensemble_with_neg_conditioning
from .Encode_Image import images_encoding_multistage
from .Diffusion import Generate_Noise_Prediction, Denoising_next_step, DDIM_Process, extract_latent_and_inversion
T = torch.tensor # Create Alias for torch.tensor to increase readability.
TN = T


# Defining a type alias for the Diffusion Inversion Process type of callable.
Diff_Inversion_Process_Callback = Callable[[StableDiffusionXLPipeline, int, T, dict[str, T]], dict[str, T]]


@torch.no_grad()
def DDIM_Inversion_Process(model: StableDiffusionXLPipeline, x0: list[np.ndarray], blending_weights: list[float], prompts: list[str], num_inference_steps: int, guidance_scale: float) -> T:
    """
    Perform the DDIM inversion process with multi-stage refinement using separate prompts for each stage.

    Args:
    - model: The StableDiffusionXLPipeline model.
    - x0: A list of numpy arrays, each representing a reference image.
    - blending_weights: A list of floats representing the blending weights for each image.
    - prompts: A list of prompts corresponding to each reference image.
    - num_inference_steps: Number of inference steps for the diffusion process.
    - guidance_scale: Guidance scale to control the influence of the prompt on the generation.

    Returns:
    - final_latent_repr: The final latent representation after multi-stage refinement.
    """

    # Encode Images: Encode the input images into latent representations using the model's VAE.
    latent_imgs = images_encoding_multistage(model, x0, blending_weights)

    # Set Timesteps: Set the timesteps for the diffusion process.
    model.scheduler.set_timesteps(num_inference_steps, device=latent_imgs[0].device)

    # Multi-Stage Refinement: Start with the first latent representation and refine iteratively.
    refined_latent_repr = latent_imgs[0].clone()

    for i in range(1, len(latent_imgs)):
        # Blend the current refined latent representation with the next latent representation.
        weight = blending_weights[i]

        # Alpha blending for the latent representations.
        refined_latent_repr = refined_latent_repr * (1 - weight) + latent_imgs[i] * weight

        # Optionally: Perform DDIM steps or intermediate steps using the prompt for this stage.
        # You could refine the current latent representation using a partial diffusion process with the current prompt.
        refined_latent_repr = DDIM_Process(model, refined_latent_repr, prompts[i], guidance_scale)

    # Return Final Latent Representation: Return the final latent representation after refinement.
    return refined_latent_repr