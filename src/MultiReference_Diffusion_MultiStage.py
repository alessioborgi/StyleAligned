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
def DDIM_Inversion_Process(model, x0, blending_weights, prompts, num_inference_steps, guidance_scale):
    """
    Perform the DDIM inversion process with multi-stage refinement using separate prompts for each stage.
    """
    latent_imgs = images_encoding_multistage(model, x0, blending_weights)

    # Convert all latent images to the correct dtype (float16) if needed
    latent_imgs = [latent_img.to(model.unet.dtype) for latent_img in latent_imgs]

    model.scheduler.set_timesteps(num_inference_steps, device=latent_imgs[0].device)

    # Start with the first latent representation and refine iteratively
    refined_latent_repr = latent_imgs[0].clone()

    for i in range(1, len(latent_imgs)):
        weight = blending_weights[i]

        for step in tqdm(range(num_inference_steps // len(latent_imgs))):
            timestep = model.scheduler.timesteps[step]

            # Ensure refined_latent_repr and timestep are in the correct dtype
            refined_latent_repr = refined_latent_repr.to(model.unet.dtype)
            timestep = timestep.to(model.unet.dtype)

            # Perform a partial diffusion step
            latent_pred = model.unet(refined_latent_repr, timestep, encoder_hidden_states=prompts[i])

            # Ensure latent_pred is in the correct dtype
            latent_pred = latent_pred.to(model.unet.dtype)

            # Blend with the current refined latent representation
            refined_latent_repr = refined_latent_repr * (1 - weight) + latent_pred * weight

    return refined_latent_repr