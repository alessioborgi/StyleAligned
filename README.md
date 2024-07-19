<a href="https://colab.research.google.com/github/alessioborgi/StyleAlignedDiffModels/blob/main/StyleAligned_Explanation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# StyleAligned: Zero-Shot Style Alignment among a Series of Generated Images via Attention Sharing

### **Authors**: ***Borgi Alessio***, ***Danese Francesco***

### **Abstract**
In this notebook we aim to reproduce and enhance **[StyleAligned](https://arxiv.org/abs/2312.02133)**, a novel technique introduced by **Google Research**, for achieving **Style Consistency** in large-scale Text-to-Image (T2I) generative models. While current T2I models excel in creating visually compelling images from textual descriptions, they often struggle to maintain a consistent style across multiple images generated. Traditional methods to address this require extensive fine-tuning and manual intervention.

**StyleAligned** addresses this challenge by introducing minimal **Attention Sharing** during the **Diffusion Process**, ensuring **Style Alignment among generated images** without the need for optimization or fine-tuning (**Zero-Shoot Inference**). The method operates by leveraging a straightforward inversion operation to apply a reference style across various generated images, maintaining high-quality synthesis and fidelity to the provided text prompts.
