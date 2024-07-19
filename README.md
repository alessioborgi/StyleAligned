<a href="https://colab.research.google.com/github/alessioborgi/StyleAlignedDiffModels/blob/main/StyleAligned_Explanation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# StyleAligned: Zero-Shot Style Alignment among a Series of Generated Images via Attention Sharing

### Copyright © 2024 Alessio Borgi, Francesco Danese

### **Abstract**
In this notebook we aim to reproduce and enhance **[StyleAligned](https://arxiv.org/abs/2312.02133)**, a novel technique designed to achieve **Zero-Shot Style Alignment in Text-to-Image (T2I) Generative Models**,introduced by **Google Research**. While current T2I models excel in creating visually compelling images from textual descriptions, they often struggle to maintain a consistent style across multiple images generated. Traditional methods to address this require extensive fine-tuning and manual intervention. **StyleAligned**, addresses this issue by incorporating minimal **Shared Attention** during the diffusion process, allowing for consistent style transfer without extensive fine-tuning (**Zero-Shot Inference**). This technique involves a straightforward inversion operation that enforces stylistic coherence while maintaining high fidelity to the text prompts. 

Its main **Features** are: 
-   **Zero-Shot Style Alignment**: Achieve consistent style alignment without the need for optimization or fine-tuning.
-	**Minimal Attention Sharing**: Introduces attention sharing during the diffusion process for seamless style transfer.
-	**High-Quality Synthesis**: Maintains high fidelity to text prompts while ensuring stylistic coherence.
-	**Ease of Use**: Simplifies the process of generating a series of stylistically aligned images.
-   **Inversion Operation**: Used to apply reference styles, ensuring stylistic coherence.

We propose three primary applications of StyleAligned:
1.	**StyleAligned with Prompts Only**: Demonstrates the simplicity and effectiveness of achieving style alignment using only **Input Text Prompts**.
2.	**StyleAligned with Reference Image**: Utilizes **Reference Images** (in order to **Input Text Prompts**) to guide the style alignment process, ensuring consistent style transfer across multiple outputs.
3.	**StyleAligned with ControlNet**: Incorporates **ControlNet**, which can be provided with **Depth Images** or **Edge Images (Canny Images)**, to enhance control over the style alignment process.

Our approach shows that high-quality, stylistically aligned image sets can be achieved with minimal intervention, enhancing the utility of T2I models for applications such as visual storytelling, artistic creation, and design. The method operates without the need for extensive optimization or fine-tuning, distinguishing it as a zero-shot solution. Evaluation across diverse styles and text prompts demonstrates the high-quality synthesis and fidelity of our method, underscoring its efficacy in achieving consistent style across various inputs. 

An extensive **Metrics Analysis** has also been provided w.r.t. the following, demonstrating the valuable insights of this technique.
- **Style Consistency (DINO Embedding Similarity)**
- **CLIP Score** 
- **Inception Score (IS)**

### **Installation**

To get started with StyleAligned, follow these steps:
1.	Clone the Repository: `git clone https://github.com/alessioborgi/StyleAlignedDiffModels.git`
2.  Navigate to the project directory:    `cd StyleAlignedDiffModels`
3.  Install the required dependencies:    `pip install -r requirements.txt`

### **License**

This project is licensed under the MIT License - see the LICENSE file for details.

### **Acknowledgments**

We would like to thank Google Research for introducing the original concept of StyleAligned.
