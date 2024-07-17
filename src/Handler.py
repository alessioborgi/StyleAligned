from __future__ import annotations
import torch

from diffusers import StableDiffusionXLPipeline
import StyleAlignedArgs
import Shared_Attention

T = torch.tensor # Create Alias for torch.tensor to increase readability.
TN = T

class Handler:

    def register(self, style_aligned_args: StyleAlignedArgs, ):
        self.norm_layers = Shared_Attention.register_shared_norm(self.pipeline, style_aligned_args.share_group_norm,
                                                style_aligned_args.share_layer_norm)
        Shared_Attention.init_attention_processors(self.pipeline, style_aligned_args) # modify the pretrained architecture adding AdaIN & Shared Attention

    def remove(self): # this "restore" the original architecture, removing any altered layer
        for layer in self.norm_layers:
            layer.forward = layer.orig_forward
        self.norm_layers = []
        Shared_Attention.init_attention_processors(self.pipeline, None)

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        self.pipeline = pipeline
        self.norm_layers = []