import torch

def get_vision_encoder(name="vggt", device=None, dtype=None):
    
    if name == "vggt":
        from vggt.models.vggt import VGGT
        
        device = "cuda" if torch.cuda.is_available() else "cpu" if device is None else device
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16 if dtype is None else dtype

        # Initialize the model and load the pretrained weights.
        # This will automatically download the model weights the first time it's run, which may take a while.
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    else:
        raise ValueError(f"Unknown vision encoder: {name}")

    return model



class VisionEncoderWrapper(torch.nn.Module):
    def __init__(self, vision_encoder_name="vggt", device=None, dtype=None):
        super().__init__()
        self.vision_encoder_name = vision_encoder_name
        self.vision_encoder = get_vision_encoder(vision_encoder_name, device, dtype)
        

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        for vggt, the input shape is [B, S, 3, H, W]
        Args:
            x (torch.Tensor): Input tensor.
            return_features (bool): Whether to return intermediate features.
        Returns:
            torch.Tensor or list of torch.Tensor: Output tensor or list of feature maps.
        """
        if self.vision_encoder_name == "vggt":
            assert x.dim() == 5, "Input tensor must have shape [B, S, 3, H, W]"
            # in shape [B, S, 3, H, W]
            # out shape [B, S, patch_num, 2048]
            # if return_features: list of feature maps [ [B, S, patch_num, 2048], ... ]
            # patch_num = H/patch_size * W/patch_size + num_register_tokens (4) + camera_token (1)
            return self.vision_encoder(x, return_features=return_features)
        else:
            raise ValueError(f"Unknown vision encoder: {self.vision_encoder_name}")

# if __name__ == "__main__":
#     # Example usage
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = get_vision_encoder("vggt", device=device)
#     dummy_input = torch.randn(2, 3, 3, 224, 224).to(next(model.parameters()).device)
#     outputs = model(dummy_input, return_features=True)

#     print(f"Input shape: {dummy_input.shape}")
#     for i, features in enumerate(outputs):
#         print(f"Feature map {i} shape: {features.shape}")