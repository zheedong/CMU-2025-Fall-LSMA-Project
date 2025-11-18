# =========================
# Dummy vision encoder & dataset for testing
# =========================

class DummyVisionEncoder(nn.Module):
    """
    Very simple vision encoder for testing.

    Input:  images [B, 3, 224, 224]
    Output: features [B, V_DIM]
    """

    def __init__(self, vision_hidden_size: int = 1024):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, vision_hidden_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        return self.backbone(x)  # [B, V_DIM]
