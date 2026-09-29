import torch
import torch.nn as nn
from typing import Optional

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("Warning: efficientnet-pytorch not installed. Using fallback CNN.")


class VoiceShieldModel(nn.Module):
    """
    VoiceShield Model using EfficientNet-B0 for binary classification
    (bonafide vs spoof).
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet-b0',
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        if EFFICIENTNET_AVAILABLE and 'efficientnet' in model_name.lower():
            self._build_efficientnet(model_name, pretrained, freeze_backbone)
        else:
            self._build_fallback_cnn()
        
        self._replace_classifier()
    
    def _build_efficientnet(self, model_name: str, pretrained: bool, freeze_backbone: bool):
        """Build EfficientNet-B0 backbone."""
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
        
        self.backbone_type = 'efficientnet'
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            for param in self.backbone._fc.parameters():
                param.requires_grad = True
    
    def _build_fallback_cnn(self):
        """Build fallback CNN architecture."""
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.backbone_type = 'fallback_cnn'
        self.feature_dim = 128
    
    def _replace_classifier(self):
        """Replace final classification layer."""
        if self.backbone_type == 'efficientnet':
            in_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(in_features, self.num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(64, self.num_classes),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 128, 251)
            
        Returns:
            Logits of shape (batch_size, 1)
        """
        if self.backbone_type == 'efficientnet':
            x = self.backbone.extract_features(x)
            x = self.backbone._avg_pooling(x)
            x = x.flatten(start_dim=1)
            x = self.backbone._dropout(x)
            x = self.backbone._fc(x)
        else:
            x = self.backbone(x)
            x = x.flatten(start_dim=1)
            x = self.classifier(x)
        
        return x
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        if self.backbone_type == 'efficientnet':
            x = self.backbone.extract_features(x)
            x = self.backbone._avg_pooling(x)
            x = x.flatten(start_dim=1)
        else:
            x = self.backbone(x)
            x = x.flatten(start_dim=1)
        return x


class SimpleCNN(nn.Module):
    """
    Simple 3-layer CNN as alternative architecture.
    """
    
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


def create_model(
    model_type: str = 'efficientnet-b0',
    num_classes: int = 1,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function to create model.
    
    Args:
        model_type: 'efficientnet-b0', 'efficientnet-b1', or 'simple_cnn'
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        freeze_backbone: Whether to freeze backbone (for EfficientNet)
        pretrained: Whether to use pretrained weights
        
    Returns:
        Model instance
    """
    if model_type == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        return VoiceShieldModel(
            model_name=model_type,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to .pth file
        device: Device to load on
        strict: Whether to enforce strict loading
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    from config import DROPOUT_RATE, FREEZE_LAYERS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nTesting EfficientNet-B0 model...")
    model = create_model(
        model_type='efficientnet-b0',
        dropout_rate=DROPOUT_RATE,
        freeze_backbone=FREEZE_LAYERS,
    ).to(device)
    
    print(f"Model type: {model.backbone_type}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 1, 128, 251).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    print("\nTesting SimpleCNN model...")
    model2 = create_model(model_type='simple_cnn', dropout_rate=DROPOUT_RATE).to(device)
    print(f"Trainable parameters: {count_parameters(model2):,}")
    
    with torch.no_grad():
        out2 = model2(x)
    print(f"Output shape: {out2.shape}")