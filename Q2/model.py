import torch
import torch.nn as nn

class ColorizationNet(nn.Module):
    """
    CNN Encoder-Decoder for Image Colorization[cite: 135].
    Follows the architecture diagram from the assignment.
    """
    def __init__(self, nic=1, nf=32, nc=24, kernel_size=3):
        super(ColorizationNet, self).__init__()
        
        # Ensure kernel size is odd for 'same' padding
        assert kernel_size % 2 == 1, "Kernel size must be odd for 'same' padding"
        padding = (kernel_size - 1) // 2
        
        # Input: [B, NIC, 32, 32]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(nic, nf, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> [B, NF, 16, 16] [cite: 199]
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(nf, nf*2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> [B, 2*NF, 8, 8] [cite: 201]
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> [B, 4*NF, 4, 4] [cite: 204]
        )
        
        # Decoder: Learned Upsampling [cite: 138]
        self.decoder1 = nn.Sequential(
            # kernel=2, stride=2 doubles spatial dims [cite: 205]
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(inplace=True) # -> [B, 2*NF, 8, 8] [cite: 206]
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(nf*2, nf, kernel_size=2, stride=2),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True) # -> [B, NF, 16, 16] [cite: 209]
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(nf, nc, kernel_size=2, stride=2),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True) # -> [B, NC, 32, 32] [cite: 212]
        )
        
        # Final 1x1 Conv Classifier [cite: 213]
        self.classifier = nn.Conv2d(nc, nc, kernel_size=1, stride=1, padding=0)
        # Output: [B, NC, 32, 32] [cite: 214]

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Decoder
        d1 = self.decoder1(e3)
        d2 = self.decoder2(d1)
        d3 = self.decoder3(d2)
        
        # Classifier
        out = self.classifier(d3)
        
        return out

if __name__ == "__main__":
    # Test the model with a sample input
    print("Testing Model...")
    
    # Test with hyperparams from prompt [cite: 235, 236]
    nf_test = 32
    k_test = 3
    
    model = ColorizationNet(nic=1, nf=nf_test, nc=24, kernel_size=k_test)
    
    # Create a dummy batch
    # [B, NIC, 32, 32] [cite: 196]
    dummy_input = torch.randn(16, 1, 32, 32) 
    
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Expected output:
    # Input shape: torch.Size([16, 1, 32, 32])
    # Output shape: torch.Size([16, 24, 32, 32])
    
    assert output.shape == (16, 24, 32, 32)
    print("Model test passed!")