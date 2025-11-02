import numpy as np

def calculate_metrics(NIC, NF, NC, H_in, W_in, K_enc=3, K_dec=2, B=1):
    """
    Calculates weights, outputs, and connections for the given architecture.
    
    Assumptions:
    - K_enc: Kernel size for all Conv2d layers in the encoder (e.g., 3x3)
    - K_dec: Kernel size for all ConvTranspose2d layers (fixed at 2x2) [cite: 205]
    - K_final: Kernel size for final classifier (fixed at 1x1) [cite: 213]
    - Padding: 'same' for encoder ( (K_enc-1)//2 )
    - Stride: 1 for encoder Conv2d, 2 for MaxPool2d, 2 for ConvTranspose2d
    - "Weights" count ignores biases and Batch Norm parameters [cite: 227]
    """
    
    K_final = 1
    
    # --- 1. Number of Weights [cite: 227] ---
    # Format: (K_in, K_in, C_in) * C_out
    
    # Encoder
    w_enc1 = (K_enc * K_enc * NIC) * NF
    w_enc2 = (K_enc * K_enc * NF) * (NF * 2)
    w_enc3 = (K_enc * K_enc * (NF * 2)) * (NF * 4)
    
    # Decoder (ConvTranspose2d)
    # Format: (K_dec, K_dec, C_in) * C_out
    w_dec1 = (K_dec * K_dec * (NF * 4)) * (NF * 2)
    w_dec2 = (K_dec * K_dec * (NF * 2)) * NF
    w_dec3 = (K_dec * K_dec * NF) * NC
    
    # Classifier
    w_final = (K_final * K_final * NC) * NC
    
    total_weights = w_enc1 + w_enc2 + w_enc3 + w_dec1 + w_dec2 + w_dec3 + w_final
    
    weights_dict = {
        "Encoder 1 (Conv)": w_enc1,
        "Encoder 2 (Conv)": w_enc2,
        "Encoder 3 (Conv)": w_enc3,
        "Decoder 1 (ConvT)": w_dec1,
        "Decoder 2 (ConvT)": w_dec2,
        "Decoder 3 (ConvT)": w_dec3,
        "Classifier (Conv)": w_final,
        "TOTAL WEIGHTS": total_weights
    }
    
    # --- 2. Number of Outputs (Activation Elements) [cite: 228] ---
    # Format: B * C * H * W for *all* layers (Conv, BN, ReLU, Pool)
    
    # Input: [B, NIC, H_in, W_in]
    
    # Encoder 1 Block
    h, w = H_in, W_in
    out_conv1 = B * NF * h * w
    out_bn1 = B * NF * h * w
    out_relu1 = B * NF * h * w
    h, w = h // 2, w // 2
    out_pool1 = B * NF * h * w
    
    # Encoder 2 Block
    out_conv2 = B * (NF * 2) * h * w
    out_bn2 = B * (NF * 2) * h * w
    out_relu2 = B * (NF * 2) * h * w
    h, w = h // 2, w // 2
    out_pool2 = B * (NF * 2) * h * w
    
    # Encoder 3 Block
    out_conv3 = B * (NF * 4) * h * w
    out_bn3 = B * (NF * 4) * h * w
    out_relu3 = B * (NF * 4) * h * w
    h, w = h // 2, w // 2
    out_pool3 = B * (NF * 4) * h * w
    
    # Decoder 1 Block
    h, w = h * 2, w * 2
    out_convT1 = B * (NF * 2) * h * w
    out_bn_d1 = B * (NF * 2) * h * w
    out_relu_d1 = B * (NF * 2) * h * w
    
    # Decoder 2 Block
    h, w = h * 2, w * 2
    out_convT2 = B * NF * h * w
    out_bn_d2 = B * NF * h * w
    out_relu_d2 = B * NF * h * w
    
    # Decoder 3 Block
    h, w = h * 2, w * 2
    out_convT3 = B * NC * h * w
    out_bn_d3 = B * NC * h * w
    out_relu_d3 = B * NC * h * w
    
    # Classifier
    out_final = B * NC * h * w
    
    total_outputs = (
        out_conv1 + out_bn1 + out_relu1 + out_pool1 +
        out_conv2 + out_bn2 + out_relu2 + out_pool2 +
        out_conv3 + out_bn3 + out_relu3 + out_pool3 +
        out_convT1 + out_bn_d1 + out_relu_d1 +
        out_convT2 + out_bn_d2 + out_relu_d2 +
        out_convT3 + out_bn_d3 + out_relu_d3 +
        out_final
    )
    
    # --- 3. Number of Connections [cite: 229] ---
    # Interpretation: For each layer, (Number of Weights) * (Output Activations / Output Channels)
    # This represents (Weight Kernel) * (Number of times kernel is applied)
    # e.g., (K*K*C_in * C_out) * (H_out * W_out * B)
    # This is equivalent to (Number of MACs) or (FLOPs / 2)
    
    # Input: [B, NIC, H_in, W_in]
    h_out, w_out = H_in, W_in
    conn_enc1 = (w_enc1) * (h_out * w_out * B)
    h_out, w_out = h_out // 2, w_out // 2 # After Pool1
    
    conn_enc2 = (w_enc2) * (h_out * w_out * B)
    h_out, w_out = h_out // 2, w_out // 2 # After Pool2
    
    conn_enc3 = (w_enc3) * (h_out * w_out * B)
    h_out, w_out = h_out // 2, w_out // 2 # After Pool3
    
    h_out, w_out = h_out * 2, w_out * 2 # After ConvT1
    conn_dec1 = (w_dec1) * (h_out * w_out * B)
    
    h_out, w_out = h_out * 2, w_out * 2 # After ConvT2
    conn_dec2 = (w_dec2) * (h_out * w_out * B)
    
    h_out, w_out = h_out * 2, w_out * 2 # After ConvT3
    conn_dec3 = (w_dec3) * (h_out * w_out * B)
    
    conn_final = (w_final) * (h_out * w_out * B)
    
    total_connections = conn_enc1 + conn_enc2 + conn_enc3 + conn_dec1 + conn_dec2 + conn_dec3 + conn_final
    
    return total_weights, total_outputs, total_connections, weights_dict

if __name__ == "__main__":
    
    # Symbolic parameters (B=1)
    # NIC, NF, NC are symbolic
    # K_enc = 3 (default)
    
    print("--- Calculating for H=32x32 Input (B=1) [cite: 230] ---")
    W_32, O_32, C_32, W_dict_32 = calculate_metrics(
        NIC='NIC', NF='NF', NC='NC', H_in=32, W_in=32, K_enc=3, B=1
    )
    
    print("\nSymbolic Weight Calculations (K_enc=3, K_dec=2, K_final=1):")
    for name, w in W_dict_32.items():
        # Use .format() to avoid issues with symbolic multiplication
        print(f"  {name+':':<20} {str(w).replace('**', '^')}")
        
    print(f"\nTotal Weights: {str(W_32).replace('**', '^')}")
    print(f"Total Outputs (Activations): {str(O_32).replace('**', '^')}")
    print(f"Total Connections (MACs): {str(C_32).replace('**', '^')}")
    
    print("\n\n--- Calculating for H=64x64 Input (B=1) [cite: 230] ---")
    W_64, O_64, C_64, W_dict_64 = calculate_metrics(
        NIC='NIC', NF='NF', NC='NC', H_in=64, W_in=64, K_enc=3, B=1
    )
    
    print("\nSymbolic Weight Calculations (K_enc=3, K_dec=2, K_final=1):")
    # Weights are independent of input size, so this is just a check
    for name, w in W_dict_64.items():
        print(f"  {name+':':<20} {str(w).replace('**', '^')}")
        
    print(f"\nTotal Weights: {str(W_64).replace('**', '^')}")
    print(f"Total Outputs (Activations): {str(O_64).replace('**', '^')}")
    print(f"Total Connections (MACs): {str(C_64).replace('**', '^')}")

    print("\n\n--- Numerical Example (NIC=1, NF=32, NC=24, K_enc=3) ---")
    print("\n--- H=32x32 ---")
    W_num_32, O_num_32, C_num_32, _ = calculate_metrics(
        NIC=1, NF=32, NC=24, H_in=32, W_in=32, K_enc=3, B=1
    )
    print(f"Total Weights: {W_num_32:,}")
    print(f"Total Outputs (Activations): {O_num_32:,}")
    print(f"Total Connections (MACs): {C_num_32:,}")
    
    print("\n--- H=64x64 ---")
    W_num_64, O_num_64, C_num_64, _ = calculate_metrics(
        NIC=1, NF=32, NC=24, H_in=64, W_in=64, K_enc=3, B=1
    )
    print(f"Total Weights: {W_num_64:,}")
    print(f"Total Outputs (Activations): {O_num_64:,}")
    print(f"Total Connections (MACs): {C_num_64:,}")