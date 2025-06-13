import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipeline import best_model, AudioCNN

def occlude_input(input_tensor: torch.tensor, top: int, left: int, height: int, width: int, fill_value: float=0.0) -> torch.Tensor:
    """
    This function occludes a portion of the input tensor by replacing it with a fill value.
    This is done to test the effectiveness of the model's predictions when parts of the input are missing.

    Args:
        input_tensor: The input tensor of shape [B, C, H, W].
        top: The top coordinate of the occlusion.
        left: The left coordinate of the occlusion.
        height: The height of the occlusion.
        width: The width of the occlusion.
        fill_value: The value to fill in the occluded area. Default is 0.0.
    """
    occluded = input_tensor.clone()
    occluded[..., top:top+height, left:left+width] = fill_value
    return occluded

def plotting(occluded: torch.Tensor) -> None:
    """
    This function plots the occluded spectrogram from the input tensor.
    """
    occluded_np = occluded.squeeze().cpu().numpy()  # shape: [H, W]

    plt.figure(figsize=(10, 4))
    plt.imshow(occluded_np, aspect='auto', origin='lower', cmap='magma')
    plt.title('Occluded Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Amplitude')
    plt.show()


def get_class_prob(model: AudioCNN, input_tensor: torch.tensor, class_idx: int) -> float:
    """
    This function computes the probability of a specific class for the given input tensor using the model.
    
    Args:
        model: The trained AudioCNN model.
        input_tensor: The input tensor of shape [B, C, H, W].
        class_idx: The index of the class for which to compute the probability.
    """
    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)[0, class_idx].item()
    return prob


def explainable() -> None:
    """
    This function generates an occlusion sensitivity map for a sample spectrogram. 
    """
    # Load model and set to eval
    model, batch_size = best_model()
    model_path = "trained_model/CNN.pt"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
    model.eval()
    
    # Load sample spectrogram
    sample_path = "data/cleaned/809574_3.csv" # you can change this to any sample you want to test by changing the path to the
    # wanted sample. For example:

    # 819798_6 for bat_type 3
    # 921896_2 for bat_type 0
    # 934884_39 for bat_type 1
    # 809574_3 for bat_type 2

    sample = pd.read_csv(sample_path)
    np_sample = sample.values

    # we need the minimum value from the spectrogram to fill the occluded area
    minimum_value = np.min(np_sample)
    data_tensor = torch.tensor(np_sample, dtype=torch.float32)
    data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

    # Get original prediction
    output = model(data_tensor)
    orig_class = torch.argmax(output, dim=1).item()
    orig_prob = get_class_prob(model, data_tensor, orig_class)

    _, patch_w = 8, 1024 # we select these values because we want to check only the horizontal oclusion of the spectograms
    # the images are 1024 pixels wide, so we divide it by 8 to get the width of each patch

    _, _, H, W = data_tensor.shape
    map_h, map_w = H, W // patch_w

    importance_map = np.zeros((map_h, map_w))
    counts = np.zeros_like(importance_map)

    for index in range(H):
        top = H - index
        left = 0
        
        # this shows the occluded area by plotting it every 100 iterations
        # if index % 100 == 0:
        #     print(f"Processing row {index + 1}/{H}...")
        #     occluded = occlude_input(data_tensor, top - 50, left, 100, patch_w, fill_value=0)
        #     plotting(occluded)
        occluded = occlude_input(data_tensor, top - 50, left, 100, patch_w, fill_value=minimum_value)
        occ_prob = get_class_prob(model, occluded, orig_class)
        drop = orig_prob - occ_prob
        i, j = index, left // patch_w
        importance_map[i, j] += drop
        counts[i, j] += 1

    # Normalize the importance map by the counts
    importance_map = importance_map / np.maximum(counts, 1)

    # Visualization
    spectrogram = sample.values

    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Now your original code will work
    im1 = axes[0].imshow(importance_map, cmap='hot', interpolation='nearest', aspect='auto', extent=[0, 2, 0, 500])
    axes[0].set_title('Occlusion Importance Map (8 Horizontal patches)')
    axes[0].set_xlabel('Patch column')
    axes[0].set_ylabel('Patch row')
    fig.colorbar(im1, ax=axes[0], label='Importance')

    im2 = axes[1].imshow(spectrogram, aspect='auto', origin='lower', cmap='magma', extent=[0, 2, 0, 80000])
    axes[1].set_title('Spectrogram Image')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Hertz')
    fig.colorbar(im2, ax=axes[1], label='Amplitude')


    plt.tight_layout()
    plt.show()
