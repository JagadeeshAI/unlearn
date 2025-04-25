import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification
from tqdm import tqdm
import copy
import pandas as pd

# Paths and constants
DATA_DIR = "./data"
MODEL_PATH = "./model/vit_pets_best.pth"
RESULTS_PATH = "./results"
TARGET_UPPER_LAYERS = True  # Focus on upper layers (8-12) where semantic knowledge is stored
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

def prepare_dataset():
    """Prepare retain and forget dataloaders."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load full test dataset
    full_dataset = OxfordIIITPet(root=DATA_DIR, split="test", transform=transform, download=False)
    class_names = full_dataset.classes
    
    # Define forget classes - exact class names from the dataset
    forget_classes = ["Egyptian Mau"]
    
    # Get class indices for forget classes
    forget_class_indices = []
    for fc in forget_classes:
        matches = [i for i, cn in enumerate(class_names) if fc == cn or fc in cn]
        if matches:
            forget_class_indices.extend(matches)
            print(f"Found class '{fc}' at indices: {matches}")
        else:
            print(f"Warning: Class '{fc}' not found in dataset")
    
    # Get forget and retain indices
    forget_indices = []
    retain_indices = []
    
    for i, (_, class_idx) in enumerate(full_dataset):
        if class_idx in forget_class_indices:
            forget_indices.append(i)
        else:
            retain_indices.append(i)
    
    # Create subsets
    forget_dataset = Subset(full_dataset, forget_indices)
    retain_dataset = Subset(full_dataset, retain_indices)
    
    # Create dataloaders
    forget_loader = DataLoader(forget_dataset, batch_size=32, shuffle=False, num_workers=2)
    retain_loader = DataLoader(retain_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Forget dataset size: {len(forget_dataset)} images")
    print(f"Retain dataset size: {len(retain_dataset)} images")
    
    return forget_loader, retain_loader, class_names

def load_model():
    """Load the ViT model with pretrained weights."""
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=37,  # Oxford Pets has 37 classes
        ignore_mismatched_sizes=True
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model

def evaluate_model(model, dataloader):
    """Evaluate model accuracy on a given dataloader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def get_ffn_parameter_positions(model):
    """Extract positions of FFN parameters in the model, excluding input and output layers."""
    ffn_positions = []
    
    # For ViT models from HuggingFace, FFN layers are in the vit.encoder.layer.*.mlp structure
    total_params = 0
    param_positions = []
    
    # First pass: calculate total parameters and positions
    for name, param in model.named_parameters():
        # Skip classifier/output layers
        if 'classifier' in name or 'head' in name or 'pooler' in name:
            continue
            
        param_positions.append((name, total_params, total_params + param.numel(), param.shape))
        total_params += param.numel()
    
    # Second pass: identify FFN parameters with layer information
    for name, start_pos, end_pos, shape in param_positions:
        # Extract layer number if present
        layer_num = -1
        if 'layer.' in name:
            parts = name.split('layer.')
            if len(parts) > 1:
                layer_part = parts[1].split('.')[0]
                if layer_part.isdigit():
                    layer_num = int(layer_part)
        
        # Check for FFN patterns but exclude embedding layers
        if ('intermediate.dense' in name or 'output.dense' in name or 'mlp.dense' in name) and 'embeddings' not in name:
            ffn_positions.append((name, start_pos, end_pos, shape, layer_num))
            print(f"Found FFN layer: {name} (layer {layer_num}) with {end_pos - start_pos} parameters")
    
    return ffn_positions, total_params

def get_flattened_parameters(model):
    """Get flattened parameters of the model."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def load_flattened_parameters(model, flat_params):
    """Load flattened parameters back into the model."""
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = flat_params[pointer:pointer + num_param].view_as(param)
        pointer += num_param

def corrupt_weights_with_zero(weights):
    """Zero out weights for targeted forgetting."""
    return torch.zeros_like(weights)

def save_metrics(metrics, filename="forgetting_metrics.json"):
    """Save metrics to JSON file."""
    # Convert metrics to serializable form (handle tensor values)
    serializable_metrics = []
    for m in metrics:
        serializable = {k: v for k, v in m.items()}
        serializable_metrics.append(serializable)
        
    path = os.path.join(RESULTS_PATH, filename)
    with open(path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)

def plot_forgetting_heatmap(metrics, filename="forgetting_heatmap.png"):
    """Create a 2D heatmap of forgetting effectiveness across layers and positions."""
    # Extract data for plotting
    df = pd.DataFrame([
        {
            'layer': m.get('layer_num', -1),
            'position': i,
            'forget_reduction': m.get('initial_forget_acc', 0) - m.get('forget_accuracy', 0),
            'retain_impact': m.get('initial_retain_acc', 0) - m.get('retain_accuracy', 0),
            'effectiveness': (m.get('initial_forget_acc', 0) - m.get('forget_accuracy', 0)) - 
                           0.5 * (m.get('initial_retain_acc', 0) - m.get('retain_accuracy', 0))
        }
        for i, m in enumerate(metrics) if 'layer_num' in m and m.get('layer_num', -1) >= 0
    ])
    
    if len(df) == 0:
        print("No data to plot heatmap")
        return
    
    # Create 2D heatmap for visualization
    plt.figure(figsize=(15, 8))
    pivot_table = df.pivot(index='layer', columns='position', values='effectiveness')
    sns.heatmap(pivot_table, cmap='viridis', annot=False, fmt=".2f", cbar_kws={'label': 'Forgetting Effectiveness'})
    plt.title('Heatmap of Forgetting Effectiveness (Layer vs Position)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, filename))
    plt.close()
    
    # Plot forget reduction vs retain impact
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['retain_impact'], df['forget_reduction'], 
                         c=df['layer'], cmap='viridis', 
                         s=50, alpha=0.7)
    plt.colorbar(scatter, label='Layer Number')
    plt.xlabel('Retain Accuracy Impact (lower is better)')
    plt.ylabel('Forget Accuracy Reduction (higher is better)')
    plt.title('Forgetting Performance by Layer')
    plt.grid(True, alpha=0.3)
    
    # Add ideal region
    plt.axvspan(0, 5, ymin=0, ymax=0.5, alpha=0.2, color='green', label='Ideal Region')
    
    # Label best points
    best_points = df[df['forget_reduction'] > 15]
    for _, row in best_points.iterrows():
        plt.annotate(f"L{int(row['layer'])},P{int(row['position'])}",
                    (row['retain_impact'], row['forget_reduction']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'forgetting_scatter.png'))
    plt.close()

def visualize_best_windows(metrics, initial_params, model, filename_prefix="best_window"):
    """Visualize the most effective forgetting windows."""
    # Find the most effective windows
    effective_windows = []
    for m in metrics:
        if 'layer_num' not in m or m['layer_num'] < 0:
            continue
            
        forget_reduction = m.get('initial_forget_acc', 0) - m.get('forget_accuracy', 0)
        retain_impact = m.get('initial_retain_acc', 0) - m.get('retain_accuracy', 0)
        
        if forget_reduction > 15 and retain_impact < 5:
            effective_windows.append(m)
    
    if not effective_windows:
        print("No effective windows found for visualization")
        return
    
    # Sort by effectiveness
    effective_windows.sort(key=lambda x: (x.get('initial_forget_acc', 0) - x['forget_accuracy']) - 
                                       0.5 * (x.get('initial_retain_acc', 0) - x['retain_accuracy']), 
                          reverse=True)
    
    # Visualize top 3 windows
    for i, window in enumerate(effective_windows[:3]):
        layer_name = window['layer_name']
        layer_num = window['layer_num']
        window_start = window['window_start']
        window_end = window['window_end']
        
        # Create parameter heatmap
        for name, param in model.named_parameters():
            if name == layer_name:
                # Flatten parameter to visualize its pattern
                param_data = param.data.cpu().numpy()
                
                if len(param_data.shape) == 2:
                    plt.figure(figsize=(12, 10))
                    plt.subplot(2, 1, 1)
                    sns.heatmap(param_data, cmap='viridis')
                    plt.title(f'Weight Matrix for Layer {layer_num}')
                    
                    # Highlight window region in histogram
                    flattened = param_data.flatten()
                    plt.subplot(2, 1, 2)
                    plt.hist(flattened, bins=50, alpha=0.7)
                    plt.title('Weight Distribution')
                    plt.tight_layout()
                    plt.savefig(os.path.join(RESULTS_PATH, f"{filename_prefix}_{i+1}_layer{layer_num}.png"))
                    plt.close()
                    break

def run_top_down_search(model, forget_loader, retain_loader, ffn_positions, original_params, initial_retain_acc, initial_forget_acc):
    """Run a top-down search strategy to find knowledge-containing weights."""
    all_metrics = []
    promising_regions = []
    
    # Define search stages from coarse to fine
    search_stages = [
        {"name": "coarse", "window_size_percent": 10.0, "stride_divisor": 2},
        {"name": "medium", "window_size_percent": 5.0, "stride_divisor": 2},
        {"name": "fine", "window_size_percent": 1.0, "stride_divisor": 2},
        {"name": "precise", "window_size_percent": 0.5, "stride_divisor": 2}
    ]
    
    total_ffn_params = sum(end - start for _, start, end, _, _ in ffn_positions)
    
    # Initial baseline metric
    all_metrics.append({
        "stage": "baseline",
        "window_start": 0,
        "window_end": 0,
        "retain_accuracy": initial_retain_acc,
        "forget_accuracy": initial_forget_acc,
        "score": abs(initial_forget_acc - initial_retain_acc),
        "layer_num": -1,
        "initial_retain_acc": initial_retain_acc,
        "initial_forget_acc": initial_forget_acc
    })
    
    # Stage 1: Coarse search across all layers
    print("\n=== Stage 1: Coarse Search ===")
    coarse_stage = search_stages[0]
    coarse_metrics = []
    
    # Calculate window size (ensure minimum size)
    window_size = max(100, int(total_ffn_params * coarse_stage["window_size_percent"] / 100))
    stride = max(50, window_size // coarse_stage["stride_divisor"])
    
    print(f"Coarse window size: {window_size} parameters ({coarse_stage['window_size_percent']}% of FFN params)")
    print(f"Stride: {stride} parameters")
    
    # For each layer, test with a large window
    for name, start_pos, end_pos, shape, layer_num in ffn_positions:
        layer_params_count = end_pos - start_pos
        
        # Skip if layer is too small
        if layer_params_count < window_size:
            print(f"Skipping layer {layer_num}: too small ({layer_params_count} < {window_size})")
            continue
            
        print(f"Testing layer {layer_num} with {layer_params_count} parameters")
        
        # Calculate number of windows for this layer
        num_windows = max(1, (layer_params_count - window_size) // stride + 1)
        
        for i in range(num_windows):
            # Calculate window boundaries
            window_start_offset = i * stride
            window_end_offset = min(window_start_offset + window_size, layer_params_count)
            
            window_start_idx = start_pos + window_start_offset
            window_end_idx = start_pos + window_end_offset
            
            # Copy original parameters
            corrupted_params = original_params.clone()
            
            # Extract and zero out window weights
            window_indices = list(range(window_start_idx, window_end_idx))
            corrupted_params[window_indices] = torch.zeros_like(corrupted_params[window_indices])
            
            # Load corrupted parameters
            load_flattened_parameters(model, corrupted_params)
            
            # Evaluate
            retain_acc = evaluate_model(model, retain_loader)
            forget_acc = evaluate_model(model, forget_loader)
            score = abs(forget_acc - retain_acc)
            
            # Calculate metrics
            forget_reduction = initial_forget_acc - forget_acc
            retain_impact = initial_retain_acc - retain_acc
            
            print(f"Layer {layer_num}, Window {i+1}/{num_windows}: "
                  f"Retain Acc: {retain_acc:.2f}%, "
                  f"Forget Acc: {forget_acc:.2f}%, "
                  f"Reduction: {forget_reduction:.2f}%, "
                  f"Impact: {retain_impact:.2f}%")
            
            # Store metrics
            metric = {
                "stage": "coarse",
                "window_start": window_start_idx,
                "window_end": window_end_idx,
                "retain_accuracy": retain_acc,
                "forget_accuracy": forget_acc,
                "score": score,
                "layer_num": layer_num,
                "layer_name": name,
                "window_size": window_size,
                "window_index": i,
                "initial_retain_acc": initial_retain_acc,
                "initial_forget_acc": initial_forget_acc,
                "forget_reduction": forget_reduction,
                "retain_impact": retain_impact
            }
            
            coarse_metrics.append(metric)
            all_metrics.append(metric)
            
            # Identify promising regions (good forget reduction with minimal retain impact)
            if forget_reduction > 10 and retain_impact < 7:
                promising_regions.append({
                    "layer_num": layer_num,
                    "layer_name": name,
                    "region_start": window_start_idx,
                    "region_end": window_end_idx,
                    "forget_reduction": forget_reduction,
                    "retain_impact": retain_impact,
                    "metric": metric
                })
            
            # Restore original parameters
            load_flattened_parameters(model, original_params)
    
    # Sort promising regions by effectiveness
    promising_regions.sort(key=lambda x: x["forget_reduction"] - 0.5 * x["retain_impact"], reverse=True)
    
    # Save coarse metrics
    save_metrics(coarse_metrics, "coarse_search_metrics.json")
    
    # Stage 2: Fine-grained search in promising regions
    finer_metrics = []
    best_windows = []
    
    if promising_regions:
        print(f"\n=== Stage 2: Fine-grained Search in {len(promising_regions)} Promising Regions ===")
        
        # For each promising region, conduct finer searches
        for region_idx, region in enumerate(promising_regions[:min(5, len(promising_regions))]):
            layer_num = region["layer_num"]
            layer_name = region["layer_name"]
            region_start = region["region_start"]
            region_end = region["region_end"]
            
            print(f"\nSearching within promising region {region_idx+1}: Layer {layer_num}, {region_start} to {region_end}")
            
            # Apply progressively finer windows within this region
            for stage_idx, stage in enumerate(search_stages[1:]):  # Skip coarse stage
                stage_name = stage["name"]
                window_size_percent = stage["window_size_percent"]
                stride_divisor = stage["stride_divisor"]
                
                # Calculate window size relative to the promising region
                region_size = region_end - region_start
                window_size = max(10, int(region_size * window_size_percent / 100))
                stride = max(5, window_size // stride_divisor)
                
                print(f"  {stage_name.capitalize()} search: Window size {window_size} ({window_size_percent}% of region)")
                
                # Calculate number of windows
                num_windows = max(1, (region_size - window_size) // stride + 1)
                
                for i in range(num_windows):
                    # Calculate window boundaries within the region
                    window_start_offset = i * stride
                    window_end_offset = min(window_start_offset + window_size, region_size)
                    
                    window_start_idx = region_start + window_start_offset
                    window_end_idx = region_start + window_end_offset
                    
                    # Copy original parameters
                    corrupted_params = original_params.clone()
                    
                    # Extract and zero out window weights
                    window_indices = list(range(window_start_idx, window_end_idx))
                    corrupted_params[window_indices] = torch.zeros_like(corrupted_params[window_indices])
                    
                    # Load corrupted parameters
                    load_flattened_parameters(model, corrupted_params)
                    
                    # Evaluate
                    retain_acc = evaluate_model(model, retain_loader)
                    forget_acc = evaluate_model(model, forget_loader)
                    score = abs(forget_acc - retain_acc)
                    
                    # Calculate metrics
                    forget_reduction = initial_forget_acc - forget_acc
                    retain_impact = initial_retain_acc - retain_acc
                    
                    print(f"    Window {i+1}/{num_windows}: "
                          f"Forget: {forget_acc:.2f}% (-{forget_reduction:.2f}%), "
                          f"Retain: {retain_acc:.2f}% (-{retain_impact:.2f}%)")
                    
                    # Store metrics
                    metric = {
                        "stage": stage_name,
                        "window_start": window_start_idx,
                        "window_end": window_end_idx,
                        "retain_accuracy": retain_acc,
                        "forget_accuracy": forget_acc,
                        "score": score,
                        "layer_num": layer_num,
                        "layer_name": layer_name,
                        "window_size": window_size,
                        "window_index": i,
                        "region_index": region_idx,
                        "initial_retain_acc": initial_retain_acc,
                        "initial_forget_acc": initial_forget_acc,
                        "forget_reduction": forget_reduction,
                        "retain_impact": retain_impact
                    }
                    
                    finer_metrics.append(metric)
                    all_metrics.append(metric)
                    
                    # Check if this is a good forgetting window
                    if forget_reduction > 15 and retain_impact < 5:
                        best_windows.append(metric)
                    
                    # Restore original parameters
                    load_flattened_parameters(model, original_params)
    
    # Save all metrics
    save_metrics(all_metrics, "all_metrics.json")
    save_metrics(finer_metrics, "fine_search_metrics.json")
    
    return all_metrics, best_windows

def main():
    # Prepare data and model
    print("Preparing dataset...")
    forget_loader, retain_loader, class_names = prepare_dataset()

    print("Loading model...")
    model = load_model()
    
    # Initial evaluation
    print("Evaluating initial model performance...")
    initial_retain_acc = evaluate_model(model, retain_loader)
    initial_forget_acc = evaluate_model(model, forget_loader)
    initial_score = abs(initial_forget_acc - initial_retain_acc)
    
    print(f"Initial Retain Accuracy: {initial_retain_acc:.2f}%")
    print(f"Initial Forget Accuracy: {initial_forget_acc:.2f}%")
    print(f"Initial Score: {initial_score:.4f}")
    
    # Extract FFN parameter positions with layer information
    print("Extracting FFN parameter positions...")
    ffn_positions, total_params = get_ffn_parameter_positions(model)
    
    # Filter upper layers (8+) if specified
    if TARGET_UPPER_LAYERS:
        print("Targeting only upper layers (8+) where semantic knowledge is stored...")
        upper_ffn_positions = [(name, start, end, shape, layer) for name, start, end, shape, layer in ffn_positions 
                              if layer >= 8]  # Focus on upper layers based on paper
        ffn_positions = upper_ffn_positions
        if not ffn_positions:
            print("Warning: No FFN positions found in upper layers, reverting to all layers")
            ffn_positions, _ = get_ffn_parameter_positions(model)
    
    total_ffn_params = sum(end - start for _, start, end, _, _ in ffn_positions)
    print(f"Total targeted FFN parameters: {total_ffn_params} ({100*total_ffn_params/total_params:.2f}% of all params)")
    
    # Get original flattened parameters
    original_params = get_flattened_parameters(model)
    
    # Run top-down search
    all_metrics, best_windows = run_top_down_search(
        model, forget_loader, retain_loader, ffn_positions, 
        original_params, initial_retain_acc, initial_forget_acc
    )
    
    # Create visualizations
    plot_forgetting_heatmap(all_metrics)
    visualize_best_windows(all_metrics, original_params, model)
    
    # Sort best windows by effectiveness
    best_windows.sort(key=lambda x: x["forget_reduction"] - 0.5 * x["retain_impact"], reverse=True)
    
    # Report best windows
    if best_windows:
        print("\nTop effective forgetting windows found:")
        for i, window in enumerate(best_windows[:5]):  # Show top 5
            print(f"\n{i+1}. Layer: {window['layer_name']} (layer {window['layer_num']})")
            print(f"   Window: {window['window_start']} to {window['window_end']}")
            print(f"   Forget Accuracy: {window['forget_accuracy']:.2f}% (reduced by {window['forget_reduction']:.2f}%)")
            print(f"   Retain Accuracy: {window['retain_accuracy']:.2f}% (impact of {window['retain_impact']:.2f}%)")
            print(f"   Search stage: {window['stage']}")
        
        # Test combined forgetting with top windows
        if len(best_windows) >= 3:
            print("\nTesting combined forgetting with top windows...")
            combined_params = original_params.clone()
            
            for window in best_windows[:3]:
                window_indices = list(range(window['window_start'], window['window_end']))
                combined_params[window_indices] = torch.zeros_like(combined_params[window_indices])
            
            # Apply combined forgetting
            load_flattened_parameters(model, combined_params)
            
            # Evaluate
            combined_retain_acc = evaluate_model(model, retain_loader)
            combined_forget_acc = evaluate_model(model, forget_loader)
            
            print(f"Combined forgetting results:")
            print(f"Forget Accuracy: {combined_forget_acc:.2f}% (reduced by {initial_forget_acc - combined_forget_acc:.2f}%)")
            print(f"Retain Accuracy: {combined_retain_acc:.2f}% (impact of {initial_retain_acc - combined_retain_acc:.2f}%)")
    else:
        print("\nNo highly effective forgetting windows found.")
    
    print("Experiment completed. Results saved to:", RESULTS_PATH)

if __name__ == "__main__":
    main()