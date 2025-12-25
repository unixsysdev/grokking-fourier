"""
Generate analysis compendium for Addition Transformer.

Produces visualizations to understand HOW the model computes addition:
1. Attention patterns (does it look at corresponding digit positions?)
2. Position-wise accuracy (which digit positions are hardest?)
3. Carry analysis (does it handle carries correctly?)
4. Training history
5. Embedding analysis
"""

import warnings
# Suppress ROCm/HIP warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with memory efficient attention.*")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*HIPBLAS_STATUS_NOT_SUPPORTED.*")

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns

from model import AdditionTransformer


def find_checkpoint(epoch: str = "latest"):
    """Find checkpoint file."""
    checkpoint_dir = Path("checkpoints")
    
    if epoch == "latest" or epoch == "final":
        checkpoints = list(checkpoint_dir.glob("model_e*.pt"))
        final = checkpoint_dir / "model_final.pt"
        if final.exists():
            checkpoints.append(final)
        
        if not checkpoints:
            return None
        
        def get_epoch_num(p):
            name = p.stem
            if "final" in name:
                return float('inf')
            try:
                return int(name.split('e')[-1])
            except:
                return -1
        
        return max(checkpoints, key=get_epoch_num)
    else:
        for pattern in [f"model_e{epoch}.pt", f"model_{epoch}.pt"]:
            p = checkpoint_dir / pattern
            if p.exists():
                return p
        return None


def plot_training_history(output_dir: Path):
    """Plot training curves."""
    history_path = Path("checkpoints/history.json")
    if not history_path.exists():
        print("No history.json found")
        return
    
    with open(history_path) as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history.get('epoch', [])
    
    # Loss with curriculum phase markers
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.semilogy(epochs, history['train_loss'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log)')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        
        # Add curriculum phase markers
        for phase_epoch, label in [(5000, 'Phase 2'), (15000, 'Phase 3'), (30000, 'Phase 4')]:
            if max(epochs) > phase_epoch:
                ax.axvline(x=phase_epoch, color='red', linestyle='--', alpha=0.5)
                ax.text(phase_epoch, ax.get_ylim()[1], label, fontsize=8, ha='left', va='top')
    
    # Accuracies with curriculum phases
    ax = axes[0, 1]
    if 'train_acc' in history:
        ax.plot(epochs, history['train_acc'], label='Train', alpha=0.8)
    if 'interp_acc' in history:
        ax.plot(epochs, history['interp_acc'], label='Interpolation', alpha=0.8)
    if 'extrap_acc' in history:
        ax.plot(epochs, history['extrap_acc'], label='Extrapolation (6-digit)', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add curriculum phase markers
    for phase_epoch, label in [(5000, 'Phase 2\n(3-digit)'), (15000, 'Phase 3\n(4-digit)'), (30000, 'Phase 4\n(5-digit)')]:
        if max(epochs) > phase_epoch:
            ax.axvline(x=phase_epoch, color='red', linestyle='--', alpha=0.5)
            ax.text(phase_epoch, 0.95, label, fontsize=8, ha='left', va='top')
    
    # Learning rate
    if 'lr' in history:
        axes[1, 0].semilogy(epochs, history['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = "Final Metrics:\n\n"
    if history.get('train_acc'):
        summary += f"Train Accuracy:         {history['train_acc'][-1]:.1%}\n"
    if history.get('interp_acc'):
        summary += f"Interpolation Accuracy: {history['interp_acc'][-1]:.1%}\n"
    if history.get('extrap_acc'):
        summary += f"Extrapolation Accuracy: {history['extrap_acc'][-1]:.1%}\n"
    
    ax.text(0.1, 0.5, summary, fontsize=14, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()
    print("Saved training_history.png")


def analyze_attention_patterns(model, device, output_dir: Path):
    """
    Visualize attention patterns to see what the model attends to.
    
    Note: PyTorch's built-in TransformerDecoder doesn't expose attention weights
    easily. This function attempts to capture them but may not work with all
    configurations.
    """
    model.eval()
    
    # Test cases
    test_cases = [
        "12+34=",
        "99+1=",
        "123+456=",
        "999+1=",
    ]
    
    fig, axes = plt.subplots(len(test_cases), 1, figsize=(10, 3 * len(test_cases)))
    if len(test_cases) == 1:
        axes = [axes]
    
    for idx, eq in enumerate(test_cases):
        with torch.no_grad():
            src = torch.tensor([model.tokenize(eq)], device=device)
            output = model.generate(src, max_len=12)
            generated = model.detokenize(output[0].tolist())
        
        # Get the expected answer
        parts = eq.replace('=', '').split('+')
        expected = str(int(parts[0]) + int(parts[1]))
        correct = "✓" if generated == expected else "✗"
        
        # Since we can't easily get attention weights from nn.TransformerDecoder,
        # visualize the input/output alignment instead
        ax = axes[idx]
        
        # Create a simple visualization showing input -> output mapping
        input_tokens = list(eq)
        output_tokens = list(generated)
        
        # Plot as text comparison
        ax.text(0.1, 0.7, f"Input:    {eq}", fontsize=14, family='monospace', transform=ax.transAxes)
        ax.text(0.1, 0.4, f"Output:   {generated}", fontsize=14, family='monospace', transform=ax.transAxes)
        ax.text(0.1, 0.1, f"Expected: {expected} {correct}", fontsize=14, family='monospace', transform=ax.transAxes,
                color='green' if generated == expected else 'red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"Test case {idx + 1}")
    
    plt.suptitle("Model Predictions (Attention visualization requires custom model hooks)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "attention_patterns.png", dpi=150)
    plt.close()
    print("Saved attention_patterns.png (simplified - showing predictions)")


def analyze_carry_handling(model, device, output_dir: Path, epoch_num: int = 99999):
    """
    Analyze how well the model handles carries.
    
    Test cases specifically designed to require carries at different positions.
    Curriculum-aware: only tests number ranges the model has seen.
    """
    model.eval()
    
    # Determine max digits based on curriculum
    if epoch_num < 5000:
        max_val = 99  # 2-digit
        curriculum_note = "(Phase 1: 2-digit max)"
    elif epoch_num < 15000:
        max_val = 999  # 3-digit
        curriculum_note = "(Phase 2: 3-digit max)"
    elif epoch_num < 30000:
        max_val = 9999  # 4-digit
        curriculum_note = "(Phase 3: 4-digit max)"
    else:
        max_val = 99999  # 5-digit
        curriculum_note = "(Phase 4: 5-digit)"
    
    # Generate carry test cases appropriate for curriculum
    test_cases = {
        'no_carry': [],
        'single_carry': [],
        'multiple_carry': [],
        'cascade_carry': []  # e.g., 999+1
    }
    
    import random
    random.seed(42)
    
    # No carry cases (2-digit, always in curriculum)
    for _ in range(100):
        a = random.randint(10, 44)
        b = random.randint(10, 55 - a % 10)  # Ensure no carry in ones place
        if (a % 10 + b % 10) < 10 and (a // 10 % 10 + b // 10 % 10) < 10:
            test_cases['no_carry'].append((a, b))
    
    # Single carry (2-digit)
    for _ in range(100):
        a = random.randint(15, min(99, max_val))
        b = random.randint(10 - a % 10, min(99, max_val))
        if (a % 10 + b % 10) >= 10:
            test_cases['single_carry'].append((a, b))
    
    # Multiple carries
    for _ in range(100):
        a = random.randint(55, 99)
        b = random.randint(55, 99)
        test_cases['multiple_carry'].append((a, b))
    
    # Cascade carries (999...+1)
    for digits in range(2, 6):
        a = int('9' * digits)
        test_cases['cascade_carry'].append((a, 1))
        test_cases['cascade_carry'].append((a, random.randint(1, 100)))
    
    # Evaluate each category
    results = {}
    
    for category, cases in test_cases.items():
        correct = 0
        total = len(cases)
        
        with torch.no_grad():
            for a, b in cases[:50]:  # Limit for speed
                eq = f"{a}+{b}="
                expected = str(a + b)
                
                src = torch.tensor([model.tokenize(eq)], device=device)
                output = model.generate(src, max_len=12)
                generated = model.detokenize(output[0].tolist())
                
                if generated == expected:
                    correct += 1
        
        results[category] = correct / min(50, total) if total > 0 else 0
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = list(results.keys())
    accuracies = [results[c] for c in categories]
    colors = ['green' if a > 0.9 else 'orange' if a > 0.5 else 'red' for a in accuracies]
    
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.7)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Carry Handling Analysis')
    ax.set_ylim(0, 1.05)
    ax.legend()
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.0%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "carry_analysis.png", dpi=150)
    plt.close()
    print("Saved carry_analysis.png")
    
    return results


def analyze_digit_embeddings(model, device, output_dir: Path):
    """
    Visualize digit embeddings to see if the model learned meaningful representations.
    
    Questions:
    - Are digits 0-9 arranged in some structured way?
    - Is there a "number line" in embedding space?
    """
    model.eval()
    
    # Get digit embeddings
    digit_tokens = [model.TOKENS[str(i)] for i in range(10)]
    embeddings = model.token_embed.weight[digit_tokens].detach().cpu().numpy()
    
    # Try to import sklearn for PCA, fall back to simple SVD if not available
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    except ImportError:
        # Fallback: use numpy SVD for dimensionality reduction
        embeddings_centered = embeddings - embeddings.mean(axis=0)
        U, S, Vt = np.linalg.svd(embeddings_centered, full_matrices=False)
        embeddings_2d = U[:, :2] * S[:2]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2D projection
    ax = axes[0]
    for i in range(10):
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=200, zorder=5)
        ax.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                   fontsize=14, ha='center', va='center')
    
    # Draw lines connecting consecutive digits
    for i in range(9):
        ax.plot([embeddings_2d[i, 0], embeddings_2d[i+1, 0]],
               [embeddings_2d[i, 1], embeddings_2d[i+1, 1]],
               'k--', alpha=0.3)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Digit Embeddings (PCA/SVD projection)')
    ax.grid(True, alpha=0.3)
    
    # Cosine similarity matrix
    ax = axes[1]
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    
    # Cosine similarity
    similarity = normalized @ normalized.T
    
    sns.heatmap(similarity, ax=ax, cmap='RdBu_r', center=0,
                xticklabels=range(10), yticklabels=range(10),
                annot=True, fmt='.2f')
    ax.set_title('Digit Embedding Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(output_dir / "digit_embeddings.png", dpi=150)
    plt.close()
    print("Saved digit_embeddings.png")


def analyze_position_accuracy(model, device, output_dir: Path, epoch_num: int = 99999):
    """
    Analyze accuracy by output digit position.
    
    Question: Is the model worse at certain positions (e.g., leftmost digit
    which depends on carries from all positions)?
    
    Curriculum-aware: only tests numbers the model has seen.
    """
    model.eval()
    
    import random
    random.seed(42)
    
    # Determine range based on curriculum
    if epoch_num < 5000:
        min_a, max_a = 10, 99  # 2-digit
        curriculum_note = "2-digit numbers"
    elif epoch_num < 15000:
        min_a, max_a = 100, 999  # 3-digit
        curriculum_note = "3-digit numbers"
    elif epoch_num < 30000:
        min_a, max_a = 100, 9999  # up to 4-digit
        curriculum_note = "up to 4-digit numbers"
    else:
        min_a, max_a = 1000, 50000  # 4-5 digit
        curriculum_note = "4-5 digit numbers"
    
    # Generate test cases with known answer lengths
    position_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Test appropriate range for curriculum
    test_cases = []
    for _ in range(500):
        a = random.randint(min_a, max_a)
        b = random.randint(min_a, max_a)
        test_cases.append((a, b, str(a + b)))
    
    with torch.no_grad():
        for a, b, expected in tqdm(test_cases, desc="Position analysis"):
            eq = f"{a}+{b}="
            
            src = torch.tensor([model.tokenize(eq)], device=device)
            output = model.generate(src, max_len=12)
            generated = model.detokenize(output[0].tolist())
            
            # Compare position by position (from right, like actual addition)
            for pos in range(min(len(generated), len(expected))):
                # Position 0 = rightmost (ones place)
                gen_digit = generated[-(pos+1)] if pos < len(generated) else None
                exp_digit = expected[-(pos+1)] if pos < len(expected) else None
                
                position_results[pos]['total'] += 1
                if gen_digit == exp_digit:
                    position_results[pos]['correct'] += 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = sorted(position_results.keys())
    accuracies = [position_results[p]['correct'] / position_results[p]['total'] 
                  for p in positions]
    
    bars = ax.bar(positions, accuracies, alpha=0.7)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Digit Position (0=ones, 1=tens, 2=hundreds, ...)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Output Digit Position')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{p}\n({["ones","tens","hundreds","thousands","ten-thousands","hundred-thousands"][p] if p < 6 else p})' 
                        for p in positions], fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "position_accuracy.png", dpi=150)
    plt.close()
    print("Saved position_accuracy.png")



# =============================================================================
# FOURIER ANALYSIS & MECHANISTIC INTERPRETABILITY
# =============================================================================

def analyze_fourier_embeddings(model, device, output_dir: Path):
    """
    Analyze digit embeddings for Fourier/periodic structure.
    
    In Neel Nanda's grokking paper, after grokking the model learns to represent
    numbers using Fourier components - digits arranged in a circle, periodic patterns.
    
    We check:
    1. Do digits form a circle in embedding space?
    2. Is there periodic structure (FFT peaks)?
    3. Are embeddings organized by digit value mod something?
    """
    model.eval()
    
    # Get digit embeddings (0-9)
    digit_tokens = [model.TOKENS[str(i)] for i in range(10)]
    embeddings = model.token_embed.weight[digit_tokens].detach().cpu().numpy()
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. PCA projection - look for circular structure
    ax1 = fig.add_subplot(2, 3, 1)
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(embeddings_centered, full_matrices=False)
    embeddings_2d = U[:, :2] * S[:2]
    
    # Plot with color gradient to show if there's ordering
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    for i in range(10):
        ax1.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=300, c=[colors[i]], 
                   edgecolors='black', linewidth=2, zorder=5)
        ax1.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=16, ha='center', va='center', fontweight='bold')
    
    # Connect in order to see if they form a path/circle
    for i in range(9):
        ax1.plot([embeddings_2d[i, 0], embeddings_2d[i+1, 0]],
                [embeddings_2d[i, 1], embeddings_2d[i+1, 1]], 'k--', alpha=0.3)
    # Close the loop
    ax1.plot([embeddings_2d[9, 0], embeddings_2d[0, 0]],
            [embeddings_2d[9, 1], embeddings_2d[0, 1]], 'r--', alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Digit Embeddings (PCA)\nLook for circular/periodic structure')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. FFT of embeddings - look for frequency peaks
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Compute FFT along the "digit" dimension for each embedding dimension
    fft_magnitudes = np.abs(np.fft.fft(embeddings, axis=0))
    avg_fft = fft_magnitudes.mean(axis=1)  # Average across embedding dims
    
    freqs = np.fft.fftfreq(10)
    ax2.bar(range(10), avg_fft, alpha=0.7)
    ax2.set_xlabel('Frequency component')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('FFT of Digit Embeddings\nPeaks indicate periodic structure')
    ax2.set_xticks(range(10))
    ax2.grid(True, alpha=0.3)
    
    # 3. Singular values - how many dimensions are used?
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(range(len(S)), S / S.sum(), alpha=0.7)
    ax3.set_xlabel('Singular Value Index')
    ax3.set_ylabel('Normalized Singular Value')
    ax3.set_title('Embedding Dimensionality\n(How many dims are actually used?)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Check for mod-10 structure with different bases
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Project onto first 2 PCs and check angles
    angles = np.arctan2(embeddings_2d[:, 1], embeddings_2d[:, 0])
    ax4.scatter(range(10), angles, s=100)
    ax4.plot(range(10), angles, 'b--', alpha=0.5)
    
    # If Fourier, angles should increase linearly (digits on a circle)
    linear_fit = np.polyfit(range(10), angles, 1)
    ax4.plot(range(10), np.polyval(linear_fit, range(10)), 'r-', alpha=0.7, label=f'Linear fit (slope={linear_fit[0]:.3f})')
    
    ax4.set_xlabel('Digit')
    ax4.set_ylabel('Angle (radians)')
    ax4.set_title('Embedding Angles in PC1-PC2\nLinear = circular arrangement')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Pairwise distances - do similar digits cluster?
    ax5 = fig.add_subplot(2, 3, 5)
    
    distances = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
    
    sns.heatmap(distances, ax=ax5, cmap='viridis', annot=True, fmt='.1f',
                xticklabels=range(10), yticklabels=range(10))
    ax5.set_title('Pairwise Embedding Distances\nLook for patterns (e.g., 0↔9 close?)')
    
    # 6. Check modular structure - distance vs |i-j| mod 10
    ax6 = fig.add_subplot(2, 3, 6)
    
    diffs = []
    dists = []
    for i in range(10):
        for j in range(i+1, 10):
            diff = min(abs(i-j), 10 - abs(i-j))  # Circular distance
            diffs.append(diff)
            dists.append(distances[i, j])
    
    ax6.scatter(diffs, dists, alpha=0.6, s=100)
    ax6.set_xlabel('Circular Distance |i-j| (mod 10)')
    ax6.set_ylabel('Embedding Distance')
    ax6.set_title('Distance vs Circular Digit Difference\nIf periodic: V-shape or correlation')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fourier_embeddings.png", dpi=150)
    plt.close()
    print("Saved fourier_embeddings.png")


def analyze_weight_fourier(model, device, output_dir: Path):
    """
    Analyze weight matrices for Fourier structure.
    
    In grokking, the MLP and attention weights often show clear periodic/frequency
    patterns - like stripes or checkerboards indicating specific frequencies.
    """
    model.eval()
    
    fig = plt.figure(figsize=(20, 16))
    
    # Get various weight matrices
    weights_to_analyze = []
    
    # Token embeddings
    weights_to_analyze.append(('Token Embedding', model.token_embed.weight.detach().cpu().numpy()))
    
    # Output projection
    weights_to_analyze.append(('Output Projection', model.output_proj.weight.detach().cpu().numpy()))
    
    # First encoder layer weights
    enc_layer = model.encoder.layers[0]
    weights_to_analyze.append(('Encoder Self-Attn (in_proj)', enc_layer.self_attn.in_proj_weight.detach().cpu().numpy()))
    weights_to_analyze.append(('Encoder FFN (fc1)', enc_layer.linear1.weight.detach().cpu().numpy()))
    weights_to_analyze.append(('Encoder FFN (fc2)', enc_layer.linear2.weight.detach().cpu().numpy()))
    
    # First decoder layer weights
    dec_layer = model.decoder.layers[0]
    weights_to_analyze.append(('Decoder Self-Attn (in_proj)', dec_layer.self_attn.in_proj_weight.detach().cpu().numpy()))
    weights_to_analyze.append(('Decoder Cross-Attn (in_proj)', dec_layer.multihead_attn.in_proj_weight.detach().cpu().numpy()))
    weights_to_analyze.append(('Decoder FFN (fc1)', dec_layer.linear1.weight.detach().cpu().numpy()))
    
    n_weights = len(weights_to_analyze)
    cols = 4
    rows = (n_weights + cols - 1) // cols
    
    for idx, (name, W) in enumerate(weights_to_analyze):
        ax = fig.add_subplot(rows, cols, idx + 1)
        
        # Show weight matrix (truncate if too large)
        W_show = W[:min(64, W.shape[0]), :min(64, W.shape[1])]
        
        im = ax.imshow(W_show, aspect='auto', cmap='RdBu_r', 
                       vmin=-np.percentile(np.abs(W_show), 95),
                       vmax=np.percentile(np.abs(W_show), 95))
        ax.set_title(f'{name}\n{W.shape}', fontsize=10)
        ax.set_xlabel('Input dim')
        ax.set_ylabel('Output dim')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Weight Matrices (look for stripes/periodic patterns)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "weight_matrices.png", dpi=150)
    plt.close()
    
    # Now do 2D FFT of key weight matrices
    fig2 = plt.figure(figsize=(16, 12))
    
    key_weights = [
        ('Token Embedding', model.token_embed.weight.detach().cpu().numpy()),
        ('Output Projection', model.output_proj.weight.detach().cpu().numpy()),
        ('Encoder FFN (fc1)', enc_layer.linear1.weight.detach().cpu().numpy()),
        ('Decoder FFN (fc1)', dec_layer.linear1.weight.detach().cpu().numpy()),
    ]
    
    for idx, (name, W) in enumerate(key_weights):
        ax = fig2.add_subplot(2, 4, idx + 1)
        
        # Truncate for FFT
        W_trunc = W[:min(128, W.shape[0]), :min(128, W.shape[1])]
        
        ax.imshow(W_trunc, aspect='auto', cmap='RdBu_r')
        ax.set_title(f'{name}\n(Weights)', fontsize=10)
        ax.axis('off')
        
        # FFT magnitude
        ax2 = fig2.add_subplot(2, 4, idx + 5)
        fft_2d = np.fft.fft2(W_trunc)
        fft_magnitude = np.abs(np.fft.fftshift(fft_2d))
        
        # Log scale for better visualization
        fft_log = np.log1p(fft_magnitude)
        
        ax2.imshow(fft_log, aspect='auto', cmap='hot')
        ax2.set_title(f'{name}\n(2D FFT magnitude)', fontsize=10)
        ax2.axis('off')
    
    plt.suptitle('Weight FFT Analysis - Bright spots = dominant frequencies', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "weight_fft.png", dpi=150)
    plt.close()
    print("Saved weight_matrices.png and weight_fft.png")


def analyze_neuron_activations(model, device, output_dir: Path):
    """
    Analyze which neurons activate for different inputs.
    
    Key questions:
    - Are there neurons that specialize in detecting specific digits?
    - Are there neurons that detect carries?
    - Do activations show periodic patterns?
    """
    model.eval()
    
    # Hook to capture activations
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                activations[name] = output[0].detach().cpu()
        return hook
    
    # Register hooks
    hooks = []
    
    # Encoder FFN activations (after ReLU/GELU)
    for i, layer in enumerate(model.encoder.layers):
        hooks.append(layer.linear1.register_forward_hook(make_hook(f'enc_{i}_ffn1')))
    
    # Decoder FFN activations
    for i, layer in enumerate(model.decoder.layers):
        hooks.append(layer.linear1.register_forward_hook(make_hook(f'dec_{i}_ffn1')))
    
    # Generate test cases: all single digit additions
    test_cases = []
    for a in range(10):
        for b in range(10):
            test_cases.append((a, b, a + b))
    
    # Collect activations for all test cases
    all_activations = {k: [] for k in ['enc_0_ffn1', 'dec_0_ffn1']}
    
    with torch.no_grad():
        for a, b, expected in test_cases:
            activations.clear()
            eq = f"{a}+{b}="
            src = torch.tensor([model.tokenize(eq)], device=device)
            _ = model.generate(src, max_len=6)
            
            for key in all_activations.keys():
                if key in activations:
                    # Take mean activation across positions
                    act = activations[key].mean(dim=1).numpy()  # (1, hidden)
                    all_activations[key].append(act[0])
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Analyze activations
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Encoder activations heatmap (10x10 grid for a+b)
    ax1 = fig.add_subplot(2, 3, 1)
    
    if all_activations['enc_0_ffn1']:
        enc_acts = np.array(all_activations['enc_0_ffn1'])  # (100, hidden)
        
        # Reshape to 10x10xhidden
        enc_acts_grid = enc_acts.reshape(10, 10, -1)
        
        # Show mean activation for each (a, b) pair
        mean_act = enc_acts_grid.mean(axis=2)
        sns.heatmap(mean_act, ax=ax1, cmap='viridis', annot=True, fmt='.2f',
                   xticklabels=range(10), yticklabels=range(10))
        ax1.set_xlabel('b')
        ax1.set_ylabel('a')
        ax1.set_title('Encoder Mean Activation\nfor a+b=')
    
    # 2. Top neurons by variance (most "interesting")
    ax2 = fig.add_subplot(2, 3, 2)
    
    if all_activations['enc_0_ffn1']:
        variances = enc_acts.var(axis=0)
        top_neurons = np.argsort(variances)[-20:][::-1]
        
        ax2.bar(range(20), variances[top_neurons])
        ax2.set_xlabel('Neuron (ranked by variance)')
        ax2.set_ylabel('Variance')
        ax2.set_title('Top 20 Most Variable Encoder Neurons')
        ax2.set_xticks(range(20))
        ax2.set_xticklabels(top_neurons, rotation=45)
    
    # 3. Activation pattern for top neuron
    ax3 = fig.add_subplot(2, 3, 3)
    
    if all_activations['enc_0_ffn1']:
        top_neuron = top_neurons[0]
        neuron_acts = enc_acts[:, top_neuron].reshape(10, 10)
        
        sns.heatmap(neuron_acts, ax=ax3, cmap='RdBu_r', center=0,
                   xticklabels=range(10), yticklabels=range(10), annot=True, fmt='.1f')
        ax3.set_xlabel('b')
        ax3.set_ylabel('a')
        ax3.set_title(f'Top Neuron #{top_neuron} Activation\n(What pattern does it detect?)')
    
    # 4. Activation vs sum value
    ax4 = fig.add_subplot(2, 3, 4)
    
    if all_activations['enc_0_ffn1']:
        sums = np.array([a + b for a in range(10) for b in range(10)])
        
        for i, neuron_idx in enumerate(top_neurons[:5]):
            neuron_acts = enc_acts[:, neuron_idx]
            ax4.scatter(sums + i*0.1, neuron_acts, alpha=0.5, label=f'N{neuron_idx}', s=30)
        
        ax4.set_xlabel('Sum (a+b)')
        ax4.set_ylabel('Activation')
        ax4.set_title('Top 5 Neurons vs Sum Value\n(Linear = encoding sum)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    
    # 5. Activation vs carry (sum >= 10)
    ax5 = fig.add_subplot(2, 3, 5)
    
    if all_activations['enc_0_ffn1']:
        carries = (sums >= 10).astype(int)
        
        no_carry_acts = enc_acts[carries == 0].mean(axis=0)
        carry_acts = enc_acts[carries == 1].mean(axis=0)
        
        diff = carry_acts - no_carry_acts
        top_carry_neurons = np.argsort(np.abs(diff))[-20:][::-1]
        
        ax5.bar(range(20), diff[top_carry_neurons])
        ax5.set_xlabel('Neuron')
        ax5.set_ylabel('Carry - No Carry (mean activation)')
        ax5.set_title('Neurons Most Different for Carry vs No-Carry')
        ax5.set_xticks(range(20))
        ax5.set_xticklabels(top_carry_neurons, rotation=45)
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # 6. FFT of neuron activations (check for periodic response to sum)
    ax6 = fig.add_subplot(2, 3, 6)
    
    if all_activations['enc_0_ffn1']:
        # Group by sum value and average
        sum_activations = []
        for s in range(19):  # 0 to 18
            mask = sums == s
            if mask.any():
                sum_activations.append(enc_acts[mask].mean(axis=0))
        
        sum_activations = np.array(sum_activations)  # (19, hidden)
        
        # FFT along sum dimension for each neuron
        fft_mags = np.abs(np.fft.fft(sum_activations, axis=0))
        avg_fft = fft_mags.mean(axis=1)
        
        ax6.bar(range(len(avg_fft)), avg_fft)
        ax6.set_xlabel('Frequency')
        ax6.set_ylabel('FFT Magnitude')
        ax6.set_title('FFT of Activations vs Sum\nPeaks = periodic encoding of sum')
    
    plt.tight_layout()
    plt.savefig(output_dir / "neuron_activations.png", dpi=150)
    plt.close()
    print("Saved neuron_activations.png")


def analyze_periodic_neurons(model, device, output_dir: Path):
    """
    Analyze neurons for periodic/sinusoidal activation patterns.
    
    Similar to the modular arithmetic grokking analysis - looks for neurons
    that fire in sinusoidal patterns based on (a+b) or (a+b) mod N.
    
    This is the key signature of Fourier-based computation!
    """
    model.eval()
    
    # Hook to capture activations
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                activations[name] = output[0].detach().cpu()
        return hook
    
    # Register hooks on encoder FFN layers
    hooks = []
    for i, layer in enumerate(model.encoder.layers):
        hooks.append(layer.linear1.register_forward_hook(make_hook(f'enc_{i}_ffn1')))
    
    # Generate comprehensive test cases: 2-digit + 2-digit additions
    # This gives us sums from 0+0=0 to 99+99=198
    test_cases = []
    for a in range(100):
        for b in range(100):
            test_cases.append((a, b, a + b))
    
    print(f"  Collecting activations for {len(test_cases)} test cases...")
    
    # Collect activations
    all_activations = []
    all_sums = []
    
    with torch.no_grad():
        for a, b, expected in tqdm(test_cases, desc="  Collecting", leave=False):
            activations.clear()
            eq = f"{a}+{b}="
            src = torch.tensor([model.tokenize(eq)], device=device)
            _ = model.generate(src, max_len=6)
            
            if 'enc_0_ffn1' in activations:
                # Take mean activation across positions
                act = activations['enc_0_ffn1'].mean(dim=1).numpy()[0]  # (hidden,)
                all_activations.append(act)
                all_sums.append(expected)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    if not all_activations:
        print("  Warning: No activations collected")
        return
    
    all_activations = np.array(all_activations)  # (10000, hidden)
    all_sums = np.array(all_sums)  # (10000,)
    
    hidden_dim = all_activations.shape[1]
    print(f"  Collected {len(all_activations)} samples, {hidden_dim} neurons")
    
    # Find neurons with most sinusoidal response to sum
    # Method: For each neuron, compute R² of sinusoidal fit
    
    def fit_sinusoid_r2(x, y):
        """Compute R² for best sinusoidal fit y = A*sin(wx + phi) + C"""
        try:
            # Try different frequencies
            best_r2 = 0
            for freq in np.linspace(0.01, 0.5, 50):  # Frequencies from slow to fast
                # Create sin and cos features
                sin_feat = np.sin(2 * np.pi * freq * x)
                cos_feat = np.cos(2 * np.pi * freq * x)
                ones = np.ones_like(x)
                
                # Least squares fit: y = a*sin + b*cos + c
                X = np.column_stack([sin_feat, cos_feat, ones])
                try:
                    coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ coeffs
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    best_r2 = max(best_r2, r2)
                except:
                    pass
            return best_r2
        except:
            return 0
    
    print("  Finding neurons with sinusoidal patterns...")
    neuron_r2_scores = []
    for n in tqdm(range(hidden_dim), desc="  Analyzing", leave=False):
        r2 = fit_sinusoid_r2(all_sums, all_activations[:, n])
        neuron_r2_scores.append(r2)
    
    neuron_r2_scores = np.array(neuron_r2_scores)
    top_periodic_neurons = np.argsort(neuron_r2_scores)[-10:][::-1]
    
    print(f"  Top periodic neurons: {top_periodic_neurons}")
    print(f"  Their R² scores: {neuron_r2_scores[top_periodic_neurons]}")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Top 6 most periodic neurons
    for idx, neuron in enumerate(top_periodic_neurons[:6]):
        ax = fig.add_subplot(3, 3, idx + 1)
        
        # Scatter plot of activation vs sum
        ax.scatter(all_sums, all_activations[:, neuron], alpha=0.1, s=5, c='steelblue')
        
        # Compute and plot mean activation for each sum value
        unique_sums = np.unique(all_sums)
        mean_acts = []
        for s in unique_sums:
            mask = all_sums == s
            mean_acts.append(all_activations[mask, neuron].mean())
        mean_acts = np.array(mean_acts)
        
        ax.plot(unique_sums, mean_acts, 'r-', linewidth=2.5, label='Mean Activation')
        
        ax.set_xlabel('Sum (a + b)')
        ax.set_ylabel('Activation')
        ax.set_title(f'Neuron {neuron}\n(R² = {neuron_r2_scores[neuron]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Distribution of R² scores
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.hist(neuron_r2_scores, bins=50, color='steelblue', edgecolor='white')
    ax7.axvline(x=neuron_r2_scores[top_periodic_neurons[0]], color='red', 
                linestyle='--', label=f'Top neuron R²={neuron_r2_scores[top_periodic_neurons[0]]:.3f}')
    ax7.set_xlabel('Sinusoidal R² Score')
    ax7.set_ylabel('Count')
    ax7.set_title('Distribution of Neuron Periodicity\n(Higher = more sinusoidal)')
    ax7.legend()
    
    # Plot 8: Best neuron with modular analysis
    ax8 = fig.add_subplot(3, 3, 8)
    best_neuron = top_periodic_neurons[0]
    
    # Try mod 10 (ones digit periodicity)
    sums_mod10 = all_sums % 10
    for mod_val in range(10):
        mask = sums_mod10 == mod_val
        acts = all_activations[mask, best_neuron]
        ax8.scatter(np.full(len(acts), mod_val) + np.random.normal(0, 0.1, len(acts)), 
                   acts, alpha=0.1, s=3)
    
    # Mean per mod value
    means = [all_activations[sums_mod10 == m, best_neuron].mean() for m in range(10)]
    ax8.plot(range(10), means, 'r-o', linewidth=2, markersize=8, label='Mean')
    ax8.set_xlabel('(a + b) mod 10')
    ax8.set_ylabel('Activation')
    ax8.set_title(f'Neuron {best_neuron} vs Sum mod 10\n(Ones digit periodicity)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Another best neuron with mod 100 (for carry detection)
    ax9 = fig.add_subplot(3, 3, 9)
    
    # Check for "carry boundary" neurons - different behavior around 10, 20, etc.
    carries = all_sums // 10  # Number of tens in the sum
    for c in range(min(20, int(carries.max()) + 1)):
        mask = carries == c
        if mask.sum() > 0:
            acts = all_activations[mask, best_neuron]
            ax9.scatter(np.full(len(acts), c) + np.random.normal(0, 0.1, len(acts)), 
                       acts, alpha=0.1, s=3)
    
    # Mean per carry value
    unique_carries = np.unique(carries)
    means = [all_activations[carries == c, best_neuron].mean() for c in unique_carries if c < 20]
    ax9.plot(range(len(means)), means, 'r-o', linewidth=2, markersize=6, label='Mean')
    ax9.set_xlabel('Sum // 10 (tens digit)')
    ax9.set_ylabel('Activation')
    ax9.set_title(f'Neuron {best_neuron} vs Sum // 10\n(Carry/tens digit pattern)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Periodic Neuron Analysis\n(Looking for Fourier-like sinusoidal patterns)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "periodic_neurons.png", dpi=150)
    plt.close()
    print("  Saved periodic_neurons.png")
    
    # Save a second figure focusing on mod-10 patterns (like the modular arithmetic example)
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, neuron in enumerate(top_periodic_neurons[:6]):
        ax = axes[idx // 3, idx % 3]
        
        # Scatter with mean line - classic modular arithmetic grokking style
        ax.scatter(all_sums, all_activations[:, neuron], alpha=0.05, s=2, c='steelblue')
        
        # Mean activation line
        mean_acts = []
        for s in unique_sums:
            mask = all_sums == s
            mean_acts.append(all_activations[mask, neuron].mean())
        
        ax.plot(unique_sums, mean_acts, 'r-', linewidth=2.5, label='Mean Activation')
        
        ax.set_xlabel('(a + b)', fontsize=12)
        ax.set_ylabel('Activation', fontsize=12)
        ax.set_title(f'Neuron {neuron} Activation vs (a+b)\nR² = {neuron_r2_scores[neuron]:.3f}', 
                    fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Top Periodic Neurons - Sinusoidal Activation Patterns\n(Red line = mean activation, similar to modular arithmetic grokking)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "periodic_neurons_detail.png", dpi=150)
    plt.close()
    print("  Saved periodic_neurons_detail.png")


def analyze_output_logits(model, device, output_dir: Path):
    """
    Analyze the output logit patterns for different inputs.
    
    This shows what the model "thinks" for each position - are there
    interesting patterns in how confident it is for different digits?
    """
    model.eval()
    
    fig = plt.figure(figsize=(18, 12))
    
    # Test cases showing different behaviors
    test_cases = [
        ("5+3=", "8"),
        ("7+8=", "15"),
        ("99+1=", "100"),
        ("45+67=", "112"),
    ]
    
    for idx, (eq, expected) in enumerate(test_cases):
        with torch.no_grad():
            src = torch.tensor([model.tokenize(eq)], device=device)
            
            # Get encoder output
            src_mask = (src == model.TOKENS['<pad>'])
            memory = model.encode(src, src_mask)
            
            # Generate step by step and collect logits
            output = torch.full((1, 1), model.TOKENS['<sos>'], dtype=torch.long, device=device)
            all_logits = []
            
            for _ in range(len(expected) + 1):
                logits = model.decode(output, memory, memory_mask=src_mask)
                all_logits.append(logits[0, -1, :].cpu().numpy())
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                output = torch.cat([output, next_token], dim=1)
        
        # Plot logits for each output position
        ax = fig.add_subplot(2, 2, idx + 1)
        
        all_logits = np.array(all_logits)  # (steps, vocab)
        
        # Focus on digit tokens (0-9)
        digit_logits = all_logits[:, model.TOKENS['0']:model.TOKENS['9']+1]
        
        # Softmax to get probabilities
        digit_probs = np.exp(digit_logits) / np.exp(digit_logits).sum(axis=1, keepdims=True)
        
        sns.heatmap(digit_probs.T, ax=ax, cmap='Blues', vmin=0, vmax=1,
                   yticklabels=range(10), annot=True, fmt='.2f')
        
        generated = model.detokenize(output[0].tolist())
        correct = "✓" if generated == expected else "✗"
        
        ax.set_xlabel('Output Position')
        ax.set_ylabel('Digit')
        ax.set_title(f'{eq} → {generated} (expected: {expected}) {correct}')
        ax.set_xticklabels([f'pos{i}' for i in range(len(all_logits))])
    
    plt.suptitle('Output Probabilities by Position\n(Shows model confidence for each digit)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "output_logits.png", dpi=150)
    plt.close()
    print("Saved output_logits.png")


def analyze_grokking_progress(model, device, output_dir: Path, epoch_num: int):
    """
    Check for signs of grokking - the transition from memorization to generalization.
    
    Key indicators:
    1. Sudden accuracy jumps
    2. Weight norm changes
    3. Emergence of structured representations
    """
    model.eval()
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Weight norms by layer
    ax1 = fig.add_subplot(2, 2, 1)
    
    layer_names = []
    weight_norms = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_names.append(name.replace('.weight', '').replace('_', '\n'))
            weight_norms.append(param.data.norm().item())
    
    ax1.barh(range(len(layer_names)), weight_norms)
    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels(layer_names, fontsize=8)
    ax1.set_xlabel('L2 Norm')
    ax1.set_title('Weight Norms by Layer')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Embedding space structure metric
    ax2 = fig.add_subplot(2, 2, 2)
    
    digit_tokens = [model.TOKENS[str(i)] for i in range(10)]
    embeddings = model.token_embed.weight[digit_tokens].detach().cpu().numpy()
    
    # Compute "circularity" - how well do digits form a circle?
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(embeddings_centered, full_matrices=False)
    embeddings_2d = U[:, :2] * S[:2]
    
    # Fit circle to points
    center = embeddings_2d.mean(axis=0)
    radii = np.linalg.norm(embeddings_2d - center, axis=1)
    circularity = 1 - (radii.std() / radii.mean()) if radii.mean() > 0 else 0
    
    # Check if order is preserved (0,1,2,...,9 around circle)
    angles = np.arctan2(embeddings_2d[:, 1] - center[1], embeddings_2d[:, 0] - center[0])
    angle_order = np.argsort(angles)
    
    # How many digits are in correct circular order?
    order_score = 0
    for i in range(10):
        expected_next = (i + 1) % 10
        actual_pos = np.where(angle_order == i)[0][0]
        next_pos = (actual_pos + 1) % 10
        if angle_order[next_pos] == expected_next:
            order_score += 1
    order_score /= 10
    
    metrics = {
        'Circularity': circularity,
        'Order Score': order_score,
        'Top-2 Variance Ratio': (S[0] + S[1]) / S.sum() if S.sum() > 0 else 0,
    }
    
    ax2.bar(metrics.keys(), metrics.values())
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Score')
    ax2.set_title(f'Embedding Structure Metrics\n(Higher = more Fourier-like)')
    for i, (k, v) in enumerate(metrics.items()):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # 3. Generalization gap over training
    ax3 = fig.add_subplot(2, 2, 3)
    
    history_path = Path("checkpoints/history.json")
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        
        if 'train_acc' in history and 'interp_acc' in history:
            epochs = history['epoch']
            train_acc = np.array(history['train_acc'])
            interp_acc = np.array(history['interp_acc'])
            
            gap = train_acc - interp_acc
            
            ax3.plot(epochs, train_acc, label='Train', alpha=0.8)
            ax3.plot(epochs, interp_acc, label='Interpolation', alpha=0.8)
            ax3.fill_between(epochs, interp_acc, train_acc, alpha=0.3, label='Gap')
            
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Generalization Gap\n(Grokking = gap suddenly closes)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axvline(x=epoch_num, color='red', linestyle='--', alpha=0.7, label='Current')
    
    # 4. Summary assessment
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary = f"""
    GROKKING PROGRESS ASSESSMENT (Epoch {epoch_num})
    ================================================
    
    Embedding Structure:
    - Circularity:     {metrics['Circularity']:.2%} {'✓' if metrics['Circularity'] > 0.5 else '○'}
    - Order preserved: {metrics['Order Score']:.0%} {'✓' if metrics['Order Score'] > 0.7 else '○'}  
    - 2D concentration: {metrics['Top-2 Variance Ratio']:.0%}
    
    Interpretation:
    - Low scores → Still memorizing
    - High scores → Fourier structure emerging
    - Circularity > 0.8 + Order > 0.7 → Likely grokked!
    
    Weight Analysis:
    - Max norm: {max(weight_norms):.2f}
    - Mean norm: {np.mean(weight_norms):.2f}
    """
    
    ax4.text(0.1, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "grokking_progress.png", dpi=150)
    plt.close()
    print("Saved grokking_progress.png")


def generate_compendium(epoch: str = "latest"):
    """Generate full analysis compendium with Fourier analysis and mechanistic interpretability."""
    
    import sys
    sys.path.insert(0, '..')
    from device_utils import get_device
    device = get_device()
    
    # Find checkpoint
    checkpoint_path = find_checkpoint(epoch)
    if checkpoint_path is None:
        print(f"Error: No checkpoint found for epoch={epoch}")
        return
    
    # Get epoch number
    try:
        epoch_num = int(checkpoint_path.stem.split('e')[-1]) if 'e' in checkpoint_path.stem else 99999
    except:
        epoch_num = 99999  # Treat 'final' as late-stage
    
    # Output directory
    epoch_label = epoch_num if epoch_num != 99999 else "final"
    output_dir = Path(f"analysis/compendium_e{epoch_label}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine curriculum phase
    if epoch_num < 5000:
        curriculum = "Phase 1 (1-2 digit numbers)"
    elif epoch_num < 15000:
        curriculum = "Phase 2 (up to 3 digits)"
    elif epoch_num < 30000:
        curriculum = "Phase 3 (up to 4 digits)"
    else:
        curriculum = "Phase 4 (full 5 digits)"
    
    print("=" * 60)
    print(f"Generating Compendium for epoch {epoch_label}")
    print(f"Curriculum: {curriculum}")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    model = AdditionTransformer().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # ========== BASIC ANALYSES ==========
    print("\n" + "=" * 40)
    print("BASIC ANALYSES")
    print("=" * 40)
    
    print("\n[1/10] Training history...")
    try:
        plot_training_history(output_dir)
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n[2/10] Model predictions...")
    try:
        analyze_attention_patterns(model, device, output_dir)
    except Exception as e:
        print(f"  Warning: {e}")
    
    print(f"\n[3/10] Carry handling {curriculum}...")
    try:
        carry_results = analyze_carry_handling(model, device, output_dir, epoch_num)
        print(f"  Results: {carry_results}")
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n[4/10] Basic digit embeddings...")
    try:
        analyze_digit_embeddings(model, device, output_dir)
    except Exception as e:
        print(f"  Warning: {e}")
    
    print(f"\n[5/10] Position accuracy {curriculum}...")
    try:
        analyze_position_accuracy(model, device, output_dir, epoch_num)
    except Exception as e:
        print(f"  Warning: {e}")
    
    # ========== FOURIER / MECHANISTIC ANALYSES ==========
    print("\n" + "=" * 40)
    print("FOURIER & MECHANISTIC ANALYSES")
    print("=" * 40)
    
    print("\n[6/10] Fourier embedding analysis...")
    try:
        analyze_fourier_embeddings(model, device, output_dir)
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n[7/10] Weight matrix analysis & FFT...")
    try:
        analyze_weight_fourier(model, device, output_dir)
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n[8/11] Neuron activation patterns...")
    try:
        analyze_neuron_activations(model, device, output_dir)
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n[9/11] Periodic neuron analysis (sinusoidal patterns)...")
    try:
        analyze_periodic_neurons(model, device, output_dir)
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n[10/11] Output logit analysis...")
    try:
        analyze_output_logits(model, device, output_dir)
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n[11/11] Grokking progress assessment...")
    try:
        analyze_grokking_progress(model, device, output_dir, epoch_num)
    except Exception as e:
        print(f"  Warning: {e}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print(f"Compendium complete! Output: {output_dir}")
    print("=" * 60)
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")
    print("\nKey files for Fourier analysis:")
    print("  - fourier_embeddings.png     : Check for circular digit structure")
    print("  - weight_fft.png             : Look for frequency peaks in weights")
    print("  - neuron_activations.png     : Which neurons detect what?")
    print("  - periodic_neurons.png       : Sinusoidal activation patterns (like mod arith grokking!)")
    print("  - periodic_neurons_detail.png: Detailed view of top periodic neurons")
    print("  - grokking_progress.png      : Has the model grokked yet?")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate analysis compendium")
    parser.add_argument("epoch", type=str, nargs="?", default="latest",
                        help="Epoch to analyze (default: latest)")
    parser.add_argument("--periodic-only", action="store_true",
                        help="Only run the periodic neuron analysis (fast)")
    
    args = parser.parse_args()
    
    if args.periodic_only:
        # Quick mode: just run periodic neuron analysis
        import sys
        sys.path.insert(0, '..')
        from device_utils import get_device
        device = get_device()
        
        checkpoint_path = find_checkpoint(args.epoch)
        if checkpoint_path is None:
            print(f"Error: No checkpoint found for epoch={args.epoch}")
            exit(1)
        
        try:
            epoch_num = int(checkpoint_path.stem.split('e')[-1]) if 'e' in checkpoint_path.stem else 99999
        except:
            epoch_num = 99999
        
        epoch_label = epoch_num if epoch_num != 99999 else "final"
        output_dir = Path(f"analysis/compendium_e{epoch_label}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading model from {checkpoint_path}...")
        model = AdditionTransformer().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        print("Running periodic neuron analysis...")
        analyze_periodic_neurons(model, device, output_dir)
        print(f"Done! Check {output_dir}")
    else:
        generate_compendium(args.epoch)
