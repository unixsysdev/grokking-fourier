"""
Accuracy sweep for Addition Transformer.

Tests the model across different:
- Number ranges (1-digit, 2-digit, ..., 6-digit)
- Operand combinations (small+small, small+large, large+large)
- Edge cases (carries, zeros, etc.)
"""

import warnings
# Suppress ROCm/HIP warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with memory efficient attention.*")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*HIPBLAS_STATUS_NOT_SUPPORTED.*")

import torch
import random
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from model import AdditionTransformer


def find_checkpoint(epoch: str = "latest"):
    """Find checkpoint file."""
    checkpoint_dir = Path("checkpoints")
    
    if epoch == "latest" or epoch == "final":
        # Find latest
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
        # Specific epoch
        for pattern in [f"model_e{epoch}.pt", f"model_{epoch}.pt"]:
            p = checkpoint_dir / pattern
            if p.exists():
                return p
        return None


def generate_test_cases(n_per_category: int = 200, seed: int = 12345):
    """
    Generate comprehensive test cases.
    
    Categories:
    - By digit count: 1+1, 2+2, 3+3, 4+4, 5+5, 6+6 (extrapolation)
    - Mixed: 1+5, 2+4, etc.
    - Edge cases: with carries, zeros
    """
    random.seed(seed)
    
    test_cases = {}
    
    # Same digit counts
    for digits in range(1, 7):
        cases = []
        min_val = 10**(digits-1) if digits > 1 else 0
        max_val = 10**digits - 1
        
        for _ in range(n_per_category):
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)
            cases.append((f"{a}+{b}=", str(a + b)))
        
        test_cases[f"{digits}d+{digits}d"] = cases
    
    # Mixed digit counts
    for d1, d2 in [(1, 3), (1, 5), (2, 4), (2, 5), (3, 5)]:
        cases = []
        min1 = 10**(d1-1) if d1 > 1 else 0
        max1 = 10**d1 - 1
        min2 = 10**(d2-1) if d2 > 1 else 0
        max2 = 10**d2 - 1
        
        for _ in range(n_per_category):
            a = random.randint(min1, max1)
            b = random.randint(min2, max2)
            cases.append((f"{a}+{b}=", str(a + b)))
        
        test_cases[f"{d1}d+{d2}d"] = cases
    
    # Edge cases: lots of carries
    cases = []
    for _ in range(n_per_category):
        # Numbers like 999, 9999 that cause carries
        digits = random.randint(2, 5)
        a = int('9' * digits)
        b = random.randint(1, 10**digits - 1)
        cases.append((f"{a}+{b}=", str(a + b)))
    test_cases["carries"] = cases
    
    # Edge cases: zeros
    cases = []
    for _ in range(n_per_category):
        a = random.randint(0, 99999)
        cases.append((f"{a}+0=", str(a)))
        cases.append((f"0+{a}=", str(a)))
    test_cases["zeros"] = cases[:n_per_category]
    
    # Edge cases: small numbers
    cases = []
    for a in range(20):
        for b in range(20):
            cases.append((f"{a}+{b}=", str(a + b)))
    test_cases["small_exhaustive"] = cases
    
    return test_cases


def evaluate_category(model, test_cases, device):
    """Evaluate model on a category of test cases."""
    model.eval()
    
    correct = 0
    total = len(test_cases)
    errors = []
    
    with torch.no_grad():
        for eq, expected in test_cases:
            src = torch.tensor([model.tokenize(eq)], device=device)
            output = model.generate(src, max_len=12)
            generated = model.detokenize(output[0].tolist())
            
            if generated == expected:
                correct += 1
            else:
                errors.append({
                    'input': eq,
                    'expected': expected,
                    'generated': generated
                })
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'errors': errors[:10]  # Keep first 10 errors for inspection
    }


def get_curriculum_info(epoch_num: int):
    """
    Get curriculum phase information for a given epoch.
    
    Returns:
        phase: str description
        max_answer_digits: int max digits in answer for this phase
        expected_categories: list of categories that should work
    """
    if epoch_num < 5000:
        return {
            'phase': "Phase 1 (1-2 digit numbers)",
            'max_answer_digits': 3,  # 99+99=198
            'expected': ['1d+1d', '2d+2d', '1d+3d', 'small_exhaustive', 'zeros'],
            'not_yet': ['3d+3d', '4d+4d', '5d+5d', '6d+6d', '2d+4d', '2d+5d', '3d+5d']
        }
    elif epoch_num < 15000:
        return {
            'phase': "Phase 2 (up to 3 digit numbers)",
            'max_answer_digits': 4,  # 999+999=1998
            'expected': ['1d+1d', '2d+2d', '3d+3d', '1d+3d', '1d+5d', 'small_exhaustive', 'zeros'],
            'not_yet': ['4d+4d', '5d+5d', '6d+6d', '2d+5d', '3d+5d']
        }
    elif epoch_num < 30000:
        return {
            'phase': "Phase 3 (up to 4 digit numbers)",
            'max_answer_digits': 5,  # 9999+9999=19998
            'expected': ['1d+1d', '2d+2d', '3d+3d', '4d+4d', '1d+3d', '1d+5d', '2d+4d', 'small_exhaustive', 'zeros', 'carries'],
            'not_yet': ['5d+5d', '6d+6d', '2d+5d', '3d+5d']
        }
    else:
        return {
            'phase': "Phase 4 (full 5 digit numbers)",
            'max_answer_digits': 6,  # 99999+99999=199998
            'expected': ['1d+1d', '2d+2d', '3d+3d', '4d+4d', '5d+5d', '1d+3d', '1d+5d', '2d+4d', '2d+5d', '3d+5d', 'small_exhaustive', 'zeros', 'carries'],
            'not_yet': ['6d+6d']  # Always extrapolation
        }


def sweep(epoch: str = "latest", verbose: bool = True):
    """Run full accuracy sweep with curriculum awareness."""
    
    import sys
    sys.path.insert(0, '..')
    from device_utils import get_device
    device = get_device()
    
    # Find checkpoint
    checkpoint_path = find_checkpoint(epoch)
    if checkpoint_path is None:
        print(f"Error: No checkpoint found for epoch={epoch}")
        return
    
    print(f"Loading {checkpoint_path}...")
    
    # Load model
    model = AdditionTransformer().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Determine epoch number and curriculum phase
    try:
        epoch_num = int(checkpoint_path.stem.split('e')[-1]) if 'e' in checkpoint_path.stem else 99999
    except:
        epoch_num = 99999
    
    curriculum = get_curriculum_info(epoch_num)
    
    # Generate test cases
    print("Generating test cases...")
    test_cases = generate_test_cases(n_per_category=200)
    
    # Evaluate each category
    print("\n" + "=" * 60)
    print("ACCURACY SWEEP")
    print("=" * 60)
    
    results = {}
    
    print(f"\nEpoch: {epoch_num} | {curriculum['phase']}")
    print("-" * 60)
    print(f"Expected to work:    {', '.join(curriculum['expected'][:5])}...")
    print(f"Not yet in curriculum: {', '.join(curriculum['not_yet'][:3])}...")
    print("-" * 60)
    
    # Evaluate all categories
    for category, cases in tqdm(test_cases.items(), desc="Evaluating"):
        result = evaluate_category(model, cases, device)
        results[category] = result
    
    # Group by curriculum status
    in_curriculum_expected = [c for c in curriculum['expected'] if c in results]
    not_yet_in_curriculum = [c for c in curriculum['not_yet'] if c in results]
    extrapolation = [c for c in results if '6d' in c]
    other = [c for c in results if c not in in_curriculum_expected + not_yet_in_curriculum + extrapolation]
    
    # Print results - IN CURRICULUM (should work)
    print("\n=== IN CURRICULUM (should be learning these) ===")
    for cat in sorted(in_curriculum_expected):
        if cat in results:
            r = results[cat]
            bar = "█" * int(r['accuracy'] * 20) + "░" * (20 - int(r['accuracy'] * 20))
            status = "✓" if r['accuracy'] > 0.8 else "○" if r['accuracy'] > 0.4 else "✗"
            print(f"{status} {cat:20s} {bar} {r['accuracy']:6.1%} ({r['correct']}/{r['total']})")
    
    # Print results - NOT YET IN CURRICULUM (expected to be low)
    if not_yet_in_curriculum:
        print("\n=== NOT YET IN CURRICULUM (expected to be low) ===")
        for cat in sorted(not_yet_in_curriculum):
            if cat in results:
                r = results[cat]
                bar = "█" * int(r['accuracy'] * 20) + "░" * (20 - int(r['accuracy'] * 20))
                # It's actually GOOD if these are low - shows no cheating
                status = "⏳"
                print(f"{status} {cat:20s} {bar} {r['accuracy']:6.1%} ({r['correct']}/{r['total']})")
    
    # Other categories
    if other:
        print("\n=== OTHER CATEGORIES ===")
        for cat in sorted(other):
            r = results[cat]
            bar = "█" * int(r['accuracy'] * 20) + "░" * (20 - int(r['accuracy'] * 20))
            print(f"  {cat:20s} {bar} {r['accuracy']:6.1%} ({r['correct']}/{r['total']})")
    
    # Extrapolation (always 6-digit, never in training)
    print("\n=== EXTRAPOLATION (6-digit, never in training) ===")
    for cat in sorted(extrapolation):
        r = results[cat]
        bar = "█" * int(r['accuracy'] * 20) + "░" * (20 - int(r['accuracy'] * 20))
        status = "🚀" if r['accuracy'] > 0.5 else "⏳"
        print(f"{status} {cat:20s} {bar} {r['accuracy']:6.1%} ({r['correct']}/{r['total']})")
    
    # Summary with curriculum awareness
    in_curr_cats = [c for c in in_curriculum_expected if c in results]
    not_yet_cats = [c for c in not_yet_in_curriculum if c in results]
    
    in_curr_acc = sum(results[c]['correct'] for c in in_curr_cats) / sum(results[c]['total'] for c in in_curr_cats) if in_curr_cats else 0
    not_yet_acc = sum(results[c]['correct'] for c in not_yet_cats) / sum(results[c]['total'] for c in not_yet_cats) if not_yet_cats else 0
    extrap_acc = sum(results[c]['correct'] for c in extrapolation) / sum(results[c]['total'] for c in extrapolation) if extrapolation else 0
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"In-curriculum accuracy:      {in_curr_acc:6.1%} {'✓ Good!' if in_curr_acc > 0.8 else '○ Learning...' if in_curr_acc > 0.4 else '✗ Needs work'}")
    print(f"Not-yet-curriculum accuracy: {not_yet_acc:6.1%} {'(expected low)' if not_yet_acc < 0.3 else '(surprisingly high!)'}")
    print(f"Extrapolation accuracy:      {extrap_acc:6.1%} {'🚀 Generalizing!' if extrap_acc > 0.5 else '⏳ Not yet'}")
    
    # Show some errors
    # Show some errors
    if verbose:
        print("\n=== SAMPLE ERRORS ===")
        error_cats = in_curr_cats[:2] + extrapolation[:1]
        for cat in error_cats:
            if cat in results and results[cat]['errors']:
                print(f"\n{cat}:")
                for err in results[cat]['errors'][:3]:
                    print(f"  {err['input']} → {err['generated']} (expected {err['expected']})")
    
    # Save results
    output_path = Path("analysis")
    output_path.mkdir(exist_ok=True)
    
    # Convert to JSON-serializable
    json_results = {
        'epoch': epoch_num,
        'curriculum_phase': curriculum['phase'],
        'categories': {
            cat: {
                'accuracy': r['accuracy'],
                'correct': r['correct'],
                'total': r['total']
            }
            for cat, r in results.items()
        },
        'summary': {
            'in_curriculum': in_curr_acc,
            'not_yet_curriculum': not_yet_acc,
            'extrapolation': extrap_acc
        }
    }
    
    with open(output_path / f"sweep_e{epoch_num}.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_path / f'sweep_e{epoch_num}.json'}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Accuracy sweep for Addition Transformer")
    parser.add_argument("--epoch", type=str, default="latest",
                        help="Epoch to test (e.g., '10000', 'final', or 'latest')")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    sweep(epoch=args.epoch, verbose=not args.quiet)
