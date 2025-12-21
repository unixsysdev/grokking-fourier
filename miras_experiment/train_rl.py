import torch
import torch.nn as nn
from model_miras import UniversalFourierTransformer
from tqdm import tqdm
import random
import json
from pathlib import Path

def generate_multi_prime_data(primes, samples_per_prime=1000):
    all_data = []
    for p in primes:
        if p < 20:
            for a in range(p):
                for b in range(p):
                    all_data.append((p, a, b, (a + b) % p))
        else:
            for _ in range(samples_per_prime):
                a = random.randint(0, p - 1)
                b = random.randint(0, p - 1)
                all_data.append((p, a, b, (a + b) % p))
    random.shuffle(all_data)
    return all_data

def train_rl(resume: bool = False, checkpoint: str = None, start_epoch: int = 0, n_epochs: int = 100000):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Training SIN-PE Policy Gradient (RL) on: {device} ---")
    
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
    random.shuffle(all_primes)
    split_idx = int(len(all_primes) * 0.8)
    train_primes_all = sorted(all_primes[:split_idx])
    test_primes = sorted(all_primes[split_idx:])

    model = UniversalFourierTransformer(max_p=150, d_model=128, d_mem=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1.0)
    
    output_path = Path("checkpoints/rl_sinpe")
    output_path.mkdir(parents=True, exist_ok=True)
    
    history = {"epoch": [], "avg_reward": [], "unseen_acc": []}
    running_reward = 0.0
    beta_entropy = 0.01
    
    # Resume from full training state
    checkpoint_path = output_path / "training_state.pt"
    if resume and checkpoint_path.exists():
        print(f"Resuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        running_reward = ckpt["running_reward"]
        print(f"Resumed at epoch {start_epoch}, running_reward: {running_reward:.3f}")
    elif resume:
        print("No training_state.pt found, starting fresh...")
    
    # Load model weights only (for continuing from old checkpoints)
    if checkpoint:
        print(f"Loading model weights from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=device)
        # Handle both old format (just state_dict) and new format (dict with "model" key)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "running_reward" in ckpt:
                running_reward = ckpt["running_reward"]
        else:
            model.load_state_dict(ckpt)
        # Load existing history if available
        history_path = output_path / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            print(f"Loaded history with {len(history['epoch'])} entries")
            # Estimate running_reward from history if available
            if history["avg_reward"]:
                running_reward = history["avg_reward"][-1]
                print(f"Estimated running_reward from history: {running_reward:.3f}")
        print(f"Starting from epoch {start_epoch} with fresh optimizer")
    
    # Create scheduler with appropriate T_max (remaining epochs)
    remaining_epochs = n_epochs - start_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    print(f"Training from epoch {start_epoch} to {n_epochs} ({remaining_epochs} epochs, LR: {scheduler.get_last_lr()[0]:.2e})")
    
    batch_size = 512
    pbar = tqdm(range(start_epoch, n_epochs), desc="RL SinPE Training")

    for epoch in pbar:
        model.train()
        current_primes = train_primes_all if epoch > 10000 else train_primes_all[:8]
        
        batch = generate_multi_prime_data(current_primes, samples_per_prime=50)
        samples = random.sample(batch, min(len(batch), batch_size))
        
        p_v = torch.tensor([x[0] for x in samples], device=device)
        a_v = torch.tensor([x[1] for x in samples], device=device)
        b_v = torch.tensor([x[2] for x in samples], device=device)
        t_v = torch.tensor([x[3] for x in samples], device=device)
        
        # RL Pass
        logits = model(p_v, a_v, b_v)
        probs = torch.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        # Reward: 1 for correct, 0 for wrong
        reward = (action == t_v).float()
        
        # Policy Gradient Loss: -log_prob * (Reward - Baseline)
        # We use a simple moving average for baseline
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        
        # Baseline subtraction reduces variance
        advantage = reward - (running_reward if epoch > 0 else 0.5)
        loss = -(log_prob * advantage).mean() - beta_entropy * entropy.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update running reward for baseline (exponential moving average)
        batch_avg_reward = reward.mean().item()
        running_reward = 0.99 * running_reward + 0.01 * batch_avg_reward
        
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                test_data = generate_multi_prime_data(test_primes, samples_per_prime=100)
                u_samples = random.sample(test_data, min(len(test_data), batch_size))
                p_u = torch.tensor([x[0] for x in u_samples], device=device)
                a_u = torch.tensor([x[1] for x in u_samples], device=device)
                b_u = torch.tensor([x[2] for x in u_samples], device=device)
                t_u = torch.tensor([x[3] for x in u_samples], device=device)
                
                u_logits = model(p_u, a_u, b_u)
                u_acc = (u_logits.argmax(dim=-1) == t_u).float().mean().item()
                
                history["epoch"].append(epoch)
                history["avg_reward"].append(batch_avg_reward)
                history["unseen_acc"].append(u_acc)
                pbar.set_postfix({"R": f"{batch_avg_reward:.2f}", "U": f"{u_acc:.1%}", "LR": f"{scheduler.get_last_lr()[0]:.1e}"})
                
                if epoch % 5000 == 0:
                    torch.save(model.state_dict(), output_path / f"rl_e{epoch}.pt")
                    # Save full training state for resume
                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "history": history,
                        "running_reward": running_reward,
                    }, checkpoint_path)
                    with open(output_path / "history.json", "w") as f:
                        json.dump(history, f)

    torch.save(model.state_dict(), output_path / "rl_final.pt")
    # Save final training state
    torch.save({
        "epoch": n_epochs - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "running_reward": running_reward,
    }, checkpoint_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train RL model with optional resume")
    parser.add_argument("--resume", action="store_true", help="Resume from training_state.pt (full state)")
    parser.add_argument("--checkpoint", type=str, help="Load model weights from a .pt file (fresh optimizer)")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch (use with --checkpoint)")
    parser.add_argument("--n_epochs", type=int, default=100000, help="Total epochs to train")
    args = parser.parse_args()
    train_rl(resume=args.resume, checkpoint=args.checkpoint, start_epoch=args.start_epoch, n_epochs=args.n_epochs)