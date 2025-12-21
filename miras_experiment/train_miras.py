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
        # For each prime, we use all possible (a, b) pairs if p is small
        for a in range(p):
            for b in range(p):
                target = (a + b) % p
                all_data.append((p, a, b, target))
    random.shuffle(all_data)
    return all_data

def train_miras():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training Universal Titans on: {device}")
    
    # We'll use a set of primes for training
    train_primes = [11, 13, 17, 19, 23]
    # We'll test on a completely unseen prime
    test_primes = [29, 31]
    
    model = UniversalFourierTransformer(max_p=150, d_model=128, d_mem=128).to(device)
    
    # High weight decay to force the model to find the 'Algorithm' rather than memorizing
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Data
    train_data = generate_multi_prime_data(train_primes)
    unseen_data = generate_multi_prime_data(test_primes)
    
    print(f"Train samples: {len(train_data)}, Unseen Test samples: {len(unseen_data)}")
    
    history = {"epoch": [], "train_loss": [], "train_acc": [], "unseen_acc": []}
    
    batch_size = 512
    n_epochs = 10000
    
    pbar = tqdm(range(n_epochs), desc="Grokking...")
    for epoch in pbar:
        model.train()
        
        # Sample mini-batch
        samples = random.sample(train_data, min(len(train_data), batch_size))
        p_v = torch.tensor([x[0] for x in samples], device=device)
        a_v = torch.tensor([x[1] for x in samples], device=device)
        b_v = torch.tensor([x[2] for x in samples], device=device)
        t_v = torch.tensor([x[3] for x in samples], device=device)
        
        logits = model(p_v, a_v, b_v)
        loss = criterion(logits, t_v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Train Acc
                train_acc = (logits.argmax(dim=-1) == t_v).float().mean().item()
                
                # Unseen Test Acc (The real goal)
                u_samples = random.sample(unseen_data, min(len(unseen_data), batch_size))
                p_u = torch.tensor([x[0] for x in u_samples], device=device)
                a_u = torch.tensor([x[1] for x in u_samples], device=device)
                b_u = torch.tensor([x[2] for x in u_samples], device=device)
                t_u = torch.tensor([x[3] for x in u_samples], device=device)
                
                u_logits = model(p_u, a_u, b_u)
                u_acc = (u_logits.argmax(dim=-1) == t_u).float().mean().item()
                
                history["epoch"].append(epoch)
                history["train_loss"].append(loss.item())
                history["train_acc"].append(train_acc)
                history["unseen_acc"].append(u_acc)
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "unseen": f"{u_acc:.1%}"})

    # Save final
    output_path = Path("miras_experiment/checkpoints")
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / "titans_mod_final.pt")
    with open(output_path / "history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    train_miras()
