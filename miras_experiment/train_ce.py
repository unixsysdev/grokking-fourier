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

def train_ce():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Training SIN-PE Cross-Entropy on: {device} ---")
    
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
    random.shuffle(all_primes)
    split_idx = int(len(all_primes) * 0.8)
    train_primes_all = sorted(all_primes[:split_idx])
    test_primes = sorted(all_primes[split_idx:])

    model = UniversalFourierTransformer(max_p=150, d_model=128, d_mem=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000)
    criterion = nn.CrossEntropyLoss()
    
    output_path = Path("miras_experiment/checkpoints/ce_sinpe")
    output_path.mkdir(parents=True, exist_ok=True)
    
    history = {"epoch": [], "train_loss": [], "unseen_acc": []}
    
    batch_size = 1024
    n_epochs = 100000
    pbar = tqdm(range(n_epochs), desc="CE SinPE Training")
    
    for epoch in pbar:
        model.train()
        # Mix of curriculum and random
        current_primes = train_primes_all if epoch > 10000 else train_primes_all[:8]
        
        batch = generate_multi_prime_data(current_primes, samples_per_prime=50)
        samples = random.sample(batch, min(len(batch), batch_size))
        
        p_v = torch.tensor([x[0] for x in samples], device=device)
        a_v = torch.tensor([x[1] for x in samples], device=device)
        b_v = torch.tensor([x[2] for x in samples], device=device)
        t_v = torch.tensor([x[3] for x in samples], device=device)
        
        logits = model(p_v, a_v, b_v)
        loss = criterion(logits, t_v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
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
                history["train_loss"].append(loss.item())
                history["unseen_acc"].append(u_acc)
                pbar.set_postfix({"L": f"{loss.item():.4f}", "U": f"{u_acc:.1%}"})
                
                if epoch % 5000 == 0:
                    torch.save(model.state_dict(), output_path / f"ce_e{epoch}.pt")
                    with open(output_path / "history.json", "w") as f:
                        json.dump(history, f)

    torch.save(model.state_dict(), output_path / "ce_final.pt")

if __name__ == "__main__":
    train_ce()
