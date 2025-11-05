import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0) NDCG IMPLEMENTATION
# ---------------------------------------------------------

def ndcg_at_k(y_true, scores, k=None):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    if k is None or k > len(y_sorted):
        k = len(y_sorted)
    rel_k = y_sorted[:k]
    gains = (2 ** rel_k - 1) / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()
    ideal_rel = np.sort(y_true)[::-1][:k]
    ideal_gains = (2 ** ideal_rel - 1) / np.log2(np.arange(2, k + 2))
    idcg = ideal_gains.sum()
    return float(dcg / idcg) if idcg != 0 else 0.0

# ---------------------------------------------------------
# 1) LOAD DARPA DATA AND CREATE LABELS
# ---------------------------------------------------------

input_path = "/Users/skynet/Documents/Projects/GAN/Machine-Intelligence-v1.0/data/Engagement_1/cadets/pandex/ProcessEvent.csv"
groundtruth_path = "/Users/skynet/Documents/Projects/GAN/Machine-Intelligence-v1.0/data/Engagement_1/cadets/pandex/cadets_pandex_webshell.csv"

print("Loading data...")
_processes = pd.read_csv(input_path)
_labels_df = pd.read_csv(groundtruth_path)

_apt_list = _labels_df.loc[_labels_df["label"] == "AdmSubject::Node"]["uuid"]
outlier_indices = _processes[_processes["Object_ID"].isin(_apt_list)].index.tolist()
normal_indices = _processes[~_processes["Object_ID"].isin(_apt_list)].index.tolist()

df = _processes.copy()
df["label"] = 0
df.loc[outlier_indices, "label"] = 1

excluded_cols = ["Object_ID", "label"]
event_labels = [col for col in _processes.columns if col not in excluded_cols]

X = df[event_labels].values.astype(np.float32)
y = df["label"].values.astype(int)

print("Data summary:")
print(f"Features: {len(event_labels)} | Samples: {len(df)} | Positives: {sum(y)}")

# ---------------------------------------------------------
# 2) SIMPLE AUTOENCODER MODEL
# ---------------------------------------------------------

class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_factor, n_hidden):
        super().__init__()
        enc_layers = []
        hidden_dim = max(4, int(input_dim * hidden_factor))
        prev_dim = input_dim
        for _ in range(n_hidden):
            enc_layers.append(nn.Linear(prev_dim, hidden_dim))
            enc_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        enc_layers.append(nn.Linear(prev_dim, latent_dim))
        enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        dec_layers = []
        prev_dim = latent_dim
        for _ in range(n_hidden):
            dec_layers.append(nn.Linear(prev_dim, hidden_dim))
            dec_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_ae(model, train_loader, lr=1e-3, epochs=10, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="mean")
    model.train()
    for _ in range(epochs):
        for (xb,) in train_loader:
            xb = xb.to(device)
            x_hat = model(xb)
            loss = loss_fn(x_hat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def compute_anomaly_scores(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        X_hat = model(X_t)
        return ((X_t - X_hat) ** 2).mean(dim=1).cpu().numpy()

# ---------------------------------------------------------
# 3) GENETIC REPRESENTATION & NEURO-EVOLUTION LOOP
# ---------------------------------------------------------

def random_individual():
    return {
        "latent_dim": random.choice([8, 16, 32, 64]),
        "hidden_factor": random.choice([0.25, 0.5, 0.75]),
        "n_hidden": random.choice([1, 2, 3]),
        "lr": random.choice([1e-4, 5e-4, 1e-3]),
    }

def crossover(a, b):
    return {k: random.choice([a[k], b[k]]) for k in a.keys()}

def mutate(ind, p=0.3):
    for key in ind.keys():
        if random.random() < p:
            if key == "latent_dim":
                ind[key] = random.choice([8, 16, 32, 64])
            elif key == "hidden_factor":
                ind[key] = random.choice([0.25, 0.5, 0.75])
            elif key == "n_hidden":
                ind[key] = random.choice([1, 2, 3])
            elif key == "lr":
                ind[key] = random.choice([1e-4, 5e-4, 1e-3])
    return ind

def evaluate_individual(ind, X_train, y_val, X_val, batch_size=256, epochs=10, device="cpu"):
    input_dim = X_train.shape[1]
    dataset = TensorDataset(torch.from_numpy(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleAE(
        input_dim=input_dim,
        latent_dim=ind["latent_dim"],
        hidden_factor=ind["hidden_factor"],
        n_hidden=ind["n_hidden"],
    )
    model = train_ae(model, loader, lr=ind["lr"], epochs=epochs, device=device)
    scores = compute_anomaly_scores(model, X_val, device=device)
    auc = roc_auc_score(y_val, scores)
    ndcg_full = ndcg_at_k(y_val, scores, k=None)
    return auc, ndcg_full, model

def neuro_evolution_baseline(X_train, X_val, y_val, pop_size=8, generations=5, batch_size=256, epochs=5, device="cpu"):
    population = [random_individual() for _ in range(pop_size)]
    best_ind, best_auc, best_ndcg, best_model = None, -1, -1, None
    ndcg_progress = []  # track nDCG per generation

    for g in range(generations):
        print(f"\n=== Generation {g+1}/{generations} ===")
        fitness = []
        for i, ind in enumerate(population):
            print(f"  Evaluating individual {i+1}/{len(population)}: {ind}")
            auc, ndcg_full, model = evaluate_individual(
                ind, X_train, y_val, X_val, batch_size, epochs, device
            )
            print(f"    -> AUC = {auc:.4f}, nDCG = {ndcg_full:.4f}")
            fitness.append((auc, ndcg_full, ind, model))

        fitness.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_auc_gen, best_ndcg_gen, best_ind_gen, best_model_gen = fitness[0]

        ndcg_progress.append(best_ndcg_gen)  # record generation's best
        print(f"  [Best in gen {g+1}] AUC = {best_auc_gen:.4f}, nDCG = {best_ndcg_gen:.4f}")

        if best_auc_gen > best_auc or (best_auc_gen == best_auc and best_ndcg_gen > best_ndcg):
            best_auc, best_ndcg, best_ind, best_model = best_auc_gen, best_ndcg_gen, best_ind_gen, best_model_gen

        # reproduction
        elites = [f[2] for f in fitness[:max(2, pop_size // 3)]]
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = mutate(crossover(p1, p2))
            new_pop.append(child)
        population = new_pop

    # Save and plot nDCG progression
    np.save("evoae_ndcg_progress.npy", np.array(ndcg_progress))
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(ndcg_progress)+1), ndcg_progress, marker="o", lw=2)
    plt.title("Evo-AE nDCG Progression over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best nDCG")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("evoae_ndcg_curve.png", dpi=300)
    plt.show()

    print("\n=== Neuro-evolution finished ===")
    print(f"Best AUC:  {best_auc:.4f}")
    print(f"Best nDCG: {best_ndcg:.4f}")
    print(f"Best configuration: {best_ind}")
    return best_ind, best_auc, best_ndcg, best_model, ndcg_progress

# ---------------------------------------------------------
# 4) RUN THE BASELINE
# ---------------------------------------------------------

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning Evo-AE baseline on device: {device}")

    best_ind, best_auc, best_ndcg, best_model, ndcg_progress = neuro_evolution_baseline(
        X_train, X_val, y_val,
        pop_size=80, generations=30,
        batch_size=256, epochs=30,
        device=device,
    )

    print("\nFinal best configuration:")
    print(best_ind)
    print(f"Final AUC:  {best_auc:.4f}")
    print(f"Final nDCG: {best_ndcg:.4f}")
    print(f"nDCG progression saved to evoae_ndcg_progress.npy and evoae_ndcg_curve.png")
