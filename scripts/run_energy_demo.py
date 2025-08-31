import argparse, os, numpy as np, torch, matplotlib.pyplot as plt
from aps.utils import toy_corpus, cooc_ppmi, svd_embed
from aps.topology import TopologicalAutoencoder, TopoAEConfig
from aps.energy import MemoryEnergy, MemoryEnergyConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--beta', type=float, default=5.0)
    ap.add_argument('--n-mem', type=int, default=6)
    ap.add_argument('--outdir', type=str, default='outputs')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    toks = toy_corpus()
    ppmi, vocab = cooc_ppmi(toks, window=2)
    X = svd_embed(ppmi, d=20)
    X_t = torch.tensor(X, dtype=torch.float32)

    cfg = TopoAEConfig(in_dim=X.shape[1], latent_dim=2, topo_weight=0.0)
    model = TopologicalAutoencoder(cfg).fit(X_t, target_adj=torch.zeros(X.shape[0], X.shape[0]), epochs=args.epochs)
    with torch.no_grad():
        Z, _ = model(X_t)

    mem = MemoryEnergy(MemoryEnergyConfig(latent_dim=2, n_mem=args.n_mem, beta=args.beta))
    opt = torch.optim.Adam(mem.parameters(), lr=1e-2)
    for ep in range(200):
        loss = mem.loss(Z)
        opt.zero_grad(); loss.backward(); opt.step()

    E = mem.energy(Z).detach().numpy()
    Znp = Z.numpy()
    plt.figure(figsize=(6,6))
    sc = plt.scatter(Znp[:,0], Znp[:,1], c=E)
    plt.colorbar(sc, label='Energy')
    plt.title('Energy basins over 2D latent')
    plt.tight_layout()
    path = os.path.join(args.outdir, 'energy_basins.png')
    plt.savefig(path, dpi=150)
    print(f'Saved {path}')

if __name__ == '__main__':
    main()
