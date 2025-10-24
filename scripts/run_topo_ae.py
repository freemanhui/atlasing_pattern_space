import argparse
import os
import torch
import pandas as pd
from aps.utils import toy_corpus, cooc_ppmi, svd_embed, scatter_labels
from aps.topology import TopologicalAutoencoder, TopoAEConfig, knn_indices, adjacency_from_knn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--topo-k', type=int, default=8)
    ap.add_argument('--topo-weight', type=float, default=1.0)
    ap.add_argument('--outdir', type=str, default='outputs')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    toks = toy_corpus()
    ppmi, vocab = cooc_ppmi(toks, window=2)
    X = svd_embed(ppmi, d=50)
    X_t = torch.tensor(X, dtype=torch.float32)

    idx = knn_indices(torch.tensor(X, dtype=torch.float32), k=args.topo_k)
    A = adjacency_from_knn(idx, n=X.shape[0])

    cfg = TopoAEConfig(in_dim=X.shape[1], latent_dim=args.latent, topo_k=args.topo_k, topo_weight=args.topo_weight)
    model = TopologicalAutoencoder(cfg).fit(X_t, A, epochs=args.epochs)

    with torch.no_grad():
        Z, _ = model(X_t)
    Z = Z.numpy()

    # save CSV
    df = pd.DataFrame(Z, columns=[f'z{i+1}' for i in range(Z.shape[1])])
    df['word'] = vocab
    df.to_csv(os.path.join(args.outdir, 'embedding_topo_ae.csv'), index=False)

    # plot subset
    sel = ['cat','mouse','dog','lion','car','train','airplane','joy','anger','hope']
    sel_idx = [vocab.index(w) for w in sel if w in vocab]
    scatter_labels(Z[sel_idx], [vocab[i] for i in sel_idx], 'Topo AE (toy corpus)', os.path.join(args.outdir, 'topo_ae_scatter.png'))
    print(f'Saved to {args.outdir}/')

if __name__ == '__main__':
    main()
