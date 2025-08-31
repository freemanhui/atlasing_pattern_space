import argparse, os, torch, pandas as pd
from aps.utils import toy_corpus, cooc_ppmi, svd_embed
from aps.topology import TopologicalAutoencoder, TopoAEConfig, knn_indices, adjacency_from_knn

def cmd_fit_topo(args):
    os.makedirs(args.outdir, exist_ok=True)
    toks = toy_corpus(); ppmi, vocab = cooc_ppmi(toks, window=2); X = svd_embed(ppmi, d=50)
    X_t = torch.tensor(X, dtype=torch.float32)
    A = adjacency_from_knn(knn_indices(X_t, k=args.topo_k), n=X.shape[0])
    cfg = TopoAEConfig(in_dim=X.shape[1], latent_dim=args.latent, topo_k=args.topo_k, topo_weight=args.topo_weight)
    model = TopologicalAutoencoder(cfg).fit(X_t, A, epochs=args.epochs)
    with torch.no_grad(): Z, _ = model(X_t)
    df = pd.DataFrame(Z.numpy(), columns=[f'z{i+1}' for i in range(Z.shape[1])]); df['word'] = vocab
    df.to_csv(os.path.join(args.outdir, 'embedding_topo_ae.csv'), index=False)
    print(f'[aps] saved {args.outdir}/embedding_topo_ae.csv')

def main(argv=None):
    ap = argparse.ArgumentParser(prog='aps')
    sp = ap.add_subparsers(dest='cmd', required=True)
    ap_topo = sp.add_parser('fit-topo', help='fit topology-preserving AE')
    ap_topo.add_argument('--latent', type=int, default=2)
    ap_topo.add_argument('--epochs', type=int, default=100)
    ap_topo.add_argument('--topo-k', type=int, default=8)
    ap_topo.add_argument('--topo-weight', type=float, default=1.0)
    ap_topo.add_argument('--outdir', type=str, default='outputs')
    ap_topo.set_defaults(func=cmd_fit_topo)
    args = ap.parse_args(argv); args.func(args)

if __name__ == '__main__':
    main()
