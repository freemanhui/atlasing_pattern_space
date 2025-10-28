import re
import numpy as np

def tokenize(s: str): return re.findall(r"[a-z']+", s.lower())

def toy_corpus():
    docs = [
        "the cat chased the mouse and the mouse escaped",
        "a dog barked loudly while the cat slept",
        "wolves hunt in packs and communicate with howls",
        "the lion stalked the gazelle across the savannah",
        "birds migrate across continents using magnetic fields",
        "the fish swim in schools near coral reefs",
        "insects form swarms and build complex hives",
        "whales sing songs that travel long distances underwater",
        "octopus solved a puzzle opening a jar with skill",
        "the elephant remembers paths to hidden water",
        "the car drove down the highway at night",
        "a truck carried cargo through the mountain pass",
        "the train arrived at the station on time",
        "ships navigate by stars and modern gps systems",
        "the airplane climbed above the clouds during flight",
        "bicycles are efficient transportation in crowded cities",
        "the bus stopped at every station on the route",
        "rockets reach orbit carrying satellites beyond earth",
        "submarines travel silently beneath the ocean surface",
        "the scooter zipped across the plaza quickly",
        "joy spreads quickly among friends who celebrate",
        "sadness can linger during cold rainy days",
        "anger rises when expectations break suddenly",
        "curiosity drives discovery and science advances",
        "hope persists despite uncertainty in the future",
        "fear retreats when knowledge grows with light",
        "love binds communities and families together",
        "trust builds slowly and breaks quickly",
        "awe appears when seeing grand mountains",
        "calm returns after storms and conflict",
    ]
    return [tokenize(d) for d in docs]

def cooc_ppmi(tokenized, window=2):
    vocab = sorted(set([w for doc in tokenized for w in doc]))
    w2i = {w:i for i,w in enumerate(vocab)}
    V = len(vocab)
    C = np.zeros((V,V), dtype=np.float32)
    for doc in tokenized:
        idxs = [w2i[w] for w in doc]
        for i, wi in enumerate(idxs):
            s = max(0, i-window)
            e = min(len(idxs), i+window+1)
            for j in range(s, e):
                if j==i:
                    continue
                wj = idxs[j]
                C[wi, wj] += 1
                C[wj, wi] += 1
    eps = 1e-8
    row = C.sum(axis=1, keepdims=True)+eps
    col = C.sum(axis=0, keepdims=True)+eps
    tot = C.sum()+eps
    p_ij = C/tot
    p_i = row/tot
    p_j = col/tot
    ppmi = np.maximum(0, np.log((p_ij+eps)/(p_i@p_j+eps)))
    return ppmi, vocab

def svd_embed(ppmi, d=50):
    U,S,VT = np.linalg.svd(ppmi, full_matrices=False)
    X = U[:,:d]*S[:d]
    X = (X - X.mean(axis=0)) / (X.std(axis=0)+1e-6)
    return X
