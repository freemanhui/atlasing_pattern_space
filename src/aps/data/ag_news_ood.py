"""AG News dataset with domain shift for OOD generalization experiments.

Uses AG News 4-class classification (World/Sports/Business/Sci-Tech) to create
domain shift by treating categories as domains. We create synthetic sentiment
labels based on keyword heuristics.

Training domains: Sports, Business, Sci-Tech
Test domain (OOD): World

This tests whether APS can learn sentiment representations invariant to topic.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import Literal, Optional
import numpy as np


class AGNewsOOD(Dataset):
    """
    AG News with domain shift for OOD experiments.
    
    Creates binary sentiment classification across news categories:
    - Training domains: Sports (0), Business (1), Sci-Tech (2)
    - Test domain (OOD): World (3)
    
    Sentiment is synthetically created from keywords to ensure
    topic-sentiment correlation exists but is spurious.
    
    Args:
        domains: List of category IDs to include (0=World, 1=Sports, 2=Business, 3=Sci-Tech)
        split: 'train' or 'test'
        root: Cache directory for embeddings
        bert_model: BERT model name for embeddings
        use_cache: Whether to cache BERT embeddings
        max_samples_per_domain: Limit samples per domain (for faster testing)
    """
    
    # AG News categories
    CATEGORIES = {
        0: 'World',
        1: 'Sports', 
        2: 'Business',
        3: 'Sci-Tech'
    }
    
    # Sentiment keywords (simple heuristic)
    POSITIVE_WORDS = ['win', 'success', 'profit', 'growth', 'breakthrough', 'achieve', 
                      'excel', 'improve', 'gain', 'victory', 'advance', 'strong']
    NEGATIVE_WORDS = ['loss', 'fail', 'crisis', 'decline', 'problem', 'concern',
                      'worry', 'struggle', 'weak', 'defeat', 'drop', 'cut']
    
    def __init__(
        self,
        domains: list = [1, 2, 3],  # Sports, Business, Sci-Tech
        split: Literal['train', 'test'] = 'train',
        root: str = './data/ag_news_cache',
        bert_model: str = 'bert-base-uncased',
        use_cache: bool = True,
        max_samples_per_domain: Optional[int] = None
    ):
        self.domains = domains
        self.split = split
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bert_model = bert_model
        self.use_cache = use_cache
        self.max_samples_per_domain = max_samples_per_domain
        
        # Load data
        self._load_data()
        
        # Always compute or load embeddings
        self._load_or_compute_embeddings()
    
    def _load_data(self):
        """Load AG News dataset and filter by domains."""
        print(f"Loading AG News {self.split} split...")
        
        # Load from HuggingFace
        dataset = load_dataset('ag_news', split=self.split)
        
        # Filter by domains and assign synthetic sentiment
        self.samples = []
        
        for item in tqdm(dataset, desc=f"Filtering domains {self.domains}"):
            label = item['label']  # Category: 0=World, 1=Sports, 2=Business, 3=Sci-Tech
            
            if label in self.domains:
                text = item['text']
                
                # Synthetic sentiment from keywords
                text_lower = text.lower()
                pos_count = sum(1 for word in self.POSITIVE_WORDS if word in text_lower)
                neg_count = sum(1 for word in self.NEGATIVE_WORDS if word in text_lower)
                
                # Binary sentiment: positive if more positive words, else negative
                # Add some randomness to avoid perfect correlation
                if pos_count > neg_count:
                    sentiment = 1 if np.random.rand() > 0.1 else 0  # 90% correct
                elif neg_count > pos_count:
                    sentiment = 0 if np.random.rand() > 0.1 else 1
                else:
                    sentiment = np.random.randint(0, 2)  # Random if neutral
                
                self.samples.append({
                    'text': text,
                    'category': label,
                    'sentiment': sentiment,
                    'domain_id': self.domains.index(label)
                })
                
                # Limit samples per domain
                if self.max_samples_per_domain:
                    domain_counts = {}
                    for s in self.samples:
                        domain_counts[s['category']] = domain_counts.get(s['category'], 0) + 1
                    if domain_counts.get(label, 0) >= self.max_samples_per_domain:
                        continue
        
        print(f"  Loaded {len(self.samples)} samples from domains {[self.CATEGORIES[d] for d in self.domains]}")
    
    def _load_or_compute_embeddings(self):
        """Load cached embeddings or compute them."""
        cache_file = self.root / f"embeddings_{self.split}_{'_'.join(map(str, self.domains))}_{self.bert_model.replace('/', '_')}.pkl"
        
        # Try to load from cache if enabled and exists
        if self.use_cache and cache_file.exists():
            try:
                print(f"Loading cached embeddings from {cache_file}...")
                with open(cache_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"  Loaded {len(self.embeddings)} embeddings")
                return
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"  Warning: Cache file corrupted ({e}), recomputing...")
                cache_file.unlink()  # Delete corrupted file
        
        # Compute embeddings
        print(f"Computing BERT embeddings...")
        self._compute_embeddings()
        
        # Save cache only if use_cache is True
        if self.use_cache:
            print(f"Saving embeddings to {cache_file}...")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
        else:
            print(f"  Caching disabled, embeddings computed on-the-fly")
    
    def _compute_embeddings(self):
        """Compute BERT [CLS] embeddings for all texts."""
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        model = AutoModel.from_pretrained(self.bert_model)
        model.eval()
        
        # Use best available device: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        model = model.to(device)
        print(f"  Using device: {device}")
        
        # Compute in batches and store as numpy float16 for efficiency
        batch_size = 32
        embeddings_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.samples), batch_size), desc="Computing embeddings"):
                batch_samples = self.samples[i:i+batch_size]
                texts = [s['text'] for s in batch_samples]
                
                # Tokenize batch
                inputs = tokenizer(
                    texts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=128,
                    padding='max_length'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get [CLS] embeddings
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # [CLS] tokens
                
                # Store as float16 numpy for memory efficiency
                embeddings_list.append(embeddings.half().numpy())
        
        # Concatenate and convert to list of tensors
        all_embeddings = np.concatenate(embeddings_list, axis=0)
        self.embeddings = [torch.from_numpy(emb).float() for emb in all_embeddings]
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample."""
        sample = self.samples[idx]
        
        result = {
            'text': sample['text'],
            'sentiment': torch.tensor(sample['sentiment'], dtype=torch.long),
            'category': torch.tensor(sample['category'], dtype=torch.long),
            'domain_id': torch.tensor(sample['domain_id'], dtype=torch.long)
        }
        
        # Add embedding if available
        if hasattr(self, 'embeddings'):
            result['embedding'] = self.embeddings[idx]
        
        return result
    
    def __len__(self) -> int:
        return len(self.samples)


def create_ag_news_ood_loaders(
    batch_size: int = 64,
    num_workers: int = 4,
    use_cache: bool = True,
    max_samples_per_domain: Optional[int] = None
) -> dict:
    """
    Create train/test dataloaders for AG News OOD experiment.
    
    Training domains: Sports (1), Business (2), Sci-Tech (3)
    Test domain (OOD): World (0)
    
    Args:
        batch_size: Batch size
        num_workers: DataLoader workers
        use_cache: Cache BERT embeddings
        max_samples_per_domain: Limit samples (for testing)
    
    Returns:
        dict with keys 'train' and 'test_ood'
    """
    # Training: Sports, Business, Sci-Tech
    train_dataset = AGNewsOOD(
        domains=[1, 2, 3],
        split='train',
        use_cache=use_cache,
        max_samples_per_domain=max_samples_per_domain
    )
    
    # Test (OOD): World
    test_ood_dataset = AGNewsOOD(
        domains=[0],
        split='test',
        use_cache=use_cache,
        max_samples_per_domain=max_samples_per_domain
    )
    
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test_ood': DataLoader(
            test_ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return loaders


def get_domain_stats(dataset: AGNewsOOD):
    """Get statistics about domain and sentiment distribution."""
    category_counts = {}
    sentiment_counts = {}
    domain_sentiment = {}
    
    for sample in dataset.samples:
        cat = sample['category']
        sent = sample['sentiment']
        
        category_counts[cat] = category_counts.get(cat, 0) + 1
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
        
        key = (cat, sent)
        domain_sentiment[key] = domain_sentiment.get(key, 0) + 1
    
    return {
        'category_counts': category_counts,
        'sentiment_counts': sentiment_counts,
        'domain_sentiment': domain_sentiment
    }


if __name__ == '__main__':
    """Quick test of AG News OOD dataset."""
    print("Testing AG News OOD dataset...")
    
    # Create small dataset for testing
    print("\n[1/3] Creating train dataset (Sports, Business, Sci-Tech)...")
    train_ds = AGNewsOOD(
        domains=[1, 2, 3],
        split='train',
        use_cache=False,
        max_samples_per_domain=100  # Small for testing
    )
    
    print(f"  Train: {len(train_ds)} samples")
    
    print("\n[2/3] Creating test dataset (World - OOD)...")
    test_ds = AGNewsOOD(
        domains=[0],
        split='test',
        use_cache=False,
        max_samples_per_domain=100
    )
    
    print(f"  Test: {len(test_ds)} samples")
    
    print("\n[3/3] Checking sample format...")
    sample = train_ds[0]
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Sentiment: {sample['sentiment'].item()} (0=negative, 1=positive)")
    print(f"  Category: {sample['category'].item()} ({train_ds.CATEGORIES[sample['category'].item()]})")
    print(f"  Domain ID: {sample['domain_id'].item()}")
    if 'embedding' in sample:
        print(f"  Embedding shape: {sample['embedding'].shape}")
    
    print("\n[4/4] Domain statistics...")
    train_stats = get_domain_stats(train_ds)
    print("  Train category distribution:")
    for cat, count in sorted(train_stats['category_counts'].items()):
        print(f"    {train_ds.CATEGORIES[cat]}: {count} samples")
    
    print("\n✓ AG News OOD dataset working correctly!")
