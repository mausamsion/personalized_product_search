import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
import yaml
import random
import wandb
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# read config
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
config['device'] = eval(config['device'])

# set seed
random.seed(config['seed'])
np.random.seed(config['seed'])
torch.random.seed = config['seed']

class TEM(nn.Module):
    def __init__(self, vocab_size, num_items, d_model, nhead, num_layers, max_seq_length):
        super(TEM, self).__init__()
        self.d_model = d_model
        # Word embeddings (shared between query and item language model)
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        # Query representation components (Eq. 2)
        self.query_proj = nn.Linear(d_model, d_model)  # W_φ and b_φ
        # Item embeddings
        self.item_embeddings = nn.Embedding(num_items, d_model)
        # Positional embeddings
        self.pos_embeddings = nn.Embedding(max_seq_length, d_model)
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                batch_first=True
            ),
            num_layers=num_layers)
        # Item language model head (shares word embeddings)
        self.item_lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.item_lm_head.weight = self.word_embeddings.weight  # Tie weights

    def forward(self, query_words, history_items, target_items,
                review_words, mask):
        """
        Args:
            query_words: (batch_size, query_len)
            history_items: (batch_size, history_len)
        Returns:
            m_qu: (batch_size, d_model) joint query-user representation
        """
        batch_size = query_words.size(0)

        # --- Query Representation ---
        # Embed query words and average
        query_emb = self.word_embeddings(query_words)  # (batch, query_len, d_model)
        query_avg = query_emb.mean(dim=1)  # (batch, d_model)
        query_repr = torch.tanh(self.query_proj(query_avg))  # (batch, d_model)

        # --- Historical Item Embeddings ---
        history_emb = self.item_embeddings(history_items)  # (batch, history_len, d_model)

        # --- Query and Items Sequence ---
        # Add positional embeddings to the sequence [query, item1, item2, ...]
        seq = torch.cat([query_repr.unsqueeze(1), history_emb], dim=1)  # (batch, seq_len, d_model)
        seq_len = seq.size(1)
        # Positional embeddings (positions 0, 1, ..., seq_len-1)
        positions = torch.arange(seq_len, device=query_words.device).expand(batch_size, seq_len)
        pos_emb = self.pos_embeddings(positions)  # (batch, seq_len, d_model)
        seq = seq + pos_emb  # Add positional embeddings

        # Transformer output
        transformer_output = self.transformer(seq)  # (seq_len, batch, d_model)
        # Get the query's output (first token)
        m_qu = transformer_output[0]  # (batch, d_model)

        # Item Generation Model Loss
        logits1 = self.compute_item_gen_scores(m_qu)
        loss_igm = F.cross_entropy(logits1, target_items)

        # Item Language Model Loss
        logits2 = self.compute_item_lm_loss(history_items, review_words, mask)
        loss_ilm = F.cross_entropy(logits2)

        return loss_igm

    def compute_item_gen_scores(self, m_qu):
        """Scores items based on dot product with M_qu (Eq. 1)"""
        item_embeddings = self.item_embeddings.weight  # (num_items, d_model)
        return torch.matmul(m_qu, item_embeddings.T)  # (batch, num_items)

    def compute_item_lm_loss(self, items, review_words, mask):
        """
        Compute loss for item language model (Eq. 3).
        Args:
            items: (total_items,) - Flattened item indices
            review_words: (total_items, num_words) - Words from reviews (padded with -1)
        """
        # initialize item embeddings: (batch_size, emb_dim)
        item_embs = self.item_embeddings(items)
        # initialize word embeddings: (batch_size, seq_len, emb_dim)
        word_embs = self.word_embeddings(review_words)
        # calculate logits: (batch_size, seq_len)
        logits = torch.bmm(word_embs, item_embs.unsqueeze(-1)).squeeze(-1)
        log_probs = torch.log_softmax(logits, dim=1)
        # loss calculation (excluding the masked tokens)
        masked_log_probs = log_probs * mask
        loss = -1 * torch.sum(masked_log_probs) / torch.sum(mask)
        return loss

        # Get item embeddings and compute logits
        item_emb = self.item_embeddings(items)  # (total_items, d_model)
        logits = self.item_lm_head(item_emb)  # (total_items, vocab_size)

        # Flatten review words and mask padding
        flat_words = review_words.view(-1)  # (total_items * num_words,)
        mask = flat_words != -1
        valid_words = flat_words[mask]
        if valid_words.numel() == 0:
            return torch.tensor(0.0, device=items.device)

        # Expand logits for each word and compute loss
        logits_expanded = logits.unsqueeze(1).expand(-1, review_words.size(1), -1)  # (total_items, num_words, vocab_size)
        logits_flat = logits_expanded.contiguous().view(-1, logits.size(-1))  # (total_items*num_words, vocab_size)
        logits_valid = logits_flat[mask]  # (num_valid, vocab_size)

        loss = F.cross_entropy(logits_valid, valid_words)
        return loss

def main():
    # Example Usage
    vocab_size = 50000  # Example vocabulary size
    num_items = 100000   # Example number of items
    d_model = 128
    nhead = 4
    num_layers = 2
    max_seq_length = 20  # Max query + history length

    model = TEM(vocab_size, num_items, d_model, nhead, num_layers, max_seq_length)

    # Sample batch
    batch_size = 32
    query_words = torch.randint(0, vocab_size, (batch_size, 5))  # 5 words per query
    history_items = torch.randint(0, num_items, (batch_size, 10))  # 10 history items
    target_items = torch.randint(0, num_items, (batch_size,))
    history_review_words = torch.randint(0, vocab_size, (batch_size, 10, 20))  # 20 words per history item
    target_review_words = torch.randint(0, vocab_size, (batch_size, 20))  # 20 words per target

    # Forward pass
    loss_search = model(query_words, history_items, target_items)

    # Item language model loss (combine history and target items)
    all_items = torch.cat([history_items, target_items.unsqueeze(1)], dim=1).flatten()
    all_reviews = torch.cat([history_review_words, target_review_words.unsqueeze(1)], dim=1).flatten(end_dim=1)
    loss_lm = model.compute_language_model_loss(all_items, all_reviews)

    # Total loss
    total_loss = loss_search + loss_lm
    print(f"Total Loss: {total_loss.item()}")

if __name__=='__main__':
    main()


