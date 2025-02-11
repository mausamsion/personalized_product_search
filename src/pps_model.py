import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
import yaml
import random
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# read config
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
config['device'] = eval(config['device'])

# set seeds
random.seed(config['seed'])
np.random.seed(config['seed'])
torch.random.seed = config['seed']

class PPSDataset(Dataset):
    """
    Returns:
        Dict of various inputs
    """
    def __init__(self):
        super().__init__()
        self.df = None
        self.num_items = None
        self.vocab_size = None
    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        data = self.df.iloc[index]
        query_words_idx = None
        item_idx = None
        words_idx = None
        return {
            'query_words': torch.tensor(query_words_idx, dtype=torch.int), 
            'history_items': torch.tensor(item_idx, dtype=torch.int), 
            'review_words': torch.tensor(words_idx, dtype=torch.int)
        }

class TEM(nn.Module):
    def __init__(self, vocab_size, num_items):
        super().__init__()
        self.d_model = config['embedding_dim']
        # Word embeddings (shared between query and item language model)
        self.word_embeddings = nn.Embedding(vocab_size, config['embedding_dim'])
        # Query representation components
        self.query_proj = nn.Linear(config['embedding_dim'], 
                                    config['embedding_dim']) 
        # Item embeddings
        self.item_embeddings = nn.Embedding(num_items, config['embedding_dim'])
        # Positional embeddings
        self.pos_embeddings = nn.Embedding(config['max_input_length'], 
                                           config['embedding_dim'])
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['embedding_dim'], 
                nhead=config['num_heads'], 
                dim_feedforward=4*config['embedding_dim'],
                batch_first=True
            ), 
            num_layers=config['num_layers'])
        # Item language model head
        self.item_lm_head = nn.Linear(config['embedding_dim'], vocab_size, bias=False)
        self.item_lm_head.weight = self.word_embeddings.weight  # Tie weights

    def forward(self, query_words, history_items):
        """
        Args:
            query_words: (batch_size, query_len)
            history_items: (batch_size, history_len)
            review_words: (batch_size, history_len, review_len)
        Returns:
            m_qu: (batch_size, d_model) joint query-user representation
        """
        
        # --- Query Representation - Embed query words and average
        query_emb = self.word_embeddings(query_words).mean(dim=1)
        query_trans = torch.tanh(self.query_proj(query_emb))
        
        # --- Historical Item Embeddings 
        history_emb = self.item_embeddings(history_items)
        
        # --- Query and Items Sequence - 
        seq = torch.cat([query_trans.unsqueeze(1), history_emb], dim=1)
        seq_len = seq.size(1)
        
        # Positional embeddings (positions 0, 1, ..., seq_len-1)
        positions = torch.arange(
            seq_len, device=query_words.device).expand(
                config['batch_size'], seq_len)
        
        pos_emb = self.pos_embeddings(positions)
        # add positional embeddings
        seq = seq + pos_emb
        
        # --- Transformer output 
        transformer_output = self.transformer(seq)
        m_qu = transformer_output[0] 
        return m_qu
    
    def get_item_generation_model_loss(self, m_qu, items):
        item_embs = self.item_embeddings.weight
        logits = torch.matmul(m_qu, item_embs.T)
        return F.cross_entropy(logits, items)
    
    def get_item_langugage_model_loss(self, items, words):
        batch_size, _, _ = words.shape
        item_embs = self.item_embeddings(items)
        losses = []
        # Calculate loss for each item and its review words, in a batch
        for i in range(batch_size):
            word_embs = self.word_embeddings.weight
            logits = torch.matmul(item_embs[i], word_embs)
            losses.append(F.cross_entropy(logits, words[i]))
        # Return mean of all item losses
        return torch.mean(losses)

def validate_tem():
    pass

def test_tem(model, query_words, history_items, num_items, top_k):
    """
    Input: Given a sequence of query and historicallly purchased items
    Output: Ranked list of items the user is likely to purchase in future
    """
    model.eval()
    with torch.no_grad():
        m_qu = model(query_words, history_items)
        all_idx = torch.tensor(list(range(num_items)))
        # Remove the history items indices
        keep_idx = all_idx[~torch.isin(all_idx, history_items)]
        # Select the item embeddings of kept indices
        item_embs = model.item_embeddings(keep_idx)
        logits = torch.matmul(m_qu, item_embs.T)
        top_logits, top_indices = torch.topk(
            logits, min(top_k, logits.shape[0]), largest=True, sorted=True)
        # Select item indices from the kept indices based on top_k
        return [keep_idx[id] for id in top_indices]

def train_tem(rundir):
    # Create dataset and dataloader
    dataset = PPSDataset()
    dataloader = DataLoader(dataset, 
                            batch_size=config['batch_size'], 
                            num_workers=config['num_workers'],
                            shuffle=True, 
                            pin_memory=True)
    # Initialize model
    model = TEM(dataset.vocab_size, dataset.num_items)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    model.to(config['device'])
    model.train()

    # Training loop
    global_step = 0
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            # ---
            query_words = batch['query_words'].to(config['device'])
            history_items = batch['history_items'].to(config['device'])
            review_words = batch['review_words'].to(config['device'])
            # ---
            m_qu = model(query_words, history_items)
            # Item Generation Model Loss 
            loss_igm = model.get_item_generation_model_loss(m_qu, history_items)
            # Item Language Model Loss 
            loss_ilm = model.get_item_langugage_model_loss(history_items, 
                                                           review_words)
            loss = loss_igm + loss_ilm
            loss.backward()
            optimizer.step()
            # ---
            total_loss += loss.item()
            if global_step%config['log_step'] == 0:
                logging.info(f"epoch {epoch+1}, " +\
                             f"global step {global_step}, " +\
                             f"loss = {loss:.5f}")
            global_step += 1
        logging.info(f"--- END - epoch {epoch+1}/{config['epochs']}, " +\
                     f"avg. loss = {total_loss/len(dataloader):.5f}")

if __name__=='__main__':
    logdir = 'train_logs'
    model_name = 'TEM'
    rundir = f"{logdir}/{model_name}_" +\
             f"{datetime.now().strftime(format='%Y-%m-%d_%H-%M-%S')}"
    # create log directory
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f'{rundir}/logs', 
        force=False
    )
    train_tem()

