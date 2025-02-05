import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
import random
import wandb
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

os.environ["WANDB_START_METHOD"] = "thread"

# set seed
seed = 3367
random.seed(seed)
np.random.seed(seed)
torch.random.seed = seed

# global config
config = {
    'model_name': 'item_language_model',
    'embedding_dim': 128,
    'max_seq_length': 1024,
    'batch_size': 128,
    'lr': 0.001,
    'epochs': 30,
    'log_step': 100,
    'save_embs_epoch': 5,
    'num_workers': 8,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}
logging.info(f"Using device: {config['device']}")

# initialize wandb logging
wb = wandb.init(
    project='transformer_pps',
    name='run_1',
    id='run_1',
    config=config
)

class ItemWordsDataset(Dataset):
    def __init__(self, max_seq_length):
        self.df = pd.read_parquet('data/item_tokens.parquet')
        self.df = self.df[['asin', 'tokens']]
        self.max_seq_length = max_seq_length
        self._create_mappings() # create vocabulary and mappings
        
    def _create_mappings(self):
        # create item ID mappings
        self.original_item_ids = self.df['asin'].unique().tolist()
        self.item_id_to_idx = {
            id: idx for idx, id in enumerate(self.original_item_ids)}
        self.idx_to_item_id = {v: k for k, v in self.item_id_to_idx.items()}
        
        # create token vocabulary
        all_tokens = [token for sublist in list(self.df['tokens']) \
                                for token in sublist]
        token_counts = defaultdict(int)
        for token in all_tokens:
            token_counts[token] += 1
            
        # sort by frequency and create mappings
        sorted_tokens = sorted(token_counts.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        self.token_to_idx = {'[PAD]': 0, '[UNK]': 1}
        self.idx_to_token = {0: '[PAD]', 1: '[UNK]'}
        
        # create token ID mappings
        idx = 2
        for token, _ in sorted_tokens:
            if not token in self.token_to_idx:
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
                idx += 1
            
        self.vocab_size = len(self.token_to_idx)
        self.num_items = len(self.item_id_to_idx)

        # write vocab files
        with open('data/item_vocab.json', 'w') as f:
            json.dump(self.item_id_to_idx, f)
        with open('data/token_vocab.json', 'w') as f:
            json.dump(self.token_to_idx, f)
        # log information
        logging.info(f'Total items: {self.num_items}')
        logging.info(f'Total tokens: {self.vocab_size}')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_id = row['asin']
        tokens = row['tokens']
        
        # Convert to indices
        item_idx = self.item_id_to_idx[item_id]
        token_indices = [self.token_to_idx.get(t, self.token_to_idx['[UNK]']) \
                            for t in tokens]
        sorted_indices = sorted(token_indices)
        
        # Truncate/pad sequence
        if len(token_indices) > self.max_seq_length:
            token_indices = sorted_indices[:self.max_seq_length]
            mask = [1] * self.max_seq_length
        else:
            pad_len = self.max_seq_length - len(sorted_indices)
            token_indices = sorted_indices + \
                            [self.token_to_idx['[PAD]']] * pad_len
            mask = [1]*len(tokens) + [0]*pad_len
            
        return {
            'item_idx': torch.tensor(item_idx, dtype=torch.long),
            'token_indices': torch.tensor(token_indices, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float)
        }

class ItemLanguageModel(nn.Module):
    def __init__(self, num_items, vocab_size, embedding_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        logging.info(
            f'Item embeddings shape: {self.item_embeddings.weight.shape}')
        logging.info(
            f'Word embeddings shape: {self.word_embeddings.weight.shape}')
        
    def forward(self, item_ids, review_words, mask):
        # initialize item embeddings: (batch_size, emb_dim)
        item_embs = self.item_embeddings(item_ids)
        # initialize word embeddings: (batch_size, seq_len, emb_dim)
        word_embs = self.word_embeddings(review_words)
        # calculate logits: (batch_size, seq_len)
        logits = torch.bmm(word_embs, item_embs.unsqueeze(-1)).squeeze(-1)
        log_probs = torch.log_softmax(logits, dim=1)
        # loss calculation (excluding the masked tokens)
        masked_log_probs = log_probs * mask
        loss = -1 * torch.sum(masked_log_probs) / torch.sum(mask)
        return loss


def save_embeddings(model, dataset, rundir, epoch):
    # Save item embeddings
    item_embeddings = model.item_embeddings.weight.data.cpu().numpy()
    item_mapping = {
        'embeddings': item_embeddings,
        'id_to_item': dataset.idx_to_item_id,
        'item_to_id': dataset.item_id_to_idx
    }
    with open(f'{rundir}/item_embeddings_epoch={epoch}.pkl', 'wb') as f:
        pickle.dump(item_mapping, f)
    
    # Save word embeddings
    word_embeddings = model.word_embeddings.weight.data.cpu().numpy()
    token_mapping = {
        'embeddings': word_embeddings,
        'id_to_token': dataset.idx_to_token,
        'token_to_id': dataset.token_to_idx
    }
    with open(f'{rundir}/token_embeddings_epoch={epoch}.pkl', 'wb') as f:
        pickle.dump(token_mapping, f)

def train_item_language_model(rundir):
    # Create dataset and dataloader
    dataset = ItemWordsDataset(max_seq_length=config['max_seq_length'])
    dataloader = DataLoader(dataset, 
                            batch_size=config['batch_size'], 
                            num_workers=config['num_workers'],
                            shuffle=True, 
                            pin_memory=True)
    # Initialize model
    model = ItemLanguageModel(num_items=dataset.num_items,
                              vocab_size=dataset.vocab_size,
                              embedding_dim=config['embedding_dim'])
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    model.to(config['device'])
    wb.watch(model)
    
    # Training loop
    global_step = 0
    for epoch in range(config['epochs']):
        total_loss = 0
        print(f'Epoch: {epoch+1}')
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            # ---
            item_indices = batch['item_idx'].to(config['device'])
            token_indices = batch['token_indices'].to(config['device'])
            mask = batch['mask'].to(config['device'])
            # ---
            loss = model(item_indices, token_indices, mask)
            loss.backward()
            optimizer.step()
            # ---
            total_loss += loss.item()
            if global_step%config['log_step'] == 0:
                logging.info(f"epoch {epoch+1}, " +\
                             f"global step {global_step}, " +\
                             f"loss = {loss:.5f}")
            global_step += 1
            wb.log(data={'train_loss': loss}, 
                      step=global_step)
        
        # torch.cuda.empty_cache()
        logging.info(f"--- END - epoch {epoch+1}/{config['epochs']}, " +\
                     f"avg. loss = {total_loss/len(dataloader):.5f}")
        # Save embeddings with original mappings
        if (epoch+1)%config['save_embs_epoch'] == 0:
            save_embeddings(model, dataset, rundir, epoch+1)
    wandb.finish()


if __name__ == "__main__":
    logdir = 'train_logs'
    model_name = 'itemLM'
    rundir = f"{logdir}/{model_name}_" +\
             f"{datetime.now().strftime(format='%Y-%m-%d_%H-%M-%S')}"
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f'{rundir}/logs'
    )
    train_item_language_model(rundir)


"""
print(batch)
print(batch['item_idx'].shape)
print(batch['token_indices'].shape)
print(batch['mask'].shape)
print('-----------------------')
print(batch['token_indices'][0][:10])
print(batch['token_indices'][1][:10])
print(batch['token_indices'][2][:10])
print(batch['token_indices'][3][:10])
"""

