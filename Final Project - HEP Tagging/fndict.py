''' Preliminary Stuff'''
import math
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Create a torch.device object to tell pytorch where to store your tensors: cpu or gpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
------ MASK FUNCTIONS ------

Masks are useful for RNN or Transformer architecture that work with sequential data of a fixed length.

The mask function plays an essential role in the training of a transformer model, specifically during 
the pre-training phase when the model learns to understand and generate language. The two main 
purposes of the mask function are:

    1. To facilitate self-attention mechanism: 
       Transformers use self-attention mechanisms to identify 
       relationships between words in a sequence. Masking is used to prevent the model from "cheating" 
       by looking at future tokens when trying to predict the current token. In other words, the mask 
       function ensures that the model only attends to the current token and the previous tokens, not 
       the future tokens, during the training process.

    2. To enable masked language modeling (MLM): 
       Masked language modeling is a popular pre-training objective used in transformer-based models 
       like BERT. In MLM, a certain percentage of input tokens are randomly masked (usually around 15%), 
       and the model is tasked with predicting the original tokens at these masked positions. The mask 
       function serves as a way of hiding the original token from the model, forcing it to learn 
       contextual representations that can help it predict the masked tokens accurately.


The use of the mask function in both self-attention and MLM helps the transformer model learn meaningful 
context-dependent representations, making it more effective at understanding and generating natural language.
'''
class masker():
    # Helper
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # Define your masking function
    def create_mask(src, tgt, pad_index):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = masker.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

        src_padding_mask = (src == pad_index).transpose(0, 1)
        tgt_padding_mask = (tgt == pad_index).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


    def pad_and_create_mask(batch):
        max_len = max(sequence.shape[0] for sequence in batch)
        padded_batch = np.zeros((len(batch), max_len, batch[0].shape[1]))
        mask = np.zeros((len(batch), max_len))

        # Pad each sequence and create its corresponding mask
        for i, sequence in enumerate(batch):
            padded_batch[i, :sequence.shape[0], :] = sequence
            mask[i, :sequence.shape[0]] = 1  # Elements where the mask is 1 represent actual values, 0s represent padding

        # Invert the mask
        mask = mask == 0

        return padded_batch, mask



'''
------ COLLATION FUNCTIONS ------

The collation function is what converts our strings into batches of tensors that can be processed 
by our model, based on the vocabularies and tokenization functions we have built up thus far.

Again, this is something we can do manually, but at some point the data transformations get so 
complicated that we might as well put them all into a function. Moreover, defining our 
transformation as a *function* allows us to use some more built-in PyTorch functionality that makes 
our jobs a whole lot easier. 

See: torch.utils.data.DataLoader.
'''
class collation:
    # Define helper function to club together sequential operations
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # Define function to add BOS/EOS and create a tensor for input sequence indices
    def tensor_transform(token_ids, bos_idx, eos_idx):
        return torch.cat((torch.tensor([bos_idx]),
                        torch.tensor(token_ids),
                        torch.tensor([eos_idx])))
    

    # Define your "collation" function to collate data samples into batch tensors
    def collate_fn(batch):
        # Separate the features and labels from the batch
        features = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Pad the features to the maximum length in the batch
        features = pad_sequence([torch.tensor(event) for event in features], 
                                       batch_first=True, padding_value=0)

        # Convert the labels to tensor
        labels = torch.stack(labels)

        return features, labels
    


'''
------ TRAINING FUNCTIONS ------
'''
class trainer:
    # Define a function to train the model for a single epoch
    def train_epoch(model, optimizer, batch_size, trainers):
        model.train()
        loss_list = []
        train_dataloader = DataLoader(trainers, batch_size=batch_size, collate_fn=collation.collate_fn)

        for src, tgt in train_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            # Since we're training, recall we need to mask our input
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = masker.create_mask(src, tgt_input)

            # What do you think the model does with the masks when it's in evaluation mode?
            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = math.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            loss_list.append(loss.item())

        return loss_list

    # Define a function to evaluate the model
    def evaluate(model, batch_size, valids, loss_fn):
        model.eval()
        loss_list = []

        val_dataloader = DataLoader(valids, batch_size=batch_size, collate_fn=collation.collate_fn)

        for src, tgt in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            # # This is meant for sequence to sequence tasks:
            # tgt_input = tgt[:-1, :]

            # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = masker.create_mask(src, tgt_input)

            # logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            # tgt_out = tgt[1:, :]
            # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            # loss_list.append(loss.item())

            # This is meant for classification tasks:
            logits = model(src)
            loss = loss_fn(logits, tgt.long())
            loss_list.append(loss.item())

        return loss_list
    


'''
------ MISCELLANEOUS ------
'''

def highest_divisor(n):
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return i
    return 1