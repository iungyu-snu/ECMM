import torch
import esm
from esm import (
    Alphabet,
    FastaBatchedDataset,
    ProteinBertModel,
    pretrained,
    MSATransformer,
)
import torch.nn as nn
import torch.nn.functional as F
from transformer import LayerNorm, FeedForwardLayer, Attention, TransBlock
from protein_embedding import ProteinEmbedding
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ECT(nn.Module):
    def __init__(
        self, model_location, output_dim, num_blocks, dropout_rate=0
    ):
        super(ECT,self).__init__()

        self.model_location = model_location
        if self.model_location == "esm2_t48_15B_UR50D":
            self.embed_dim = 5120
            self.num_layers = 47
        elif self.model_location == "esm2_t36_3B_UR50D":
            self.embed_dim = 2560
            self.num_layers = 35
        elif self.model_location == "esm2_t33_650M_UR50D":
            self.embed_dim = 1280
            self.num_layers =32
        elif self.model_location == "esm2_t30_150M_UR50D":
            self.embed_dim = 640
            self.num_layers = 29
        elif self.model_location == "esm2_t12_35M_UR50D":
            self.embed_dim = 480
            self.num_layers = 11
        elif self.model_location == "esm2_t6_8M_UR50D":
            self.embed_dim = 320
            self.num_layers = 5
        else:
            raise ValueError("Provide an accurate esm_embedder name")

        self.num_blocks = num_blocks
        self.output_dim = output_dim

        # ========
        # layers
        self.layer_norm = nn.LayerNorm(self.embed_dim).to(
            device
        )  
        self.class_fc = nn.Linear(self.embed_dim, self.output_dim).to(
            device
        ) 
        self.dropout = nn.Dropout(dropout_rate).to(device)
        # ========
        # blocks
        self.transformer_blocks = nn.ModuleList(
            [TransBlock(self.model_location, self.output_dim).to(device) for _ in range(self.num_blocks)]
        )

    def forward(self, fasta_files):
        """
        Args:
            fasta_files (list of str): Batch of fasta file sequences
        Returns:
            torch.Tensor: [batch_size, output_dim]
        """
        batch_size = len(fasta_files)
        prot_embeds = []
        # Obtain embeddings for the batch
        for fasta_file in fasta_files:
            # prot_embed only can handle one file
            # This is how to use ProteinEmbedding
            prot_embed = ProteinEmbedding(self.model_location, fasta_file, self.num_layers).forward()
            prot_embed = prot_embed.to(device)  
            prot_embeds.append(prot_embed)
        
        x = self.pad_and_stack(prot_embeds)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1) 
        x = self.class_fc(x)  # x shape: [batch_size, output_dim]
        x = F.softmax(x, dim=-1)
        return x

    def pad_and_stack(self, prot_embeds):
        max_seq_length = max([embed.shape[0] for embed in prot_embeds])
        batch_embeds = torch.zeros((len(prot_embeds), max_seq_length, self.embed_dim)).to(device)

        for i, embed in enumerate(prot_embeds):
            seq_length = embed.shape[0]
            if seq_length < max_seq_length:
                pad_length = max_seq_length - seq_length
                padding = torch.zeros((pad_length, self.embed_dim)).to(device)
                embed = torch.cat((embed, padding), dim=0)
            batch_embeds[i] = embed
        
        return batch_embeds
        
