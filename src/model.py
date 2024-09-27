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
from layers import LayerNorm, Linear_piece, Linear_block
from protein_embedding import ProteinEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Linear_esm(nn.Module):
    def __init__(
        self, model_location, output_dim, num_blocks, dropout_rate=0
    ):
        super().__init__()
        self.model_location = model_location



        if self.model_location == "esm2_t48_15B_UR50D":
            self.embed_dim = 5120
            self.num_layers = 47
        elif self.model_location == "esm2_t36_3B_UR50D":
            self.embed_dim = 2560
            self.num_layers = 35
        elif self.model_location == "esm2_t33_650M_UR50D":
            self.embed_dim = 1280
            self.num_layers = 32
        elif self.model_location == "esm2_t30_150M_UR50D":
            self.embed_dim = 640
            self.num_layers = 29
        elif self.model_location == "esm2_t12_35M_UR50D":
            self.embed_dim = 480
            self.num_layers = 11
        elif self.model_location == "esm2_t6_8M_UR50D":
            self.embed_dim = 320
            self.num_layers =5
        else:
            raise ValueError("Provide an accurate esm_embedder name")

        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.layer_norm = nn.LayerNorm(self.embed_dim).to(
            device
        )  # Ensure layer_norm is on the correct device
        self.class_fc = nn.Linear(self.embed_dim, self.output_dim).to(
            device
        )  # Ensure fc layer is on the correct device
        self.dropout = nn.Dropout(dropout_rate).to(device)

    def forward(self, fasta_file):
        # Load and process fasta file (this might already be on the correct device)
        prot_embed = (
            ProteinEmbedding(self.model_location, fasta_file, self.num_layers)
            .forward()
            .to(device)
        )

        length = prot_embed.shape[0]

        # Create blocks on the correct device
        fc_blocks = nn.ModuleList(
            [
                Linear_block(self.model_location, self.embed_dim, length).to(device)
                for _ in range(self.num_blocks)
            ]
        )

        x = prot_embed.to(device)  # Ensure the embedding is on the correct device
        for fc_block in fc_blocks:
            x = fc_block(x)
            x = self.dropout(x)
        x = self.layer_norm(x)
        x = x.mean(dim=0)
        x = self.class_fc(x)
        x = F.softmax(x, dim=-1)
        return x
