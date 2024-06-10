## import moules
import csv
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GATConv, GCNConv, SAGPooling
from torch_geometric.nn import global_max_pool , global_mean_pool,global_add_pool
from transformer import *

device = torch.device('cuda:0')

class MDSyn(nn.Module):
    def __init__(
        self,
        molecule_channels: int = 78,
        hidden_channels: int = 128,
        middle_channels: int = 64,
        layer_count: int = 2,
        out_channels: int = 2,
        dropout_rate: int = 0.3
    ):

        super().__init__()
        self.dropout_rate = dropout_rate

        # activation function
        self.relu = nn.ReLU()

        # pooling method
        self.attn = TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=64,
                                            dropout=0.2, activation='relu')

        self.trans = TransformerEncoder(encoder_layer=self.attn, num_layers=2)

        # GCN
        self.gcn_conv1 = GCNConv(molecule_channels, 512)
        self.gcn_conv2 = GCNConv(512, hidden_channels)
        self.drug1_fc_g1 = torch.nn.Linear(hidden_channels , hidden_channels)

        # classification
        self.fc1 = nn.Sequential(
            nn.Linear(2048-128, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 2)
        )

        # cell line
        self.fc2 = nn.Sequential(
            nn.Linear(954, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, hidden_channels * 2)
        )


    def forward(self, molecules_left, molecules_right, lincs):
        x1, edge_index1, batch1, mask1, smile_embedding1, cell = (molecules_left.x, molecules_left.edge_index, molecules_left.batch, molecules_left.mask,
                                                                  molecules_left.smiles_embedding, molecules_left.ccle_embedding)
        x2, edge_index2, batch2, mask2, smile_embedding2 = (molecules_right.x, molecules_right.edge_index, molecules_right.
                                                            batch, molecules_right.mask, molecules_right.smiles_embedding)

        batch_size = smile_embedding1.size(0)

        mask1 = mask1.reshape((batch_size, 100)) # size:[128, 100]
        mask2 = mask2.reshape((batch_size, 100))

        # 2D gcn smiles embedding
        drug1 = self.gcn_conv1(x1, edge_index1)
        drug1 = self.relu(drug1)
        drug1 = self.gcn_conv2(drug1, edge_index1)
        drug1 = self.relu(drug1)

        drug2 = self.gcn_conv1(x2, edge_index2)
        drug2 = self.relu(drug2)
        drug2 = self.gcn_conv2(drug2, edge_index2)
        drug2 = self.relu(drug2)

        # 2-D Attention pooling
        drug1 = drug1.reshape(batch_size, 100, 128)
        drug2 = drug2.reshape(batch_size, 100, 128)

        lincs = lincs.expand(batch_size, 978, 128) #shape:[128, 978, 128]
        fine_input = torch.cat([drug1, drug2, lincs], dim=1)

        lincs_mask = torch.zeros(batch_size, 978)  ### lincs mask

        lincs_mask = lincs_mask.to(device)
        mask_input = torch.cat([mask1, mask2, lincs_mask], dim=-1)  # [batch_size, 1178]
        attn_mask = mask_input
        # print("attn_mask.shape: ", attn_mask.shape)

        fine_enc = fine_input.permute(1, 0, 2)

        fine_out, weight, weight1 = self.trans(fine_enc, src_key_padding_mask=attn_mask.bool())
        fine_out = fine_out.permute(1, 0, 2)
        fine_out = torch.mean(fine_out, dim=1)
        fine_enc1 = self.drug1_fc_g1(fine_out)
        fine_enc1 = self.relu(fine_out)

        # 1-D cell line
        cell = F.normalize(cell, 2, 1)
        ccle_redu = self.fc2(cell)

        combine_feature = torch.cat([smile_embedding1, smile_embedding2, ccle_redu, fine_enc1], dim=1)
        combine_feature = F.normalize(combine_feature, 2, 1)
        out = self.fc1(combine_feature)

        return out, weight

