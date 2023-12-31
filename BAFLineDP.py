import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class BAFLineDP(nn.Module):
    def __init__(self, embed_dim, gru_hidden_dim, gru_num_layers, bafn_output_dim, dropout, device):
        super(BAFLineDP, self).__init__()

        # define Bi-GRU module
        self.gru = nn.GRU(
            embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True, dropout=dropout
        )

        # define BAFN module
        self.bafn = weight_norm(
            BAFN(l_dim=embed_dim, c_dim=2 * gru_hidden_dim, h_dim=bafn_output_dim, h_out=2),
            name='q_mat', dim=None
        )

        # define prediction layer
        self.fc = nn.Linear(bafn_output_dim, 1)

        self.device = device
        # self.sig = nn.Sigmoid()

    def forward(self, code_tensor):
        sent_lengths = [len(code) for code in code_tensor]
        sent_lengths = torch.tensor(sent_lengths).to(self.device)

        code_tensor = pad_sequence(code_tensor, batch_first=True)

        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        code_tensor = code_tensor[sent_perm_idx]

        packed_sents = pack_padded_sequence(code_tensor, lengths=sent_lengths.tolist(), batch_first=True)

        # extract line-level context
        line_level_contexts, _ = self.gru(packed_sents)
        line_level_contexts, _ = pad_packed_sequence(line_level_contexts, batch_first=True)

        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        code_tensor = code_tensor[sent_unperm_idx]
        line_level_contexts = line_level_contexts[sent_unperm_idx]

        # construct features that incorporate source code line semantics, line-level contexts and local interation information
        file_level_embeds, sent_att = self.bafn(code_tensor, line_level_contexts)

        # obtain line-level attention for line-level defect prediction
        sent_att = sent_att.sum(dim=1)
        sent_att_weights = [item.diag() for item in sent_att]
        sent_att_weights = torch.stack(sent_att_weights, dim=0)
        sent_att_weights = sent_att_weights / torch.sum(sent_att_weights, dim=1, keepdim=True)

        # compute file-level defect prediction score
        scores = self.fc(file_level_embeds)

        return scores, sent_att_weights

class BAFN(nn.Module):
    def __init__(self, l_dim, c_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BAFN, self).__init__()

        self.k = k
        self.h_out = h_out

        # define two weight matrices U and M
        self.U = FCNet([l_dim, h_dim * self.k], act=act, dropout=dropout)
        self.M = FCNet([c_dim, h_dim * self.k], act=act, dropout=dropout)

        # define the weight vector q
        self.q_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
        self.q_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())

        if k > 1:
            self.pooling = nn.AvgPool1d(self.k, stride=self.k)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        # integrate code line semantics, global context information and local interaction information
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if self.k > 1:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.pooling(fusion_logits).squeeze(1) * self.k   # sum pooling

        return fusion_logits

    def forward(self, l, c, softmax=False, v_mask=True, mask_with=float(0)):
        # ----------Bilinear Interaction Module----------
        l_num = l.size(1)
        c_num = c.size(1)

        l_ = self.U(l)
        c_ = self.M(c)

        # compute bilinear interaction attention maps
        att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.q_mat, l_, c_)) + self.q_bias

        if v_mask:
            mask = (l.abs().sum(2).unsqueeze(1).unsqueeze(3).expand(att_maps.size()) == 0)
            att_maps.data.masked_fill_(mask.data, mask_with)

        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, l_num * c_num), 2)
            att_maps = p.view(-1, self.h_out, l_num, c_num)

        # ----------Bilinear Pooling Module----------
        logits = self.attention_pooling(l_, c_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(l_, c_, att_maps[:, i, :, :])
            logits += logits_i

        logits = self.bn(logits)

        return logits, att_maps

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0.):
        super(FCNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act != '':
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)