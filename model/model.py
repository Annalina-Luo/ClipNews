import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import spacy
import numpy as np
import random
import math
import time
from build_vocab import Vocabulary
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from build_vocab import Vocabulary
import sys
from torch.autograd import Variable
from cider.cider import Cider
import json
import clip
from transformers import RobertaTokenizer, RobertaModel
from transformers import logging
logging.set_verbosity_error()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
text_model = RobertaModel.from_pretrained('roberta-base')


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


class CLIP_encoder(nn.Module):
    def __init__(self, hid_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(512, hid_dim)

    def forward(self, imgs):
        out = clip_model.encode_image(imgs)  # [batch_size, 512]
        out = self.l1(out)

        return out


class Positional_Encoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(Positional_Encoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # print("position", position.shape)
        # position [max_len, dim]
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.pe = pe
        # self.register_buffer('pe', pe)

    def forward(self, x):
        # x [batch_size, len, dim]
        # pe [1, len, dim]
        # print("x", x.shape)
        # print("pe shape", self.pe.shape)
        # print(self.pe[:, :x.shape[1], :].shape)
        # print("result", (x + self.pe[:, :x.size(1), :]).shape)
        # pe [max_length, 1, text_embedding]
        return x + self.pe[:, :x.size(1), :]


class Encoder_text(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=303):
        super().__init__()

        self.hid_dim = hid_dim
        # self.tok_embedding = text_model
        # self.text_embedding = 768
        self.pos_embedding = Positional_Encoding(
            hid_dim, max_length)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        # self.l1 = nn.Linear(self.text_embedding, hid_dim)

        self.scale = torch.sqrt(torch.FloatTensor(
            [hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len, embedding_dim]

        # pos = torch.arange(0, src_len).unsqueeze(
        #     0).repeat(batch_size, 1).to(device)
        # print("pos", (pos.shape))
        # print("after scale", (src * self.scale).shape)

        # pos [batch_Size, src_len], [64,84]
        # (src * self.scale) [64, 84, 768]
        # src = torch.LongTensor(src)
        # src = self.l1(src)
        # print("src_before", src.shape) src_before torch.Size([64, 84, 512])

        src = self.dropout(self.pos_embedding(src * self.scale))
        # print("src_after", src.shape)
        # src [batch_size, len, text_embedding]

        for layer in self.layers:
            src = layer(src, src_mask)

        # print("src", src.shape)

        # src = [batch size, src len, hid dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attn_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm2 = nn.LayerNorm(hid_dim)
        # self.ff_layer_norm3 = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        # self.multimodel_attention = MultiHeadAttentionLayer(
        #     hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)

        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim] hid dim=768

        # document inside self attention
        # print('src:', src.size())
        _src, _ = self.self_attention(src, src, src, src_mask)
        # print('src:', src.size())

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # imgs1 = torch.mean(imgs, dim=1).unsqueeze(1)
        # src1 = torch.mean(src, dim=1).unsqueeze(1)
        # # imgs1 = self.attention1(imgs).unsqueeze(1)
        # # src1 = self.attention2(src).unsqueeze(1)
        # src1_ = torch.cat([imgs1, src1], dim=1)

        # multimodal attention
        # src2 = self.multimodel_attention(src1, src1_, src1_)
        # src2_ = self.tanh(
        #     self.l2(src.expand(src.size(0), src.size(1), src.size(2))))
        # src2_ = self.self_attn_layer_norm1(src0 + self.dropout(src2_))

        # src3 = torch.mul(src, src2_)
        _src = self.positionwise_feedforward(src)
        src4 = self.ff_layer_norm1(src + self.dropout(_src))

        # print("src4", src4.shape)
        # src4 = [batch size, query len, dim] # torch.Size([64, 203, 768])

        return src4


# class Mutihead_Attention(nn.Module):
#     def __init__(self, d_model, dim_k, dim_v, n_heads):
#         super(Mutihead_Attention, self).__init__()
#         self.dim_v = dim_v
#         self.dim_k = dim_k
#         self.n_heads = n_heads

#         self.q = nn.Linear(d_model, dim_k)
#         self.k = nn.Linear(d_model, dim_k)
#         self.v = nn.Linear(d_model, dim_v)

#         self.o = nn.Linear(dim_v, d_model)
#         self.norm_fact = 1 / math.sqrt(d_model)
#         self.feed_forward = Feed_Forward(d_model)
#         self.add_norm = Add_Norm()

#     def generate_mask(self, dim):
#         # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
#         # padding mask 在数据输入模型之前完成。
#         matirx = np.ones((dim, dim))
#         mask = torch.Tensor(np.tril(matirx))

#         return mask == 1

#     def forward(self, x, y, requires_mask=False):
#         assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
#         # size of x : [batch_size * seq_len * batch_size]
#         # 对 x 进行自注意力
#         # n_heads * batch_size * seq_len * dim_k
#         Q = self.q(x).reshape(-1, x.shape[0],
#                               x.shape[1], self.dim_k // self.n_heads)
#         # n_heads * batch_size * seq_len * dim_k
#         K = self.k(x).reshape(-1, x.shape[0],
#                               x.shape[1], self.dim_k // self.n_heads)
#         # n_heads * batch_size * seq_len * dim_v
#         V = self.v(y).reshape(-1, y.shape[0],
#                               y.shape[1], self.dim_v // self.n_heads)
#         # print("Attention V shape : {}".format(V.shape))
#         attention_score = torch.matmul(
#             Q, K.permute(0, 1, 3, 2)) * self.norm_fact
#         if requires_mask:
#             mask = self.generate_mask(x.shape[1])
#             # 注意这里的小Trick，不需要将Q,K,V 分别MASK,只MASKSoftmax之前的结果就好了
#             attention_score.masked_fill(mask, value=float("-inf"))
#         output = torch.matmul(attention_score, V).reshape(
#             y.shape[0], y.shape[1], -1)
#         # print("Attention output shape : {}".format(output.shape))

#         output = self.o(output)
#         output = self.add_norm(output)

#         output = self.add_norm(x, self.muti_atten, y=x)
#         output = self.add_norm(output, self.feed_forward)
#         return output


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, q_dim, k_dim, v_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, q_dim)
        self.fc_k = nn.Linear(hid_dim, k_dim)
        self.fc_v = nn.Linear(hid_dim, v_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        # self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.scale = np.sqrt(self.head_dim)
        self.l1 = nn.Linear(hid_dim * 2, hid_dim)
        self.l2 = nn.Linear(hid_dim * 2, hid_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.p = nn.Parameter(torch.FloatTensor(1, self.hid_dim))
        # self.init_weights()

    # def init_weights(self):
        # nn.init.normal_(self.p, 0, 1 / self.hid_dim)

    def generate_mask(self, dim):
        # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
        # padding mask 在数据输入模型之前完成。
        matirx = np.ones((dim, dim))
        mask = torch.Tensor(np.tril(matirx))

        return mask == 1

    def forward(self, query, key, value, mask=None):
        # print('query:', query.size())
        # print('key:', key.size())
        # print('value:', value.size())
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        # print("Q", Q.shape)
        K = self.fc_k(key)
        # print("K", Q.shape)
        V = self.fc_v(value)
        # print("V", Q.shape)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # print("energy", energy.shape)

        # energy torch.Size([64, 8, 56, 56])
        # energy = [batch size, n heads, query len, key len]

        # if mask is not None:
        #     energy = energy.masked_fill(mask == 0, -1e10)

        # if requires_mask:
        #     mask = self.generate_mask(query.shape[1])
        #     # print("mask", mask.shape)
        #     # 注意这里的小Trick，不需要将Q,K,V 分别MASK,只MASKSoftmax之前的结果就好了
        #     energy.masked_fill(mask, value=float("-inf"))

        if mask is not None:
            # print(mask.shape)
            energy = energy.masked_fill(mask == 0, -1e10)
        # output = torch.matmul(energy, V).reshape(y.shape[0], y.shape[1], -1)

        # print("energy", energy.shape)

        attention = torch.softmax(energy, dim=-1)
        # print("attention", attention.shape)

        # attention torch.Size([64, 8, 56, 56])
        # attention = [batch size, n heads, query len, key len]

        # x = torch.matmul(self.dropout(attention), V)
        # print(attention.shape)
        # print(V.shape)
        x = torch.matmul(attention, V)
        # x = torch.matmul(attention, V).reshape(
        #     value.shape[0], query.shape[1], -1)
        # output = torch.matmul(attention_score, V).reshape(
        #     y.shape[0], y.shape[1], -1)
        # print("x", x.shape)
        # x torch.Size([64, 84, 512])
        # x = [batch size, n heads, query len, head dim]

        # x = self.fc_o(x)
        # print("x", x.shape)
        # x = [batch size, query len, dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # print("x", x.shape)
        # x torch.Size([64, 84, 8, 64])
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # print("x", x.shape)
        # x torch.Size([64, 56, 768])
        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        # print("x", x.shape)
        # x torch.Size([64, 56, 768])
        # query torch.Size([64, 84, 768])

        # x = [batch size, query len, hid dim]

        # print("Attention output shape : {}".format(output.shape))

        x = torch.cat([x, query], dim=2)
        # print('x:', x.size())
        # x = self.fc_o(x)
        # print("x", x.shape)
        x1 = self.sigmoid(self.l1(x))
        # print("x1", x1.shape)
        x2 = self.l2(x)
        # print("x2", x2.shape)

        x = torch.mul(x1, x2)
        # print("x", x.shape) # torch.Size([64, 203, 768])

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = torch.relu(self.fc_2(x))

        # x = [batch size, seq len, hid dim]

        return x


class Decoder(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=130):
        super().__init__()

        # self.device = device

        # self.tok_embedding = text_model
        self.pos_embedding = Positional_Encoding(hid_dim, max_length)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        # output_dim = 512
        self.embedding_weight = text_model.get_input_embeddings().weight

        self.fc_out = nn.Linear(hid_dim, self.embedding_weight.shape[0])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        # self.scale = np.sqrt(hid_dim)
        self.l1 = nn.Linear(hid_dim*3, 1)
        self.l2 = nn.Linear(hid_dim * 3, 1)

    def forward(self, trg, trg_mask, enc_image, enc_src, src, src_mask, src_ids):
        index1 = src_ids
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        # index2 = reference

        # print('imgs:', imgs.size())
        # pos = torch.arange(0, trg_len).unsqueeze(
        #     0).repeat(batch_size, 1).to(device)
        trg = self.dropout(self.pos_embedding(trg * self.scale))
        # print("trg", trg.shape)
        # pos = [batch size, trg len]

        # trg = self.dropout(
        #     (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        trg1 = trg

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, trg_src, trg_image, attention_src = layer(
                trg, enc_image, enc_src, src_mask, trg_mask)
        # print(attention_src.shape)  # [batch size, n heads, trg len, src len]

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        # print(output.shape)

        # output = [batch size, trg len, output dim]

        attention_src = torch.mean(attention_src, dim=1)
        # torch.Size([batch_size, trg_len, src_len])
        # print(attention_src.shape)
        # print("index1", index1.shape)  # [batch_size, src_len, hid_dim]

        index1 = index1.expand(attention_src.size(1), index1.size(
            0), index1.size(1)).permute(1, 0, 2)
        index1 = torch.tensor(index1, dtype=torch.int64)
        # print("index1", index1.shape)  # [batch_size,src_len,src_len]

        attn_value = torch.zeros(
            [output.size(0), output.size(1), output.size(2)]).to(device)
        # self.embedding_weight = self.embedding_weight.unsqueeze(1)
        # print(self.embedding_weight.shape)

        attn_value = attn_value.scatter_add_(2, index1, attention_src)
        # print(attn_value.shape)
        p = torch.sigmoid(
            self.l1(torch.cat([trg1, trg_src, trg_image], dim=2)))

        output = (1 - p) * output + p * attn_value
        # print("output", output.shape) #[64, 46, 50265]

        return output


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        # self.enc_attn_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        self.encoder_attention1 = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        self.encoder_attention2 = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        # self.v = nn.Linear(512, 512)
        # self.v1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()

    def forward(self, trg, enc_image, enc_src, src_mask, trg_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # self attention
        # print('imgs:', imgs.size())
        # imgs = self.relu(self.v(imgs))
        # print('imgs:', imgs.size())
        # print("start first self attention in decoder")
        _trg, _ = self.self_attention(trg, trg, trg, mask=trg_mask)
        # print("_trg", _trg.shape)
        # print("trg", trg.shape)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # print("finish first self attention in decoder")
        # trg = [batch size, trg len, hid dim]

        # print("start second self attention in decoder with src")
        # encoder attention
        _trg0, attention_src = self.encoder_attention(
            trg, enc_src, enc_src, src_mask)
        # print("trg", _trg0.shape)

        # print("start second self attention in decoder with image")
        _trg1, _ = self.encoder_attention1(
            trg, enc_image, enc_image)
        # print("trg", _trg1.shape)
        # trg torch.Size([64, 46, 512])

        # dropout, residual connection and layer norm
        trg1_ = _trg0
        trg2_ = _trg1
        # trg3_ = _trg2
        trg = self.enc_attn_layer_norm(
            trg + self.dropout(_trg0) + self.dropout(_trg1))
        # print("trg", trg.shape)
        # trg torch.Size([64, 46, 512])
        # trg1 = self.enc_attn_layer_norm1(trg + self.dropout(_trg1))
        # print("finish second self attention in decoder")
        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # print("_trg", trg.shape)
        # trg torch.Size([64, 46, 512])

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # print("trg", trg.shape)
        # _trg torch.Size([64, 46, 512])

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, trg1_, trg2_, attention_src


class NewsTransformer(nn.Module):
    def __init__(self,
                 encoder_text,
                 encoder_image,
                 decoder,
                 hidden_dim,
                 src_pad_idx,
                 trg_pad_idx):
        super().__init__()

        self.encoder_text = encoder_text,  # GPT2_token
        self.encoder_image = encoder_image,  # CLIP
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.l1 = nn.Linear(2048, hidden_dim)
        # self.l2 = nn.Linear(768, 512)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def make_src_mask(self, src):
        # src = [batch size, src len]
        # print('self.src_pad_idx:', self.src_pad_idx)
        src = torch.sum(src, dim=2)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # print(src_mask.shape)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        # print('self.trg_pad_idx:', self.trg_pad_idx)
        # print("trg", trg.shape)
        # trg = torch.sum(trg, dim=2)
        # print("trg", trg.shape)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # print("trg_pad_mask", trg_pad_mask.shape)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=device)).bool()
        # print("trg_sub_mask", trg_sub_mask.shape)

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, arts_ids, arts_mask, arts_emb, caps_mask, caps_emb, imgs):
        # src: [batch_size, 302, 768]
        # trg: [batch_size, len, 768]
        # imgs: [batch size, 3,224,224]

        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(caps_embedding)

        # src = self.l2(src)
        # trg = self.l2(trg)
        # print("src", src.shape)
        # print("trg", trg.shape)

        enc_image = self.encoder_image[0](imgs)
        # enc_image [batch_size, 768]

        arts_mask = arts_mask.unsqueeze(1).unsqueeze(2)
        # caps_mask = torch.sum(caps_mask, dim=2)
        caps_pad_mask = (caps_mask != 0).unsqueeze(1).unsqueeze(2)
        caps_mask_len = caps_mask.shape[-1]
        caps_sub_mask = torch.tril(torch.ones(
            (caps_mask_len, caps_mask_len), device=device)).bool()
        caps_mask = caps_pad_mask & caps_sub_mask

        # print("arts_mask", arts_mask.shape)
        # print("caps_mask", caps_mask.shape)

        enc_src = self.encoder_text[0](arts_emb, arts_mask)  # encode article

        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)

        # print("finish encoding part")

        # enc_image torch.Size([64, 512])
        # enc_src torch.Size([64, 84, 512])

        output = self.decoder(
            caps_emb, caps_mask, enc_image, enc_src, arts_emb, arts_mask, arts_ids)

        return output


def translate_sentence(model, src, src_mask, src_emb, caplens, imgs, device):
    model.eval()
    # print("enc_image")
    enc_image = model.encoder_image[0](imgs)

    # imgs = model.relu(model.l1(imgs))
    # src_mask = model.make_src_mask(src)
    # ref_mask = model.make_src_mask(enc_ref)
    # reference = enc_ref
    src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    # caps_mask = torch.sum(caps_mask, dim=2)

    # caps_pad_mask = (caps_mask != 0).unsqueeze(1).unsqueeze(2)
    # caps_mask_len = caps_mask.shape[-1]
    # caps_sub_mask = torch.tril(torch.ones(
    #     (caps_mask_len, caps_mask_len), device=device)).bool()
    # caps_mask = caps_pad_mask & caps_sub_mask
    # print("enc_src")
    with torch.no_grad():
        enc_src = model.encoder_text[0](src_emb, src_mask)

    max_length = max(caplens)
    outputs = text_model.get_input_embeddings().weight[0]
    outputs = outputs.unsqueeze(0).unsqueeze(1).to(device)
    # outputs = outputs.expand([len(caplens), -1, -1])
    # results = torch.zeros([len(caplens), 1])
    # mask = torch.ones([len(caplens), 1])
    results = torch.zeros([1])
    mask = torch.ones([1, 1])
    # print(mask.shape)
    # print("output", outputs.shape)
    print("decoder")
    for i in range(max_length):
        # trg_tensor = torch.LongTensor(
        #     outputs).unsqueeze(0).unsqueeze(1).to(device)
        # trg_tensor = trg_tensor.expand([len(caplens), 1])
        print("trg_tensor", outputs.shape)
        trg_pad_mask = (mask != 0).unsqueeze(
            1).unsqueeze(2)
        trg_len = mask.shape[1]
        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = model.make_trg_mask(trg_tensor)
        print("trg_mask", trg_mask.shape)

        with torch.no_grad():
            output = model.decoder(
                outputs, trg_mask, enc_image, enc_src, src_emb, src_mask, src)
            # output = model(sentence_tensor, trg_tensor)
        print(output.shape)

        best_guess = output.argmax(2)[:, -1]
        print("best_guess", best_guess.shape)
        results = torch.cat((results, best_guess), dim=0)
        print(i, best_guess)
        outputs_1 = text_model.get_input_embeddings(
        ).weight[best_guess.unsqueeze(0)]
        print(best_guess.unsqueeze(0).shape)
        outputs = torch.cat((outputs, outputs_1), dim=1)
        # outputs = text_model.get_input_embeddings().weight[0]
        # outputs = outputs.unsqueeze(0).unsqueeze(1)
        # outputs = outputs.expand([len(caplens), -1, -1])
        # results = torch.zeros([len(caplens), 1])
        # outputs = torch.cat((outputs, ), dim=1)
        # print("outputs", outputs.shape)

        if best_guess == 2:
            break

            # translated_sentence = [word_map.idx2word[idx] for idx in outputs]
    print("output", results)
    translated_sentence = text_tokenizer.decode(results)
    # remove start token
    return translated_sentence[:]


def bleu(model, src, src_mask, src_emb, caplens, imgs, device):

    prediction = translate_sentence(
        model, src, src_mask, src_emb, caplens, imgs, device)
    prediction = prediction  # remove <eos> token
    return prediction


def ciderScore(gts_file, res):
    gts = json.dump(open(gts_file, 'r'))
    gts_dic = {}
    res_dic = {}
    for i in gts:
        gts_dic[i["id"]] = i["caption"]
    for i in res:
        res_dic[res["image_id"]] = i["caption"]
    scorer = Cider()
    (score, scores) = scorer.compute_score(gts_dic, res_dic)
    return score
