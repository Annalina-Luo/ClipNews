import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
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
        out = torch.tensor(out, dtype=torch.float32)
        out = self.l1(out)

        return out


class Positional_Encoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(Positional_Encoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # position [max_len, dim]
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.pe = pe

    def forward(self, x):
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
        self.pos_embedding = Positional_Encoding(
            hid_dim, max_length)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor(
            [hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src [batch size, src len, embedding_dim]
        # pos [batch_Size, src_len], [64,84]

        src = self.dropout(self.pos_embedding(src * self.scale))
        # src [batch_size, len, text_embedding]

        for layer in self.layers:
            src = layer(src, src_mask)
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
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim] hid dim=768
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        _src = self.positionwise_feedforward(src)
        src4 = self.ff_layer_norm(src + self.dropout(_src))
        # src4  [batch size, query len, dim] # torch.Size([64, 203, 768])

        return src4


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

        self.scale = np.sqrt(self.head_dim)
        self.l1 = nn.Linear(hid_dim * 2, hid_dim)
        self.l2 = nn.Linear(hid_dim * 2, hid_dim)
        self.sigmoid = nn.Sigmoid()
        self.p = nn.Parameter(torch.FloatTensor(1, self.hid_dim))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

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
        # energy  [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        # attention [batch size, n heads, query len, key len]

        x = torch.matmul(attention, V)
        # x [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x [batch size, query len, hid dim]

        x = self.fc_o(x)
        # x [batch size, query len, hid dim]

        x = torch.cat([x, query], dim=2)
        x1 = self.sigmoid(self.l1(x))
        x2 = self.l2(x)
        x = torch.mul(x1, x2)

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

        self.pos_embedding = Positional_Encoding(hid_dim, max_length)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.embedding_weight = text_model.get_input_embeddings().weight
        self.fc_out = nn.Linear(hid_dim, self.embedding_weight.shape[0])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.l1 = nn.Linear(hid_dim*3, 1)
        self.l2 = nn.Linear(hid_dim * 3, 1)

    def forward(self, trg, trg_mask, enc_image, enc_src, src, src_mask, src_ids):
        index1 = src_ids
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        trg = self.dropout(self.pos_embedding(trg * self.scale))
        # pos = [batch size, trg len]
        trg1 = trg
        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, trg_src, trg_image, attention_src = layer(
                trg, enc_image, enc_src, src_mask, trg_mask)
        # [batch size, n heads, trg len, src len]

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]

        attention_src = torch.mean(attention_src, dim=1)
        # torch.Size([batch_size, trg_len, src_len])

        index1 = index1.expand(attention_src.size(1), index1.size(
            0), index1.size(1)).permute(1, 0, 2)
        index1 = torch.tensor(index1, dtype=torch.int64)
        # [batch_size,src_len,src_len]
        attn_value = torch.zeros(
            [output.size(0), output.size(1), output.size(2)]).to(device)
        attn_value = attn_value.scatter_add_(2, index1, attention_src)
        p = torch.sigmoid(
            self.l1(torch.cat([trg1, trg_src, trg_image], dim=2)))
        output = (1 - p) * output + p * attn_value

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
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, trg, enc_image, enc_src, src_mask, trg_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        _trg, _ = self.self_attention(trg, trg, trg, mask=trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg0, attention_src = self.encoder_attention(
            trg, enc_src, enc_src, src_mask)

        _trg1, _ = self.encoder_attention1(
            trg, enc_image, enc_image)
        # trg torch.Size([64, 46, 512])

        # dropout, residual connection and layer norm
        trg1_ = _trg0
        trg2_ = _trg1
        trg = self.enc_attn_layer_norm(
            trg + self.dropout(_trg0) + self.dropout(_trg1))
        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # trg torch.Size([64, 46, 512])

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
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

    def forward(self, arts_ids, arts_mask, arts_emb, caps_mask, caps_emb, imgs):
        # src: [batch_size, 302, 768]
        # trg: [batch_size, len, 768]
        # imgs: [batch size, 3,224,224]

        enc_image = self.encoder_image[0](imgs)
        # enc_image [batch_size, 768]

        arts_mask = arts_mask.unsqueeze(1).unsqueeze(2)
        caps_pad_mask = (caps_mask != 0).unsqueeze(1).unsqueeze(2)
        caps_mask_len = caps_mask.shape[-1]
        caps_sub_mask = torch.tril(torch.ones(
            (caps_mask_len, caps_mask_len), device=device)).bool()
        caps_mask = caps_pad_mask & caps_sub_mask

        enc_src = self.encoder_text[0](arts_emb, arts_mask)  # encode article

        # enc_image torch.Size([64, 512])
        # enc_src torch.Size([64, 84, 512])

        output = self.decoder(
            caps_emb, caps_mask, enc_image, enc_src, arts_emb, arts_mask, arts_ids)

        return output


def translate_sentence(model, src, src_mask, src_emb, caplens, imgs, device):
    model.eval()
    enc_image = model.encoder_image[0](imgs)

    src_mask = src_mask.unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        enc_src = model.encoder_text[0](src_emb, src_mask)

    max_length = max(caplens)
    outputs = text_model.get_input_embeddings().weight[0]
    outputs = outputs.unsqueeze(0).unsqueeze(1).to(device)
    results = torch.zeros([1])
    mask = torch.ones([1, 1])
    for i in range(max_length):
        trg_pad_mask = (mask != 0).unsqueeze(
            1).unsqueeze(2)
        trg_len = mask.shape[1]
        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask

        with torch.no_grad():
            output = model.decoder(
                outputs, trg_mask, enc_image, enc_src, src_emb, src_mask, src)
        print(output.shape)

        best_guess = output.argmax(2)[:, -1]
        results = torch.cat((results, best_guess), dim=0)
        print(i, best_guess)
        outputs_1 = text_model.get_input_embeddings(
        ).weight[best_guess.unsqueeze(0)]
        print(best_guess.unsqueeze(0).shape)
        outputs = torch.cat((outputs, outputs_1), dim=1)

        if best_guess == 2:
            break

    translated_sentence = text_tokenizer.decode(results)
    return translated_sentence[:]


def bleu(model, src, src_mask, src_emb, caplens, imgs, device):

    prediction = translate_sentence(
        model, src, src_mask, src_emb, caplens, imgs, device)
    prediction = prediction
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
