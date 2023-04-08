import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from torch.autograd import Variable
from cider.cider import Cider
import json
# import clip
from transformers import RobertaTokenizer, RobertaModel
from transformers import logging
logging.set_verbosity_error()

# device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the RobertaTokenizer and RobertaModel
text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
text_model = RobertaModel.from_pretrained('roberta-base')


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


class Positional_Encoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        # create a tensor of shape (max_len, d_model) filled with zeros
        pe = torch.zeros(max_len, d_model).to(device)
        # create a tensor of shape (max_len, 1) containing values from 0 to max_len-1
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        # position [max_len, dim]

        # create a tensor of shape (d_model/2,) containing the exponential terms used to calculate the positional encoding
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        # calculate the sine and cosine positional encoding values and assign them to the pe tensor
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.pe = pe

    def forward(self, x):
        # add the positional encoding to the input tensor x
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, q_dim, k_dim, v_dim, n_heads, dropout):
        """
        hid_dim: default 768
        q_dim: dim of query, default 768
        k_dim: dim of key, default 768
        v_dim: dim of value, default 768
        n_heads: number of heads in multi-head attention module
        dropout: default 0.1
        """
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

        # calculate the similarity of K and Q
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy  [batch size, n heads, query len, key len]

        if mask is not None:
            # add the mask
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        # attention [batch size, n heads, query len, key len]

        # weighted sum of value and the attention
        x = torch.matmul(attention, V)
        # x [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x [batch size, query len, hid dim]

        x = self.fc_o(x)
        # x [batch size, query len, hid dim]

        # concatenate the attention and input
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


class CLIP_encoder(nn.Module):
    def __init__(self,  hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout, clip_model,
                 is_attention) -> None:
        super().__init__()
        """
        hid_dim: default 768
        n_layers: number of encoder blocks
        n_heads: number of heads in multi-head attention module
        pf_dim: dim for positionwise feedforward layer
        dropout: default 0.1
        is_attention: whether add the AoA module
        """
        self.clip_model = clip_model
        self.layers = nn.ModuleList([ImageLayer(hid_dim,
                                                n_heads,
                                                pf_dim,
                                                dropout)
                                     for _ in range(n_layers)])
        self.is_attention = is_attention
        self.dropout = nn.Dropout(dropout)

        self.l1 = nn.Linear(512, hid_dim)

    def forward(self, imgs):
        # encode the image using the CLIP model
        # out = self.clip_model.encode_image(imgs).to(
        #     device, dtype=torch.float32)  # [batch_size, 512]
        out = self.clip_model.encode_image(imgs)
        out = torch.tensor(out, dtype=torch.float32)
        out = self.l1(out)

        if self.is_attention:
            out = out.unsqueeze(1)
            # apply multi-head self-attention on the encoded image
            for layer in self.layers:
                out = layer(out)

        return out


class ImageLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()
        """
        hid_dim: default 768
        n_heads: number of heads in multi-head attention module
        pf_dim: dim for positionwise feedforward layer
        dropout: default 0.1
        """
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attn_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm1 = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # multi-head attention-on-attention layer
        _src, _ = self.self_attention(src, src, src)
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # positionwise feedforward layer
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm1(src + self.dropout(_src))

        return src


class Encoder_text(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 is_attention,
                 max_length=303):
        super().__init__()
        """
        hid_dim: default 768
        n_layers: number of encoder blocks
        n_heads: number of heads in multi-head attention module
        pf_dim: dim for positionwise feedforward layer
        dropout: default 0.1
        is_attention: whether add the AoA module
        """
        self.hid_dim = hid_dim
        self.pos_embedding = Positional_Encoding(
            hid_dim, max_length)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.is_attention = is_attention
        self.scale = torch.sqrt(torch.FloatTensor(
            [hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src [batch size, src len, embedding_dim]
        # pos [batch_Size, src_len]

        # add position embedding
        src = self.dropout(self.pos_embedding(src * self.scale))
        # src [batch_size, len, text_embedding]

        # Multi-Head AoA layer
        if self.is_attention:
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
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src  [batch size, query len, dim]

        return src


class Decoder(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=160):
        super().__init__()
        """
        hid_dim: default 768
        n_layers: number of encoder blocks
        n_heads: number of heads in multi-head attention module
        pf_dim: dim for positionwise feedforward layer
        dropout: default 0.1
        """
        self.pos_embedding = Positional_Encoding(
            hid_dim, max_length)  # position embedding

        # Create n_layers number of decoder layers using DecoderLayer module
        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        # Get the embedding weights of the text model, and create an output linear layer
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

        # Apply the positional encoding and scaling factor to the target sequence
        trg = self.dropout(self.pos_embedding(trg * self.scale))
        # pos = [batch size, trg len]
        trg1 = trg
        # trg = [batch size, trg len, hid dim]

        # Pass the target sequence through each of the decoder layers
        for layer in self.layers:
            trg, trg_src, trg_image, attention_src = layer(
                trg, enc_image, enc_src, src_mask, trg_mask)
        # [batch size, n heads, trg len, src len]

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        # Compute the output probability distribution over the vocabulary using a linear layer
        output = self.fc_out(trg)
        # output = torch.softmax(output, dim=-1)
        # output = [batch size, trg len, output dim]

        # Compute the average of the attention over the source sequence
        attention_src = torch.mean(attention_src, dim=1)
        # torch.Size([batch_size, trg_len, src_len])

        # Create an index tensor for the attention value
        index1 = index1.expand(attention_src.size(1), index1.size(
            0), index1.size(1)).permute(1, 0, 2)
        index1 = torch.tensor(index1, dtype=torch.int64)
        # [batch_size,src_len,src_len]

        # Create an empty tensor for the attention values and add the attention weights
        attn_value = torch.zeros(
            [output.size(0), output.size(1), output.size(2)]).to(device)
        attn_value = attn_value.scatter_add_(2, index1, attention_src)

        # Compute the probability of selecting the attention or output distribution using a sigmoid
        p = torch.sigmoid(
            self.l1(torch.cat([trg1, trg_src, trg_image], dim=2)))
        output = (1 - p) * output + p * attn_value
        # output = torch.softmax(output, dim=-1)

        return output


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()

        # Layer normalization for self-attention and encoder attention
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        # Multi-head self-attention layer, encoder attention layer, and a second encoder attention layer
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        self.encoder_attention1 = MultiHeadAttentionLayer(
            hid_dim, hid_dim, hid_dim, hid_dim, n_heads, dropout)
        # Position-wise feedforward layer
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, trg, enc_image, enc_src, src_mask, trg_mask):
        """
        trg: the target input sequence, shape [batch size, trg len, hid dim]
        enc_image: the encoded image feature map, shape [batch size, img len, hid dim]
        enc_src: the encoded source sequence, shape [batch size, src len, hid dim]
        src_mask: a mask for the source sequence, shape [batch size, src len]
        trg_mask: a mask for the target sequence, shape [batch size, trg len]
        """

        # Multi-head self-attention on the target sequence
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
        # arts_ids: [batch_size, max_seq_len] - article ids
        # arts_mask: [batch_size, max_seq_len] - article mask
        # arts_emb: [batch_size, max_seq_len, 768] - article embeddings
        # caps_mask: [batch_size, max_cap_len] - caption mask
        # caps_emb: [batch_size, max_cap_len, 768] - caption embeddings
        # imgs: [batch size, 3,224,224] - image input

        # Encode the image
        enc_image = self.encoder_image[0](imgs)
        # enc_image [batch_size, 768]

        # Create mask for caption input
        arts_mask = arts_mask.unsqueeze(1).unsqueeze(2)
        caps_pad_mask = (caps_mask != 0).unsqueeze(1).unsqueeze(2)
        caps_mask_len = caps_mask.shape[-1]
        caps_sub_mask = torch.tril(torch.ones(
            (caps_mask_len, caps_mask_len), device=device)).bool()
        caps_mask = caps_pad_mask & caps_sub_mask

        # Encode the article
        enc_src = self.encoder_text[0](arts_emb, arts_mask)  # encode article

        # Decoder
        output = self.decoder(
            caps_emb, caps_mask, enc_image, enc_src, arts_emb, arts_mask, arts_ids)

        return output


def translate_sentence(model, src, src_mask, src_emb, caplens, imgs, device):
    # Set model to evaluation mode
    model.eval()
    enc_image = model.encoder_image[0](imgs)  # encode image using CLIP model

    # Add mask for source input
    src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    # [batch size, 1, 1, src len]

    with torch.no_grad():
        # Encode the source article
        enc_src = model.encoder_text[0](src_emb, src_mask)

    max_length = max(caplens)
    # Initialize tensor for output sequence
    outputs = text_model.get_input_embeddings().weight[0].to(device)
    outputs = outputs.unsqueeze(0).unsqueeze(1).to(device)
    results = torch.zeros([1]).to(device)
    # print("results",results.shape)

    # Generate the caption one token at a time
    for i in range(max_length):
        # mask = torch.ones([i+1, i+1]).to(device)
        # Create mask for current target input
        # print("results", results.unsqueeze(0).shape)
        trg_pad_mask = (results.unsqueeze(0) != 0).unsqueeze(
            1).unsqueeze(2).to(device)
        trg_len = len(results)
        # print("trg_len",trg_len)
        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        with torch.no_grad():
            # Generate next token in the sequence
            output = model.decoder(
                outputs, trg_mask, enc_image, enc_src, src_emb, src_mask, src)
            # print("output",output[0,0,:10])
            # [batch_size, trg_len, src_len]

        best_guess = output.argmax(2)[:, -1]
        # print("best_guess", i, best_guess)
        results = torch.cat((results, best_guess), dim=0)
        # print("results_2", results)
        outputs_1 = text_model.get_input_embeddings(
        ).weight[best_guess.unsqueeze(0)].to(device)
        outputs = torch.cat((outputs, outputs_1), dim=1)
        # print("outputs_2", outputs.shape)

        if best_guess == 2:
            break

    translated_sentence = text_tokenizer.decode(results[1:-1])
    return translated_sentence[:]


def bleu(model, src, src_mask, src_emb, caplens, imgs, device):

    prediction = translate_sentence(
        model, src, src_mask, src_emb, caplens, imgs, device)
    prediction = prediction
    return prediction


def ciderScore(gts_file, res):
    gts = json.load(open(gts_file, 'r'))
    gts_dic = {}
    res_dic = {}
    for i in gts:
        gts_dic[i["id"]] = i["caption"]
    for i in res:
        res_dic[i["image_id"]] = i["caption"]
    scorer = Cider()
    (score, scores) = scorer.compute_score(gts_dic, res_dic)
    return score
