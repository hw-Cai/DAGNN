import numpy as np
import torch.nn as nn
from datasets import *

import variables as var

d_model = 6   # 字 Embedding 的维度  512
d_ff = 16    # 前向传播隐藏层维度  2048
d_k = d_v = 16 # K(=Q), V的维度  64
n_layers = 3    # 有多少个encoder和decoder  6    # MAM 单独用transformer layers=3,head=3 d_model=6  AUC=85
n_heads = 3     # Multi-Head Attention设置为8


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])           # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])           # 字嵌入维度为奇数时
        # self.pos_table = torch.FloatTensor(pos_table).cuda()        # enc_inputs: [seq_len, d_model]
        self.pos_table = torch.FloatTensor(pos_table).to(var.device)

    def forward(self, enc_inputs):                                  # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        # return self.dropout(enc_inputs.cuda())
        return self.dropout(enc_inputs.to(var.device))

def get_attn_pad_mask(seq_q, seq_k):                                # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)                   # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)           # 扩展成多维度


def get_attn_subsequence_mask(seq):                                 # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)            # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()    # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):                              # Q: [batch_size, n_heads, len_q, d_k]
                                                                        # K: [batch_size, n_heads, len_k, d_k]
                                                                        # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)    # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)                            # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                 # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):    # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)    # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)    # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)       # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)                                # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)             # context: [batch_size, n_heads, len_q, d_v]
                                                                                    # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)                    # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                   # [batch_size, len_q, d_model]
        # return nn.LayerNorm(d_model).cuda()(output + residual), attn
        return nn.LayerNorm(d_model).to(var.device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):                                  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        # return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model).to(var.device)(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                   # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                      # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):              # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V            # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                                    # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                     # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

#
# class EncoderLayer(nn.Module):
#     def __init__(self):
#         super(EncoderLayer, self).__init__()
#         self.enc_self_attn = MultiHeadAttention()       # 多头注意力机制
#         self.pos_ffn = PoswiseFeedForwardNet()          # 前馈神经网络
#
#     def forward(self, enc_inputs, enc_self_attn_mask):  # enc_inputs: [batch_size, src_len, d_model]
#         # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V             # enc_self_attn_mask: [batch_size, src_len, src_len]
#         enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
#                                                                         # enc_outputs: [batch_size, src_len, d_model],
#                                                enc_self_attn_mask)      # attn: [batch_size, n_heads, src_len, src_len]
#         enc_outputs = self.pos_ffn(enc_outputs)                         # enc_outputs: [batch_size, src_len, d_model]
#         return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(10, d_model)                     # 把字转换字向量
        self.pos_emb = PositionalEncoding(d_model)                               # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs)                                  # enc_outputs: [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):                                             # dec_inputs: [batch_size, tgt_len, d_model]
                                                                                # enc_outputs: [batch_size, src_len, d_model]
                                                                                # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                                                                                # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs,
                                                        dec_self_attn_mask)     # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs,
                                                      dec_enc_attn_mask)        # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)                                 # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(10, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):                         # dec_inputs: [batch_size, tgt_len]
                                                                                    # enc_intpus: [batch_size, src_len]
                                                                                    # enc_outputs: [batsh_size, src_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)                                      # [batch_size, tgt_len, d_model]
        # dec_outputs = self.pos_emb(dec_outputs).cuda()                              # [batch_size, tgt_len, d_model]
        # dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()   # [batch_size, tgt_len, tgt_len]
        # dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        # dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
        #                                dec_self_attn_subsequence_mask), 0).cuda()   # [batch_size, tgt_len, tgt_len]
        dec_outputs = self.pos_emb(dec_outputs).to(var.device)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(var.device)   # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(var.device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).to(var.device)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)               # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:                                                   # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                    # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
                                                                                    # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # self.Encoder = Encoder().cuda()
        # self.Decoder = Decoder().cuda()
        # self.projection = nn.Linear(d_model, 3, bias=False).cuda()  #tgt_vocab_size
        self.Encoder = Encoder().to(var.device)
        self.Decoder = Decoder().to(var.device)
        self.projection = nn.Linear(d_model, 6, bias=False).to(var.device)  #tgt_vocab_size

    def forward(self, enc_inputs, dec_inputs):                          # enc_inputs: [batch_size, src_len]
                                                                        # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)          # enc_outputs: [batch_size, src_len, d_model],
                                                                        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)                        # dec_outpus    : [batch_size, tgt_len, d_model],
                                                                        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
                                                                        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)                       # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # print('dec_outputs.size: ', dec_outputs.size())
        # print('dec_logits.size(0): ', dec_logits.size(0))
        # print('dec_logits.view_test: ', dec_logits.view(dec_logits.size(0), -1).size())
        # return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

        return dec_logits.view(dec_logits.size(0), -1), enc_self_attns, dec_self_attns, dec_enc_attns
        # return dec_outputs.view(dec_logits.size(0), -1), enc_self_attns, dec_self_attns, dec_enc_attns