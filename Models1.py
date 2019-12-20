import torch
import torch.nn as nn 
from Layersff import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayersff import Norm,MultiHeadAttention,FeedForward
import copy
import numpy as np

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        print("src:", src.size())
        x = self.embed(src)
        print("x.embed:",x.size())
        x = self.pe(x)
        print("x.pe:",x.size())
        for i in range(self.N):
            x = self.layers[i](x, mask)
        print("x.attention:", x.size())
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        #print("trg:",trg.size())
        x = self.embed(trg)
        #print("x.embed:",x.size())
        x = self.pe(x)
        #print("x.pe:",x.size())
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        #print("x.attention:",x.size())
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
        self.out1 = nn.Linear(d_model, src_vocab)
        self.out2 = nn.Linear(d_model, 1)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        mean_pooled = torch.mean(e_outputs, dim=1)
        print("mean.e_output:", mean_pooled.size())
        y = self.out2(mean_pooled)
        y = torch.mean(y, dim=0)
        y = torch.sigmoid(y)
        print(y)
        exit()
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    #if opt.device == 0:
    #    model = model.cuda()
    
    return model
    
