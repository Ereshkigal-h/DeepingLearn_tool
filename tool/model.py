import math
import torch
from torch import nn
import torch.nn.functional as F
class transformer(nn.Module):
    def __init__(self,vocab_size,layer=6,d_model=512,num_heads=8,d_ff=2048,dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding1 = nn.Embedding(vocab_size, d_model)
        self.embedding2 = nn.Embedding(vocab_size, d_model)
        self.encoder = transformer_encoder(d_model,num_heads,d_ff,dropout,layer=layer)
        self.decoder = transformer_decoder(d_model,num_heads,d_ff,dropout,layer=layer)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self,src,tar,src_mask=None,tar_mask=None):
        src=self.embedding1(src)
        tar=self.embedding2(tar)
        combined_mask = self.combine_mask(tar,tar_mask)
        q_k=self.encoder(src,src_mask)
        x=self.decoder(tar,q_k,mask=tar_mask,conb_mask= combined_mask)
        x=self.fc_out(x)
        return x
    def combine_mask(self,x,mask=None):
        #x:[batch,seq_len_q,d_k]
        #mask:[]
        seq_len=x.size(1)
        row=torch.arange(seq_len,device=x.device).unsqueeze(1)
        column=torch.arange(seq_len,device=x.device).unsqueeze(0)
        causal_mask_bool= column >row
        causal_mask_bool = causal_mask_bool.unsqueeze(0).unsqueeze(0)
        #causal_mask_bool[1,1,seq_len,seq_len]
        #mask[batch_size,1,1,seq_len]
        if mask is None:
            combined_mask_bool = causal_mask_bool
        else:
            combined_mask_bool = causal_mask_bool | mask
        return combined_mask_bool

class transformer_decoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1,layer=6):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.decoder_layer = nn.ModuleList([transformer_decoder_layer(d_model,num_heads,d_ff,dropout)
                                            for _ in range(layer)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self,x,q_k,mask=None,conb_mask=None):
        for layer in self.decoder_layer:
            x=layer(x,q_k,mask=mask,conb_mask=conb_mask)
        x=self.norm(x)
        return x




class transformer_decoder_layer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1,mask=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.cross_attention = MultiHeadAttention(d_model,num_heads)
        self.mask_attention = MultiHeadAttention(d_model,num_heads)
        self.ffn = FFN(d_model,d_ff,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1,self.norm2,self.norm3= nn.LayerNorm(d_model),nn.LayerNorm(d_model),nn.LayerNorm(d_model)
    def forward(self,x,q_k,mask=None,conb_mask=None):
        norm_x = self.norm1(x)
        hidden = self.mask_attention(query=norm_x, key=norm_x, value=norm_x, mask=conb_mask)
        x = x + self.dropout(hidden)
        norm2_x = self.norm2(x)
        hidden2=self.cross_attention(query=norm2_x, key=q_k, value=q_k, mask=mask)
        x = x + self.dropout(hidden2)
        norm3_x = self.norm3(x)
        x = x + self.dropout(self.ffn(norm3_x))
        return x






class transformer_encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, layer=6):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            transformer_encoder_layer(d_model, num_heads, d_ff, dropout)
            for _ in range(layer)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self,x,mask=None):
        for layer in self.layers:
            x=layer(x,mask)
        x=self.norm(x)
        return x
class transformer_encoder_layer(nn.Module):
    def __init__(self,d_model=512,num_heads=8,d_ff=2048,dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout=nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.multihead_attention=MultiHeadAttention(self.d_model,self.num_heads)
        self.FFN=FFN(self.d_model,self.d_ff,dropout)
    def forward(self,x,mask=None):
        norm_x=self.norm1(x)
        hidden=self.multihead_attention(query=norm_x,key=norm_x,value=norm_x,mask=mask)
        x=x+self.dropout(hidden)
        norm2_x=self.norm2(x)
        x=x+self.dropout(self.FFN(norm2_x))
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model
        self.singe_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def scaled_dot_product_attention(self,query, key, value, mask=None):
        """
           计算缩放点积注意力
           参数维度说明：
           :param query: 形状通常是 [batch_size, num_heads, seq_len_q, d_k] (你先当它是三维也行 [batch, seq, dim])
           :param key: 形状和 query 类似，维度 d_k 必须一样
           :param value: 形状和 key 类似，维度 d_v (通常 d_k == d_v)
           :param mask: 用于遮掩某些位置(如 padding 不齐的 0，或未来信息)
           :return: 注意力计算结果 和 注意力权重矩阵
           """
        d_k=query.size(-1)
        #dot_product=torch.bmm(query,key.transpose(-2,-1)) / d_k*0.05
        dot_product=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            #dot_product[mask]=-1e9这个有风险，最好用体系化函数
            dot_product=dot_product.masked_fill(mask==1,-1e9)
        score=F.softmax(dot_product,dim=-1) @ value
        return score
    def forward(self, query, key, value,mask=None):
        batch_size,seq_len_q,d_k =query.size()
        q=self.W_q(query)
        k=self.W_k(key)
        v=self.W_v(value)
        q,k,v=(q.view(batch_size, -1, self.num_heads, self.singe_dim).transpose(1, 2),
               k.view(batch_size,-1,self.num_heads,self.singe_dim).transpose(1,2),
               v.view(batch_size,-1,self.num_heads,self.singe_dim).transpose(1,2))
        if mask is not None:
            if mask.dim()==2:
                mask=mask.unsqueeze(1).unsqueeze(2)
            if mask.dim()==3:
                mask=mask.unsqueeze(1)
        dot_product=self.scaled_dot_product_attention(q,k,v,mask).transpose(1,2).contiguous().view(batch_size,-1,d_k)
        output=self.out_proj(dot_product)
        return output
class FFN(nn.Module):
    def __init__(self, d_model,hidden,dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x





    

