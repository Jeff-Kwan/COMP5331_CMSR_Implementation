import torch
from torch import nn
from math import sqrt


def get_activation(activation_name):
    activations = {
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
        'swish': nn.SiLU,
        'silu': nn.SiLU,
        'gelu': nn.GELU,
        'none': nn.Identity  # No activation
        }
    # Default activation is ReLU
    return activations.get(activation_name.lower(), nn.ReLU)() # New instance when called


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, activation="relu", bn=False):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),     # Only 1 linear projection
            nn.BatchNorm1d(out_dim) if bn else nn.Identity(),
            get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        
    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, 
                 layer_norm_eps, q_weight, k_weight, v_weight, with_mlp=False):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))
        self.n_heads = n_heads
        self.d = hidden_size // n_heads
        self.sqrt_d = sqrt(self.d)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # MLP adjustments
        self.with_mlp = with_mlp
        if with_mlp:
            self.mlp_query = MLP(hidden_size, hidden_size, bn=False)
            self.mlp_key = MLP(hidden_size, hidden_size, bn=False)
        else:
            self.mlp_query = nn.Identity()
            self.mlp_key = nn.Identity()

        # Setting QKV weights
        if q_weight is not None:
            self.query.weight = q_weight
        if k_weight is not None:
            self.key.weight = k_weight
        if v_weight is not None:
            self.value.weight = v_weight

        self.softmax = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(attn_dropout_prob))
        self.out = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(hidden_dropout_prob))
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x, attn_mask):
        # x: [batch_size, seq_len, hidden_size]
        # attn_mask: [batch_size, seq_len, seq_len]
        B, S, D = x.size()

        # Linear projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # MLP adjustments (Identity if initialized with with_mlp=False)
        q = self.mlp_query(q)
        k = self.mlp_key(k)

        # Reshape Q, K, V
        q = q.view(B, S, self.n_heads, self.d).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_d 
        attn_scores = attn_scores + attn_mask.unsqueeze(1)  # Same mask over each head
        attn_probs = self.softmax(attn_scores)  # Softmax w/ dropout

        # Contextual embeddings
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection
        attn_output = self.out(attn_output)
        output = self.LayerNorm(x + attn_output)
        return output        
    

class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps, ffn_dict=None):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            get_activation(hidden_act),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(hidden_dropout_prob))
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        if ffn_dict is not None:
            self.layers[0].weight = ffn_dict.get('layer1_weight')
            self.layers[0].bias = ffn_dict.get('layer1_bias')
            self.layers[2].weight = ffn_dict.get('layer2_weight')
            self.layers[2].bias = ffn_dict.get('layer2_bias')
        
    def forward(self, x):
        return self.norm(x + self.layers(x))
    

class TransformerLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, 
                 hidden_act, layer_norm_eps, q_weight, k_weight, v_weight, with_mlp, ffn_dict):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, 
                                            layer_norm_eps, q_weight, k_weight, v_weight, with_mlp)
        self.feedforward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps, ffn_dict)

    def forward(self, x, attn_mask):
        x = self.attention(x, attn_mask)
        x = self.feedforward(x)
        return x
    

class TransformerSingleEncoder(nn.Module):
    def __init__(self, n_layers=1, n_heads=2, hidden_size=64, inner_size=256, 
                 hidden_dropout_prob=0.5, attn_dropout_prob=0.5, hidden_act="gelu", 
                 layer_norm_eps=1e-12, q_weight=None, k_weight=None, v_weight=None, 
                 with_mlp=False, ffn_dict=None):
        super(TransformerSingleEncoder, self).__init__()

        # In code implementation, there is no n_layers implementation hence n_layers=1 by default there
        self.layers = nn.ModuleList([TransformerLayer(n_heads, hidden_size, inner_size, 
                                        hidden_dropout_prob, attn_dropout_prob, hidden_act, 
                                        layer_norm_eps, q_weight, k_weight, v_weight, with_mlp, 
                                        ffn_dict) for _ in range(n_layers)])

    def forward(self, x, attn_mask):
        outputs = []
        for layer in self.layers:
            x = layer(x, attn_mask)
            outputs.append(x)   
        return outputs