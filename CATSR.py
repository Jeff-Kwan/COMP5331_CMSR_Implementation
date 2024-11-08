import torch
from torch import nn
from torch.nn.parameter import Parameter
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from layers import TransformerSingleEncoder
from loss_fct import element_weighted_loss, calculate_item_freq, weight_decay

class CATSR(SequentialRecommender):
    def __init__(self, config, dataset, weight_dict=None):
        super(CATSR, self).__init__(config, dataset)

        self.attn_dropout_prob = config['attn_dropout_prob']
        self.fix = config['fix']
        self.hidden_act=config['hidden_act']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_size=config['hidden_size']
        self.initializer_range=config['initializer_range']
        self.inner_size=config['inner_size']
        self.layer_norm_eps=config['layer_norm_eps']
        self.log_base=config['log_base']
        self.loss_type=config['loss_type']
        self.n_layers=config['n_layers']
        self.n_heads=config['n_heads']
        self.with_mlp=config['with_adapter']

        if weight_dict is not None:
            q_weight=weight_dict['q_weight'] #writing styles different
            if q_weight is not None:
                self.q_weight=Parameter(q_weight)
                if self.fix:
                    self.q_weight.requires_grad=False
            k_weight=weight_dict['k_weight'] #writing styles different
            if k_weight is not None:
                self.k_weight=Parameter(k_weight)
                if self.fix:
                    self.k_weight.requires_grad=False
            v_weight=weight_dict['v_weight'] #writing styles different
            if v_weight is not None:
                self.v_weight=Parameter(v_weight)
                if self.fix:
                    self.v_weight.requires_grad=False
        else:
            self.q_weight=None
            self.k_weight=None
            self.v_weight=None

        self.item_embedding=nn.Embedding.from_pretrained(dataset.item_feat.item_emb)
        self.hidden_size=dataset.item_feat.item_emb.size(1) #writing style different
        self.position_embedding=nn.Embedding(self.max_seq_len, self.hidden_size)
        self.trm_encoder=TransformerSingleEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            q_weight=self.q_weight,
            k_weight=self.k_weight,
            v_weight=self.v_weight,
            with_mlp=self.with_mlp
        )
        self.LayerNorm=nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(p=self.hidden_dropout_prob)

        if self.loss_type=='BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type=='CE':
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type=='WCE':
            self.loss_fct = element_weighted_loss()
            #
            self.alpha = config['alpha']
            self.beta = config['beta']
        else:
            raise NotImplementedError("Loss type not implemented, only BPR CE WCE")
        self.apply(self._init_weights)

        def _init_weights(self, module): #writing styles different
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad:
                    module.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                if module.weight.requires_grad:
                    module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        def forward(self, item_seq, item_seq_len, status='train'):
            position_ids=torch.arange(item_seq.size(1), dtype=torch.long, device=self.device)#[seq_length,]
            position_ids=position_ids.unsqueeze(0).expand_as(item_seq)#[1,seq_length] [batch_size,seq_length]
            position_embedding=self.position_embedding(position_ids)#[batch_size,seq_length]
            item_emb=self.item_embedding(item_seq)#[batch_size,seq_length]
            input_emb=item_emb+position_embedding#[batch_size,seq_length]
            input_emb=self.LayerNorm(input_emb)
            input_emb=self.dropout(input_emb)
            extended_attention_mask=self.get_attention_mask(item_seq_len)

            trm_output=self.trm_encoder(input_emb, extended_attention_mask)
            output=trm_output[-1]
            output=self.gather_indexes(output,item_seq_len-1)
            return output #[batch_size,seq_length]

        def calculate_loss(self,interaction):
            item_seq=interaction[self.ITEM_SEQ]
            item_seq_len=interaction[self.ITEM_SEQ_LEN]
            seq_output=self.forward(item_seq, item_seq_len)
            pos_items=interaction[self.POS_ITEM_ID]
            if self.loss_type=='BPR':
                neg_items=interaction[self.NEG_ITEM_ID]
                pos_items_emb=self.item_embedding(pos_items)
                neg_items_emb=self.item_embedding(neg_items)
                pos_score=torch.sum(seq_output * pos_items_emb, dim=-1) #[batch_size,seq_length]*[batch_size,seq_length], and then sum up of seq_length dimension, [batch_size]
                neg_score=torch.sum(seq_output * neg_items_emb, dim=-1) #[batch_size]
                loss=self.loss_fct(pos_score, neg_score)
            elif self.loss_type=='CE':
                test_item_emb = self.item_embedding.weight
                logits=torch.matmul(seq_output, test_item_emb.transpose(0,1)) #[batch_size,seq_length]*[seq_length,batch_size]
                loss=self.loss_fct(logits, pos_items)
            elif self.loss_type=='WCE':
                test_item_emb = self.item_embedding.weight
                logits=torch.matmul(seq_output, test_item_emb.transpose(0,1))
                item_weight=torch.clone(pos_items).cpu().double()
                item_weight.apply_(lambda x: self.item_weight_dict[x])
                item_weight.apply_(lambda x: weight_decay(x,self.alpha,self.beta))
                item_weight=item_weight.to(pos_items.device)
                loss=self.loss_fct(logits, pos_items,item_weight)
            else:
                test_item_emb = self.item_embedding.weight
                logits=torch.matmul(seq_output, test_item_emb.transpose(0,1))
                item_weight=torch.clone(pos_items).cpu().double()
                item_weight.apply_(lambda x: self.item_weight_dict[x])
                item_weight.apply_(lambda x:inverse_frequency(x,self.item_average_freq,self.log_base))
                item_weight=item_weight.to(pos_items.device)
                loss=self.loss_fct(logits, pos_items,item_weight)
            return loss

        def predict(self, interaction):
            item_seq=interaction[self.ITEM_SEQ]
            item_seq_len=interaction[self.ITEM_SEQ_LEN]
            test_item=interaction[self.ITEM_ID]
            seq_output=self.forward(item_seq, item_seq_len,status='eval')
            test_item_emb=self.item_embedding(test_item)
            scores=torch.mul(seq_output, test_item_emb).sum(dim=1)
            return scores

        def full_sort_predict(self, interaction):
            item_seq=interaction[self.ITEM_SEQ]
            item_seq_len=interaction[self.ITEM_SEQ_LEN]
            seq_output=self.forward(item_seq, item_seq_len,status='eval')
            test_item_emb=self.item_embedding.weight
            scores=torch.matmul(seq_output, test_item_emb.transpose(0,1))
            return scores








