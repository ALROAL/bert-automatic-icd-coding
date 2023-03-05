from hydra import initialize, compose
try:
    initialize('../')
except ValueError:
    pass
cfg = compose(config_name='config')
import torch
import pandas as pd
from transformers import AutoModel
from src.PATHS import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from transformers import AutoModel, AutoConfig
from .transformers_chunking import apply_chunking_to_forward

def build_model(mode=cfg.dataset):
    if mode == "full":
        Y = len(pd.read_csv(PROCESSED_DATA_PATH / "FULL_CODES.csv", header=None))
    else:
        Y = len(pd.read_csv(PROCESSED_DATA_PATH / f"TOP_{mode}_CODES.csv", header=None))

    if cfg.model == "HiBERT":
        model = HiBERT(Y)
    elif cfg.model == "PLM-ICD":
        if cfg.model_type == "BERT":
            model = BertForMultilabelClassification(Y)
        elif cfg.model_type == "RoBERTa":
            model = RobertaForMultilabelClassification(Y)
    model.to(cfg.device)

    return model

def load_model(model=cfg.model, mode=cfg.dataset):
    path = MODELS_PATH / model / f"model_{mode}.pth"
    model = build_model(mode)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def save_model(model, model_name):
    model_path = MODELS_PATH / cfg.model
    model_path.mkdir(parents=True, exist_ok=True)
    model_path = model_path / model_name
    torch.save(model.state_dict(), model_path.as_posix())

#Models from the paper Does the Magic of BERT Apply to Medical Code Assignment? A Quantitative Study
#https://agit.ai/jsx/MCA_BERT
class OutputLayer(nn.Module):
    def __init__(self, Y, embed_dim):
        super(OutputLayer, self).__init__()
        self.U = nn.Linear(embed_dim, Y)
        xavier_uniform_(self.U.weight)
        self.final = nn.Linear(embed_dim, Y)
        xavier_uniform_(self.final.weight)

    def forward(self, x):
        att = self.U.weight.matmul(x.transpose(1, 2)) # [bs, Y, seq_len]
        alpha = F.softmax(att, dim=2)
        m = alpha.matmul(x) # [bs, Y, dim]
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y

class BERT(nn.Module):
    def __init__(self, Y):
        super(BERT, self).__init__()
        self.decoder = cfg.decoder
        self.Y = Y
        self.apply(self.init_bert_weights)

        config = AutoConfig.from_pretrained(cfg.bert_model)

        self.bert = AutoModel.from_pretrained(cfg.bert_model, config=config)
        if cfg.decoder == "fcn":
            self.classifier = nn.Linear(self.bert.config.hidden_size, self.Y)
        elif cfg.decoder == "lan":
            self.classifier = OutputLayer(self.Y, self.bert.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        # x: [bs, seq_len]
        hidden_state = self.bert(input_ids, attention_mask).last_hidden_state # hidden_states: [bs, seq_len, 768]
        if self.decoder == 'fcn':
            final_features = hidden_state[:,0,:] # the 0-th hidden state is used for classification
            y = self.classifier(final_features)
        elif self.decoder == 'lan':
            y = self.classifier(hidden_state)
        return y

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class ChunkBERT(nn.Module):
    """
    A BERT operating on documents chunked based on their lenght to chunks of lenght max 512.
    Forward returns document representation in format [batch size, number of features (768, hidden size), number of chunks (sequence length)]
    """
    def __init__(self):
        super(ChunkBERT, self).__init__()
        self.apply(self.init_bert_weights)
        config = AutoConfig.from_pretrained(cfg.bert_model)
        self.bert = AutoModel.from_pretrained(cfg.bert_model, config=config)
    
    def forward_chunk(self, input_ids, attention_mask):
        # x: [bs, seq_len]
        hidden_states = self.bert(input_ids, attention_mask).last_hidden_state # hidden_states: [bs, seq_len, 768]
        final_features = hidden_states[:,0,:] #the 0-th hidden state is used for classification
        return final_features

    def forward(self, input_ids, attention_mask):
        # Document in format [batch size, number of features (768), number of chunks (sequence length)]
        document = apply_chunking_to_forward(self.forward_chunk, cfg.chunk_size, 1, input_ids, attention_mask)

        return document

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class HiBERT(nn.Module):
    def __init__(self, Y):
        super(HiBERT, self).__init__()
        self.Y = Y
        self.decoder = cfg.decoder        
        self.apply(self.init_bert_weights)
        self.chunk_bert = ChunkBERT().to(cfg.device)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.chunk_bert.bert.config.hidden_size, nhead=cfg.n_heads, dropout=self.chunk_bert.bert.config.hidden_dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=cfg.transformer_encoder_num_layers)

        if self.decoder == "fcn":
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.chunk_bert.bert.config.hidden_dropout_prob),
                nn.Linear(self.chunk_bert.bert.config.hidden_size, Y)
                )
        elif self.decoder == "lan":
            self.classifier = OutputLayer(self.Y, self.chunk_bert.bert.config.hidden_size, self.chunk_bert.bert)
            
    def forward(self, input_ids, attention_mask):
        # x: [bs, seq_len]
        document = self.chunk_bert(input_ids, attention_mask)

        document = document.permute(2,0,1)
        transformer_output = self.transformer_encoder(document)
        transformer_output = transformer_output.permute(1,0,2)

        if self.decoder == 'fcn':
            y = self.classifier(transformer_output.max(dim=1)[0])
        elif self.decoder == 'lan':
            y = self.classifier(transformer_output)

        return y

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


#Models from the paper PLM-ICD: Automatic ICD Coding with Pretrained Language Models
#https://github.com/MiuLab/PLM-ICD/tree/764ca73473df3f948857fb52f4db2e65b5d8c995
class BertForMultilabelClassification(nn.Module):
    def __init__(self, Y):
        super(BertForMultilabelClassification, self).__init__()
        self.num_labels = Y
        self.model_mode = cfg.model_mode

        config = AutoConfig.from_pretrained(cfg.bert_model)
        self.bert = AutoModel.from_pretrained(cfg.bert_model, config=config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if "cls" in self.model_mode:
            self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        elif "laat" in self.model_mode:
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, self.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, self.num_labels)
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

    def forward(self, input_ids, attention_mask):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.bert(input_ids.view(-1, chunk_size), attention_mask=attention_mask.view(-1, chunk_size))
        
        if "cls" in self.model_mode:
            pooled_output = outputs[1].view(batch_size, num_chunks, -1)
            if self.model_mode == "cls-sum":
                pooled_output = pooled_output.sum(dim=1)
            elif self.model_mode == "cls-max":
                pooled_output = pooled_output.max(dim=1).values
            else:
                raise ValueError(f"model_mode {self.model_mode} not recognized")
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        elif "laat" in self.model_mode:
            if self.model_mode == "laat":
                hidden_output = outputs[0].view(batch_size, num_chunks*chunk_size, -1)
            elif self.model_mode == "laat-split":
                hidden_output = outputs[0].view(batch_size*num_chunks, chunk_size, -1)
            weights = torch.tanh(self.first_linear(hidden_output))
            att_weights = self.second_linear(weights)
            att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
            weighted_output = att_weights @ hidden_output
            logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
            if self.model_mode == "laat-split":
                logits = logits.view(batch_size, num_chunks, -1).max(dim=1).values
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        return logits

class RobertaForMultilabelClassification(nn.Module):
    def __init__(self, Y):
        super(RobertaForMultilabelClassification, self).__init__()
        self.num_labels = Y
        self.model_mode = cfg.model_mode

        config = AutoConfig.from_pretrained(cfg.bert_model)
        self.bert = AutoModel.from_pretrained(cfg.bert_model, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if "cls" in self.model_mode:
            self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        elif "laat" in self.model_mode:
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, self.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, self.num_labels)
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

    def forward(self,input_ids,attention_mask):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """

        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.bert(input_ids.view(-1, chunk_size), attention_mask=attention_mask.view(-1, chunk_size))
        
        if "cls" in self.model_mode:
            pooled_output = outputs[1].view(batch_size, num_chunks, -1)
            if self.model_mode == "cls-sum":
                pooled_output = pooled_output.sum(dim=1)
            elif self.model_mode == "cls-max":
                pooled_output = pooled_output.max(dim=1).values
            else:
                raise ValueError(f"model_mode {self.model_mode} not recognized")
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        elif "laat" in self.model_mode:
            if self.model_mode == "laat":
                hidden_output = outputs[0].view(batch_size, num_chunks*chunk_size, -1)
            elif self.model_mode == "laat-split":
                hidden_output = outputs[0].view(batch_size*num_chunks, chunk_size, -1)
            weights = torch.tanh(self.first_linear(hidden_output))
            att_weights = self.second_linear(weights)
            att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
            weighted_output = att_weights @ hidden_output
            logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
            if self.model_mode == "laat-split":
                logits = logits.view(batch_size, num_chunks, -1).max(dim=1).values
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        return logits