# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import math
from ....utils import get_or_create_path
from ....log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ....model.base import Model
from ....data.dataset import DatasetH
from ....data.dataset.handler import DataHandlerLP
import sys
sys.path.append("/home/wyy/workspace/ms/qlib/qlib_cur/contrib/model/customer")
from ns_layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from ns_layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
from ns_layers.t2v import T2VEncoding


class TransformerModel(Model):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 2048, #8192
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        model_type="transformerT2V",
        **kwargs
    ):

        # set hyper-parameters.
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("ProTransformerModel")
        self.logger.info("Naive Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))
        self.model_type = model_type
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        if model_type == "transformer":
            self.model = Transformer(d_feat, d_model, nhead, num_layers, dropout, self.device)
        if model_type == "transformerT2V":
            self.model = TransformerT2V(d_feat, d_model, nhead, num_layers, dropout, self.device)
        # if model_type == "nstransformer":
            # self.model = NSTransformer(d_feat, d_model, nhead, num_layers, dropout, self.device)
        if model_type =="nstransformerT2V":
            self.model = NSTransformerT2V(d_feat, d_model, nhead, num_layers, dropout, self.device)
        


        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):

        self.model.train()

        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            # print(feature.shape)
            pred = self.model(feature.float())  # .float()
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):

        self.model.eval()

        scores = []
        losses = []

        for data in data_loader:

            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float())  # .float()
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        print("--------------------fit")
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        
        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y

class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, T, F], [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()


class TransformerT2V(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(TransformerT2V, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.t2v_encoder = T2VEncoding("sin", d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model*2, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model*2, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, T, F], [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src) # [T, N, F]
        src = src.transpose(1, 0) 

        time_embedding = self.t2v_encoder(src) # [T, F]
        time_embedding = time_embedding.unsqueeze(0).repeat(src.shape[0], 1, 1) # [N, T, F]

        src = torch.cat((src,time_embedding),dim=-1) # [N, T, F*2]
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        output = self.decoder_layer(output[:, -1, :])  # [512, 1]

        return output.squeeze()



class NSTransformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(NSTransformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        d_model_encoder = d_model 
        self.transformer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, 1, attention_dropout=dropout,
                                      output_attention=True), d_model_encoder, nhead),
                    d_model_encoder,
                    d_ff=None,
                    dropout=dropout,
                ) for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model_encoder)
        )
        self.d_model = d_model
        self.device = device
        self.d_feat = d_feat
        self.seq_len = 20
        self.tau_learner   = Projector(enc_in=d_feat, seq_len=self.seq_len, hidden_dims=[d_model_encoder, d_model_encoder], hidden_layers=num_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=self.d_feat, seq_len=self.seq_len, hidden_dims=[d_model_encoder,d_model_encoder], hidden_layers=num_layers, output_dim=self.seq_len)
        self.denorm_pre = nn.Linear(d_model_encoder,self.seq_len)
        self.decoder_layer = nn.Linear(self.seq_len, 1)

    def forward(self, src):

        x_raw = src.clone().detach()

        # Normalization
        mean_src = src.mean(1, keepdim=True).detach() # B x 1 x E
        src = src - mean_src
        std_src = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        src = src / std_src

        tau = self.tau_learner(x_raw, std_src).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
        delta = self.delta_learner(x_raw, mean_src)      # B x S x E, B x 1 x E -> B x S

        # Model Inference
        # src [N, T, F], [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None
        src = self.pos_encoder(src)
        src = src.transpose(1, 0)
        # print(time_embedding.shape)
        # print(src.shape)

        # output = self.transformer_encoder(src, mask)  # [60, 512, 8]
        output, attn = self.transformer_encoder(src, attn_mask=mask, tau=tau, delta=delta)
        # De-normalization
        output = self.denorm_pre(output) * std_src + mean_src #[4096,20,20]
        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output[:, -1, :])  # [512, 1]
        return output.squeeze()



class NSTransformerT2V(nn.Module):
    '''transformer + ns + t2v'''
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(NSTransformerT2V, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.t2v_encoder = T2VEncoding("sin", d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        d_model_encoder = d_model * 2
        self.transformer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, 1, attention_dropout=dropout,
                                      output_attention=True), d_model_encoder, nhead),
                    d_model_encoder,
                    d_ff=None,
                    dropout=dropout,
                ) for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model_encoder)
        )
        self.d_model = d_model
        self.device = device
        self.d_feat = d_feat
        self.seq_len = 20
        self.tau_learner   = Projector(enc_in=d_feat, seq_len=self.seq_len, hidden_dims=[d_model_encoder, d_model_encoder], hidden_layers=num_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=self.d_feat, seq_len=self.seq_len, hidden_dims=[d_model_encoder,d_model_encoder], hidden_layers=num_layers, output_dim=self.seq_len)
        self.denorm_pre = nn.Linear(d_model_encoder,self.seq_len)
        self.decoder_layer = nn.Linear(self.seq_len, 1)

    def forward(self, src):

        x_raw = src.clone().detach()

        # Normalization
        mean_src = src.mean(1, keepdim=True).detach() # B x 1 x E
        src = src - mean_src
        std_src = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        src = src / std_src

        tau = self.tau_learner(x_raw, std_src).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
        delta = self.delta_learner(x_raw, mean_src)      # B x S x E, B x 1 x E -> B x S

        # Model Inference
        # src [N, T, F], [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None
        src = self.pos_encoder(src)
        # print(src.shape)
        src = src.transpose(1, 0)
        # print(src.shape)
        time_embedding = self.t2v_encoder(src)
        # print(time_embedding.shape)

        time_embedding = time_embedding.unsqueeze(0).repeat(src.shape[0], 1, 1)
        src = torch.cat((src,time_embedding),dim=-1)
        # print(time_embedding.shape)
        # print(src.shape)

        # output = self.transformer_encoder(src, mask)  # [60, 512, 8]
        output, attn = self.transformer_encoder(src, attn_mask=mask, tau=tau, delta=delta)
        # De-normalization
        output = self.denorm_pre(output) * std_src + mean_src #[4096,20,20]
        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output[:, -1, :])  # [512, 1]
        return output.squeeze()


