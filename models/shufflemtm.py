
__all__ = ['Model']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from models.layers.pos_encoding import *
from models.layers.basics import *
from models.layers.attention import *
from models.layers.revin import RevIN
from models.layers.enc_dec import *
from models.layers.heads import *

# Cell

class Model(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, 
                 e_layers:int=3, d_layers=1, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'

        # Backbone
        self.backbone = ShuffleMTMEncoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                n_layers=e_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        
        self.head_type = head_type
        
        if head_type=='pretrain':
            self.head = ShuffleMTMDecoder(d_model, n_heads=n_heads, patch_len=patch_len, d_ff=d_ff,
                                   norm=norm, attn_dropout=attn_dropout, dropout=dropout, activation=act,
                                   res_attention=res_attention, n_layers=d_layers)
        else:
            self.head = DownstreamHead(c_in, target_dim, num_patch, d_model, head_dropout, individual, head_type, y_range)
            
    def forward(self, x, x_permute=None, ca_query=None):                             
        """
        x: tensor [bs x num_patch x n_vars x patch_len]
        """   
        z = self.backbone(x)                                                                # z: [bs x nvars x d_model x num_patch]
        if self.head_type=='pretrain':
            z_permute = self.backbone(x_permute)
            if ca_query=='permute':
                pred = self.head(z_permute, z)
            elif ca_query=='original':
                pred = self.head(z, z_permute)
        else:
            pred = self.head(z)
        return pred
    
    
    
class DownstreamHead(nn.Module):
    def __init__(self, c_in, target_dim, num_patch, d_model, head_dropout, individual, head_type, y_range=None):
        super().__init__()
        if head_type == "prediction":
            self.head = PredictionHead(individual, c_in, d_model, num_patch, target_dim, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(c_in, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(c_in, d_model, target_dim, head_dropout)
        
    def forward(self, x):
        """
        x : [bs x nvars x d_model x num_patch]
        """
        pred = self.head(x)
        # pred: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        return pred
    







