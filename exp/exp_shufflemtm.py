from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, transfer_weights
from utils.metrics import metric
from models.layers.revin import RevIN

from torch.optim import lr_scheduler

import patch_mask
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import random

warnings.filterwarnings('ignore')

class Exp_ShuffleMTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_ShuffleMTM, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
            
    def _build_model(self):
        normalizer = RevIN(self.args.c_in, affine=False)
        encoder = self.model_dict['ShuffleMTM'].Model(self.args.c_in, 
                                                             target_dim=self.args.pred_len, 
                                                             patch_len=self.args.patch_len,
                                                             stride=self.args.stride,
                                                             num_patch=self.args.num_patch,
                                                             e_layers=self.args.e_layers,
                                                             d_layers=self.args.d_layers,
                                                             n_heads=self.args.n_heads,
                                                             d_model=self.args.d_model,
                                                             shared_embedding=True,
                                                             d_ff=self.args.d_ff,
                                                             dropout=self.args.dropout,
                                                             head_dropout=self.args.head_dropout,
                                                             act='relu',
                                                             head_type=self.args.head_type,
                                                             res_attention=False,
                                                             ).float()
        
        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            print(self.device)
            print('torch.cuda.device_count()', torch.cuda.device_count())
            encoder = transfer_weights(self.args.load_checkpoints, encoder, device=self.device)
                
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # print out the model size
        
        print('number of params', sum(p.numel() for p in encoder.parameters() if p.requires_grad))
        return encoder, normalizer

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, learning_rate):
        model_optim = optim.Adam(list(self.normalizer.parameters())+list(self.encoder.parameters()), lr=learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def criterion_mask_reconstruct(self, preds, target, mask=None):
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        loss = (preds - target) ** 2
        if mask is not None:
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss
    
    def pretrain(self, setting):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # show cases
        self.train_show = next(iter(train_loader))
        self.valid_show = next(iter(vali_loader))

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data, setting, str(self.args.seed))
        if not os.path.exists(path):
            os.makedirs(path)
            
        # optimizer
        model_optim = self._select_optimizer(self.args.pretrain_learning_rate)
        #model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                T_max=self.args.train_epochs)

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss, train_loss_masked, train_loss_permuted = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            vali_loss, vali_loss_masked, vali_loss_permuted = self.valid_one_epoch(vali_loader)

            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f} ({4:.4f}, {5:.4f}) Val Loss: {6:.4f} ({7:.4f}, {8:.4f})".format(
                    epoch, model_scheduler.get_lr()[0], end_time - start_time, 
                    train_loss, train_loss_masked, train_loss_permuted, vali_loss, vali_loss_masked, vali_loss_permuted))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_loss_masked' : train_loss_masked,
                'train_loss_permuted' : train_loss_permuted,
                'vali_loss': vali_loss,
                'vali_loss_masked' : vali_loss_masked,
                'vali_loss_permuted' : vali_loss_permuted,
            }

            self.writer.add_scalars("/{}/{}".format(self.args.data, setting), loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))

                min_vali_loss = vali_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.encoder.state_dict().items():
                    # if 'encoder' in k or 'embedding' in k:
                    if 'module.' in k:
                        k = k.replace('module.', '')  # multi-gpu
                    self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.encoder.state_dict().items():
                    # if 'encoder' in k or 'embedding' in k:
                    if 'module.' in k:
                        k = k.replace('module.', '')
                    self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))


    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        train_loss = []
        train_loss_masked = []
        train_loss_permuted = []
        self.encoder.train()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            loss, loss_masked, loss_permuted = self.pretrain_one_iter(batch_x)
            loss.backward()
            model_optim.step()
            model_scheduler.step()
            
            # record
            train_loss.append(loss.item())
            train_loss_masked.append(loss_masked.item())
            train_loss_permuted.append(loss_permuted.item())
        

        train_loss = np.average(train_loss)
        train_loss_masked = np.average(train_loss_masked)
        train_loss_permuted = np.average(train_loss_permuted)

        return train_loss, train_loss_masked, train_loss_permuted
    
    def valid_one_epoch(self, vali_loader):
        valid_loss = []
        valid_loss_masked = []
        valid_loss_permuted = []

        self.encoder.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            loss, loss_masked, loss_permuted = self.pretrain_one_iter(batch_x)
            # Record
            valid_loss.append(loss.item())
            valid_loss_masked.append(loss_masked.item())
            valid_loss_permuted.append(loss_permuted.item())
            
        valid_loss = np.average(valid_loss)
        valid_loss_masked = np.average(valid_loss_masked)
        valid_loss_permuted = np.average(valid_loss_permuted)

        self.encoder.train()
        return valid_loss, valid_loss_masked, valid_loss_permuted
    
    def pretrain_one_iter(self, batch_x):
        
        # reversible instance normalization
        batch_x = batch_x.float().to(self.device)
        normed_x = self.normalizer(batch_x, 'norm')
        # patch
        x_patch, _ = patch_mask.create_patch(normed_x, self.args.patch_len)
        x_patch_permute = patch_mask.permute_patch(x_patch)
        # mask
        x_patch_masked, mask = patch_mask.random_masking(x_patch, self.args.mask_ratio)
        x_patch_permute_masked = patch_mask.random_masking(x_patch_permute, mask=mask)
        
        # forward: siamese encoder & cross-attention decoder
        pred = self.encoder(x_patch_masked, x_patch_permute_masked, ca_query=self.args.ca_query)
        
        # loss
        if self.args.rec=='shuffle':
            loss_masked = self.criterion_mask_reconstruct(pred, x_patch, mask)
            loss_permuted = self.criterion_mask_reconstruct(pred, x_patch_permute, 1-mask)
        elif self.args.rec=='original':
            loss_masked = self.criterion_mask_reconstruct(pred, x_patch, None)
            loss_permuted = torch.zeros((1,)).to(self.device)
        elif self.args.rec=='mask':
            loss_masked = self.criterion_mask_reconstruct(pred, x_patch, mask)
            loss_permuted = torch.zeros((1,)).to(self.device)
            
        loss = loss_masked + loss_permuted 
        
        return loss, loss_masked, loss_permuted
    
    
    def finetune(self, train_loader, vali_loader):
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Optimizer
        model_optim = self._select_optimizer(self.args.finetune_learning_rate)
        self.criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.finetune_learning_rate)    
        
        for epoch in range(self.args.train_epochs):
            start_time = time.time()
            train_loss = self.finetune_one_epoch(train_loader, model_optim, scheduler)
            vali_loss = self.finetune_valid_one_epoch(vali_loader)
            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.encoder, self.path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.encoder.load_state_dict(torch.load(best_model_path))

        self.lr = model_optim.param_groups[0]['lr']

        return self.encoder
    
    def freeze_and_finetune(self, train_loader, vali_loader):
        # PatchTST style fine-tuning
        
        self.freeze()
        model_optim = self._select_optimizer(self.args.finetune_learning_rate)
        self.criterion = self._select_criterion()
        optim_scheduler = lr_scheduler.OneCycleLR(model_optim, 
                                                  steps_per_epoch=len(train_loader),
                                                  pct_start = self.args.pct_start,
                                                  epochs=self.args.freeze_epochs,
                                                  max_lr=self.args.finetune_learning_rate)
        early_stopping = EarlyStopping(patience=self.args.freeze_epochs+self.args.finetune_epochs, verbose=True)
        
        print('Finetune the head')
        for epoch in range(self.args.freeze_epochs):
            start_time = time.time()
            train_loss = self.finetune_one_epoch(train_loader, model_optim, optim_scheduler)
            vali_loss = self.finetune_valid_one_epoch(vali_loader)
            end_time = time.time()
            print(
            "Epoch: {0}, Time: {1:.2f}s | Train Loss: {2:.7f} Vali Loss: {3:.7f} | lr={4:7f}".format(
                epoch + 1, end_time - start_time, train_loss, vali_loss, model_optim.param_groups[0]['lr']))
            early_stopping(vali_loss, self.encoder, self.path)

        self.unfreeze()
        optim_scheduler = lr_scheduler.OneCycleLR(model_optim,
                                                  steps_per_epoch=len(train_loader),
                                                  pct_start=self.args.pct_start, 
                                                  epochs=self.args.finetune_epochs,
                                                  max_lr=self.args.finetune_learning_rate/2)
        
        print('Finetune the entire network')
        for epoch in range(self.args.finetune_epochs):
            start_time = time.time()
            train_loss = self.finetune_one_epoch(train_loader, model_optim, optim_scheduler)
            vali_loss = self.finetune_valid_one_epoch(vali_loader)
            end_time = time.time()
            print(
            "Epoch: {0}, Time: {1:.2f}s | Train Loss: {2:.7f} Vali Loss: {3:.7f} | lr={4:7f}".format(
                epoch + 1, end_time - start_time, train_loss, vali_loss, model_optim.param_groups[0]['lr']))
            early_stopping(vali_loss, self.encoder, self.path)

            # adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.encoder.load_state_dict(torch.load(best_model_path))
        self.lr = model_optim.param_groups[0]['lr']
        return self.encoder
    
    def finetune_one_epoch(self, train_loader, model_optim, scheduler):
        train_loss = []
        self.encoder.train()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            # to device
            loss = self.finetune_one_iter(batch_x, batch_y)     
            loss.backward()
            model_optim.step()
            scheduler.step()
            train_loss.append(loss.item())
            
        train_loss = np.average(train_loss)
        
        return train_loss
    
    def finetune_valid_one_epoch(self, vali_loader):
        total_loss = []
        self.encoder.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # to device
                loss = self.finetune_one_iter(batch_x, batch_y).detach().cpu()     
                # record
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.encoder.train()
        return total_loss
    
    def finetune_one_iter(self, batch_x, batch_y):
        # to device
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        
        # reversible instance normalization 
        normed_x = self.normalizer(batch_x, 'norm')
        # patch
        batch_xp, _ = patch_mask.create_patch(normed_x, self.args.patch_len)
        
        # encoder
        outputs = self.encoder(batch_xp)
        outputs = self.normalizer(outputs, 'denorm')
        
        f_dim = -1 if self.args.features == 'MS' else 0

        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # loss
        loss = self.criterion(outputs, batch_y)
        return loss
    
    def freeze(self):
        for param in self.encoder.parameters(): param.requires_grad = False
        for param in self.encoder.head.parameters(): param.requires_grad = True
        
    def unfreeze(self):
        for param in self.encoder.parameters(): param.requires_grad = True

    def train(self, setting):
        
        self.path = os.path.join(self.args.checkpoints, setting, str(self.args.seed))
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        if self.args.finetune=='finetune_only':
            self.finetune(train_loader, vali_loader)
        
        elif self.args.finetune=='freeze_finetune':
            self.freeze_and_finetune(train_loader,vali_loader)


    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/{}/'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.encoder.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                normed_x = self.normalizer(batch_x, 'norm')
                batch_xp, _ = patch_mask.create_patch(normed_x, self.args.patch_len)
                
                # encoder
                outputs = self.encoder(batch_xp)
                outputs = self.normalizer(outputs, 'denorm')
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f = open(folder_path + "{}.txt".format(setting), 'a')
        # if self.args.pred_len==96:
        #     f.write(setting+'\n')
        f.write('{} -> {} | {:.3f}, {:.3f} \n'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f.close()
    
