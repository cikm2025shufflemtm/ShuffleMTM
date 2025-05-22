import os
import torch
from models import shufflemtm


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {'ShuffleMTM': shufflemtm}
        self.device = self._acquire_device()
        self.encoder, self.normalizer = self._build_model()
        self.encoder = self.encoder.to(self.device)
        self.normalizer = self.normalizer.to(self.device)
        
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            
            device = torch.device('cuda:{}'.format(self.args.gpu) if not self.args.use_multi_gpu else f'cuda:{self.args.device_ids[0]}')
            print('Use GPU: {}'.format(device))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
