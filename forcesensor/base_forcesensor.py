import torch
from torch import device
from torch.functional import tensordot
from forcesensor.SoftFinger.src.models import VisualForceVAE
from forcesensor.SoftFinger.src.datamodules.SoftFinger_datamodule import SoftFingerDataModule
from forcesensor.SoftFinger.src.datamodules.SoftFinger_datamodule import Bone_finger, Bone_finger_test

from forcesensor.SoftFinger.src.callbacks.printing_callback import MyPrintingCallback, GenerateCallback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import time

from skimage import io
from torchvision.transforms import transforms
import numpy as np

scale_y = np.array([10, 10, 2, 1, 1, 0.2])

class BaseForceSensor:
    def __init__(self, svae_checkpoint, device) -> None:

        # initialise svae
        self.model = VisualForceVAE(latent_dim = 32,VAE_weight=1)
        self.model = self.model.load_from_checkpoint(svae_checkpoint).to(device).eval()

        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize([224, 224]),
            ])
        self.device = device
        self.template9 = np.load('/home/fang/Documents/Track-Anything/cam9_template.npy')[:,140:500]/255
        self.template10 = np.load('/home/fang/Documents/Track-Anything/cam10_template.npy')[:,140:500]/255

    def measure(self, mask, finger): # frame ~ or [360,360]
        if finger == 9:
            frame = np.stack([mask,self.template9],axis=2)
        else:
            frame = np.stack([mask,self.template10],axis=2)

        x = self.trans(frame).to(self.device)
        # to [1,1,224,224] batch,chanels,width,height
        y = self.model.predict_force(x.unsqueeze(0)).squeeze()
        return np.multiply(y.detach().cpu().numpy(),scale_y)

