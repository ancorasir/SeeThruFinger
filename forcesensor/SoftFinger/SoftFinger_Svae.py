import torch
import pytorch_lightning as pl
from torchvision import models
from src.datamodules.SoftFinger_datamodule import SoftFingerDataModule

from src.models import VisualForceVAE
from src.callbacks.printing_callback import MyPrintingCallback, GenerateCallback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np

if __name__ == "__main__":
    # SVAE Learning for Soft Finger
    
    ## Training Data
    dm = SoftFingerDataModule(data_dir=['/home/fang/Documents/Track-Anything/forcesensor/train-2-0524-112734',
                                        '/home/fang/Documents/Track-Anything/forcesensor/train-3-0525-193122',
                                        '/home/fang/Documents/Track-Anything/forcesensor/train-2-cam10-0531-154909',
                                        '/home/fang/Documents/Track-Anything/forcesensor/train-3-cam10-0531-162540/',

                                        ], 
                            train_test_split=(1, 0), num_workers = 8, batch_size = 64,)
    dm.setup()

    ## Training Pipeline
    trainer = pl.Trainer(max_epochs = 50,gpus = [0],callbacks=[ModelCheckpoint(
        save_weights_only=True,),
        LearningRateMonitor("epoch")],)

   
    ## Model Parameter Explanation: 
    #  kl_coeff, coresponds to beta in the paper
    #  VAE_weight, coresponds to alpha in the paper
    #  latent_dim, defines the dimension of the latent space
    model = VisualForceVAE(kl_coeff=0.1, VAE_weight = 1, latent_dim = 32)
    
    ## Model Training
    trainer.fit(model, dm)

    ## Model Evaluation
    model.eval()        
    trainer.test(model,dm)