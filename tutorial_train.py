from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks.base import Callback
import subprocess
from setup import conf_p
from config_loader import load_config


ckpt_save_path = '/export/data/msturm/CNet_deep_track'
gpu_train=7
gpu_samp=3
prompt='cell, microscopy, image'

class ExternalScriptCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        subprocess.call(["python", "val_sampling.py", "--ckpt_path", ckpt_save_path + '/last.ckpt', "--prompt", prompt, "--epoch", str(epoch),"--gpu",str(gpu_samp)])



# Define a ModelCheckpoint callback.
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_save_path,
    save_weights_only=True,  # default is False, change to True if you only want to save model weights
    verbose=True,
    filename='last-{epoch:02d}',
    save_last=True
    #every_n_epochs=1  # if you want to ensure that the last model is always saved
)



logger_freq = 300
logger = ImageLogger(batch_frequency=logger_freq)
callbacks =[logger, checkpoint_callback] #[logger, checkpoint_callback, ExternalScriptCallback()]


def main():
    # Configs
    resume_path = '/export/data/msturm/CNet_deep_track/last-epoch=42.ckpt' #./models/control_sd15_cell.ckpt'#/export/data/msturm/CNet_deep/last.ckpt' #'/export/data/msturm/CNet_deep_track/last.ckpt'  

    learning_rate = 5e-6 #5e-6
    sd_locked = False
    only_mid_control = False

    wandb_logger = WandbLogger(name="ControlNet", project="CNet_cells_track",save_dir='./wandb_logs')

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    #dataset = MyDataset('CNet_cells_track')
    #dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    
    trainer = pl.Trainer(
    gpus=[gpu_train], 
    precision=32, 
    callbacks=callbacks,
    min_steps=150000, 
    min_epochs=0,
    logger=wandb_logger )# Set Weights & Biases logger here
#)#,accumulate_grad_batches=8)

    # Train!
    #trainer.fit(model, dataloader)
    trainer.fit(model)

if __name__ == '__main__':
    main()
