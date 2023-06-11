from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict




# Define a ModelCheckpoint callback.
checkpoint_callback = ModelCheckpoint(
    dirpath='/export/data/msturm/CNet',
    save_weights_only=True,  # default is False, change to True if you only want to save model weights
    verbose=True,
    save_last=True,  # if you want to ensure that the last model is always saved
)

def main():
    # Configs
    resume_path = './models/control_sd15_cell.ckpt'
    batch_size = 4
    logger_freq = 300
    learning_rate = 2e-6
    sd_locked = False
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset = MyDataset('CNet_cells')
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=[1], precision=32, callbacks=[logger, checkpoint_callback],min_steps=150000, min_epochs=0)#,accumulate_grad_batches=8)

    # Train!
    #trainer.fit(model, dataloader)
    trainer.fit(model)

if __name__ == '__main__':
    main()
