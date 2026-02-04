
import os
from pytorch_lightning.loggers import CSVLogger

class TransientLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        # Force log directory to the environment variable (or current dir)
        # Force name/version to empty to keep structure flat: {TEMP_DIR}/checkpoints/
        target_dir = os.environ.get('ABCDE_LOG_DIR', '.')
        super().__init__(save_dir=target_dir, name='output', version='')

# Alias classes to our Shim
WandbLogger = TransientLogger
TensorBoardLogger = TransientLogger


import os
# from pytorch_lightning.loggers import CSVLogger

class TransientLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        # Force log directory to the environment variable (or current dir)
        # Force name/version to empty to keep structure flat: {TEMP_DIR}/checkpoints/
        target_dir = os.environ.get('ABCDE_LOG_DIR', '.')
        super().__init__(save_dir=target_dir, name='output', version='')

# Alias classes to our Shim
WandbLogger = TransientLogger
TensorBoardLogger = TransientLogger


import os
# # from pytorch_lightning.loggers import CSVLogger

class TransientLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        # Force log directory to the environment variable (or current dir)
        # Force name/version to empty to keep structure flat: {TEMP_DIR}/checkpoints/
        target_dir = os.environ.get('ABCDE_LOG_DIR', '.')
        super().__init__(save_dir=target_dir, name='output', version='')

# Alias classes to our Shim
WandbLogger = TransientLogger
TensorBoardLogger = TransientLogger


import os
# # # from pytorch_lightning.loggers import CSVLogger

class TransientLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        # Force log directory to the environment variable (or current dir)
        # Force name/version to empty to keep structure flat: {TEMP_DIR}/checkpoints/
        target_dir = os.environ.get('ABCDE_LOG_DIR', '.')
        super().__init__(save_dir=target_dir, name='output', version='')

# Alias classes to our Shim
WandbLogger = TransientLogger
TensorBoardLogger = TransientLogger


import os
# # # # from pytorch_lightning.loggers import CSVLogger

class TransientLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        # Force log directory to the environment variable (or current dir)
        # Force name/version to empty to keep structure flat: {TEMP_DIR}/checkpoints/
        target_dir = os.environ.get('ABCDE_LOG_DIR', '.')
        super().__init__(save_dir=target_dir, name='output', version='')

# Alias classes to our Shim
WandbLogger = TransientLogger
TensorBoardLogger = TransientLogger


import os
# # # # # from pytorch_lightning.loggers import CSVLogger

class TransientLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        # Force log directory to the environment variable (or current dir)
        # Force name/version to empty to keep structure flat: {TEMP_DIR}/checkpoints/
        target_dir = os.environ.get('ABCDE_LOG_DIR', '.')
        super().__init__(save_dir=target_dir, name='output', version='')

# Alias classes to our Shim
WandbLogger = TransientLogger
TensorBoardLogger = TransientLogger


import os
# # # # # # from pytorch_lightning.loggers import CSVLogger

class TransientLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        # Force log directory to the environment variable (or current dir)
        # Force name/version to empty to keep structure flat: {TEMP_DIR}/checkpoints/
        target_dir = os.environ.get('ABCDE_LOG_DIR', '.')
        super().__init__(save_dir=target_dir, name='output', version='')

# Alias classes to our Shim
WandbLogger = TransientLogger
TensorBoardLogger = TransientLogger

from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
# # # # # # # from pytorch_lightning.loggers import CSVLogger, CSVLogger, CSVLogger

from abcde.data import GraphDataModule
from abcde.models import ABCDE
from abcde.util import fix_random_seed, ExperimentSetup


# Fix the seed for reproducibility
fix_random_seed(42)
experiment = ExperimentSetup(name='vanilla_abcde', create_latest=True, long_description="""
Use PReLU activation
Use Adam optimizer with big learning rate
Try to have variable number of edges in the generated graphs
Try dropping edges while training
Graphs are only of 'powerlaw' type.
Use unique convolutions.
Use blocks of convolutions followed with max pooling and skip connections
Use gradient clipping
""")
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    loggers = [
        CSVLogger(experiment.log_dir, name='history'),
        CSVLogger(experiment.log_dir, name=experiment.name),
        CSVLogger(save_dir="experiments", name="abcde_logs"),
        # AimLogger(experiment=experiment.name),
    ]
    model = ABCDE(nb_gcn_cycles=(4, 4, 6, 6, 8, 8),
                  conv_sizes=(48, 48, 32, 32, 24, 24),
                  drops=(0.3, 0.3, 0.2, 0.2, 0.1, 0.1),
                  lr_reduce_patience=2, dropout=0.1)
    data = GraphDataModule(min_nodes=4000, max_nodes=5000, nb_train_graphs=160, nb_valid_graphs=240,
                           batch_size=16, graph_type='powerlaw', repeats=8, regenerate_epoch_interval=10,
                           cache_dir=Path('datasets') / 'cache')
    trainer = Trainer(logger=loggers, gradient_clip_val=1,
                      accelerator='gpu', devices=-1 if torch.cuda.is_available() else None, 
                      max_epochs=50,  callbacks=[
                          EarlyStopping(monitor='val_kendal', patience=5, verbose=True, mode='max'),
                          ModelCheckpoint(dirpath=experiment.model_save_path, filename='model-{epoch:02d}-{val_kendal:.2f}', monitor='val_kendal', save_top_k=5, verbose=True, mode='max'),
                          LearningRateMonitor(logging_interval='epoch'),
                      ])
    trainer.fit(model, datamodule=data)
    print(trainer.callback_metrics)
