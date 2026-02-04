from pathlib import Path
import torch
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from abcde.data import GraphDataModule
from abcde.models import ABCDE
from abcde.util import fix_random_seed, ExperimentSetup

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--name", default="vanilla_abcde", help="Experiment name")
args = parser.parse_args()

fix_random_seed(args.seed)

# Disable create_latest to prevent concurrent job crashes
experiment = ExperimentSetup(
    name=f"{args.name}_{args.seed}", 
    create_latest=False, 
    long_description="ABCDE Training"
)
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    loggers = [
        CSVLogger(experiment.log_dir, name='history'),
    ]
    model = ABCDE(nb_gcn_cycles=(4, 4, 6, 6, 8, 8),
                  conv_sizes=(48, 48, 32, 32, 24, 24),
                  drops=(0.3, 0.3, 0.2, 0.2, 0.1, 0.1),
                  lr_reduce_patience=2, dropout=0.1)
    
    # Passed args.seed to GraphDataModule to prevent cache collisions during concurrent execution
    data = GraphDataModule(min_nodes=4000, max_nodes=5000, nb_train_graphs=160, nb_valid_graphs=240,
                           batch_size=16, graph_type='powerlaw', repeats=8, regenerate_epoch_interval=10,
                           cache_dir=Path('datasets') / 'cache',
                           seed=args.seed)
    
    trainer = Trainer(
        logger=loggers, 
        gradient_clip_val=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=50, 
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            EarlyStopping(monitor='val_kendal', patience=5, verbose=True, mode='max'),
            ModelCheckpoint(dirpath=experiment.model_save_path, filename='model-{epoch:02d}-{val_kendal:.2f}', monitor='val_kendal', save_top_k=5, verbose=True, mode='max'),
            LearningRateMonitor(logging_interval='epoch'),
        ]
    )
    trainer.fit(model, datamodule=data)
    print(trainer.callback_metrics)