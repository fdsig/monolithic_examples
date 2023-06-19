import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback

global args
parser = argparse.ArgumentParser(description='PyTorch Lightning CIFAR-10 Training with WandB')
parser.add_argument('--devices', type=int, default=4, help='Number of GPUs to use for distributed training')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_type', default='allow', type=str)
parser.add_argument('--wandb_run_id',type=str, default=None, help='unique run id from wandb run eg https://wandb.ai/demonstrations/lightning_logs/runs/vhmehnf9')
args = parser.parse_args()


# Define data augmentation and normalization transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and prepare CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define a simple convolutional neural network
class SimpleCNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer

class WandbInitCallback(Callback):
    ''' place holder for custom callback if you want this'''
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(args)
            ###

def main(args):
    model = SimpleCNN()

    # Create PyTorch Lightning DataModule
    class CIFAR10DataModule(pl.LightningDataModule):
        def __init__(self, train_dataset, test_dataset, batch_size=128, num_workers=4):
            super().__init__()
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            self.batch_size = batch_size
            self.num_workers = num_workers

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    data_module = CIFAR10DataModule(train_dataset, test_dataset)

   # Create WandB logger for only 0th rank one run for all ranks
        
    if not args.resume:
        wandb_logger = WandbLogger(log_model=True)
    else:
        if args.resume_type=='allow':
            _id = wandb.util.generate_id()
            wandb_logger = WandbLogger(log_model=True,resume=args.resume_type,id=_id)
            wandb_logger.watch(model, log_freq=500)
        else:
            wandb_logger = WandbLogger(log_model=True,resume=args.resume_type,id=args.wandb_run_id)
            wandb_logger.watch(model, log_freq=500)
            
        

    # Initialize the wandb callback
    wandb_init_callback = WandbInitCallback()

    # Create PyTorch Lightning Trainer for distributed training
    trainer = Trainer(accelerator="gpu", devices=args.devices, strategy="ddp", logger=wandb_logger, max_epochs=5, callbacks=[wandb_init_callback])
    
    # Close the wandb run if initialized on rank 0
    trainer.fit(model, datamodule=data_module)
    
    if trainer.global_rank == 0:
        wandb.finish()

if __name__ == '__main__':
    main(args)


