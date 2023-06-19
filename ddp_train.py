# Import necessary modules
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from typing import Tuple

# Simple Convolutional Neural Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define the layers
        # Model is split across two GPUs
        self.conv1 = nn.Conv2d(3, 6, 5).to('cuda:0')  
        self.pool = nn.MaxPool2d(2, 2).to('cuda:0')
        self.conv2 = nn.Conv2d(6, 16, 5).to('cuda:1')
        self.fc1 = nn.Linear(16 * 5 * 5, 120).to('cuda:1')
        self.fc2 = nn.Linear(120, 84).to('cuda:1')
        self.fc3 = nn.Linear(84, 10).to('cuda:1')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to calculate accuracy
def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Function to get the dataloader
def get_dataloader(rank: int, world_size: int) -> DataLoader:
    # Define the transformation
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the CIFAR10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    # Create a DistributedSampler to split the dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank
    )

    # Create the DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=False, num_workers=2, sampler=train_sampler)

    return trainloader

# Main function for training
def main(rank: int, world_size: int, log_all_ranks: bool, num_epochs: int, lr: float) -> None:
    # Initialize the distributed environment
    dist.init_process_group(
        "gloo",
        rank=rank,
        init_method='tcp://localhost:12355',
        world_size=world_size
    )

    # Initialize the model
    model = SimpleCNN()
    model = model.to(rank)
    # Make the model distributed
    model = DistributedDataParallel(model, device_ids=[rank])

    # Initialize wandb run
    if log_all_ranks:
        group_name = 'ddp_model_all_ranks' 
        run = wandb.init(project='model_distributed', entity='demonstrations', group=group_name)
        run.watch(model)
    elif rank==0 and not log_all_ranks:
        group_name = 'ddp_model_0_rank'
        run = wandb.init(project='model_distributed', entity='demonstrations', group=group_name)
        run.watch(model) 
    else:
        run = None

    # Get the dataloader
    trainloader = get_dataloader(rank, world_size)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs): 
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            # Move inputs and labels to the GPU
            inputs, labels = data[0].to(rank), data[1].to(rank)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Compute accuracy
            acc = accuracy(outputs, labels)

            # Log granular loss and accuracy if rank is 0
            if run:
                run.log({'granular_loss': loss, 'granular_acc': acc})
        
            # Perform optimization step
            optimizer.step()

            # Update running loss and accuracy
            running_loss += loss.item()
            running_acc += acc

            # Log average loss and accuracy every 2000 mini-batches
            if i % 2000 == 1999:
                avg_loss = running_loss / 2000
                avg_acc = running_acc / 2000
                print(f'Rank {rank}, Epoch {epoch+1}, Step {i+1}, Loss {avg_loss}, Accuracy {avg_acc}')
                if rank == 0:
                    run.log({
                        'avg_loss': avg_loss,
                        'avg_accuracy': avg_acc,
                        'epoch': epoch,
                        'step': i
                    })
                    running_loss = 0.0
                    running_acc = 0.0

    # Finish wandb run if rank is 0
    if run:
        run.finish()

    print('Finished Training')

    # Cleanup the distributed environment
    dist.destroy_process_group()
    
# Function to run the main function
def run(rank: int, size: int, log_all_ranks: bool, num_epochs: int, lr: float) -> None:
    print(f"Running main() on rank {rank}.")
    main(rank, size, log_all_ranks, num_epochs, lr)
    print("End of main()")

# Function to start processes
def start_processes(log_all_ranks: bool, num_epochs: int, lr: float) -> None:
    wandb.setup()
    size = 4
    processes = []
    for rank in range(size):
        p = torch.multiprocessing.Process(target=run, args=(rank, size, log_all_ranks, num_epochs, lr))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

# If this file is being run as the main program, start the processes
# If this file is being run as the main program, start the processes
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument('--log-all-ranks', action='store_true', help='Log on all ranks.')
    parser.add_argument('--epochs', default=2, type=int, help='Number of epochs to train.')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate for the optimizer.')
    
    args = parser.parse_args()

    wandb.login()
    start_processes(args.log_all_ranks, args.epochs, args.lr)


