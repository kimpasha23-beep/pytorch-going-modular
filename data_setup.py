
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

Num_workers = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transforms: transforms.Compose,
    batch_size: int,
    num_workers=Num_workers):
  
  train_data = datasets.ImageFolder(train_dir, transforms)
  test_data = datasets.ImageFolder(test_dir, transforms)

  class_name = train_data.classes

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
  )

  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
  )

  return train_dataloader, test_dataloader, class_name

