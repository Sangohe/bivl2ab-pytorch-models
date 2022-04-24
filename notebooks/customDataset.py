import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os
from glob import glob

#Dataset directory reading
dset_dir = "../smooth_pursuit"

#-----------------------------------------------
############## KNOWING OUR DATASET ##############

label_map = {0: "Control", 1: "Parkinson"}

def read_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_label_from_path(path):
    return 0 if "C" in sample_path.split("/")[-2] else 1

def plot_img(img, label=None):
    plt.imshow(img)
    if label is not None: plt.title(label_map[label])

sample_path = "../smooth_pursuit/C00_0_0/00001.jpg"

#plotting image
img = read_image(sample_path)
label = get_label_from_path(sample_path)
plt.imshow(img)
plt.title(label_map[label])

#------------------------------------------------
############## DATA LOADER ##############
from typing import List, Optional, Any

class EyesDataset(Dataset):
    
    def __init__(
        self, 
        patient_dirs: List[str],
        transform: Optional[Any] = None
    ):
        
        self.patient_dirs = patient_dirs
        self.transform = transform

        # Create a list with all the frame paths.
        self.frame_paths = []
        for patient_dir in patient_dirs:
            patient_frame_paths = [os.path.join(patient_dir, p) for p in os.listdir(patient_dir)]
            self.frame_paths.extend(patient_frame_paths)
    
    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        curr_path = self.frame_paths[idx]
        img = read_image(curr_path)
        label = get_label_from_path(curr_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
#----------------------------------------
############### DATA LOADER FOR ALL PATHS ##############
patient_dirs = [os.path.join(dset_dir, p) for p in os.listdir(dset_dir)]
patient_dirs = sorted([p for p in patient_dirs if os.path.isdir(p)])
dset = EyesDataset(patient_dirs)

#Dataset behaviour
dataset_loader = DataLoader(dset, batch_size=4, shuffle=True)

for batch in dataset_loader:
    images_batch, labels_batch = batch
    break

print(type(images_batch), images_batch.shape, type(labels_batch), labels_batch.shape)

#----------------------------------------
############### CREATING A DATALOADER FOR TRAIN AND TEST SPLISTS ##############
batch_size = 4

target_size = (224, 224)
train_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize(target_size), 
    transforms.GaussianBlur((3, 3))
])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize(target_size)]
)

# Leave one patient out, leave the first patient for test.
test_patients = patient_dirs[:1]
train_patients = patient_dirs[1:]

# Create the datasets.
train_dset = EyesDataset(train_patients, transform=train_transform)
test_dset = EyesDataset(test_patients, transform=test_transform)

# Create the dataloaders.
train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)

#----------------------------------------
############### CHECKING IF THE DATASET IS BALANCED ##############
patient_dirs = [os.path.join(dset_dir, p) for p in os.listdir(dset_dir)]
patient_dirs = sorted([p for p in patient_dirs if os.path.isdir(p)])

# How many left and right eyes videos are in the dset.
left_eyes_counter, right_eyes_counter = 0, 0
for patient_dir in patient_dirs:
    if patient_dir[-1] == "0":
        left_eyes_counter += 1
    elif patient_dir[-1] == "1":
        right_eyes_counter += 1
print(f"Left eye videos: {left_eyes_counter}, Right eye videos: {right_eyes_counter}")

# How many frames per right and left video.
left_eyes_frame_counter, right_eyes_frame_counter = 0, 0
for patient_dir in patient_dirs:
    if patient_dir[-1] == "0":
        left_eyes_frame_counter += len(os.listdir(patient_dir))
    elif patient_dir[-1] == "1":
        right_eyes_frame_counter += len(os.listdir(patient_dir))
print(f"Left eye frames: {left_eyes_frame_counter}, Right eye frames: {right_eyes_frame_counter}")

# How many frames per contrl and pd video.
control_frame_counter, pd_frame_counter = 0, 0
for patient_dir in patient_dirs:
    if patient_dir.split("/")[-1][0] == "C":
        control_frame_counter += len(os.listdir(patient_dir))
    if patient_dir.split("/")[-1][0] == "P":
        pd_frame_counter += len(os.listdir(patient_dir))
print(f"Control frames: {control_frame_counter}, Parkinson frames: {pd_frame_counter}")
