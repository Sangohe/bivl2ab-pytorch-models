from torch.utils.data import Dataset

import os
from typing import List, Optional, Any

from utils import read_image, get_label_from_path, get_patient_id_from_path, get_eye_laterality_from_path

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
        patient_id = get_patient_id_from_path(curr_path)
        eye_laterality = get_eye_laterality_from_path(curr_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, patient_id, eye_laterality