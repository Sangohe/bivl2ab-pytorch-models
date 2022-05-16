import cv2

label_map = {0: "Control", 1: "Parkinson"}
laterality_map = {0: "Right", 1: "Left"}


def read_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_label_from_path(path):
    return 0 if "C" in path.split("/")[-2] else 1


def get_patient_id_from_path(path):
    return path.split("/")[-2][:3]


def get_eye_laterality_from_path(path):
    return path.split("/")[-2][-1]
