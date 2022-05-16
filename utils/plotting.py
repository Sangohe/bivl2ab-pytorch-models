import matplotlib.pyplot as plt

label_map = {0: "Control", 1: "Parkinson"}


def plot_img(img, label=None):
    plt.imshow(img)
    if label is not None:
        plt.title(label_map[label])
