import numpy as np
import matplotlib.pyplot as plt
import os


def load_mnist_images(filename):
    with open(filename, "rb") as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)

    return data.reshape(-1, 28, 28)


file_path = "imnist_dataset.idx3-ubyte"

save_dir = "./examples/"
os.makedirs(save_dir, exist_ok=True)

images = load_mnist_images(file_path)

for i in range(100):
    plt.imsave(os.path.join(save_dir, f"image_{i + 1:003}.png"), images[i], cmap="gray")
