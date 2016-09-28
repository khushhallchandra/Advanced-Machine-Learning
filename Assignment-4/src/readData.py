from skimage import io 
import os

image = io.imread()

for file in os.listdir("../train"):
    if file.endswith(".png"):
        image = io.imread(file)
