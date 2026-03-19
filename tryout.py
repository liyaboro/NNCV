from PIL import Image
import numpy as np

img = Image.open("local_output/aachen_000006_000019_leftImg8bit.png")
arr = np.array(img)

print(arr.min(), arr.max(), np.unique(arr))