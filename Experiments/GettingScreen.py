import pyscreeze as ps
import numpy as np
import matplotlib.pyplot as plt

imageGrayscale = (np.array(ps.screenshot().convert("L").getdata())-128)/128
image = imageGrayscale.reshape(768, 1366)

plt.imshow(image)
plt.show()
