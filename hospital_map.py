from image_modefy import get_binary_map, rgb_constraint
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

map_size = [200, 200]
hospital_map_image = Image.open('snu_hospital.png')
hospital_map_image_red, image_name = \
    rgb_constraint([100, 0, 0], [255, 40, 40],
                   hospital_map_image=hospital_map_image)

obstacle, hospital_map_image_bin = get_binary_map(image_name, map_size)

plt.scatter(obstacle[0], obstacle[1], s=np.pi*0.1)
plt.show()
