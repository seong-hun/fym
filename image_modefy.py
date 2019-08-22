import cv2
import numpy as np


def rgb_constraint(lb=[0, 0, 0], ub=[255, 255, 255],
                                 let=(255, 255, 255, 0), **kwargs):
    """
    rgb_constraint is used for getting image editted RGB constraints.

    Parameters
    ----------
    lb : list
        ``lb`` is lower bound of RGB. It has three components which represent
        lower bound of red, blue, green respectively. Default is [0, 0, 0].
    ub : list
        ``ub`` is upper bound of RGB. It has three components which represent
        upper bound of red, blue, green respectively.
        Default is [255, 255, 255].
    let : tuple
        ``let`` is what will be RGBA value of image's pixels if the RGB of
        pixels don't satisfy RGB constraints. Default is (255, 255, 255, 0)
        which means white.
    kwargs : image
        ``kwargs`` is image loaded like Imgge.open('image.png').
        Keyward will be name of image saved with '_rgb_ediited'.

    Returns
    -------
    new_image : image
        ``new_image`` is the image editted according to RGB constraints.
    """
    new_image = []
    for name, value in kwargs.items():
        img = value
        data = img.getdata()
        newData = []
        for item in data:
            if lb[0] <= item[0] <= ub[0] and \
                    lb[1] <= item[1] <= ub[1] and \
                    lb[1] <= item[1] <= ub[1]:
                newData.append(item)
            else:
                newData.append(let)

        img.putdata(newData)
        img.save(f"{name}_rgb_editted.png", "PNG")
        new_image.append(img)
    return new_image, f'{name}_rgb_editted.png'


def get_binary_map(img_name, map_size):
    """
    get_binary_map is used for getting binary map and obstacle's coordinate.

    Parameters
    ----------
    img_name : str
        ``img_name`` is name of file what we want to get binary map
        and obstacle's cooridnate.

    map_size : list
        ``map_size`` is size of map

    Returns
    -------
    obstacle_coordinate : ndarray
        ``np.array(obstacle_coordinate)`` is ndarray of obstacle's coordinate.

    bw_img : image
        ``bw_img`` is the image converted from ``img`` to binary image.
    """
    img = cv2.imread(img_name, 2)
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    column_length = bw_img.shape[0]
    row_length = bw_img.shape[1]
    i, j = np.where(bw_img == 0)

    row_size = map_size[0]
    column_size = map_size[1]

    d_row = row_size / row_length
    d_column = column_size / column_length

    obstacle_x = []
    obstacle_y = []
    for i in np.arange(column_length):
            for j in np.arange(row_length):
                if bw_img[i][j] == 0:
                    obstacle_x.append(j * d_row)
                    obstacle_y.append((column_length - i) * d_column)
    obstacle_coordinate = np.array([obstacle_x, obstacle_y])
    return obstacle_coordinate, bw_img
