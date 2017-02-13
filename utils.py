import scipy.misc
import numpy as np
import pprint

pp = pprint.PrettyPrinter()


def center_crop(x, crop_h, crop_w, resize_h, resize_w):
    h, w = x.shape[:2]
    j = int(round(h - crop_h)/2.)
    i = int(round(w - crop_w)/2.)
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height, resize_width,
              is_crop=True):
    if is_crop:
        transform_image = center_crop(image, input_height, input_width,
                                      resize_height, resize_width)
    else:
        transform_image = scipy.misc.imresize(image,
                                              [resize_height, resize_width])
    return np.array(transform_image)/127.5 - 1


def inverse_transform(images):
    return (images + 1) * 127.5


def get_image(image_path, resize_height, resize_width, is_crop, is_grayscale):
    if is_grayscale:
        image = scipy.misc.imread(image_path, 'L').astype(np.float)
    else:
        image = scipy.misc.imread(image_path).astype(np.float)
    return transform(image, image.shape[0], image.shape[1],
                     resize_height, resize_width, is_crop)


def get_images(path, data_file, idxs, resize_height, resize_width, is_crop):
    data = []
    with open(data_file, 'r') as fin:
        lines = fin.readlines()
        for idx in idxs:
            line = lines[idx]
            pair = line.split(' ')
            # Assumes that image file extension is 4 characters long
            image1 = get_image(path + pair[0], resize_height, resize_width,
                               True, False)
            image1_x = get_image(path + pair[0][:-4] + '_x' + pair[0][-4:],
                                 resize_height, resize_width, True, True)
            image1_y = get_image(path + pair[0][:-4] + '_y' + pair[0][-4:],
                                 resize_height, resize_width, True, True)
            image2 = get_image(path + pair[1], resize_height, resize_width,
                               True, False)
            image2_x = get_image(path + pair[1][:-4] + '_x' + pair[1][-4:],
                                 resize_height, resize_width, True, True)
            image2_y = get_image(path + pair[1][:-4] + '_y' + pair[1][-4:],
                                 resize_height, resize_width, True, True)
            data.append(np.concatenate(image1, image1_x, image1_y, image2,
                                       image2_x, image2_y), axis=2)

    # Returns a 4D tensor of shape (idxs, height, width, 10)
    return np.array(data).astype(np.float32)
