import numpy as np
import colorsys, cv2
from PIL import Image


class Processing(object):

	def colors(self, class_labels):
	    hsv_tuples = [(x / len(class_labels), 1., 1.) for x in range(len(class_labels))]
	    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	    class_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
	    np.random.seed(43)
	    np.random.shuffle(colors)
	    np.random.seed(None)
	    class_colors = np.tile(class_colors, (16, 1))
	    return class_colors


	def get_classes(self, classes_path):
	    with open(classes_path, encoding='utf-8') as f:
	        class_names = f.readlines()
	    class_names = [c.strip() for c in class_names]
	    return class_names, len(class_names)


	def preprocess_input(self, image):
	    image = image / 255.0
	    return image


	def cvtColor(self, image):
	    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
	        return image 
	    else:
	        image = image.convert('RGB')
	        return image


	def resize_image(self, image, size):
	    iw, ih  = image.size
	    w, h    = size
	    scale   = min(w/iw, h/ih)
	    nw      = int(iw*scale)
	    nh      = int(ih*scale)
	    image   = image.resize((nw,nh), Image.BICUBIC)
	    new_image = Image.new('RGB', size, (128,128,128))
	    new_image.paste(image, ((w-nw)//2, (h-nh)//2))    
	    return new_image
