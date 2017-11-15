from scipy.signal import convolve2d
from PIL import Image, ImageTk
import numpy as np
import cv2

"""
Given the location of the optimal (low energy) seam, 
insert a new seam into the input image.
"""
def insert(image, seam):
    h,w,d = image.shape
    new_image = np.zeros((h,w+1,d), dtype=image.dtype)
    for i in range(h):
        for c in range(d):
            x = seam[i]
            left = max(x-1, 0)
            right = min(x+1, w-1)
            rest = min(right+1, w-1)
            avg = np.floor(np.average(image[i,left:right,c]))
      
            if x == w-1: # if seam is in last column
                new_image[i,:x,c] = image[i,:x,c]
                new_image[i,x,c] = avg
                new_image[i,right,c] = image[i,x,c]
            else:
                new_image[i,:right,c] = image[i,:right,c]
                new_image[i,right,c] = avg
                new_image[i,rest:,c] = image[i,right:,c]

    return new_image

"""
Insert given value into 2D index matrix
"""
def insert_val(idx, seam, val):
    h,w = idx.shape
    new_idx = np.zeros((h,w+1), dtype=idx.dtype)

    for i in range(h):
        x = seam[i]
        left = max(x-1, 0)
        right = min(x+1, w-1)
        rest = min(right+1, w-1)
  
        if x == w-1: # if seam is in last column
            new_idx[i,:x] = idx[i,:x]
            new_idx[i,x] = -1
            new_idx[i,right] = idx[i,x]
        else:
            new_idx[i,:right] = idx[i,:right]
            new_idx[i,right] = val
            new_idx[i,rest:] = idx[i,right:]

    return new_idx 

"""
Remove a single seam from an input image.
"""
def carve(image, seam):
    mask = np.ones_like(image, dtype=np.bool)
    
    if len(image.shape) == 2:
        h,w = image.shape
        mask[np.arange(h),seam] = False
        return image[mask].reshape(h,w-1)
    
    h,w,d = image.shape
    mask[np.arange(h),seam,:] = False
    return image[mask].reshape(h,w-1,d)

"""
Highlight seams on an input image.
"""
def highlight(image, seams, energy=False):
    if energy:
        image = image.astype(np.float64)
        image /= np.amax(image)

    h,w = image.shape[:2]
  
    for seam in seams:
        if len(image.shape) == 2:
            image[np.arange(image.shape[0]),seam] = 255
        else: 
            if np.amax(seam) >= w: seam[seam >= w] = w-1
            image[np.arange(h),seam,:] = (0,0,255)
  
    return image

"""
Helper function to transpose an image
"""
def transpose(image):
    if len(image.shape) == 2: return image.T
    return np.transpose(image, (1,0,2))

"""
Convolve a kernel over each color channel of an image
"""
def conv(image, kernel):
    b, g, r = cv2.split(image)
    return np.absolute(convolve2d(b, kernel, 'same')) + \
          np.absolute(convolve2d(g, kernel, 'same')) + \
          np.absolute(convolve2d(r, kernel, 'same'))

"""
Find dimensions of a bit mask filled with 0s and 1s
"""
def get_mask_dimensions(mask):
    h,w = mask.shape
    mn_x,mn_y = w,h
    mx_x,mx_y = 0,0
    for i in range(h):
        for j in range(w):
            if mask[i,j] == 1: 
                mx_x = max(mx_x, j)
                mn_x = min(mn_x, j)
                mx_y = max(mx_y, i)
                mn_y = min(mn_y, i)
    return (mx_y-mn_y, mx_x-mn_x)

"""
Given an input text area size, finds optimal cv2 text size, creates new text area
"""
def get_text_area(text, size, dtype, font=cv2.FONT_HERSHEY_SIMPLEX):
    h,w,d = 0,0,0
    text_area = None
    if len(size) == 2: 
        h,w = size
        text_area = np.ones((h,w), dtype=dtype) * 255
    else:
        h,w,d = size
        text_area = np.ones((h,w,d), dtype=dtype) * 255

    # first determine a text size that fits
    sz = 1
    text_size = cv2.getTextSize(text, font, sz, 1)[0]
    while(text_size[0] >= w):
        sz -= 0.05
        text_size = cv2.getTextSize(text, font, sz, 1)[0]

    # now put text on image
    t_x = int((w-text_size[0])/2)
    t_y = int((h+text_size[1])/2)
    cv2.putText(text_area, text, (t_x,t_y), font,sz, (0,0,0), 1, cv2.LINE_AA)
    
    return text_area

"""
Singleton pattern from: 
https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons
"""
class Singleton:
    def __init__(self, decorated):
        self._decorated = decorated
    def Instance(self):    
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
        return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)

"""
Custom singleton printer class
"""
@Singleton
class Printer(object):
    def __init__(self):
        self.on = True
  
    def turnOn(self):
        self.on = True

    def turnOff(self):
        self.on = False

    def _print_(self, s, end='\n'):
        if self.on: print(s, end=end)

"""
Load and process an input image
"""
def process_image(imagepath):
    # Load image
    image = np.array(cv2.imread(imagepath), dtype=np.uint8)
    height, width, _ = image.shape

    original = Image.fromarray(image)

    while height > 500 or width > 800:
        width = int(width/2)
        height = int(height/2)
        original = original.resize((width, height), Image.BILINEAR)

    return (np.array(original), height, width)

"""
# determine order to resize in each direction
def get_resize_order(self, new_size):
    temp = image
    if image is None: 
        image = self.image.copy()
        temp = self.image.copy()

    h,w,_ = image.shape
    h1,w1,d = new_size
    r = h-h1
    c = w-w1
    T = np.zeros((r+1,c+1), dtype=np.int32)

    # get seam order bitmap
    for i in range(r+1):
        tmp = image.copy()
        prev = image.copy()
        carve_h, seams_h = self.carve_seams(image=transpose(tmp))
        e_h = self.energy(carve_h)
        cost_h = self.cost(e_h, seams_h[0])
        for j in range(c+1):
            if i != 0 or j != 0: 
                prev, seams_v = self.carve_seams(image=prev)
                e_v = self.energy(prev)
                cost_v = self.cost(e_v, seams_v[0])
                T[i,j] = min(T[i,j-1] + cost_v, T[i-1,j] + cost_h)
        
        image = transpose(carve_h)

    # restore unaltered image
    self.image = temp

    # backtrack to determine best order
    order = np.zeros((r+c), dtype=np.int32)
    y = r
    x = c
    count = 0

    while count < r + c:
        if x == 0: 
            order[-(count+1)] = 1
            y -= 1
        elif y == 0: 
            order[-(count+1)] = -1
            x -= 1
        else: 
            if T[y,x-1] < T[y-1,x]: 
                order[-(count+1)] = -1
                x -= 1
            else:
                order[-(count+1)] = 1
                y -=1
        count += 1

    return order, T
"""
