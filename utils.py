from scipy.signal import convolve2d
from PIL import Image, ImageTk
from os import path
import numpy as np
import pickle
import cv2

############################################################
# Functions
############################################################

def backward_energy(energy):
    """
    Compute energy map of image using 'backward' energy

    energy: 2D energy image
    """
    h,w = energy.shape[:2]

    K = np.ones((h,w+2), dtype=energy.dtype) * 1e10
    K[:,1:-1] = energy

    E = np.ones_like(K) * 1e10
    E[0,1:-1] = energy[0,:]
    T = np.zeros((3,w), dtype=energy.dtype)

    for i in range(1,h):
        for j in range(3):
            T[j,:] = E[i-1,j:j+w]
        E[i,1:-1]  = K[i,1:-1] + np.amin(T, axis=0)

    return E[:,1:-1]
    
def carve(image, seam):
    """
    Remove a single seam from an input image.

    image: input image
    seam: indices of seam to remove in input image
    """
    mask = np.ones_like(image, dtype=np.bool)
    h,w = image.shape[:2]
    shape = (h,w-1)
    
    if len(image.shape) == 3:
        shape = (h,w-1,image.shape[2])

    mask[np.arange(h),seam] = False
    return image[mask].reshape(shape)

def carve_seams(image, n_seams=1, use_FE=False):
    """
    Remove seams from an image.

    image: input image
    n_seams: number of seams to carve
    use_FE: whether or not to use 'forward' energy
    """
    seams = []
    carved_image = image.copy()
    energy_map = None

    s_print_('Carving %d seams from image.' % n_seams)
    for n in range(n_seams):
        s_print_('Carving seam %d of %d.' % (n+1,n_seams), end='\r')    
        
        # compute e1 energy of image
        e1 = e1_energy(carved_image)
        
        # compute image energy map
        if use_FE: energy_map = forward_energy(carved_image)
        else: energy_map = backward_energy(e1)

        # find seam of minimum energy, append to list
        seam = find_seam(energy_map)
        seams.append(seam)

        # remove the seam from the image
        carved_image = carve(carved_image, seam)

    s_print_('\ndone.')
    
    return (carved_image, seams)

def center(mat, image):
    """
    Center an image in another image.
    
    mat: background image
    image: image to overlay in background image
    """
    mh,mw = mat.shape[:2]
    h,w = image.shape[:2]

    y = int((mh - h) / 2)
    x = int((mw - w) / 2)

    mat[y:y+h,x:x+w] = image
    return mat

def conv2(image, kernel, mode='same'):
    """
    Convolve a kernel over each color channel of an image, 
    and return the sum the results.

    image: 3D or 2D input image
    kernel: input kernel for convolution
    """
    if len(image.shape) == 2:
        return convolve2d(image, kernel, mode)

    C = conv3(image, kernel, mode)
    return np.sum([np.abs(C[:,:,i]) for i in range(3)], axis=0)

def conv3(image, kernel, mode='same'):
    """
    Convolve a kernel independently over each color 
    channel of an image.

    image: input image
    kernel: input kernel for convolution
    """
    b, g, r = cv2.split(image)
    cb = convolve2d(b, kernel, mode)
    cg = convolve2d(g, kernel, mode)
    cr = convolve2d(r, kernel, mode)
    return cv2.merge([cb, cg, cr])

def crop(image, n_pixels, side='both', horizontal=True):
    """
    Crop pixels from image.

    image: 3D input image
    n_pixels: number of pixels to crop
    side: side(s) from which to remove pixels
    horizontal: whether to crop horizontally (width) or vertically (height)
    """
    if side not in ['both', 'left', 'right']:
        error_str = 'Invalid entry for \'side\'. Should be either' \
                    '\'both\', \'left\', or \'right\'.'
        raise ValueError(error_str)
    
    if n_pixels == 1 and side == 'both':
        raise ValueError('Cannot remove 1 pixel from both sides if n_pixels = 1')
    
    # if cropping widthwise, transpose
    if horizontal: image = transpose(image)

    cropped = None
    if side == 'both':
        left    = int(n_pixels/2)
        right   = n_pixels-left
        cropped = image[left:-right,:,:]
    elif side == 'left':
        cropped = image[n_pixels:,:,:]
    else: 
        cropped = image[:-n_pixels,:,:]

    # if cropping widthwise, transpose
    if horizontal: cropped = transpose(cropped)

    return cropped

def e1_energy(image):
    """
    e1 energy function. computes sum of image x,y gradients

    image: input image
    """
    x_grad = np.array([[-3,0,3],[-10,0,10],[-3,0,3]])
    y_grad = np.array([[-3,-10,-3],[0,0,0],[3,10,3]])
    return conv2(image, x_grad) + conv2(image, y_grad)

def find_seam(energy_map):
    """
    Find a seam of minimum energy on an input energy map.

    energy_map: 'backward' or 'forward' 2D cumulative energy map
    """
    h,w = energy_map.shape

    # vertical seam with length = image height
    seam = np.zeros((h), dtype=np.int32)

    # backtrack from min of last row of energy map
    seam[-1] = np.argmin(energy_map[-1,:])
    for i in range(h-2,-1,-1):
        x       = seam[i+1]
        left    = int(max(x-1, 0))
        right   = int(min(x+1, w-1))
        idx     = [left, right, x]
        seam[i] = idx[np.argmin([energy_map[i,left], 
                                energy_map[i,right], 
                                energy_map[i,x]])]
    return seam

def forward_energy(gray, p=None):
    """
    Compute energy map of image using 'forward' energy.

    image: input image
    p: 2D energy of input image
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    h,w  = gray.shape
    if p is None: p = np.zeros((h,w))

    K = np.zeros((h,w+2), dtype=np.float64)
    K[:,0]  = 9e9
    K[:,-1] = 9e9

    M = np.pad(gray, ((0,0),(1,1)), 'edge').astype(np.float64)
    K[0,1:-1] = np.abs(M[0,2:]-M[0,:-2])
    T = np.zeros((w,3), dtype=np.float64)

    for i in range(1,h):
        left   = M[i,:-2]
        right  = M[i,2:]
        up     = M[i-1,1:-1]
        T[:,0] = K[i-1,:-2]  + np.abs(up-left) + np.abs(right-left)
        T[:,1] = K[i-1,1:-1] + np.abs(right-left)
        T[:,2] = K[i-1,2:]   + np.abs(up-right) + np.abs(right-left)
        K[i,1:-1] = np.amin(T,axis=1) + p[i,:]

    return K[:,1:-1]
    
def get_mask_dimensions(mask):
    """
    Find dimensions of a bit mask filled with 0s and 1s

    mask: 2D input mask
    """
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

    
def get_text_area(text, size, dtype, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Given an input text area size, find optimal cv2 text size, 
    and create new text area.

    text: text string
    size: tuple size of text area
    dtype: dtype of output text area image
    font: cv2 font type
    """
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
    
def highlight(image, seams, energy=False):
    """
    Highlight seams on an input image.

    image: 2D or 3D input image
    seams: list of seams to highlight
    energy: whether or not 'image' is an energy image.
            if so, then output image will be 2-dimensional
    """
    if energy:
        image = image.astype(np.float64)
        image /= np.amax(image)

    h,w = image.shape[:2]
  
    for seam in seams:
        if len(image.shape) == 2:
            image[np.arange(image.shape[0]),seam]  = 255
        else: 
            if np.amax(seam) >= w: seam[seam >= w] = w - 1
            image[np.arange(h),seam,:] = (255,0,0)
  
    return image

def image_to_patches(image, patch_sz):
    """
    Split an image into non-overlapping patches.

    image: input image
    patch_sz: size of patch
    """
    h,w = image.shape[:2]
    rows  = int(h / patch_sz)
    if h % patch_sz > 0: rows += 1

    cols = int(w / patch_sz)
    if w % patch_sz > 0: cols += 1

    patches = np.zeros((rows, cols, patch_sz, patch_sz))
    if len(image.shape) == 3: 
        patches = np.zeros((rows, cols, patch_sz, patch_sz, 3))

    for i in range(rows):
        m  = i * patch_sz
        m1 = patch_sz if m + patch_sz <= h else h - m
        for j in range(cols):
            n  = j * patch_sz
            n1 = patch_sz if n + patch_sz <= w else w - n
            patches[i,j,:m1,:n1] = image[m:m+m1,n:n+n1]

    if len(image.shape) == 3: 
        patches = patches.reshape(rows, cols, patch_sz, patch_sz, 3)
    else:
        patches = patches.reshape(rows, cols, patch_sz, patch_sz)

    return patches

def insert(image, seam, use_average=True):
    """
    Given the location of the optimal (low energy) seam, 
    insert a new seam into the input image.

    image: input image
    seam: indices of seam in input image
    use_average: whether or not to replace neighbor of
                 seam with average of its neighbors.
    """
    h,w = image.shape[:2]
    shape = (h, w+1)
    if len(image.shape) == 3:
        shape = (h, w+1, image.shape[2])

    new_image = np.zeros(shape, dtype=image.dtype)

    for i in range(h):
        x  = seam[i]
        xm = max(x-1,0)
        xp = min(x+1,w-1)        
        if use_average: # insert averages of two seams into image
            new_image[i,:x]    = image[i,:x]
            new_image[i,x]     = np.mean([image[i,xm],image[i,x]], axis=0)
            new_image[i,xp]    = np.mean([image[i,x],image[i,xp]], axis=0)
            new_image[i,xp+1:] = image[i,xp:]
        elif x == w-1: # insert seam left
            new_image[i,:x]    = image[i,:x]
            new_image[i,x]     = np.mean([image[i,xm],image[i,x]], axis=0)
            new_image[i,xp]   = image[i,x]
        else:# insert seam right
            new_image[i,:xp]     = image[i,:xp]
            new_image[i,xp]    = np.mean([image[i,x],image[i,xp]], axis=0)
            new_image[i,xp+1:] = image[i,xp:]
            
    return new_image

def insert_seams(image, n_seams=1, use_FE=False): 
    """
    Insert seams into an image.

    image: input image
    n_seams: number of seams to insert
    use_FE: whether or not to use 'forward' energy
    """
    temp = image.copy()
    _, seams = carve_seams(image, n_seams=n_seams, use_FE=use_FE)
    enlarged_image = temp

    s_print_('Inserting %d seams' % n_seams)
    for i in range(len(seams)):
        s_print_('inserting seam %d of %d' % (i+1,n_seams), end='\r')
        seam = seams.pop(0)
        enlarged_image = insert(enlarged_image, seam)
        seams = update_seams(seams, seam)

    s_print_('\ndone.')

    return (enlarged_image, seams)

def insert_val(image, seam, val):
    """
    Insert given value into a seam in an image

    image: input image
    seam: indices of seam in input image
    val: value to insert
    """
    h,w = image.shape[:2]
    shape = (h, w+1)
    if len(image.shape) == 3:
        shape = (h, w+1, image.shape[2])
    
    new_image = np.zeros(shape, dtype=image.dtype)

    for i in range(h):
        x = seam[i]
        new_image[i,:x]   = image[i,:x]
        new_image[i,x]    = val    
        new_image[i,x+1:] = image[i,x:]

    return new_image

def L1(A,B):
    """
    Sum of absolute values of pixel differences, aka L1-norm

    A: scalar or array
    B: scalar or array, same shape as 'A'
    """
    return np.sum(np.abs(A-B))

def L2(A,B):
    """
    Sum of squared differences (SSD) of pixel values, aka L2-norm^2.

    A: scalar or array
    B: scalar or array, same shape as 'A'
    """
    # grayscale, return squared pixel value difference
    if len(A.shape) == 0:
        return (A-B)**2
    # RGB, LAB tuples of pixel values for each channel
    else:
        diff  = (A-B).reshape(np.prod(A.shape))
        return np.linalg.norm(diff,2)**2

def load_and_process_image(imagepath, max_height=400, 
                            max_width=600, swap_channels=False):
    """
    Load and process an input image.

    imagepath: relative path to image file
    max_width: maximum desired width of output image
    max_height: maximum desired height of output image
    swap_channels: swap from BGR to RGB
    """
    # Read in image
    image = np.array(cv2.imread(imagepath), dtype=np.uint8)

    # swap from BGR to RGB if necessary
    if swap_channels:
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])

    h,w,d = image.shape

    if max_height == 0: max_height = 9e9
    if max_width == 0: max_width = 9e9

    # if height is too large...
    if h > max_height:
        ratio = max_height / h
        h     = max_height
        w     = int(ratio * w)
        image = scale(image, (w, h))

    # if width is too large
    if w > max_width:
        ratio = max_width / w
        w     = max_width
        h     = int(ratio * h)
        image = scale(image, (w, h))

    return image

def load_image_maps(image_name, with_FE=False):
    """
    Load multi-size image maps if they exist.

    image_name: name of image file.
    """
    # If maps with forward energy exist
    if with_FE:   
        filename = 'pics/image_maps/' + image_name + '_f.pkl'
        if path.exists(filename):
            with open(filename, 'rb') as image_maps:
                maps = pickle.load(image_maps)
                return filename, maps
    # Else if normal maps exist
    else:
        filename = 'pics/image_maps/' + image_name + '.pkl'
        if path.exists(filename):
            with open(filename, 'rb') as image_maps:
                maps = pickle.load(image_maps)
                return filename, maps

    # Otherwise, return None
    return None

def make_image_grid(images, spacing=16):
    """
    Create a grid displaying each image in a list of images. 

    images: a list of images with the same type and dimensions (2D or 3D)
    """
    n_images = len(images)
    c = int(np.sqrt(n_images))
    r = int(np.ceil(n_images / c))
    mxh = max([img.shape[0] for img in images]) + spacing
    mxw = max([img.shape[1] for img in images]) + spacing

    img   = images[0]
    msize = (mxh, mxw)
    h_size = (mxh, c*mxw)
    g_v   = np.zeros((0, c*mxw), dtype=img.dtype)
    g_h   = np.zeros(h_size, dtype=img.dtype)

    if len(img.shape) == 3:
        d = img.shape[2]
        msize = (mxh, mxw, d)
        h_size = (mxh, c*mxw, d)
        g_v   = np.zeros((0, c*mxw, d), dtype=img.dtype)
        g_h   = np.zeros(h_size, dtype=img.dtype)

    cnt = 0
    mat = np.ones(msize, dtype=img.dtype) * 255

    for i in range(r):
        x = 0
        for j in range(c):
            if cnt >= n_images: break
            centered = center(mat.copy(), images[cnt])
            g_h[:,x:x+mxw] = centered
            x   += mxw
            cnt += 1

        g_v = np.vstack((g_v, g_h))
        g_h = np.zeros(h_size, dtype=img.dtype)

    return g_v

def pre_process_multi(image, use_FE=False):
    """
    Pre-process an image for fast resizing.

    image: input image
    use_FE: whether or not to use 'forward' energy
    """
    h,w,_ = image.shape
    enlarged_image = image.copy()

    map_remove = np.zeros((h,w), dtype=np.int32)
    _, idx     = np.indices((h,w), dtype=np.int32)
    map_insert = idx.copy()

    if use_FE: s_print_('Pre-processing image with forward energy...')
    else: s_print_('Pre-processing image...')

    for i in range(w):
        s_print_('Seam %d of %d' % (i+1, w), end='\r')
        
        e1 = e1_energy(image)
        energy_map = forward_energy(image) if use_FE else backward_energy(e1)
        seam = find_seam(energy_map)
        
        image = carve(image, seam)
        map_remove[np.arange(h), idx[np.arange(h), seam]] = i
        idx = carve(idx, seam)

        enlarged_image = insert(enlarged_image, seam, use_average=False)
        map_insert = insert_val(map_insert, seam, -i-1)
        
    s_print_('\ndone.')
    
    return (map_remove, enlarged_image, map_insert)

def save_image_maps(image_name, maps, did_use_FE=False):
    """
    Save image maps for a multi-size image.

    image_name: name of image file
    maps: tuple containing 2D image removal map,
          enlarged image map for insertion, and 
          2D image insertion map.
    did_use_FE: whether or not 'forward' energy was used
    """
    (map_remove, enlarged_image, map_insert) = maps
    image_maps = {
        'map_remove' : map_remove, 
        'enlarged_image': enlarged_image, 
        'map_insert': map_insert
    }

    filename = 'pics/image_maps/' + image_name
    if did_use_FE: filename += '_f'
    filename += '.pkl'

    if not path.exists(filename):
        open(filename, 'wb').close()

    s_print_('Saving image maps to file \'%s\'.' % filename)

    with open(filename, 'wb') as map_file:
        pickle.dump(image_maps, map_file, protocol=0)

    s_print_('done.')
    
def scale(image, new_size):
    """
    Scale an image using bicubic interpolation.

    image: input image
    new_size: size of resized output image
    """
    original = Image.fromarray(image)
    resized  = original.resize(new_size, Image.BICUBIC)
    return np.array(resized, dtype=image.dtype)

def transpose(image):
    """
    Transpose an image.

    image: input image
    """
    if len(image.shape) == 2: return image.T
    return np.transpose(image, (1,0,2))

def update_seams(seams, current_seam):
    """
    Correct indices that changed during seam removal.

    seams: list of seams to update
    current_seam: current seam to check against
    """
    for i in range(len(seams)):
        seams[i][np.where(seams[i] >= current_seam)] += 2
    return seams

def s_print_(str, end='\n'):
    """
    Print using the 'Printer' singleton.

    str: string to print
    """
    Printer.Instance()._print_(str, end=end)

############################################################
# Classes
############################################################

class Singleton:
    """
    Singleton pattern from: 
    https://stackoverflow.com/questions/31875/ ...
    is-there-a-simple-elegant-way-to-define-singletons
    """
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

@Singleton
class Printer(object):
    """
    Custom singleton printer class
    """
    def __init__(self):
        self.on = False
  
    def turnOn(self):
        self.on = True

    def turnOff(self):
        self.on = False

    def _print_(self, s, end='\n'):
        if self.on: print(s, end=end)
