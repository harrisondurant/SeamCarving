from scipy.signal import convolve2d
from utils import *
import numpy as np

class SeamCarver(object):
    def __init__(self, imagepath, use_forward_energy=False):
        self.imagepath = imagepath
        self.image,_,_ = process_image(imagepath)

        # keep a copy of original image for comparison
        self.original = self.image

        # set function based on mode 
        self.carve_fn = self.insert_seams if insert else self.carve_seams

        # kernels for computing image energy (gradient)
        self.x_grad = np.array([[-3,0,3],[-10,0,10],[-3,0,3]])
        self.y_grad = np.array([[-3,-10,-3],[0,0,0],[3,10,3]])

        # kernels for computing forward energy
        self.x_f = np.array([[0,0,0],[-1,0,1],[0,0,0]], dtype=np.float64)
        self.x_f_left = np.array([[0,0,0],[0,0,1],[0,-1,0]], dtype=np.float64)
        self.x_f_right = np.array([[0,0,0],[1,0,0],[0,-1,0]], dtype=np.float64)

        # whether or not to use forward energy
        self.energy_fn = self.backward_energy
        if use_forward_energy:
            self.energy_fn = self.forward_energy

        # Printer singleton
        self.p = Printer.Instance()
        self.p.turnOn()

        # constantfor object removal
        self.C = 1e5

        # number of seams for object removal. will be set when function is called
        self.n_restore_seams = 0

        # need to know of image has been transposed
        self.did_transpose = False

        self.bmap_r = None
        self.bmap_i = None
        self.I = None

    # cost function for a seam
    def cost(self, energy_map, seam):
        return np.sum(energy_map[np.arange(seam.shape[0]),seam])

    # default image energy function
    def energy(self):
        return conv(self.image, self.x_grad) + conv(self.image, self.y_grad)

    # compute energy map for image using backward energy
    def backward_energy(self, E=None):
        h,w,_ = self.image.shape
        
        if E is None: E = self.energy()
        M = E.copy()

        for i in range(1,h):
            for j in range(w):
                up = M[i-1,j]
                up_left = M[i-1,max(j-1,0)]
                up_right = M[i-1,min(j+1,w-1)]
                M[i,j] = E[i,j] + min(up_left, up, up_right)
        return M

    # helper function to retrieve forward energy matrices  
    def get_forward_matrices(self):
        M_x = conv(self.image, self.x_f)
        M_x_left = conv(self.image, self.x_f_left)
        M_x_right = conv(self.image, self.x_f_right)
        return M_x, M_x_left, M_x_right

    # compute energy map for image using forward energy
    def forward_energy(self, E=None):
        h,w,_ = self.image.shape
        
        if E is None: E = self.energy()
        M = E.copy()
        M_x, M_x_left, M_x_right = self.get_forward_matrices()

        for i in range(1,h):
            for j in range(w):
                r = min(j+1,w-1)
                l = max(j-1,0)
                e_right = M[i-1,r] + M_x[i-1,r] + M_x_right[i-1,r]
                e_left = M[i-1,l] + M_x[i-1,l] + M_x_left[i-1,l]
                e_up = M[i-1,j] + M_x[i-1,j]        
                if j == 0: e_left = e_up
                if j == w-1: e_right = e_up
                M[i,j] = E[i,j] + min(e_left, e_right, e_up)
        return M

    def find_seam(self, M=None):
        if M is None: M = self.energy_fn()
        h,w = M.shape

        seam = np.zeros((h), dtype=np.int32)
        seam[-1] = np.argmin(M[-1,:])

        for i in range(h-2,-1,-1):
            x = seam[i+1]
            left = int(max(x-1, 0))
            right = int(min(x+1, w-1))
            idx = [left, right, x]
            seam[i] = idx[np.argmin([M[i,left], M[i,right], M[i,x]])]

        return seam

    # remove seams from an image
    def carve_seams(self, n_seams=1):
        seams = []
        
        for n in range(n_seams):
            self.p._print_('Processing seam {} of {}.'.format(n+1,n_seams), end='\r')
            seam = self.find_seam()
            seams.append(seam)
            self.image = carve(self.image, seam)

        self.p._print_('\ndone.')
        return seams

    # correct indices that changed during removal
    def update_seams(self, seams, current_seam):
        for i in range(len(seams)):
            seams[i][np.where(seams[i] >= current_seam)] += 2
        return seams

    # insert seams into an image
    def insert_seams(self, n_seams=1):
        temp = self.image.copy()
        seams = self.carve_seams(n_seams=n_seams)
        self.image = temp

        for i in range(len(seams)):
            seam = seams.pop(0)
            self.image = insert(self.image, seam)
            seams = self.update_seams(seams, seam)

        return seams

    # resize an image
    def resize(self, new_size):
        h,w,_ = new_size
        if h < 1 or w < 1:
            raise ValueError('ERROR. Invalid image dimension(s) {}x{}'.format(h,w))

        # no energy map lookahead - just resize horizontal, then vertical
        nh = self.image.shape[0] - h
        nw = self.image.shape[1] - w
        
        if nh > 0:
            self.did_transpose = True
            self.image = transpose(self.image)
            _ = self.carve_seams(n_seams=nh)
        elif nh < 0:
            self.did_transpose = True
            self.image = transpose(self.image)
            _ = self.insert_seams(n_seams=-1*nh)

        self.maybe_transpose()

        if nw > 0: 
            _ = self.carve_seams(n_seams=nw)
        elif nw < 0: 
            _ = self.insert_seams(n_seams=-1*nw)      

    # perform object removal given input mask
    def remove_object(self, mask):
        self.p._print_('Removing object...')
    
        ns_y, n_seams = get_mask_dimensions(mask)
        if ns_y < n_seams:
            n_seams = ns_y
            self.did_transpose = True
            self.image = transpose(self.image)
            mask = transpose(mask)
            
        h,w,_ = self.image.shape

        # while mask still has columns with ones
        for n in range(n_seams):
            self.p._print_('Processing seam {} of {}.'.format(n+1,n_seams), end='\r')
            E = self.energy()
            # multiply object pixels by large negative constant
            E[np.where(mask[:,:] == 1)] *= -1*int(self.C)
            M = self.energy_fn(E=E)
            # get seam, remove it from image and mask
            seam = self.find_seam(M=M)
            self.image = carve(self.image, seam)
            mask = carve(mask, seam)

        self.p._print_('\ndone.')

        # set number of seams for insertion
        self.n_restore_seams = n_seams
        
        self.maybe_transpose()

    # can be called once object removal has completed
    def restore_image(self):
        self.p._print_('Restoring image...')

        self.maybe_transpose()
        self.insert_seams(n_seams=self.n_restore_seams)
        self.maybe_transpose()

    def process(self):
        h,w,_ = self.image.shape
        temp = self.image.copy()
        self.I = self.image.copy()

        self.bmap_r = np.zeros((h,w), dtype=np.int32)
        _, Z = np.indices((h,w), dtype=np.int32)
        self.bmap_i = Z.copy()

        print('Processing')
        for i in range(w):
            print('seam {} of {}'.format(i+1, w), end='\r')
            seam = self.find_seam()
            
            self.image = carve(self.image, seam)
            self.bmap_r[np.arange(h), Z[np.arange(h), seam]] = i
            Z = carve(Z, seam)

            self.I = insert(self.I, seam)
            self.bmap_i = insert_val(self.bmap_i, seam, -i-1)
            
        print('\ndone.')
        self.image = temp

    # helper function to transpose image if necessary
    def maybe_transpose(self):
        if self.did_transpose: self.image = transpose(self.image)

    # reset original image
    def set_image(self, image):
        self.image = image
        self.original = image
