from utils import *
import numpy as np

class SeamCarver(object):
    def __init__(self, image, imagepath, 
                    use_forward_energy=False,
                    obj_remove_direction='horizontal',
                    verbose=False):

        # keep image name
        self.image_name = imagepath.split('/')[-1].split('.')[0]

        # keep a copy of original image for comparison
        self.original = image
        self.image = image
        self.orig_h, self.orig_w, self.d = image.shape

        # whether or not to use forward energy for energy map
        self.use_FE = use_forward_energy

        # if verbose, turn on printer singleton
        if verbose: Printer.Instance().turnOn()

        # constantfor object removal
        self.C = -1*1e5

        # number of seams for object removal. will be set when function is called
        self.n_restore_seams = 0

        # direction of seam removal for objects
        if obj_remove_direction not in ['vertical', 'horizontal']:
            raise ValueError('ERROR: Invalid entry for ' \
                                'object removal direction.')
        self.obj_remove_direction = obj_remove_direction

        # need to know of image has been transposed
        self.did_transpose = False

        # for multi-size images
        self.did_pre_process = False
        self.map_remove = None
        self.map_insert = None
        self.enlarged_image = None

    # remove seams from an image
    def carve_seams(self, n_seams=1):
        self.image, seams = carve_seams(self.image, 
                                    n_seams=n_seams,
                                    use_FE=self.use_FE)
        return seams

    # insert seams into an image
    def insert_seams(self, n_seams=1):
        self.image,_ = insert_seams(self.image, 
                                    n_seams=n_seams,
                                    use_FE=self.use_FE)

    # resize an image
    def resize(self, new_height, new_width):
        h,w = new_height, new_width
        if h < 1 or w < 1:
            raise ValueError('ERROR. Invalid image dimension(s) %dx%d' % (h,w))
        
        # Resize vertically
        nh = self.image.shape[0] - h
        if nh > 0:
            self.did_transpose = True
            self.maybe_transpose()
            self.carve_seams(n_seams=nh)
        elif nh < 0:
            self.did_transpose = True
            self.maybe_transpose()
            self.insert_seams(n_seams=-1*nh)

        # Re-transpose image if necessary
        self.maybe_transpose()

        # Resize horizontally
        nw = self.image.shape[1] - w
        if nw > 0: 
            self.carve_seams(n_seams=nw)
        elif nw < 0: 
            self.insert_seams(n_seams=-1*nw)  

    # Fast resizing for multi-size image format
    def fast_resize_width(self, new_width):
        if new_width < 1 or new_width > 2*self.orig_w:
            raise ValueError('ERROR. Invalid image width %d' % new_width)

        # pre-process image if not already done
        if not self.did_pre_process: self.pre_process()

        # determine whether to reduce or increase size
        diff = self.orig_w - new_width
        
        # if reducing width
        if diff > 0:
            self.image = self.original[np.where(self.map_remove >= diff)]. \
                                    reshape(self.orig_h, new_width, self.d)
        elif diff < 0:
            self.image = self.enlarged_image[np.where(self.map_insert >= diff)]. \
                                    reshape(self.orig_h, new_width, self.d)
    
    # perform object removal given input mask
    def remove_object(self, mask):
        s_print_('Removing object...')

        # Get dimensions of input mask
        ns_y, n_seams = get_mask_dimensions(mask)

        # If vertical seam removal
        if self.obj_remove_direction == 'vertical':
            n_seams = ns_y
            self.did_transpose = True
            self.maybe_transpose()
            mask = transpose(mask)
        
        image = self.image
        h,w,_ = image.shape

        energy     = None
        energy_map = None

        # while mask still has columns with ones
        for n in range(n_seams):
            s_print_('Processing seam %d of %d.' % (n+1,n_seams), end='\r')
            
            if self.use_FE:
                # use forward energy with mask as a weight matrix
                p = mask.astype(np.float64) * self.C
                energy_map = forward_energy(image.copy(), p)
            else:
                energy = e1_energy(image)
                # multiply object pixels by large negative constant
                energy[np.where(mask[:,:] == 1)] *= self.C
                # compute cumulative energy map
                energy_map = backward_energy(energy)
            
            # get seam, remove it from image and mask
            seam  = find_seam(energy_map)
            image = carve(image, seam)
            mask  = carve(mask, seam)

        # Reset image, transpose if necessary
        self.image = image
        self.maybe_transpose()

        # set number of seams for insertion
        self.n_restore_seams = n_seams        

    # can be called once object removal has completed
    def restore_image(self):
        s_print_('Restoring image...')

        self.maybe_transpose()
        self.insert_seams(n_seams=self.n_restore_seams)
        self.maybe_transpose()

    # pre-process image for multi-size functionality
    def pre_process(self): 
        # check for existing maps
        s_print_('Checking for existing image maps...')
        image_maps = load_image_maps(self.image_name, with_FE=self.use_FE)
        
        # if none exist, create them, and save to disk
        if image_maps is None:
            s_print_('No existing maps found.')
            maps = pre_process_multi(self.image.copy(), use_FE=self.use_FE)
            (self.map_remove, self.enlarged_image, self.map_insert) = maps
            save_image_maps(self.image_name, maps, did_use_FE=self.use_FE)
        else:
            filename, maps = image_maps
            s_print_('Found file \'%s\'.' % filename)
            self.map_remove = maps['map_remove']
            self.enlarged_image = maps['enlarged_image']
            self.map_insert = maps['map_insert']
        
        self.did_pre_process = True

    # helper function to transpose image if necessary
    def maybe_transpose(self):
        if self.did_transpose: self.image = transpose(self.image)

    # reset original image
    def reset_image(self, image):
        self.image = image
        self.original = image
