from seamcarver import *
from utils import *

import numpy as np
import argparse
import cv2

# Ask if user wants to continue resizing image
def query_continue(string='Continue? (y/n): ', target_answer='y'):
    return target_answer == input(string).lower().replace(' ', '')[0]

# Ask user for new image size for resizing
def query_image_size(curr_size, min_h=10, min_w=10):
    h,w = curr_size
    h_,w_ = min_h,min_w
    h_str = 'Current height = {}. Please enter new image height: '.format(h)
    w_str = 'Current width = {}. Please enter new image width: '.format(w)
    while True:
        h_ = input(h_str)
        if h_.isdigit() and int(h_) > min_h: break
        print('new height must be a number greater than {}'.format(min_h))
    while True:
        w_ = input(w_str)
        if w_.isdigit() and int(w_) > min_w: break
        print('new width must be a number greater than {}'.format(min_w))
    return int(h_), int(w_)

def main(args):
    # Create SeamCarver
    sc = SeamCarver(args.image, use_forward_energy=args.forward)
    sc.p.turnOff()

    # default 1 seam at a time
    nw, nh = 0, 0

    h,w,d = sc.image.shape
    spacing = 16
    half = int(spacing / 2)

    # create mat 
    mat = np.ones((2*(h+spacing), w+spacing, d), dtype=sc.image.dtype) * 255

    # create cv2 named window to display images
    window_name = 'Seam Carving - {}'.format(sc.imagepath)
    cv2.namedWindow(window_name)

    # while user has not quit, or image is too small
    while sc.image.shape[1] > 1:

        # if continuous mode, remove one vertical seam at a time.
        # otherwise, query user for new image size
        if args.continuous: 
            nh, nw = h, sc.image.shape[1] - 1
        else:
            nh, nw = query_image_size(sc.image.shape[:2])

        # Resize window in case seams are inserted
        mat = np.ones((2*(max(h,nh)+spacing),
                    max(w,nw)+spacing, d), dtype=sc.image.dtype) * 255

        # Perform seam carving/inserting operations
        sc.resize((nh, nw, d))

        # Display results on window
        mat[half:nh+half,half:nw+half,:] = sc.image
        mat[spacing+half+h:spacing+half+2*h,half:w+half,:] = sc.original

        cv2.imshow(window_name, mat)

        if cv2.waitKey(1) == ord('q'): break
        if not args.continuous and not query_continue(): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True,
              help='path to input image file')
    ap.add_argument('-c', dest='continuous', action='store_true', 
                  default=False, help='continuous removal mode')
    ap.add_argument('-f', dest='forward', action='store_true', 
                  default=False, help='use forward energy')
    args = ap.parse_args()

    main(args)
