from seamcarver import *
from utils import *

import numpy as np
import argparse
import cv2

# Ask if user wants to continue resizing image
def query_continue(string='Enter \'y\' to continue, or any other key to quit: ', 
                    target_answer='y'):
    user_input = input(string).lower().replace(' ', '')
    if len(user_input) == 0: return False
    return target_answer == user_input[0]

# Ask user for new image size for resizing
def query_image_size(curr_size, max_w, max_h, min_h=10, min_w=10):
    h,w = curr_size
    h_,w_ = min_h,min_w
    h_str = 'Current height = %d. Please enter new image height: ' % h
    w_str = 'Current width = %d. Please enter new image width: ' % w
    while True:
        h_ = input(h_str)
        if h_.isdigit() and int(h_) > min_h and int(h_) < max_h: break
        print('new height must be a number > %d and < %d' % (min_h, max_h))
    while True:
        w_ = input(w_str)
        if w_.isdigit() and int(w_) > min_w and int(w_) < max_w: break
        print('new width must be a number > %d and < %d' % (min_w, max_w))
    return int(h_), int(w_)

def main(args):
    # Create SeamCarver
    image = load_and_process_image(args.image)

    verbose = not args.continuous
    sc = SeamCarver(image, args.image, verbose=verbose, 
                    use_forward_energy=args.forward)

    # default 1 seam at a time
    nw, nh = 0, 0

    h,w,d = sc.image.shape
    spacing = 16
    half = int(spacing / 2)

    mat = make_image_grid([sc.image, sc.original])

    # create cv2 named window to display images
    window_name = 'Seam Carving - %s' % sc.image_name
    cv2.namedWindow(window_name)

    cv2.imshow(window_name, mat)
    cv2.waitKey(1)

    # while image is not too small
    while sc.image.shape[1] > 1:

        # if continuous mode, remove one vertical seam at a time.
        # otherwise, query user for new image size
        if args.continuous: 
            nh, nw = h, sc.image.shape[1] - 1
        else:
            nh, nw = query_image_size(sc.image.shape[:2], 2*w, 2*h)

        # Resize window in case seams are inserted
        mat = np.ones((2*(max(h,nh)+spacing),
                    max(w,nw)+spacing, d), dtype=sc.image.dtype) * 255

        # Perform seam carving/inserting operations
        sc.resize(nh, nw)

        # Display results on window
        mat = make_image_grid([sc.original, sc.image])
        cv2.imshow(window_name, mat)
        if cv2.waitKey(1) == ord('q'): break

        # ask if user wants to continue
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
