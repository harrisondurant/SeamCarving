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

# Ask user for percentage amplification
def query_percent():
    q = 'Please enter percent amplification (-100 to 100): '
    while True:
        try:
            p = int(input(q))
            if p >= -100 and p <= 100: return p
        except ValueError:
            print('Percent amp. must be a number >= %d and =< %d' % (-100,100))

def main(args):
    # Create SeamCarver
    image = load_and_process_image(args.image)
    sc = SeamCarver(image, args.image, verbose=True, 
                    use_forward_energy=args.forward)

    # create cv2 named window to display images
    window_name = 'Content Amplification - %s' % sc.image_name
    cv2.namedWindow(window_name)

    # display original image
    cv2.imshow(window_name, sc.original)
    cv2.waitKey(1)

    while True:
        # ask user for percent amplification
        percent = query_percent()

        # now perform amplification
        sc.content_amplify(percent)

        # display results
        cv2.imshow(window_name, sc.image)
        cv2.waitKey(1)

        # ask if user wants to continue
        if not query_continue(): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True,
              help='path to input image file')
    ap.add_argument('-f', dest='forward', action='store_true', 
                  default=False, help='use forward energy')
    args = ap.parse_args()

    main(args)
