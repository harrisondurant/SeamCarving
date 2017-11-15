from seamcarver import *
from utils import *

import numpy as np
import argparse
import cv2

done_drawing = True
x_, y_ = 0, 0
points = []
mask = None

# get all points on a line between two points
def get_points(x1,y1,x2,y2):
    if x1 == x2 and y1 == y2: return [(x1,y1)]
    if x1 == x2: return [(x1,y) for y in range(min(y1,y2),max(y1,y2)+1)]
    if y1 == y2: return [(x,y1) for x in range(min(x1,x2),max(x1,x2)+1)]
    
    f = lambda y: (y-y1)*(x2-x1)/(y2-y1)+x1
    x,y = x1,y1
    xp,yp = x2,y2
    if y2 < y1: 
        x,y = x2,y2
        xp,yp = x1,y1

    return [(x,y)] + [(round(f(y)), y) for y in range(min(y1+1,y2+1),max(y1,y2))] + [(xp,yp)]

# function to handle user drawing
def draw(event,x,y,flags,params):
    global done_drawing,x_,y_,points
    
    # unpack params
    mat,h,orig = params

    # now starting to draw
    if event == cv2.EVENT_LBUTTONDOWN:
        # reset canvas
        mat[:h,:,:] = orig
        points = [(x,y)]
        x_,y_ = x,y
        done_drawing = False

    # currently drawing mask
    if event == cv2.EVENT_MOUSEMOVE and not done_drawing:
        # make sure user not drawing in text area
        if x < orig.shape[1] and y < orig.shape[0]:
            cv2.line(mat,(x_,y_),(x,y),(0,0,255),1)
            points.extend(get_points(x_,y_,x,y))
            x_,y_ = x,y

    # finished creating mask
    if event == cv2.EVENT_LBUTTONUP and not done_drawing:
        cv2.line(mat, points[-1], points[0],(0,0,255),1)
        points.extend(get_points(x_,y_,points[0][0],points[0][1]))
        create_mask(mat.shape)
        done_drawing = True

# create object mask given list of points
def create_mask(shape):
    global points, mask
    # get unique points
    points = list(set(points))

    # get min and max x,y values
    s_x = min(points, key = lambda p: p[0])[0]
    s_y = min(points, key = lambda p: p[1])[1]
    e_x = max(points, key = lambda p: p[0])[0]
    e_y = max(points, key = lambda p: p[1])[1]

    # create 2D mask filled with zeros
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)

    # fill mask with ones where user has drawn
    for y in range(s_y, e_y+1):
        pts = [p[0] for p in points if p[1] == y]
        min_x = min(pts)
        max_x = max(pts)
        mask[y,min_x:max_x+1] = 1

# dummy function for resetting cv2 mouse callback
def nothing(event,x,y,flags,param):
    pass

def main(args):
    global mask

    # Create SeamCarver
    sc = SeamCarver(args.image, use_forward_energy=args.forward)

    # create cv2 named window to display images
    h,w,d = sc.image.shape
    text_area_height = 50
    spacing = 16
    half = int(spacing / 2)
    
    # create mat
    orig = sc.image.copy()
    mat = np.ones((h+text_area_height,w,d), dtype=sc.image.dtype) * 255
    mat[:h,:,:] = orig

    # get text area, display it
    text = 'draw object mask using mouse. Press \'enter\' when finished.'
    text_area = get_text_area(text, (text_area_height,w,d), dtype=sc.image.dtype)
    mat[h:,:,:] = text_area

    # create window
    window_name = 'Object removal - {}'.format(sc.imagepath)
    cv2.namedWindow(window_name)

    # set callback for drawing
    cv2.setMouseCallback(window_name, draw, (mat,h,orig))

    # while user has not drawn mask, display image, wait...
    while True:
        cv2.imshow(window_name, mat)
        if cv2.waitKey(1) == 13: # 'Enter' key
            print('Object mask created.')
            break

    # no need to draw anymore, reset callback
    cv2.setMouseCallback(window_name, nothing)

    # create 'please wait...' text area
    text = 'removing object...'
    text_area = get_text_area(text, (text_area_height,w,d), dtype=sc.image.dtype)
    mat[h:,:,:] = text_area

    # redisplay image with new text area
    cv2.imshow(window_name, mat)
    cv2.waitKey(1)

    # remove object
    sc.set_image(orig)
    mask = mask[:h,:]
    sc.remove_object(mask)

    # done with object removal, display result
    nh, nw, _ = sc.image.shape
    mat = np.ones((nh+text_area_height,nw,d), dtype=sc.image.dtype) * 255
    mat[:nh,:nw,:] = sc.image
    text = 'restoring image to original size...'
    text_area = get_text_area(text, (text_area_height,nw,d), dtype=sc.image.dtype)
    mat[nh:,:,:] = text_area

    # display image with new text area
    cv2.imshow(window_name, mat)
    cv2.waitKey(1)

    # restore image to original size
    sc.restore_image()

    # redisplay new image along with original image
    mat = np.ones((2*(h+spacing),w+spacing, d), dtype=sc.image.dtype) * 255
    mat[half:h+half,half:w+half,:] = sc.image
    mat[spacing+half+h:spacing+half+2*h,half:w+half,:] = sc.original

    # until user quits, display image with object removed
    while True:
        cv2.imshow(window_name, mat)
        if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True,
              help='path to input image file')
    ap.add_argument('-f', dest='forward', action='store_true', 
                  default=False, help='use forward energy')
    args = ap.parse_args()

    main(args)
