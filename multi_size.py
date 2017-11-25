# -*- coding: utf-8 -*-
import numpy as np
import argparse

from seamcarver import *
from utils import *

from PIL import Image as IMG
from PIL import ImageTk
from tkinter import *

class ResizeableImage(Frame):
    def __init__(self, parent, seam_carver,
                 direction='horizontal'):
        Frame.__init__(self, parent)
        
        self.parent = parent
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)

        self.seam_carver = seam_carver

        self.h, self.w, self.d = self.seam_carver.image.shape

        self.image = ImageTk.PhotoImage(
            IMG.fromarray(self.seam_carver.image))

        self.dims = {'max_w': int(3 * self.w / 2),
                    'min_w': int(self.w / 2),
                    'max_h': int(self.h * 2),
                    'min_h': int(self.h / 2)}

        self.set_resize_direction(direction)

        self.display = Canvas(self, bd=0, highlightthickness=0)
        self.display.create_image(0, 0, image=self.image, anchor=NW, tags="IMG")
        self.display.grid(row=0, sticky=W+E+N+S)
        self.pack(fill=BOTH, expand=1)
        self.bind("<Configure>", self.on_resize)

    def set_resize_direction(self, direction):
        if direction == 'horizontal':
            max_w = self.dims['max_w']
            min_w = self.dims['min_w']
            max_h = self.h
            min_h = self.h
        else:
            max_w = self.w
            min_w = self.w
            max_h = self.dims['max_h']
            min_h = self.dims['min_h']

        self.parent.wm_minsize(width=min_w, height=min_h)
        self.parent.wm_maxsize(width=max_w, height=max_h)

    def on_resize(self, event):
        self.seam_carver.fast_resize_width(event.width)
        resized = IMG.fromarray(self.seam_carver.image)
        self.image = ImageTk.PhotoImage(resized)

        self.display.delete("IMG")
        self.display.create_image(0, 0, image=self.image, anchor=NW, tags="IMG")

def main(args):
    # Create window
    root = Tk()
    root.title(args.image)

    # create seamcarver, run pre-processing
    image = load_and_process_image(args.image, swap_channels=True)
    sc = SeamCarver(image, args.image, verbose=True,
                    use_forward_energy=args.forward)
    sc.pre_process()

    # Create image frame
    frame = ResizeableImage(root, sc)

    # set frame size
    root.geometry('{}x{}'.format(frame.w, frame.h))

    # run gui
    root.mainloop()
        
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                help='path to input image file')
    ap.add_argument('-f', dest='forward', action='store_true', 
                  default=False, help='use forward energy')
    # ap.add_argument('-d', '--direction', type=str,
    #            default='horizontal', help='seam removal (resize) direction')
    args = ap.parse_args()

    main(args)

    
