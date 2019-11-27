import sys
import os
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image

args = sys.argv
slide_path = args[1]
dist_path = args[2]
zoom_level = int(args[3])
tile_size = int(args[4])
offset = [int(i) for i in args[5].split(',')]
limit = [int(i) for i in args[6].split(',')]

osr = open_slide(slide_path)

dz = DeepZoomGenerator(osr, tile_size, 0, False)

for x_origin in range(offset[0], (offset[0] + limit[0])):
    os.system("mkdir -p '%(dist_path)s'/x%(x)s" % { "dist_path":dist_path, "x": x_origin})
    for y_origin in range(offset[1], (offset[1] + limit[1])):
        img = dz.get_tile(zoom_level, (x_origin, y_origin))
        img = img.convert("RGB")
        file_params = { "dist_path":dist_path, "x": x_origin, "y": y_origin, "w": tile_size, "h": tile_size, "z": zoom_level }
        file_path = "%(dist_path)s/x%(x)s/x%(x)s.y%(y)s.w%(w)s.h%(h)s.z%(z)s.jpg" % file_params
        img.save(file_path, format='JPEG', subsampling=0, quality=100)
