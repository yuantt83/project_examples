import glob
from PIL import Image
import os
from random import shuffle
path_spiral = os.getcwd() + os.sep + 'dataset/training/spirals/*.jpg'
path_disk = os.getcwd() + os.sep + 'dataset/training/disk_nonspirals/*.jpg'
path_early = os.getcwd() + os.sep + 'dataset/training/early_types/*.jpg'

spiral_out = os.getcwd() + os.sep + 'spiral.gif'
disk_out = os.getcwd() + os.sep + 'disk_nonspiral.gif'
early_out = os.getcwd() + os.sep + 'early.gif'


file_spirals = glob.glob(path_spiral)
shuffle(file_spirals)
file_disk = glob.glob(path_disk)
shuffle(file_disk)
file_early = glob.glob(path_early)
shuffle(file_early)

img, *imgs = [Image.open(f) for f in file_spirals[0:40]]
img.save(fp=spiral_out, format='GIF', append_images=imgs, save_all=True, duration=300, loop=2)

img, *imgs = [Image.open(f) for f in file_disk[0:40]]
img.save(fp=disk_out, format='GIF', append_images=imgs, save_all=True, duration=300, loop=2)

img, *imgs = [Image.open(f) for f in file_early[0:40]]
img.save(fp=early_out, format='GIF', append_images=imgs, save_all=True, duration=300, loop=2)
