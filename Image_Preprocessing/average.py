import os
import numpy
from PIL import Image

path = 'D:/Pycharm/projects/faces/tfaces3crop/sum'
allfiles = os.listdir(path)
imlist=[path+'/'+filename for filename in allfiles if filename[-4:] in [".jpg"]]

w,h=Image.open(imlist[0]).size
N=len(imlist)

arr=numpy.zeros((h,w,3),numpy.float)
stdev=numpy.zeros((h,w,3),numpy.float)

for im in imlist:
    imarr=numpy.array(Image.open(im),dtype=numpy.float)
    arr=arr+imarr/N

for im in imlist:
    imarr = numpy.array(Image.open(im), dtype=numpy.float)
    stdev = stdev+((imarr-arr)**2)/N

op1=numpy.array(numpy.round(arr),dtype=numpy.uint8)
op2=numpy.array(numpy.round(stdev),dtype=numpy.uint8)

Image.fromarray(op1,mode="RGB").save("average.jpg")
Image.fromarray(op2,mode="RGB").save("stdev.jpg")

