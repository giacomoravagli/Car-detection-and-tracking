import os, sys, shutil, glob, binascii, string
import numpy as np

input = open("data/modified/weights/yolov3-weights.txt", 'r')
output = open("data/modified/weights_out/yolov3-weights.txt", 'w')



#for i in range(15403511):
    #output.write('0000 0000 0000 0000 0000 0000 0000 0000\n')

for i in range(15403511):
    a = binascii.b2a_hex(os.urandom(2))
    a = a.decode('utf-8')
    b = binascii.b2a_hex(os.urandom(2))
    b = b.decode('utf-8')
    c = binascii.b2a_hex(os.urandom(2))
    c = c.decode('utf-8')
    d = binascii.b2a_hex(os.urandom(2))
    d = d.decode('utf-8')
    e = binascii.b2a_hex(os.urandom(2))
    e = e.decode('utf-8')
    f = binascii.b2a_hex(os.urandom(2))
    f = f.decode('utf-8')
    g = binascii.b2a_hex(os.urandom(2))
    g = g.decode('utf-8')
    h = binascii.b2a_hex(os.urandom(2))
    h = h.decode('utf-8')
    string = a + ' ' + b + ' ' + c + ' ' + d + ' ' + e + ' ' + f + ' ' + g + ' ' + h + '\n'
    output.write(string)