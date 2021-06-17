from fastai.vision.all import *
path = untar_data(URLs.PETS)

print(path.ls())
print((path/'images').ls()[:5])
