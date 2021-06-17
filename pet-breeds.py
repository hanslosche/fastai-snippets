from fastai.vision.all import *
path = untar_data(URLs.PETS)

#print(path.ls())
#print((path/'images').ls()[:5])

fname = (path/'images').ls()[0]
test = re.findall(r'(.+)_\d+.jpg$', fname.name)
print(test)
