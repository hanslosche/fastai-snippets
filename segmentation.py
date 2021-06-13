from fastai.vision.all import *
import matplotlib.pyplot as plt

path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str),
    num_workers=0
)
learn = unet_learner(dls, resnet34)
#learn.fine_tune(8)
learn.show_results(max_n=6, figsize=(7,8))
plt.show()
