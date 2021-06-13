import fastbook
from fastai.vision.all import *
from fastbook import *

def search_images_ddg(key,max_n=200):
     """Search for 'key' with DuckDuckGo and return a unique urls of 'max_n' images
        (Adopted from https://github.com/deepanprabhu/duckduckgo-images-api)
     """
     url        = 'https://duckduckgo.com/'
     params     = {'q':key}
     res        = requests.post(url,data=params)
     searchObj  = re.search(r'vqd=([\d-]+)\&',res.text)
     if not searchObj: print('Token Parsing Failed !'); return
     requestUrl = url + 'i.js'
     headers    = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:71.0) Gecko/20100101 Firefox/71.0'}
     params     = (('l','us-en'),('o','json'),('q',key),('vqd',searchObj.group(1)),('f',',,,'),('p','1'),('v7exp','a'))
     urls       = []
     while True:
         try:
             res  = requests.get(requestUrl,headers=headers,params=params)
             data = json.loads(res.text)
             for obj in data['results']:
                 urls.append(obj['image'])
                 max_n = max_n - 1
                 if max_n < 1: return L(set(urls))     # dedupe
             if 'next' not in data: return L(set(urls))
             requestUrl = url + data['next']
         except:
             pass

bear_types = 'grizzly', 'black', 'teddy'
main_path = Path('bears')

for types in bear_types:
    dest_path = Path(f'bears/{types}')
    urls = search_images_ddg(types, max_n=62)
    
    if not dest_path.exists():
        print(f'Processing .. {types}')
        dest_path.mkdir()
        dest_path.mkdir(exist_ok=True)
    else:
        print('Already Done!')
    for num, path in enumerate(urls):
            dest_new = f'bears/{types}/{types}_{num}.jpg' 
            print(dest_new)
            download_url(path, dest_new )

bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=4),
    get_y=parent_label,
    item_tfms=Resize(128))

bears = bears.new(item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms())
failed = verify_images(main_path)
failed.map(main_path.unlink)
dls = bears.dataloaders(main_path, num_workers=0)
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(2)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(5, nrows=1)
plt.show()
