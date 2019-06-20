from io import BytesIO
from PIL import Image
from utils import Dictionary, get_words, load_model
import base64
import difflib
import numpy as np
import torch.nn.functional as F
import torch


net_recog = load_model('../recog_params.pkl')
ug_dict = Dictionary('../../../data/ug_words.txt') 

def img2str(pic):
    # pic: numpy array
    figfile = BytesIO()
    pic = Image.fromarray(pic).convert('RGBA').save(figfile, format='PNG')
    figfile.seek(0, 0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    return figdata_png


def compute(test_id='test'):
    pic_path = 'static/' + test_id + '.png'
    truth_path = 'static/' + test_id + '.txt'
    base_image, pic_with_box, word_pics = get_words(pic_path)
    word_pics = np.stack(word_pics)
    pred = F.softmax(net_recog(torch.from_numpy(word_pics)), dim=1)
    _, idxes = torch.max(pred, dim=1)

    res = []
    for idx in idxes:
        res.append(ug_dict[idx])

    f = open(truth_path, 'r')
    lines = f.readlines()
    f.close()
    lines = [l.strip() for l in lines]
    truth = []
    for l in lines:
        truth += l.split()

    for i, w in enumerate(res):
        if w not in truth:
            res[i] = '<span style="color:red">' + res[i] + '</span>'
    truth = ' '.join(truth)
    res = ' '.join(res[::-1])
    score = difflib.SequenceMatcher(a=truth, b=res).ratio()

    return img2str(base_image), img2str(pic_with_box), res, truth, score


if __name__ == "__main__":
    compute(1, 0.1, 1, 20)
