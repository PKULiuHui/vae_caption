import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from    models import triu_mask
import  os
# Parameters
data_folder = '/home/liuhui/vae_caption/data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
ckpt_folder = './ckpt3/'
standard_gaussian = False
checkpoint = ckpt_folder + 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = '/home/liuhui/vae_caption/data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=3, shuffle=False, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    references = list()
    hypotheses = list()
    for i, (image, _, _, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        image = image.to(device)
        # caps = caps.to(device)
        # caplens = caplens.to(device)
        output = model.generate(beam_size, image, word_map['<start>'], word_map['<end>'])
        # print(img_caps)
        img_captions = list(map(lambda t: [[w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}] for c in t], allcaps.tolist()))  # remove <start> and pads
        references.extend(img_captions)
        # Hypotheses
        hypotheses.extend([[w for w in line  if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}] for line in output])
        # print(' '.join([rev_word_map[x] for x in hypotheses[-1]]))
        
        assert len(references) == len(hypotheses)
    bleu4 = corpus_bleu(references, hypotheses)
    return bleu4



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    beam_size = 10
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))