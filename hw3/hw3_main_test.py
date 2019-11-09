import torch
import sys
from hw3_util import test_hw3, layer3_CNN
import pdb

'''
python3 hw3_main.py ./data/test_img/ ./output.csv test
bash hw3_test.sh <testing data folder> <prediction.csv>
'''

def test():
    file = open(sys.argv[2], 'w')
    print('id,label', file=file)
    test_dataset = test_hw3(sys.argv[1])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = layer3_CNN()
    model.load_state_dict(torch.load('./model_CNN3v2_19.pkl'))
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    model.eval()

    prediction = []
    with torch.no_grad():
        for batch_idx, (img, index) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            _, pred_label = torch.max(out, 1)
            prediction.append((index.item(), pred_label.item()))
    for pred in prediction:
        print('{:d},{:d}'.format(pred[0], pred[1]), file=file)
    file.close()

if __name__=='__main__':
    test()