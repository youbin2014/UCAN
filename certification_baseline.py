import argparse
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from tqdm import tqdm
from core_baseline import Smooth
# from noisegenerator import NoiseGenerator
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
# parser.add_argument('model_path', type=str, help='folder to save model and training log)')
# parser.add_argument('noisegenerator', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=196, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--sigma', type=float, default=1,
                    help='sigma')
parser.add_argument('--noise_name',type=str)
args = parser.parse_args()

if __name__ == '__main__':

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
    #                           num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)
    # if args.dataset=='cifar10':
    #     model.load_state_dict(torch.load(args.model_path))
    # else:
    model.load_state_dict(torch.load("./model_saved/CIFAR10_iso_model_{}_sigma{}_best.pth".format(args.noise_name,args.sigma)))
    model.cuda()

    if args.dataset=='cifar10':
        SIZE=32
        CLASS=10
        skip=1
    elif args.dataset=='imagenet':
        SIZE=224
        CLASS=1000
        skip=100
    else:
        print("unknown dataset")
    # NoiseGenerator = NoiseGenerator(in_nc=3, out_nc=3, BlockNum=1, size=SIZE)
    # NoiseGenerator.cuda()
    # NoiseGenerator.load_state_dict(torch.load('./model_saved/CIFAR10_3.0_No_Variance_NoiseGenerator_sigma{}_last.pth'.format(args.sigma)))

    Smoother=Smooth(model,CLASS)

    print('start')
    if os.path.exists('./results/cifar10/CIFAR_iso_10000_{}_sigma{}_results_pA.npy'.format(args.noise_name,args.sigma)):
        pA_list = np.load('./results/cifar10/CIFAR_iso_10000_{}_sigma{}_results_pA.npy'.format(args.noise_name,args.sigma)).tolist()
        # sigmas=np.load('./results/cifar10/CIFAR_iso_{}_lambda{}_results_pA.npy'.format(args.sigma)).tolist()
        dif=len(pA_list)
    else:
        pA_list = []
        # sigmas=[]
        dif=0
    valid=0
    count=0
    for j in tqdm(range(10000-dif)):
        i=(j+dif)*skip
        # i=np.random.randint(10000)
        print('fig {} certifying'.format(i))
        (x, y) = test_dataset[i]
        X = x.cuda().unsqueeze(0)
        prediction, pA=Smoother.certify(X,n0=100,n=100000,alpha=0.001,batch_size=args.batch,noise_name=args.noise_name,sigma=args.sigma)
        correct = int(prediction == y)
        if correct:
            print('pA={}'.format(pA))
            pA_list.append(pA)
            # sigmas.append(sigma)
            valid+=1
        else:
            print('pA={}'.format(-1))
            pA_list.append(-1)
            # sigmas.append(sigma)
        count+=1
        print('acc: {}'.format(valid/count))
        np.save('./results/cifar10/CIFAR_iso_10000_{}_sigma{}_results_pA.npy'.format(args.noise_name,args.sigma), pA_list)
        # np.save('./results/cifar10/CIFAR_trainone_iso_var_sigmas_{}'.format(args.sigma), sigmas)
