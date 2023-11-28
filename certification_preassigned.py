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
from core import Smooth_Preassigned
from noisegenerator import Generator
import numpy as np
from noises import transform_lambda
from scipy.stats import norm, binom_test

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
# parser.add_argument('model_path', type=str, help='folder to save model and training log)')
# parser.add_argument('noisegenerator', type=str, help='folder to save model and training log)')
parser.add_argument('--method',type=str)
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
parser.add_argument('--norm', type=int, default=2)
parser.add_argument('--noise_name', type=str,default="UniNoise")
parser.add_argument('--pattern',type=str,default='/home/cc/NoiseGenerator/results/cifar10_radial_preassign_pattern.npy')
parser.add_argument('--IsoMeasure', type=bool,default=False)

args = parser.parse_args()

if __name__ == '__main__':

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dataset=='cifar10':
        SIZE=32
        CLASS=10
        skip=1
        d=32*32*3
        c=3
    elif args.dataset=='imagenet':
        SIZE=224
        CLASS=1000
        skip=100
        d=224*224*3
        c=3
    elif args.dataset=='mnist':
        SIZE=28
        CLASS=10
        skip=1
        d=28*28
        c=1
    else:
        print("unknown dataset")

    lambd = transform_lambda(args.noise_name, args.sigma)

    if args.noise_name=='Laplace' and args.norm==1:
        def radius_iso(pa):
            return -lambd*np.log(2*(1-pa))
    elif args.noise_name=='Gaussian' and args.norm==2:
        def radius_iso(pa):
            return lambd*norm.ppf(pa)
    elif args.noise_name=='Expinf' and args.norm==1:
        def radius_iso(pa):
            return 2*d*lambd*(pa-0.5)
    elif args.noise_name=='Expinf' and args.norm==-1:
        def radius_iso(pa):
            return lambd*np.log(1/(2*(1-pa)))
    elif args.noise_name=='Uniform' and args.norm==1:
        def radius_iso(pa):
            return 2*lambd*(pa-0.5)
    elif args.noise_name=='Uniform' and args.norm==-1:
        def radius_iso(pa):
            return 2*lambd*(1-(3/2-pa)**(1/d))
    elif args.noise_name=='PowerLaw' and args.norm==1:
        def radius_iso(pa):
            a=4000
            return 2*d*lambd/(a-d)*(pa-0.5)

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
    model.load_state_dict(torch.load("./model_saved/mnist_{}_ourmodel_{}_sigma{}_best.pth".format(args.method,args.noise_name,args.sigma)))
    model.cuda()


    # NoiseGenerator1 = Generator(3*32*32)
    # NoiseGenerator2 = Generator(3 * 32 * 32)
    # NoiseGenerator1.load_state_dict(torch.load("./model_saved/CIFAR10_{}_NoiseGenerator1_{}_sigma{}_last.pth".format(args.method,args.noise_name,args.sigma)))
    # NoiseGenerator2.load_state_dict(torch.load("./model_saved/CIFAR10_{}_NoiseGenerator2_{}_sigma{}_last.pth".format(args.method,args.noise_name, args.sigma)))
    # NoiseGenerator1.cuda()
    # NoiseGenerator2.cuda()
    anisotropic_pattern=np.load(args.pattern)
    anisotropic_pattern=torch.from_numpy(anisotropic_pattern).repeat((c,1,1)).cuda()
    Smoother=Smooth_Preassigned(model,CLASS,anisotropic_pattern)

    print('start')
    if os.path.exists('./results/mnist/mnist_{}_{}_sigma{}_results_pA.npy'.format(args.method,args.noise_name,args.sigma)):
        pA_list = np.load('./results/mnist/mnist_{}_{}_sigma{}_results_pA.npy'.format(args.method,args.noise_name,args.sigma)).tolist()
        R_list=np.load('./results/mnist/mnist_{}_{}_sigma{}_results_radius.npy'.format(args.method,args.noise_name,args.sigma)).tolist()
        variance_term_list=np.load('./results/mnist/mnist_{}_{}_sigma{}_results_varianceterm.npy'.format(args.method,args.noise_name,args.sigma)).tolist()
        dif=len(pA_list)
    else:
        pA_list = []
        R_list=[]
        variance_term_list=[]
        dif=0
    valid=0
    count=0
    # dif=0
    for j in tqdm(range(10000-dif)):
        i=(j+dif)*skip
        # i=np.random.randint(10000)
        print('fig {} certifying'.format(i))
        (x, y) = test_dataset[i]
        X = x.cuda().unsqueeze(0)
        prediction, pA,variance=Smoother.certify(X,n0=100,n=100000,alpha=0.001,batch_size=args.batch,noise_name=args.noise_name,sigma=args.sigma)
        variance_logsum=torch.sum(torch.log(torch.abs(variance))).cpu().data.numpy()

        correct = int(prediction == y)
        if correct:
            pA_list.append(pA)
            if args.IsoMeasure:
                variance_term=torch.min(variance).cpu().data.numpy()
                R = radius_iso(pA) * variance_term
            else:
                variance_term=np.exp(variance_logsum/d)
                variance_term_list.append(variance_term)
                R=radius_iso(pA)*variance_term
            R_list.append(R)
            valid+=1
            print('pA={},variance_term={}, R={}'.format(pA,variance_term,R))
        else:
            pA_list.append(-1)
            if args.IsoMeasure:
                variance_term=torch.min(variance).cpu().data.numpy()
                R = radius_iso(pA) * variance_term
            else:
                variance_term=np.exp(variance_logsum/d)
                variance_term_list.append(variance_term)
                R=radius_iso(pA)*variance_term
            R_list.append(R)
            print('pA={},variance_term={}, R={}'.format(-1,variance_term,R))
        count+=1
        print('acc: {}'.format(valid/count))
        np.save('./results/mnist/mnist_{}_{}_sigma{}_results_pA.npy'.format(args.method,args.noise_name,args.sigma), pA_list)
        np.save('./results/mnist/mnist_{}_{}_sigma{}_results_radius.npy'.format(args.method,args.noise_name,args.sigma), R_list)
        np.save('./results/mnist/mnist_{}_{}_sigma{}_results_varianceterm.npy'.format(args.method,args.noise_name,args.sigma), variance_term_list)