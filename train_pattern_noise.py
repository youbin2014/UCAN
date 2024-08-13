# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.
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
from noisegenerator import NoiseGenerator, Generator
# from train_baselines import transform_lambda, noise_baselines
import numpy as np
from matplotlib import pyplot as plt
from noises import *
from utils.plot_examples import plot_sample


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--method',type=str)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=10,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--model_path', type=str, default='./model_saved/ResNet101_CIFAR10_our_2,0.707_best.pth')
parser.add_argument('--noisegenerator1_path', type=str, default='')
parser.add_argument('--noisegenerator2_path', type=str, default='')
parser.add_argument('--input_size', type=int, default=32 * 32 * 3)
parser.add_argument('--sigma', type=float, default=1,
                    help='sigma')
parser.add_argument('--noise_name',type=str,default='Gaussian')
parser.add_argument('--train',type=int,default=1)
parser.add_argument('--pattern',type=str,default='/home/cc/NoiseGenerator/results/cifar10_radial_preassign_pattern.npy')








args = parser.parse_args()

def train(model,optimizer,trainloader,epoch,pattern):
    model.train()
    # NoiseGenerator1.train()
    # NoiseGenerator2.train()
    # print('training noise generator')
    pbar=tqdm(enumerate(trainloader))
    for batch_idx,(x,y) in pbar:
        X = x.cuda()
        label = y.cuda()

        optimizer.zero_grad()
        # mean=NoiseGenerator1(torch.ones(X.shape[0]).cuda().unsqueeze(1))
        # mean=2*(mean-0.5)
        # variance=NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1))*5 # cifar10
        # variance = NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1)) * 4  # imagenet
        # variance=torch.abs(variance)
        # mean=0*mean
        # variance=0*variance
        # mean=mean*0
        # variance=torch.ones_like(variance).cuda()
        # mean=mean.view(X.shape)
        # variance=variance.view(X.shape)
        # noise=mean+torch.randn_like(X)*variance
        clean_outputs = model(X)
        pred_clean = torch.max(clean_outputs, 1)[1]
        # loss_clean = F.cross_entropy(clean_outputs, label)
        # loss_clean=F.cross_entropy(pred_clean, label)
        loss_smooth=0
        loss_pert=0
        loss_variance=0
        step_score_smoothed=0

        for i in range(5):
            lambd=transform_lambda(args.noise_name,args.sigma)
            noise=noise_baselines(args.noise_name,X,lambd=lambd)-X
            # noise=torch.randn_like(X)
            noise_input = X.float()+noise.float() * pattern.repeat((X.shape[0],1,1,1)).reshape(X.shape).float()
            smoothed_outputs = model(noise_input.cuda().reshape(X.shape))
            pred_smoothed =torch.max(smoothed_outputs, 1)[1]
            step_score_prf = accuracy_score(label.cpu().data.squeeze().numpy(),
                                            pred_smoothed.cpu().data.squeeze().numpy())
            step_score_smoothed+=step_score_prf
            loss_smooth += F.cross_entropy(smoothed_outputs, label)
            # loss_pert -= torch.mean(torch.abs(mean))
            # loss_variance-=torch.mean(torch.abs(variance))
            # loss_variance += torch.mean(torch.abs(args.sigma-torch.min(torch.abs(variance.view(variance.size()[0], -1)), dim=1)[0]))/args.sigma

        loss=loss_smooth
        loss.backward()
        optimizer.step()
        # if batch_idx%100==0: #cifar10 20, imagenet 100
        #     optimizer1.step()
        #     optimizer2.step()
        #     optimizer1.zero_grad()
        #     optimizer2.zero_grad()

        step_score = accuracy_score(label.cpu().data.squeeze().numpy(), pred_clean.cpu().data.squeeze().numpy())
        step_score_smoothed=step_score_smoothed/5
        pbar.set_description('E|{}|Lm:{:.4f}|Lv{:.4f}|Ls{:.4f}|C:{:.2f}|S{:.2f}'.format(epoch+1,0,0,loss_smooth.item()/5,step_score*100,step_score_smoothed*100))
    # print(torch.exp(torch.mean(torch.log(variance))))
    # M = mean[0].data.cpu().numpy().reshape(X[0].shape)
    # M = np.swapaxes(np.swapaxes(M, 0, 2), 0, 1)
    # plt.figure()
    # plt.imshow(M)
    # plt.colorbar()
    # plt.savefig('./visualization/{}_mean_map.png'.format(args.noise_name))
    # V = variance[0].data.cpu().numpy().reshape(X[0].shape)
    # V = np.swapaxes(np.swapaxes(V, 0, 2), 0, 1)[:,:,0]
    # plt.figure()
    # plt.imshow(V)
    # plt.colorbar()
    # plt.savefig('./visualization/{}_variance_map.png'.format(args.noise_name))

def test(model,optimizer,testloader,pattern):
    model.eval()
    # NoiseGenerator1.eval()
    # NoiseGenerator2.eval()
    all_label = []
    all_pred = []

    with torch.no_grad():
        pbar=tqdm(enumerate(testloader))
        for batch_idx,(x,y) in pbar:
            # print(y)
            X = x.cuda()
            label = y.cuda()
            optimizer.zero_grad()
            # optimizer2.zero_grad()
            # optimizer3.zero_grad()
            # mean = NoiseGenerator1(torch.ones(X.shape[0]).cuda().unsqueeze(1))
            # mean = 2 * (mean - 0.5)
            # variance = NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1))*5  # cifar10
            # variance = NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1)) * 4  # imagenet
            # variance = torch.abs(variance)
            # mean=mean*0
            # variance = torch.ones_like(variance).cuda()
            lambd=transform_lambda(args.noise_name,args.sigma)
            noise=noise_baselines(args.noise_name,X,lambd=lambd)-X
            # noise_input = X+mean.reshape(X.shape) + (noise-X) * variance.reshape(X.shape)-X
            # noise = torch.randn_like(X)
            noise_input = X+noise * pattern.repeat((X.shape[0],1,1,1)).reshape(X.shape).float()

            # plot_sample(X,noise_input,None,None,'PreAssigned_Square','imagenet',[0,1,2,3])


            smoothed_outputs = model(noise_input.cuda())
            pred_smoothed = torch.max(smoothed_outputs, 1)[1]
            all_label.extend(label)
            all_pred.extend(pred_smoothed)

    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)


    test_score = accuracy_score(all_label.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    return test_score*100

if __name__ == '__main__':
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)


    pin_memory = (args.dataset == "imagenet")
    if args.train:
        train_dataset = get_dataset(args.dataset, 'train')
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                  num_workers=args.workers, pin_memory=pin_memory)
    test_dataset = get_dataset(args.dataset, 'test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)
    if args.model_path!="":

        checkpoint = torch.load(args.model_path)
        # if 'state_dic' in checkpoint:
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model.load_state_dict(checkpoint)
    if args.dataset=='cifar10':
        SIZE=32
        c=3
    elif args.dataset=='imagenet':
        SIZE=224
        c=3
    elif args.dataset=='mnist':
        SIZE=28
        c=1
    else:
        print("unknown dataset")

    # Constants
    kappa = 0.2
    iota = 0.8

    # Generate a 32x32 grid of (a, b) coordinates, centered at (0,0)
    x = np.linspace(-(SIZE-1)/2, (SIZE-1)/2, SIZE)
    y = np.linspace(-(SIZE-1)/2, (SIZE-1)/2, SIZE)
    a, b = np.meshgrid(x, y)


    # Function to calculate sigma for different p norm
    def calculate_sigma(a, b, p, kappa, iota):
        a=a/(SIZE-1)/2
        b=b/(SIZE-1)/2
        if p == 1:
            norm = np.abs(a) + np.abs(b)
        elif p == 2:
            norm = np.sqrt(a ** 2 + b ** 2)
        elif p == np.inf:
            norm = np.maximum(np.abs(a), np.abs(b))
        else:
            raise ValueError("Unsupported norm")

        return kappa * norm ** 2 + iota


    # Calculate sigma for p = 1, 2, infinity
    sigma_1 = calculate_sigma(a, b, 1, kappa, iota)
    sigma_2 = calculate_sigma(a, b, 2, kappa, iota)
    sigma_inf = calculate_sigma(a, b, np.inf, kappa, iota)

    if args.pattern=="l1":
        anisotropic_pattern=sigma_1
    elif args.pattern=="l2":
        anisotropic_pattern = sigma_2
    elif args.pattern=="linf":
        anisotropic_pattern = sigma_inf
    else:
        raise NotImplementedError
    if args.dataset=="mnist":
        anisotropic_pattern = torch.from_numpy(anisotropic_pattern).repeat((1, 1, 1)).cuda()
    else:
        anisotropic_pattern = torch.from_numpy(anisotropic_pattern).repeat((c, 1, 1)).cuda()

    # NoiseGenerator = NoiseGenerator(in_nc=3, out_nc=3, BlockNum=1, size=SIZE)

    # NoiseGenerator1 = Generator(3*SIZE*SIZE)
    # NoiseGenerator2 = Generator(3*SIZE*SIZE)
    # if args.noisegenerator1_path!="":
    #     NoiseGenerator1.load_state_dict(torch.load(args.noisegenerator1_path))
    # if args.noisegenerator2_path!="":
    #     NoiseGenerator2.load_state_dict(torch.load(args.noisegenerator2_path))
    # NoiseGenerator1.cuda()
    # NoiseGenerator2.cuda()

    # NoiseGenerator.load_state_dict(torch.load(args.noisegenerator))

    # optimizer1 = SGD(NoiseGenerator1.parameters(), lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler1 = StepLR(optimizer1, step_size=args.lr_step_size, gamma=args.gamma)
    # optimizer2 = SGD(NoiseGenerator2.parameters(), lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler2 = StepLR(optimizer2, step_size=args.lr_step_size, gamma=args.gamma)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    num_epoch = args.epochs
    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epoch):
        if args.train:
            train(model,optimizer, train_loader, epoch,anisotropic_pattern)
        test_score = test(model,optimizer, test_loader,anisotropic_pattern)
        if test_score > best_acc:
            best_epoch = epoch + 1
            best_acc = test_score
            # torch.save(NoiseGenerator1.state_dict(), 'model_saved/CIFAR10_{}_NoiseGenerator1_{}_sigma{}_best.pth'.format(args.method,args.noise_name,args.sigma))
            # torch.save(NoiseGenerator2.state_dict(), 'model_saved/CIFAR10_{}_NoiseGenerator2_{}_sigma{}_best.pth'.format(args.method,args.noise_name,args.sigma))
            torch.save(model.state_dict(),
                       './model_saved/{}_{}_ourmodel_{}_sigma{}_pattern_{}_best.pth'.format(args.dataset,args.method, args.noise_name,
                                                                                       args.sigma,args.pattern))
        print(
            'Epoch:{},Test Acc:{:.4f},Best Acc:{:.4f} at epoch {}'.format(epoch + 1, test_score, best_acc, best_epoch))
        # torch.save(NoiseGenerator1.state_dict(), './model_saved/CIFAR10_{}_NoiseGenerator1_{}_sigma{}_last.pth'.format(args.method,args.noise_name,args.sigma))
        # torch.save(NoiseGenerator2.state_dict(), './model_saved/CIFAR10_{}_NoiseGenerator2_{}_sigma{}_last.pth'.format(args.method,args.noise_name,args.sigma))
        #
        torch.save(model.state_dict(), './model_saved/{}_{}_ourmodel_{}_sigma{}_pattern_{}_last.pth'.format(args.dataset,args.method,args.noise_name,args.sigma,args.pattern))

