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








args = parser.parse_args()

def train(model,NoiseGenerator1,NoiseGenerator2, optimizer1,optimizer2,optimizer3,trainloader,epoch):
    model.train()
    NoiseGenerator1.train()
    NoiseGenerator2.train()
    # print('training noise generator')
    pbar=tqdm(enumerate(trainloader))
    for batch_idx,(x,y) in pbar:
        X = x.cuda()
        label = y.cuda()

        optimizer3.zero_grad()
        mean=NoiseGenerator1(torch.ones(X.shape[0]).cuda().unsqueeze(1))
        # mean=2*(mean-0.5)
        # variance=NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1))*1.2 # cifar10
        variance = NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1)) * 1  # imagenet
        # variance = NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1)) * 3  # mnist
        variance=torch.abs(variance)
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
            noise_input = X+mean.reshape(X.shape) + noise * variance.reshape(X.shape)
            smoothed_outputs = model(noise_input.cuda().reshape(X.shape))
            pred_smoothed =torch.max(smoothed_outputs, 1)[1]
            step_score_prf = accuracy_score(label.cpu().data.squeeze().numpy(),
                                            pred_smoothed.cpu().data.squeeze().numpy())
            step_score_smoothed+=step_score_prf
            loss_smooth += F.cross_entropy(smoothed_outputs, label)
            loss_pert -= torch.mean(torch.abs(mean))
            loss_variance-=torch.min(torch.abs(variance))
            # loss_variance += torch.mean(torch.abs(args.sigma-torch.min(torch.abs(variance.view(variance.size()[0], -1)), dim=1)[0]))/args.sigma

        loss=loss_variance*5+loss_smooth
        loss.backward()
        optimizer3.step()
        if batch_idx%20==0: #cifar10 20, imagenet 100
            optimizer1.step()
            optimizer2.step()
            optimizer1.zero_grad()
            optimizer2.zero_grad()

        step_score = accuracy_score(label.cpu().data.squeeze().numpy(), pred_clean.cpu().data.squeeze().numpy())
        step_score_smoothed=step_score_smoothed/5
        pbar.set_description('E|{}|Lm:{:.4f}|Lv{:.4f}|Ls{:.4f}|C:{:.2f}|S{:.2f}'.format(epoch+1,loss_pert.item()/5,loss_variance.item()/5,loss_smooth.item()/5,step_score*100,step_score_smoothed*100))
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

def test(model,NoiseGenerator1,NoiseGenerator2, optimizer1,optimizer2,optimizer3,testloader):
    model.eval()
    NoiseGenerator1.eval()
    NoiseGenerator2.eval()
    all_label = []
    all_pred = []

    with torch.no_grad():
        pbar=tqdm(enumerate(testloader))
        for batch_idx,(x,y) in pbar:
            # print(y)
            X = x.cuda()
            label = y.cuda()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            mean = NoiseGenerator1(torch.ones(X.shape[0]).cuda().unsqueeze(1))
            # mean = 2 * (mean - 0.5)
            # variance = NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1))*1.2  # cifar10
            variance = NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1)) * 1  # imagenet
            # variance = NoiseGenerator2(torch.ones(X.shape[0]).cuda().unsqueeze(1)) * 3  # mnist

            variance = torch.abs(variance)
            # mean=mean*0
            # variance = torch.ones_like(variance).cuda()
            lambd=transform_lambda(args.noise_name,args.sigma)
            noise=noise_baselines(args.noise_name,X,lambd=lambd)-X
            # noise_input = X+mean.reshape(X.shape) + (noise-X) * variance.reshape(X.shape)-X
            # noise = torch.randn_like(X)
            noise_input = X + mean.reshape(X.shape) + noise * variance.reshape(X.shape)

            # plot_sample(X,noise_input,mean,variance,'Universal','imagenet',[0,1,2,3])


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
    # NoiseGenerator = NoiseGenerator(in_nc=3, out_nc=3, BlockNum=1, size=SIZE)

    NoiseGenerator1 = Generator(c*SIZE*SIZE)
    NoiseGenerator2 = Generator(c*SIZE*SIZE)
    if args.noisegenerator1_path!="":
        NoiseGenerator1.load_state_dict(torch.load(args.noisegenerator1_path))
    if args.noisegenerator2_path!="":
        NoiseGenerator2.load_state_dict(torch.load(args.noisegenerator2_path))
    NoiseGenerator1.cuda()
    NoiseGenerator2.cuda()

    # NoiseGenerator.load_state_dict(torch.load(args.noisegenerator))

    optimizer1 = SGD(NoiseGenerator1.parameters(), lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler1 = StepLR(optimizer1, step_size=args.lr_step_size, gamma=args.gamma)
    optimizer2 = SGD(NoiseGenerator2.parameters(), lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler2 = StepLR(optimizer2, step_size=args.lr_step_size, gamma=args.gamma)
    optimizer3 = SGD(model.parameters(), lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler3 = StepLR(optimizer3, step_size=args.lr_step_size, gamma=args.gamma)

    num_epoch = args.epochs
    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epoch):
        if args.train:
            train(model,NoiseGenerator1,NoiseGenerator2, optimizer1,optimizer2,optimizer3, train_loader, epoch)
        test_score = test(model,NoiseGenerator1,NoiseGenerator2, optimizer1,optimizer2,optimizer3, test_loader)
        if test_score > best_acc:
            best_epoch = epoch + 1
            best_acc = test_score
            torch.save(NoiseGenerator1.state_dict(), 'model_saved/{}_{}_NoiseGenerator1_{}_sigma{}_best.pth'.format(args.dataset,args.method,args.noise_name,args.sigma))
            torch.save(NoiseGenerator2.state_dict(), 'model_saved/{}_{}_NoiseGenerator2_{}_sigma{}_best.pth'.format(args.dataset,args.method,args.noise_name,args.sigma))
            torch.save(model.state_dict(), 'model_saved/{}_{}_ourmodel_{}_sigma{}_best.pth'.format(args.dataset,args.method,args.noise_name,args.sigma))
        print(
            'Epoch:{},Test Acc:{:.4f},Best Acc:{:.4f} at epoch {}'.format(epoch + 1, test_score, best_acc, best_epoch))
        torch.save(NoiseGenerator1.state_dict(), './model_saved/{}_{}_NoiseGenerator1_{}_sigma{}_last.pth'.format(args.dataset,args.method,args.noise_name,args.sigma))
        torch.save(NoiseGenerator2.state_dict(), './model_saved/{}_{}_NoiseGenerator2_{}_sigma{}_last.pth'.format(args.dataset,args.method,args.noise_name,args.sigma))

        torch.save(model.state_dict(), './model_saved/{}_{}_ourmodel_{}_sigma{}_last.pth'.format(args.dataset,args.method,args.noise_name,args.sigma))

