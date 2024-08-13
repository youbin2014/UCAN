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
from noisegenerator import NoiseGenerator
import torch.distributions as D
from noises import *
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
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
parser.add_argument('--noisegenerator', type=str, default='./model_saved/ResNet101_CIFAR10_our_2,0.707_best.pth')
parser.add_argument('--input_size', type=int, default=32 * 32 * 3)
parser.add_argument('--sigma', type=float, default=1,
                    help='sigma')
parser.add_argument('--noise_name',type=str,default='Gaussian')





args = parser.parse_args()


def train(model, optimizer1,trainloader,epoch):
    model.train()
    # print('training noise generator')
    pbar=tqdm(enumerate(trainloader))
    for batch_idx,(x,y) in pbar:
        X = x.cuda()
        label = y.cuda()
        optimizer1.zero_grad()
        # mean,variance=NoiseGenerator(X)
        # mean=mean*0
        # variance=variance*args.sigma
        # variance=torch.ones_like(variance).cuda()
        clean_outputs = model(X)
        pred_clean = torch.max(clean_outputs, 1)[1]
        # loss_clean = F.cross_entropy(clean_outputs, label)

        loss_smooth=0
        loss_pert=0
        loss_variance=0
        step_score_smoothed=0

        for i in range(3):
            lambd=transform_lambda(args.noise_name,args.sigma)
            noise_input=noise_baselines(args.noise_name,X,lambd=lambd)
            smoothed_outputs = model(noise_input.cuda().reshape(X.shape))
            pred_smoothed =torch.max(smoothed_outputs, 1)[1]
            step_score_prf = accuracy_score(label.cpu().data.squeeze().numpy(),
                                            pred_smoothed.cpu().data.squeeze().numpy())
            step_score_smoothed+=step_score_prf
            loss_smooth += F.cross_entropy(smoothed_outputs, label)
            # loss_pert += F.mse_loss(X + mean, X)
            # loss_variance += torch.mean(torch.abs(args.sigma-torch.min(torch.abs(variance.view(variance.size()[0], -1)), dim=1)[0]))/args.sigma
            # loss_variance += torch.mean(torch.clamp(torch.abs(variance.view(variance.size()[0], -1)),min=0))
        #
        # loss=loss_pert + loss_variance+loss_smooth
        loss=loss_smooth
        loss.backward()
        optimizer1.step()

        step_score = accuracy_score(label.cpu().data.squeeze().numpy(), pred_clean.cpu().data.squeeze().numpy())
        step_score_smoothed=step_score_smoothed/3
        pbar.set_description('E|{}|Ls{:.4f}|C:{:.2f}|S{:.2f}'.format(epoch+1,loss_smooth.item(),step_score*100,step_score_smoothed*100))

def test(model, optimizer1,testloader):
    model.eval()
    all_label = []
    all_pred = []

    with torch.no_grad():
        pbar=tqdm(enumerate(testloader))
        for batch_idx,(x,y) in pbar:
            # print(y)
            X = x.cuda()
            label = y.cuda()
            optimizer1.zero_grad()
            lambd = transform_lambda(args.noise_name, args.sigma)
            noise_input = noise_baselines(args.noise_name, X, lambd=lambd)
            smoothed_outputs = model(noise_input.cuda().reshape(X.shape))
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

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)
    # checkpoint = torch.load(args.model_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(torch.load(args.model_path))

    if args.dataset=='cifar10':
        SIZE=32
    elif args.dataset=='imagenet':
        SIZE=224
    else:
        print("unknown dataset")
    # NoiseGenerator = NoiseGenerator(in_nc=3, out_nc=3, BlockNum=1, size=SIZE)
    # NoiseGenerator.cuda()
    # NoiseGenerator.load_state_dict(torch.load(args.noisegenerator))

    # optimizer1 = SGD(NoiseGenerator.parameters(), lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler1 = StepLR(optimizer1, step_size=args.lr_step_size, gamma=args.gamma)
    optimizer1 = SGD(model.parameters(), lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler1 = StepLR(optimizer1, step_size=args.lr_step_size, gamma=args.gamma)

    num_epoch = args.epochs
    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epoch):

        train(model, optimizer1, train_loader, epoch)
        test_score = test(model, optimizer1, test_loader)
        if test_score > best_acc:
            best_epoch = epoch + 1
            best_acc = test_score
            torch.save(model.state_dict(), './model_saved/CIFAR10_model_{}_sigma{}_best.pth'.format(args.noise_name,1.0))
        print(
            'Epoch:{},Test Acc:{:.4f},Best Acc:{:.4f} at epoch {}'.format(epoch + 1, test_score, best_acc, best_epoch))
        torch.save(model.state_dict(), './model_saved/CIFAR10_model_{}_sigma{}_last.pth'.format(args.noise_name,1.0))

