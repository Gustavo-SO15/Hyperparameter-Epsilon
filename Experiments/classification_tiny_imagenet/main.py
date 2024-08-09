
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from torch.optim import Adam, RMSprop, SGD
from models import *
from optimizers import *
from cal_eps import *
from tqdm import tqdm, trange

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--total_epoch', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--decay_epoch', default=150, type=int, help='Number of epochs to decay learning rate')
    parser.add_argument('--model', default='resnet', type=str, help='model', choices=['resnet', 'densenet', 'vgg','alexnet'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer', choices=['sgd', 'adam','rmsprop', 'adabelief', 'adammom',])
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.1, type=float, help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=1e-3, type=float, help='convergence speed term of AdaBound')
    parser.add_argument('--cal_eps', action='store_true', default=False, help='calculate search range of epsilon hyperparameter')    
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizers')
    parser.add_argument('--reset', action = 'store_true', help='whether reset optimizer at learning rate decay')
    return parser


def build_dataset(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
      #  transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    transform_test = transforms.Compose([
       # transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dir = './data/tiny-imagenet-200/train'
    
    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
            
    test_dir = './data/tiny-imagenet-200/val'
    
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_test) 
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize, shuffle=False, num_workers=2)
        
    # classes = 200

    return train_loader, test_loader
    


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, gamma=1e-3, eps=1e-8, weight_decay=5e-4,
                  reset = False, run = 0, weight_decouple = False, rectify = False):
    name = {
        'sgd': 'lr{}-momentum{}-wdecay{}-run{}'.format(lr, momentum, weight_decay, run),
        'adam': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),
        'rmsprop': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),
        'adabelief': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),
        'adammom': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),

    }[optimizer]
    return '{}-{}-{}-reset{}'.format(model, optimizer, name, str(reset))


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet': ResNet34,
        'densenet': DenseNet121,
        'vgg':vgg11,
        'alexnet':alexnet,
    }[args.model]()
    net = net.to(device)
    print("msg"+str(device))
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)      
    elif args.optim == 'rmsprop':
        return optim.RMSprop(model_params, args.lr, alpha=args.beta2,
                          weight_decay=args.weight_decay, eps=args.eps)
 
    elif args.optim == 'adabelief':
        return AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
  
    elif args.optim == 'adammom':
        return AdamMom(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    else:
        print('Optimizer not found')
        
        
def create_optimizer_cal_eps(args, model_params, num_batches):
    args.optim = args.optim.lower()
    if args.optim == 'adam':
        return Cal_eps_Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, num_batches=num_batches)      
    elif args.optim == 'rmsprop':
        return Cal_eps_RMSprop(model_params, args.lr, alpha= args.beta2,
                          weight_decay=args.weight_decay, num_batches=num_batches) 
    elif args.optim == 'adabelief':
        return Cal_eps_AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, num_batches=num_batches) 
    elif args.optim == 'adammom':
        return Cal_eps_AdamMom(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, num_batches=num_batches) 
    else:
        print('Optimizer not found')        
        
        

def train(net, epoch, device, data_loader, optimizer, criterion, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for (inputs, targets) in tqdm(data_loader, desc="Training",position=0):
#    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f'\tTrain Loss: {train_loss/total:.5f} | Train Acc: {accuracy:.3f}%')
    
    return accuracy, train_loss/total


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f'\tTest Loss: {test_loss/total:.5f} | Test Acc: {accuracy:.3f}%')
    
    return accuracy, test_loss/total

def adjust_learning_rate(optimizer, epoch, step_size=150, gamma=0.1, reset = False):
    for param_group in optimizer.param_groups:
        if epoch % step_size==0 and epoch>0:
            param_group['lr'] *= gamma

    if  epoch % step_size==0 and epoch>0 and reset:
        optimizer.reset()

def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset(args)
    
    num_batches = len(train_loader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              final_lr=args.final_lr, momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, gamma=args.gamma,
                              eps = args.eps,
                              reset=args.reset, run=args.run,
                              weight_decay = args.weight_decay)
    print('ckpt_name')
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']

        curve = os.path.join('curve', ckpt_name)     
        curve = torch.load(curve)
        train_losses = curve['train_loss']
        train_accuracies = curve['train_acc']
        test_losses = curve['test_loss']
        test_accuracies = curve['test_acc']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()

    if args.cal_eps == True:
        args.total_epoch = 1
        optimizer = create_optimizer_cal_eps(args, net.parameters(), num_batches)          
    else:	 
        optimizer = create_optimizer(args, net.parameters()) 

    for epoch in range(start_epoch + 1, args.total_epoch):
        start = time.time()
        #scheduler.step()
        for param_group in optimizer.param_groups:        
            print('lr:'+str(param_group['lr']))
            
           
        adjust_learning_rate(optimizer, epoch, step_size=args.decay_epoch, gamma=args.lr_gamma, reset = args.reset)
        train_acc,train_loss = train(net, epoch, device, train_loader, optimizer, criterion, args)
        
        test_acc, test_loss = test(net, device, test_loader, criterion)
        end = time.time()
        print('Time: {}'.format(end-start))

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
    print(os.path.join('curve', ckpt_name))
    if not os.path.isdir('curve'):
        os.mkdir('curve')
        
    if args.cal_eps == False:
        torch.save({'train_loss': train_losses, 'train_acc': train_accuracies, 'test_loss': test_losses, 'test_acc': test_accuracies},
               os.path.join('curve', ckpt_name))

if __name__ == '__main__':
    main()
