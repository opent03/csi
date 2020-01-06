import os
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.apl import *
import gc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
cuda_avail = torch.cuda.is_available()
from sklearn.metrics import confusion_matrix
if cuda_avail:
    torch.backends.cudnn.benchmark = True

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def train(net, epoch, train_loader, optimizer, scheduler, criterion, lmbda=1):
    'Basic training function'
    net.train()
    scheduler.step()
    l = []
    for (samples, labels) in tqdm(train_loader, ncols=70):
        samples = Variable(samples)
        if cuda_avail:
            samples, labels = samples.cuda(), labels.cuda()
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        labels_act, labels_loc = Variable(labels_act), Variable(labels_loc)
        labels_loc = labels_loc - 1
        optimizer.zero_grad()
        pred_act, pred_loc, _, _, _, _, _, _, _ = net(samples)
        loss_act = criterion(pred_act, labels_act)
        loss_loc = criterion(pred_loc, labels_loc)
        loss = loss_act + lmbda*loss_loc
        l.append(loss.data.item())
        loss.backward()
        #debug
        #plot_grad_flow(net.named_parameters())
        optimizer.step()
    print('Loss at {} epoch: {:.3f}\n'.format(epoch, loss.data.item()))
    l = l[:-1]
    return l


def evaluate(net, test_loader):
    'Basic evaluation function'
    net.eval()
    correct_test_act = 0
    correct_test_loc = 0
    num_instances = test_loader.dataset.__len__()
    c_y_act, c_y_loc = [], []
    c_pred_act, c_pred_loc = [], []
    for samples, labels in tqdm(test_loader, ncols=70):


        with torch.no_grad():
            samplesV = Variable(samples)
            if cuda_avail:
                samplesV, labels = samplesV.cuda(), labels.cuda()
            labels = labels.squeeze()

            labels_act = labels[:, 0].squeeze()
            labels_loc = labels[:, 1].squeeze()
            labelsV_act = Variable(labels_act)
            labelsV_loc = Variable(labels_loc)
            labelsV_loc = labelsV_loc - 1
            predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = net(samplesV)

            prediction = predict_label_loc.data.max(1)[1]
            correct_test_loc += prediction.eq(labelsV_loc.data.long()).sum()

            prediction = predict_label_act.data.max(1)[1]
            correct_test_act += prediction.eq(labelsV_act.data.long()).sum()

            # confusion matrix
            c_y_act.append(labelsV_act.cpu().numpy())
            c_y_loc.append(labelsV_loc.cpu().numpy())
            _, predict_label_act = torch.max(predict_label_act, 1)
            _, predict_label_loc = torch.max(predict_label_loc, 1)
            c_pred_act.append(predict_label_act.cpu().numpy())
            c_pred_loc.append(predict_label_loc.cpu().numpy())

    c_y_act, c_y_loc = np.concatenate(c_y_act), np.concatenate(c_y_loc)
    c_pred_act, c_pred_loc = np.concatenate(c_pred_act), np.concatenate(c_pred_loc)

    actmat = confusion_matrix(c_y_act, c_pred_act)
    locmat = confusion_matrix(c_y_loc, c_pred_loc)
    acc_act = float(correct_test_act ) / num_instances
    acc_loc = float(correct_test_loc) / num_instances
    print("Activity accuracy: {:.2f}%".format(100 * acc_act))
    print("Location accuracy: {:.2f}%\n".format(100 * acc_loc))
    return [acc_act, acc_loc]


def load_data(data_dir):
    # format labels
    X_train, X_test = np.load(os.path.join(data_dir, 'X_train.npy')), np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train, y_test = np.load(os.path.join(data_dir, 'y_train.npy')), np.load(os.path.join(data_dir, 'y_test.npy'))
    X_train, X_test = np.transpose(X_train, (0, 2, 1)), np.transpose(X_test, (0, 2, 1))

    # remove outlier
    X_train = np.delete(X_train, 398, axis=0)
    y_train = np.delete(y_train, 398, axis=0)

    y_train_loc, y_train_act = y_train[:, 1], y_train[:, 2]
    y_test_loc, y_test_act = y_test[:, 1], y_test[:, 2]
    num_train_instances, num_test_instances = X_train.shape[0], X_test.shape[0]

    # TRAIN
    y_train_act = np.reshape(y_train_act, (num_train_instances, 1))
    y_train_loc = np.reshape(y_train_loc, (num_train_instances, 1))
    train_label = np.concatenate((y_train_act, y_train_loc), 1)
    train_data = torch.from_numpy(X_train).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)
    train_dataset = TensorDataset(train_data, train_label)
    del (X_train)
    gc.collect()

    # TEST
    y_test_act = np.reshape(y_test_act, (num_test_instances, 1))
    y_test_loc = np.reshape(y_test_loc, (num_test_instances, 1))
    test_label = np.concatenate((y_test_act, y_test_loc), 1)
    test_data = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)
    test_dataset = TensorDataset(test_data, test_label)
    del (X_test)
    gc.collect()

    # CLEANUP
    del (y_train)
    del (y_test)
    gc.collect()

    # data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# params
batch_size = 128
epochs = 30
weights_dir = 'weights'
learning_rate = 5e-4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='lab_traintest', help='self-explanatory')
    args = parser.parse_args()

    # data loading
    train_loader, test_loader = load_data(args.data_dir)

    # setup model
    net = DiscriminatorNet(block=BasicBlock, layers=[3,4,6,3], inchannel=90)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                                                                 140, 150, 160, 170, 180, 190, 200, 250, 300],
                                                     gamma=0.5)
    if cuda_avail: # literally the most important thing
        net = net.cuda()
        criterion = criterion.cuda()

    # things for funny graphs
    loss_tr = []
    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        epoch += 1
        print('Epoch:', epoch)
        print('Training...')
        loss_tr.append(train(net, epoch, train_loader, optimizer, scheduler, criterion))
        print('Evaluation on training data...')
        train_acc.append(evaluate(net, train_loader))
        print('Evaluation on validation data...')
        test_acc.append(evaluate(net, test_loader))

    train_acc, test_acc = np.array(train_acc), np.array(test_acc)
    loss_tr = np.concatenate(loss_tr)

    # save weights
    torch.save(net.state_dict(), os.path.join(weights_dir, (args.data_dir + '.pth')))
    # plot loss over epochs
    plt.figure()
    plt.title('Loss over batch #')
    plt.plot(loss_tr)
    plt.xlabel('?')
    plt.ylabel('sigma cross-entropy')

    # plot activity
    plt.figure()
    plt.title('Activity accuracy')
    plt.plot(train_acc[:,0], '-r', label='act_train')
    plt.plot(test_acc[:,0], '-b', label='act_val')
    plt.ylim(0,1)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')

    # plot location
    plt.figure()
    plt.title('Location accuracy')
    plt.plot(train_acc[:,1], '-b', label='loc_train')
    plt.plot(test_acc[:,1], '-y', label='loc_val')
    plt.ylim(0,1)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()