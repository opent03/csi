from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='lab_traintest', help='self-explanatory')
    args = parser.parse_args()
    data_dir = args.data_dir

    'default use 3-4-6-3 architecture'
    net = ResNet(block=BasicBlock, layers=[3,4,6,3], inchannel=90)
    net.load_state_dict(torch.load(os.path.join(weights_dir, (data_dir + '.pth'))))
    if cuda_avail:
        net = net.cuda()

    # SOme print statements
    print('Beginning evaluation phase')
    print('Home directory: {}'.format(data_dir))

    # Evaluation: same test set
    print('Evaluation: same test set')
    train_loader, test_loader = load_data(data_dir)
    evaluate(net, test_loader)


    # Evaluation: different test set
    if 'lab' in data_dir:
        diff_dir = 'meeting_traintest'
    else:
        diff_dir = 'lab_traintest'
    print('Evaluation: different test set')
    train_loader, test_loader = load_data(diff_dir)
    evaluate(net, test_loader)