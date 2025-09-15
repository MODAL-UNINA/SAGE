from Model.LeNet import LeNet

def get_model(args):
    net = LeNet(args).to(args.device)
    return net
