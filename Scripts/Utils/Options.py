import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=50, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=1024, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="weight_decay")
    parser.add_argument('--algorithm', type=str, default='SAGE', help="name of algorithm")
    parser.add_argument('--Drichlet_arg', type=float, default=0.1, help='Drichlet_arg')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--prox_alpha', type=float, default=0.01, help="alpha for FedProx, 1e-1, 1e-2, 1e-3, 1e-4")
    
    # SAGE optimization problem parameters 
    parser.add_argument('--a', type=float, default=0.3, help="value of a in optimization problem") 
    parser.add_argument('--b', type=float, default=0.2, help="value of b in optimization problem") 
    parser.add_argument('--c', type=float, default=0.5, help="value of c in optimization problem") 
    parser.add_argument('--k', type=int, default=2, help="minimum number of clients to be selected")
    parser.add_argument('--max_energy', type=float, default=15, help="maximum energy of clients, default is 15.0")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    p = args_parser()
