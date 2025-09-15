import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
import os
import subprocess

def get_gpu_power_consume(device=0):
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    power_draw = result.stdout.decode('utf-8').strip().split('\n')[device]
    return float(power_draw)


class DatasetSplit(Dataset):

    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        image, label = self.dataset[self.idx[item]]
        return image, label
    
class Custom_Local_SAGE(object): 


    def __init__(self, args, dataset, idx, net_glob, client_id, local_ep = None, path=None):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(DatasetSplit(dataset, idx), batch_size=args.local_bs, shuffle=True,
                                        drop_last=False)
        self.net_glob = net_glob
        self.client_id = client_id
        self.local_ep = local_ep if local_ep is not None else args.local_ep
        self.path = path
        

    def train(self, net, time_diff=None, train=True):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=self.args.lr)
        path = self.path

        os.makedirs(path, exist_ok=True)

        power_consume=[]
        total_time_append=0
        t0 = time.time()
        if train:
            t0 = time.time()
            print(f"Client {self.client_id} training...")
            print("The current number of local epoch is:", self.local_ep)
            for iter in range(self.local_ep):
                for batch_idx, (images, labels) in enumerate(self.train_loader):
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)

                    t_append_0 = time.time()
                    power_consume.append(get_gpu_power_consume())
                    t_append_1 = time.time()

                    outputs = net(images)['output']
                    loss = self.criterion(outputs, labels)
                    optimizer.zero_grad()

                    t_append_2 = time.time()
                    power_consume.append(get_gpu_power_consume())
                    t_append_3 = time.time()

                    if self.args.algorithm == 'FedProx':
                        penalize_loss = 0.0
                        global_weight_collector = [p.detach().clone() for p in self.net_glob.parameters()]

                        for param_index, param in enumerate(net.parameters()):
                            penalize_loss += (self.args.prox_alpha / 2) * torch.norm(
                                param - global_weight_collector[param_index]) ** 2

                        loss = loss + penalize_loss

                    loss.backward()
                    optimizer.step()

                    t_append_4 = time.time()
                    power_consume.append(get_gpu_power_consume())
                    t_append_5 = time.time()
                    
                    total_time_append += (t_append_1 - t_append_0 + t_append_3 - t_append_2 + t_append_5 - t_append_4)


            t1 = time.time()
            time_diff = t1 - t0
            print('Training completed in', time_diff, 'seconds (without append)')
            print('GPU power consume train: ', np.mean(power_consume), 'Watt in', time_diff, 'seconds')
            watthours_train = np.mean(power_consume) * time_diff/3600
            print('Energy consume in train: ', watthours_train, 'Wh \n')
            
            # Test

            power_test=[]
            t0 = time.time()
            t0_append = time.time()
            power_test.append(get_gpu_power_consume())
            t1_append = time.time()

            net.eval()
            soft_label = None
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(self.train_loader):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    log_prob = torch.softmax(net(images)['output'], dim=1)
                    if soft_label is None:
                        soft_label = log_prob
                    else:
                        soft_label = torch.cat((soft_label, log_prob), 0)

            soft_label = torch.sum(soft_label, dim=0) / soft_label.shape[0]
            t2_append = time.time()
            power_test.append(get_gpu_power_consume())
            t3_append = time.time()

            t1 = time.time()

            time_diff = t1 - t0 - (t1_append - t0_append) - (t3_append - t2_append)
            watthours_test = np.mean(power_test) * time_diff/3600
            print('GPU power consume test: ', np.mean(power_test), 'Watt in', time_diff, 'seconds')
            print('Energy consume in test: ', watthours_test, 'Wh \n')
            watthours = watthours_train + watthours_test
            os.mkdir(path + '/Consumed') if not os.path.exists(path + '/Consumed') else None
            file_consumed_path = os.path.join(path, f'Consumed/energy_consumed_client_{self.client_id}.txt')
            with open(file_consumed_path, "a") as f:
                f.write(str(watthours) + "\n")
            return net.state_dict(), soft_label, time_diff, watthours
        else:
            power_idle = get_gpu_power_consume()
            watthours_test = power_idle * time_diff/3600
            os.mkdir(path + '/Consumed') if not os.path.exists(path + '/Consumed') else None
            file_consumed_path = os.path.join(path, f'Consumed/energy_consumed_client_{self.client_id}.txt')           
            with open(file_consumed_path, "a") as f:
                f.write(str(watthours_test) + "\n")
            return watthours_test
