from Utils.SaveResult import save_result
from Model.Update import *
from Model.Update_custom import get_gpu_power_consume
from Model.Test import test
from Model.Fed import *
import torch
import copy
import time
import random
import os
import pickle
import pickle as pkl
import random
import pulp
from Model.Update_custom import Custom_Local_SAGE
import numpy as np


def solve_optimization_problem(selectable, a, b, c, k, r, divergence, energy_norm, renewable, args):
    valid_clients = [i for i in selectable if energy_norm[i] >= 0.2]
    m = len(valid_clients)
    if m == 0:
        return {}, 0.0
    print('SOLVE 3')
    score = [a*energy_norm[i] + b*abs(divergence[i]) + c*renewable[i] for i in valid_clients]

    prob = pulp.LpProblem("SAGE_Select", pulp.LpMaximize)

    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(m)]

    prob += pulp.lpSum(y[i]*score[i] for i in range(m))

    prob += pulp.lpSum(y[i] for i in range(m)) >= k
    prob += pulp.lpSum(y[i] for i in range(m)) <= k + r - 1

    prob += pulp.lpSum(y[i] for i in range(m)) <= int(np.floor(args.frac*m))

    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    t1 = time.time()

    power_consume = get_gpu_power_consume() * (t1 - t0) / 3600.0

    device_epochs = {}
    energy_threshold = 0.3

    for idx_local, client_id in enumerate(valid_clients):
        yi = int(pulp.value(y[idx_local]) or 0)
        if yi == 1:
            if energy_norm[client_id] < energy_threshold:
                device_epochs[client_id] = 2  # half
            else:
                device_epochs[client_id] = 1  # full

    print('Selected devices and epochs:', device_epochs)
    return device_epochs, power_consume


def compute_energy(consumed_energy, max_energy):
    if consumed_energy == 0.:
        return max_energy
    else:
        return max_energy - consumed_energy

def get_properties(trainloader, renewable, max_energy, max_energy_dict, num_classes=10):
    lab = []
    for _, labels in trainloader:
        lab.extend(labels.tolist())

    # Compute distribution for KL divergence
    # Count occurrences and normalize → class probability distribution
    counts = torch.bincount(labels.clone().detach(), minlength=num_classes)
    class_distribution = counts.float() / counts.sum()

    energy = compute_energy(0., max_energy)  # Assuming consumed_energy starts at 0
    energy_norm = energy / max_energy_dict 
    dict_properties = {'max_energy_client': max_energy, 'energy': energy, 'energy_norm': energy_norm, 'renewable': renewable, 'class_distribution': class_distribution.tolist()}
    return dict_properties


# JS divergence
def _to_tensor(p):
    """Ensure tensor is float and add epsilon for numerical stability."""
    eps = 1e-12
    t = torch.tensor(p, dtype=torch.float64)
    t = t + eps
    t = t / t.sum()
    return t

def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence D(p || q) (log base 2)"""
    # p, q already positive and sum=1
    return torch.sum(p * (torch.log(p) - torch.log(q))) / torch.log(torch.tensor(2.0))

def js_divergence(p, q) -> float:
    """
    Jensen-Shannon divergence (base 2) between p and q.
    Returns a float in [0, 1].
    """
    p_t = _to_tensor(p)
    q_t = _to_tensor(q)
    m = 0.5 * (p_t + q_t)
    js = 0.5 * kl_divergence(p_t, m) + 0.5 * kl_divergence(q_t, m)
    return float(js) 

def client_scores(distributions: list[list[float]], return_distance: bool = False
                ) -> tuple[list[float], list[float]]:
    """
    distributions: list of distributions (list of lists), one per client.
    return_distance: if True returns the Jensen-Shannon distance = sqrt(JS).
    Returns:
        - avg_js_per_client: list with the average JS divergence of each client towards all others
        - (optional) avg_js_distance_per_client: list of sqrt(avg_js) for each client
    """
    n = len(distributions)
    if n == 0:
        return ([], []) if return_distance else ([],)

    # Preconvert to normalized tensors for speed
    tensors = [_to_tensor(d) for d in distributions]

    # Symmetric JS matrix
    js_matrix = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        for j in range(i+1, n):
            m = 0.5 * (tensors[i] + tensors[j])
            js_ij = 0.5 * kl_divergence(tensors[i], m) + 0.5 * kl_divergence(tensors[j], m)
            js_matrix[i, j] = js_ij
            js_matrix[j, i] = js_ij

    avg_js = []
    for i in range(n):
        if n == 1:
            avg_js.append(0.0)
        else:
            # average only on others (N-1)
            s = js_matrix[i].sum()  # include diagonal 0
            avg = float(s / (n - 1))
            avg_js.append(avg)

    if return_distance:
        avg_js_dist = [float(torch.sqrt(torch.tensor(x, dtype=torch.float64))) for x in avg_js]
        return avg_js, avg_js_dist

    return avg_js

def SAGE(args, net_glob, train_dataset, test_dataset, dict_users):
    seed = 2024
    random.seed(seed)

    dict_of_properties = {}
    accuracy = []
    opt_times = []
    aggr_times = []
    latency_times = []
    opt_power_consume = []
    weights_power_consume = []
    append_time = 0.0
    final_rounds = 0

    print('epochs:', args.epochs)
    start_FL = time.time()
    start_read = time.time()
    if args.algorithm == 'FedAvg' or args.algorithm == 'FedProx':
        save_path = f"Results/{args.algorithm}/{args.Drichlet_arg}"
    elif args.algorithm == 'SAGE':
        save_path = f"Results/{args.algorithm}/{args.a}_{args.b}_{args.c}_{args.Drichlet_arg}"
    os.makedirs(save_path, exist_ok=True)

    for idx in range(args.num_users):
        train_loader = DataLoader(DatasetSplit(train_dataset, dict_users[idx]), batch_size=args.local_bs, shuffle=True,
                                    drop_last=False)
        
        read_power_consume = get_gpu_power_consume() 

        with open('../Dict_pkl/dict_users.pkl', 'rb') as f:
            d_users = pickle.load(f)
        with open('../Dict_pkl/dict_renewable.pkl', 'rb') as f:
            d_renewable = pickle.load(f)

        max_energy_dict = max(d_users.values())
        args.max_energy = d_users[idx]

        dict_of_properties[idx] = get_properties(train_loader, d_renewable[idx], args.max_energy, max_energy_dict)

    end_read = time.time()    

    read_time = end_read - start_read
    read_power_consume = read_power_consume*read_time/3600

    start_training_time = time.time()

    # divergence calculation
    print('Calculating divergence scores for each client...')
    clients_distributions = [dict_of_properties[i]['class_distribution'] for i in range(args.num_users)]
    div_scores = client_scores(clients_distributions)
    # dict to save div_scores with keys 0..99
    div_scores_dict = {i: div_scores[i] for i in range(args.num_users)}
    with open('div_scores.pkl', 'wb') as f:
        pkl.dump(div_scores_dict, f)

    selectable = list(range(args.num_users))

    for round in range(args.epochs): 
        print("\n #################### {} ####################".format(round))
        start_time = time.time()
        local_model = []
        lens = []
        times=[]

        renewable = [dict_of_properties[i]['renewable'] for i in range(args.num_users)]
        energy_norm = [dict_of_properties[i]['energy_norm'] for i in range(args.num_users)]
        
        m = max(int(args.frac * args.num_users), 1)
        start_opt_time = time.time()
        if args.algorithm == 'FedAvg' or args.algorithm == 'FedProx':
            power_consume = 0.0
            if m <= len(selectable):
                idxs = np.random.choice(selectable, m, replace=False)
            else:
                idx = selectable
            selected_devices = list(idxs)
            if len(selected_devices) < args.k:
                print(f'Number of selected devices {len(selected_devices)} less than k')
                break
            selected_devices_orig = {device: 1 for device in selected_devices}  # all full epochs
            not_selected_dev = list(set(range(args.num_users)) - set(selected_devices) - set(selectable))
        elif args.algorithm == 'SAGE':
            selected_devices_orig, power_consume = solve_optimization_problem(selectable, args.a, args.b, args.c, args.k, round+1, div_scores, energy_norm, renewable, args)
            selected_devices = list(selected_devices_orig.keys())
            not_selected_dev = list(set(range(args.num_users)) - set(selected_devices) - set(selectable)) 
        
        end_opt_time = time.time()

        t0_append_opt = time.time()
        opt_power_consume.append(power_consume)
        opt_times.append(end_opt_time - start_opt_time)
        t1_append_opt =	time.time()

        rand = random.random()
        rand_2 = random.random()
        percentage_malf = random.choice([0.0, 0.1, 0.2, 0.3])
        percentage_malf_2 = random.choice([0.0, 0.1, 0.2, 0.3])

        print('initial selected devices', selected_devices)

        if round >= 1 and rand < percentage_malf and len(selected_devices) > 2:  # losing connection with probability 1/10, device is not removed definitively

            max_number_indices = random.randint(1, max(1, int(len(selected_devices) * 0.3))) # at most 30% of selected devices
            devices_to_remove = random.sample(selected_devices, max_number_indices)
            print('The devices ', devices_to_remove, ' have lost connection')

            # Remove the elements corresponding to the indices from the list
            for device in devices_to_remove:
                try:
                    selected_devices.remove(device)
                except:
                    print('not eliminated')
            

        if round >= 1 and rand_2 < percentage_malf_2 and len(selected_devices) > 2:  # sudden depletion of a selected device with probability 1/10, which is removed definitively
            max_number = random.randint(1, max(1, int(len(selected_devices) * 0.3))) # at most 30% of selected devices
            devices_to_remove = random.sample(selected_devices, max_number)
            # Generate a random list of indices to remove
            print('The devices ', devices_to_remove, ' have suddenly depleted their charge')

            # Remove the elements corresponding to the indices from the list
            for device in devices_to_remove:
                dict_of_properties[device]['energy'] = 0.0  # Reset the device's charge
                try:
                    selectable.remove(device)
                    selected_devices.remove(device)
                except:
                    print('not eliminated')
            print('selectable: ', selectable)

        if round % 1 == 0:
    
            for device in range(args.num_users):
                actual_value = dict_of_properties[device]['renewable']
                values_to_add = np.arange(-actual_value, 1-actual_value, 0.1)

                new_value = np.round((actual_value + random.choice(values_to_add)), 1)
                if new_value <= 0:
                    new_value = 0.0
                if new_value > 1:
                    new_value = 1.0
                dict_of_properties[device]['renewable'] = new_value
                

        print('final selected_devices', selected_devices)
        

        if selected_devices == []:
            break
        end_opt_time = time.time()


        for selected, mode in selected_devices_orig.items():
            if selected in selected_devices:
                if mode == 2:
                    local_ep = max(1, args.local_ep // 2)  # half epochs (at least 1)
                else:
                    local_ep = args.local_ep  # full epochs
            client = Custom_Local_SAGE(args=args, dataset=train_dataset, idx=dict_users[selected], net_glob=net_glob, client_id=selected, local_ep=local_ep, path=save_path)
            w, _, time_diff, watthours = client.train(net=copy.deepcopy(net_glob).to(args.device))
            times.append(time_diff)
            local_model.append(copy.deepcopy(w))
            lens.append(len(dict_users[selected]))
            os.mkdir(save_path + '/Residual') if not os.path.exists(save_path + '/Residual') else None
            file_residual_path = os.path.join(save_path, f'Residual/residual_energy_client_{selected}.txt')
            with open(file_residual_path, "a") as f:
                f.write(str(dict_of_properties[selected]['energy']) + "\n")

            dict_of_properties[selected]['energy'] = dict_of_properties[selected]['energy'] - watthours
            # random energy recharge
            if round % 10 == 0 and round != 0:
                random_energy = random.uniform(1e-3, dict_of_properties[selected]['max_energy_client'] - dict_of_properties[selected]['energy'])
                dict_of_properties[selected]['energy'] += random_energy
                print(f'Client {selected} recharged {random_energy:.4f} Wh, new energy: {dict_of_properties[selected]["energy"]:.4f} Wh')
            dict_of_properties[selected]['energy_norm'] = dict_of_properties[selected]['energy']/max_energy_dict

        file_sel_path = os.path.join(save_path, 'selected_devices.txt')
        with open(file_sel_path, "a") as f:
            f.write(str(round) + ":" + str(selected_devices) + "\n")


        for not_selected in not_selected_dev:
            client = Custom_Local_SAGE(args=args, dataset=train_dataset, idx=dict_users[not_selected], client_id=not_selected, path=save_path)
            watthours = client.train(net=copy.deepcopy(net_glob).to(args.device), time_diff=np.max(times), train=False)
            dict_of_properties[not_selected]['energy'] = dict_of_properties[not_selected]['energy'] - watthours
            dict_of_properties[not_selected]['energy_norm'] = dict_of_properties[not_selected]['energy']/max_energy_dict
            os.mkdir(save_path + '/Residual') if not os.path.exists(save_path + '/Residual') else None
            file_notsel_path = os.path.join(save_path, f'Residual/residual_energy_client_{not_selected}.txt') 
            with open(file_notsel_path, "a") as f:
                f.write(str(dict_of_properties[not_selected]['energy']) + "\n")            

        t0_append_lat = time.time()
        latency_times.append(np.max(times)-np.min(times)) # waiting time for the client that takes the longest from the other clients
        t1_append_lat = time.time()


        start_aggr_time = time.time()
        w_avg = Avg(local_model, lens)
        w_avg_power_consume = get_gpu_power_consume() 
        net_glob.load_state_dict(w_avg)
        end_aggr_time = time.time()

        t0_append_aggr = time.time()
        aggr_times.append(end_aggr_time - start_aggr_time)
        weights_power_consume.append(w_avg_power_consume*(  - start_aggr_time)/3600)
        t1_append_aggr = time.time()

        test_loss, test_accuracy = test(args, net_glob, test_dataset)

        t0_append_acc = time.time()
        accuracy.append(test_accuracy)
        t1_append_acc =	time.time()

        append_time = append_time + (t1_append_opt - t0_append_opt) + (t1_append_lat-t0_append_lat) + (t1_append_aggr-t0_append_aggr) + (t1_append_acc-t0_append_acc)
        finish_time = time.time()
        oneRoundTime = (finish_time - start_time)

        print('test_loss: {}, accuracy: {}, time: {}'.format(test_loss, test_accuracy, oneRoundTime))

        if test_accuracy > 0.40 and final_rounds == 0: # target accuracy and final_rounds updated
            final_rounds = 21 # 20 training rounds after target accuracy reached

        if final_rounds != 0: # final_rounds updated
            final_rounds -= 1

        if final_rounds == 1: # end of training with target accuracy reached
            break

    end_training_time = time.time()
    training_time = (end_training_time - start_training_time - append_time)
    
    end_FL = time.time()
    total_time = (end_FL - start_FL)
    print('Total time for FL: ', total_time, 'min')
    
    save_result(args, accuracy, save_path, total_time, read_time, latency_times, training_time, opt_times, aggr_times, append_time,
            read_power_consume, opt_power_consume, weights_power_consume)


