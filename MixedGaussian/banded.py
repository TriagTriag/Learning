import sys  
sys.path.insert(0, './')
import aux
import warnings
warnings.filterwarnings("ignore") 
import torch as pt
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog 
import argparse
from argparse import Namespace
from time import sleep
CUDA_LAUNCH_BLOCKING=1
def get_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--log-dir', type=str, default='.', help='Directory in which the results are saved.')
    parser.add_argument('--log-name', type=str, default='data', help='File in which the results are saved.')
    parser.add_argument('--mean-1', nargs='+',  type=float, default=-0.1, help='Mean of the first Gaussian.')
    parser.add_argument('--mean-2', nargs='+', type=float, default=2.0, help='Mean of the second Gaussian.')
    parser.add_argument('--N', type=int, default=100, help='Number of samples.')
    parser.add_argument('--N-test', type=int, default=2000, help='Number of test samples.')
    parser.add_argument('--var-1', type=float, default=0.1, help='Variance of the first Gaussian.')
    parser.add_argument('--var-2', type=float, default=0.1, help='Variance of the second Gaussian.')
    parser.add_argument('--dim', type=int, default=1, help='Dimension of the data.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator.')
    parser.add_argument('--hidden-dim', type=int, default=10, help='Dimension of the hidden layer.')
    ar = parser.parse_args()
    return ar


def run():
    configs     =   get_args()    
    main(**vars(configs))



def log_args(args):
    """ print all args """
    lines = [' {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
    for line in lines:
        aux.print_out(line.rstrip())
        aux.print_out('-------------------------------------------')
def main(**kwargs):
    save_folder = Namespace(**kwargs).log_dir+'/'+'mean_1='+str(Namespace(**kwargs).mean_1)+'mean_2='+str(Namespace(**kwargs).mean_2)+'var_1='+str(Namespace(**kwargs).var_1)+'var_2='+str(Namespace(**kwargs).var_2)+'N='+str(Namespace(**kwargs).N)+'N_test='+str(Namespace(**kwargs).N_test)+'dim='+str(Namespace(**kwargs).dim)+'dim_hidden='+str(Namespace(**kwargs).hidden_dim)+'/raw_data/'+'seed='+str(Namespace(**kwargs).seed)
    save_file = Namespace(**kwargs).log_name
    import os
    #make the directory of save_folder and all unmade directories in its path
    if (not os.path.isdir(save_folder)):
        #make the directory of save_folder and all unmade directories in its path
        for i in range(1, len(save_folder.split('/'))):
            print(('/'.join(save_folder.split('/')[:i])))
            if (not os.path.isdir('/'.join(save_folder.split('/')[:i+1]))):
                os.mkdir('/'.join(save_folder.split('/')[:i+1]))
                print('/'.join(save_folder.split('/')[:i+1]))
        # os.mkdir('./'+save_folder)
    aux.folder='./'+save_folder+'/'
    aux.print_out("start!")
    log_args(Namespace(**kwargs))    

    # Set the seed for pytorch
    pt.manual_seed(Namespace(**kwargs).seed)
    # sleep(1)
    N = Namespace(**kwargs).N;
    N_test = Namespace(**kwargs).N_test;
    mu_1 = pt.tensor([Namespace(**kwargs).mean_1])
    mu_2 = pt.tensor([Namespace(**kwargs).mean_2])
    var_1 = Namespace(**kwargs).var_1;
    var_2 = Namespace(**kwargs).var_2;
    dim = Namespace(**kwargs).dim;
    dim_hidden = Namespace(**kwargs).hidden_dim;
    # pt.manual_seed(3091994)
    device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
    # Set the dimension of Xs to be dim
    X1 = np.sqrt(var_1)*pt.randn([N,dim])+mu_1*pt.ones([N, dim])
    Y1 = pt.ones([N,])
    X2 = np.sqrt(var_2)*pt.randn([N,dim])+mu_2*pt.ones([N, dim])
    Y2 = pt.zeros([N,])
    X = pt.vstack([X1, X2])
    aux.print_out(X.shape)
    Y = pt.hstack([Y1, Y2])
    aux.print_out(Y.shape)
    perm = pt.randperm(2*N)
    X = X[perm].to(device)
    Y = Y[perm].to(device)
    Y_h = (1-Y).to(device)
    X1_test = np.sqrt(var_1)*pt.randn([N_test,dim])+mu_1*pt.ones([N_test, dim])
    Y1_test = pt.ones([N_test,])
    X2_test = np.sqrt(var_2)*pt.randn([N_test,dim])+mu_2*pt.ones([N_test, dim])
    Y2_test = pt.zeros([N_test,])
    # stack X1_test a X2_test to get X_test be of size 2*N_test x dim
    X_test = pt.vstack([X1_test, X2_test])
    Y_test = pt.hstack([Y1_test, Y2_test])
    perm = pt.randperm(2*N_test)
    X_test = X_test[perm].to(device)
    Y_test = Y_test[perm].to(device)
    Y_test_h = 1-Y_test
    b = np.arange(0.05, 1.0, 0.05)
    accs = np.zeros_like(b)
    accs_order = np.zeros_like(b)
    accs_thresh_adapt = np.zeros_like(b)
    accs_thresh_adapt_offline = np.zeros_like(b)
    accs_half = np.zeros_like(b)
    accs_half_constrained = np.zeros_like(b)
    exps = np.zeros_like(b)
    exps_thresh_adapt = np.zeros_like(b)
    exps_thresh_adapt_offline = np.zeros_like(b)
    exps_half = np.zeros_like(b)
    exps_half_constrained = np.zeros_like(b)
    accs_mixed = np.zeros_like(b)
    exps_mixed = np.zeros_like(b)
    rnd_mixed  = np.zeros_like(b)
    Tb = np.zeros_like(b)
    Cb = np.zeros_like(b)
    marg = np.zeros_like(b)
    accs_mixed_val = np.zeros_like(b)
    exps_mixed_val = np.zeros_like(b)
    rnd_mixed_val  = np.zeros_like(b)
    Tb_val = np.zeros_like(b)
    Cb_val = np.zeros_like(b)
    marg_val = np.zeros_like(b)
    
    models = []
    accs_thresh = []
    exps_thresh = []
    acc_thresh = 0.0
    exp_thresh = 0.0
    thresh_model = []
    models1_val = []
    models2_val = []
    plt.figure()
    for i in range( len(b)):
        # i = j-1
        # aux.print_out(i)
        model = aux.LN(dim_hidden, dim)
    #     model_ = LN(10)
        thr, val_x, val_y = aux.train_sorted(model, X.unsqueeze(1), Y.unsqueeze(1).unsqueeze(1), Y_h.unsqueeze(1).unsqueeze(1), b[i])
        if (i%3 == 0):
            plt.plot(val_x, val_y, label='b = '+"{:.2f}".format(b[i]))
            plt.xlabel('Sorted index')
            plt.ylabel('Cumulative difference of losses')

        thresh_model.append(thr)
        accs[i], exps[i] = aux.test(model, X_test.unsqueeze(1), Y_test.unsqueeze(1).unsqueeze(1), Y_test_h.unsqueeze(1).unsqueeze(1))
        accs_order[i], accs_thresh_adapt[i], accs_thresh_adapt_offline[i], accs_half[i], accs_half_constrained[i], exps_thresh_adapt[i], exps_thresh_adapt_offline[i], exps_half[i], exps_half_constrained[i], acc_thresh, exp_thresh = aux.test_by_ordering(model, X_test.unsqueeze(1), Y_test.unsqueeze(1), Y_test_h.unsqueeze(1), b[i], thresh_model[i])
        models.append(model)
        accs_thresh.append(acc_thresh)
        exps_thresh.append(exp_thresh)
        aux.print_out("b="+str(b[i]))
        model1 = aux.LN(dim_hidden, dim)
        model2 = aux.LN(10, dim)
    #     model_ = LN(10)
        Tb_val[i], Cb_val[i], marg_val[i] = aux.train_losses(model1, model2, X.unsqueeze(1), Y.unsqueeze(1).unsqueeze(1), Y_h.unsqueeze(1).unsqueeze(1), b[i], 0.001, dim_hidden, dim, True)
    #     aux.print_out(marg[i])
        accs_mixed_val[i], exps_mixed_val[i], rnd_mixed_val[i] = aux.test_losses(model1, model2, X_test.unsqueeze(1), Y_test.unsqueeze(1).unsqueeze(1), Y_test_h.unsqueeze(1).unsqueeze(1), Tb[i], Cb[i], marg[i])
        models1_val.append(model1)
        models2_val.append(model2)
        model1 = aux.LN(dim_hidden, dim)
        model2 = aux.LN(dim_hidden, dim)
        Tb[i], Cb[i], marg[i] = aux.train_losses(model1, model2, X.unsqueeze(1), Y.unsqueeze(1).unsqueeze(1), Y_h.unsqueeze(1).unsqueeze(1), b[i], 0.01, dim_hidden, dim, False)
    #     aux.print_out(marg[i])
        accs_mixed[i], exps_mixed[i], rnd_mixed[i] = aux.test_losses(model1, model2, X_test.unsqueeze(1), Y_test.unsqueeze(1).unsqueeze(1), Y_test_h.unsqueeze(1).unsqueeze(1), Tb[i], Cb[i], marg[i])
    plt.legend()
    plt.savefig('./'+save_folder+'/'+'val_losses.pdf')
    print("Validaton losses saved in ./"+save_folder+'/'+'val_losses.pdf')
#%%
    import os
    if (not os.path.isdir('./'+save_folder)):
        os.mkdir('./'+save_folder)
    np.savez('./'+save_folder+'/'+save_file+'.npz',   accs = accs, 
        accs_order = accs_order,
        accs_thresh_adapt = accs_thresh_adapt,
        accs_thresh_adapt_offline = accs_thresh_adapt_offline,
        accs_half = accs_half,
        accs_half_constrained = accs_half_constrained,
        exps = exps,
        exps_thresh_adapt = exps_thresh_adapt,
        exps_thresh_adapt_offline = exps_thresh_adapt_offline,
        exps_half = exps_half,
        exps_half_constrained = exps_half_constrained,
        accs_mixed = accs_mixed,
        exps_mixed = exps_mixed,
        rnd_mixed  = rnd_mixed,
        Tb = Tb,
        Cb = Cb,
        marg = marg,
        accs_mixed_val = accs_mixed_val,
        exps_mixed_val = exps_mixed_val,
        rnd_mixed_val  = rnd_mixed_val,
        Tb_val = Tb_val,
        Cb_val = Cb_val,
        marg_val = marg_val,
        accs_thresh = accs_thresh,
        exps_thresh = exps_thresh)
#%%
if __name__ == '__main__':
    aux.print_out("start!")
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    aux.print_out("Current Time =" +str(current_time))
    run()
#aux.print_out("this")
