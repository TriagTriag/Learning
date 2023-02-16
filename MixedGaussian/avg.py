import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import argparse
from argparse import Namespace
import aux

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--log-dir', type=str, default='.', help='Directory in which the results are saved.')
    parser.add_argument('--log-name', type=str, default='data', help='File in which the results are saved.')
    parser.add_argument('--mean-1', nargs='+',  type=float, default=1.0, help='Mean of the first Gaussian.')
    parser.add_argument('--mean-2', nargs='+', type=float, default=4.0, help='Mean of the second Gaussian.')
    parser.add_argument('--N', type=int, default=10000, help='Number of samples.')
    parser.add_argument('--N-test', type=int, default=2000, help='Number of test samples.')
    parser.add_argument('--var-1', type=float, default=1.0, help='Variance of the first Gaussian.')
    parser.add_argument('--var-2', type=float, default=1.0, help='Variance of the second Gaussian.')
    parser.add_argument('--dim', type=int, default=1, help='Dimension of the data.')
    parser.add_argument('--hidden-dim', type=int, default=100, help='Dimension of the hidden layer.')
    parser.add_argument('--seed-size', type=int, default=20, help='Size of the seeds.')
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

    log_args(Namespace(**kwargs))    
    load_folder = Namespace(**kwargs).log_dir+'/'+'mean_1='+str(Namespace(**kwargs).mean_1)+'mean_2='+str(Namespace(**kwargs).mean_2)+'var_1='+str(Namespace(**kwargs).var_1)+'var_2='+str(Namespace(**kwargs).var_2)+'N='+str(Namespace(**kwargs).N)+'N_test='+str(Namespace(**kwargs).N_test)+'dim='+str(Namespace(**kwargs).dim)+'dim_hidden='+str(Namespace(**kwargs).hidden_dim)+'/raw_data/'+'seed='
    save_folder = Namespace(**kwargs).log_dir+'/'+'mean_1='+str(Namespace(**kwargs).mean_1)+'mean_2='+str(Namespace(**kwargs).mean_2)+'var_1='+str(Namespace(**kwargs).var_1)+'var_2='+str(Namespace(**kwargs).var_2)+'N='+str(Namespace(**kwargs).N)+'N_test='+str(Namespace(**kwargs).N_test)+'dim='+str(Namespace(**kwargs).dim)+'dim_hidden='+str(Namespace(**kwargs).hidden_dim)
    save_file = Namespace(**kwargs).log_name
    seed_size = Namespace(**kwargs).seed_size
    import os
    if (not os.path.isdir('./'+save_folder)):
        os.mkdir('./'+save_folder)
    # Warning of start!
    
    aux.folder='./'+save_folder+'/'
    aux.print_out("start!")


    data = []
    for i in range(seed_size):
        data_tmp =  np.load(load_folder+str(i)+'/data.npz')
        data.append(data_tmp)

    X_plot = pt.tensor(np.arange(-0, 10, 0.2), dtype=pt.float).unsqueeze(1)
    # print(X_plot)
    b = np.arange(0.05, 1, 0.05)
    threshs = np.arange(0.1, 1.1, 0.1)
    plt.figure()
    exps_threshs = []
    exps_threshs2 = []
    exps_threshs_std = []
    # exps_threshs_mean = []
    accs_threshs = []
    accs_threshs2 = []
    accs_threshs_std = []
    # accs_threshs_mean = []
    exps = 0
    exps_thresh_adapt = 0
    exps_thresh_adapt_offline = 0
    exps_half = 0
    exps_half_constrained = 0
    exps_mixed = 0
    accs_thresh_adapt =0
    accs_mixed =0
    accs_thresh2 = 0
    accs_mixed2 = 0
    exps2 = 0
    accs = 0
    accs2 = 0
    exps_thresh_adapt2 = 0
    exps_thresh_adapt_offline2 = 0
    exps_half2 = 0
    exps_half_constrained2 = 0
    exps_mixed2 = 0
    accs_order = 0
    accs_order2 = 0
    accs_thresh_adapt_offline = 0
    accs_thresh_adapt_offline2 = 0
    accs_half = 0
    accs_half2 = 0
    accs_half_constrained = 0
    accs_half_constrained2 = 0
    rnd_mixed = 0
    rnd_mixed2 = 0
    Tb = 0
    Tb2 = 0
    Cb = 0
    Cb2 = 0
    marg = 0
    marg2 = 0
    accs_mixed_val = 0
    accs_mixed_val2 = 0
    exps_mixed_val = 0
    exps_mixed_val2 = 0
    rnd_mixed_val = 0
    rnd_mixed_val2 = 0
    Tb_val = 0
    Tb_val2 = 0
    Cb_val = 0
    Cb_val2 = 0
    marg_val = 0
    marg_val2 = 0 
    for j in range(len(threshs)):
        exps_threshs.append(np.zeros(len(b)))
        exps_threshs2.append(np.zeros(len(b)))
        accs_threshs.append(np.zeros(len(b)))
        accs_threshs2.append(np.zeros(len(b)))
        exps_threshs_std.append(np.zeros(len(b)))
        accs_threshs_std.append(np.zeros(len(b)))
    for i in range(seed_size):
    #    print(data[i])
        exps+=data[i]['exps']
        exps2 += data[i]['exps']**2
        accs += data[i]['accs']
        accs2 += data[i]['accs']**2
        exps_thresh_adapt +=data[i]['exps_thresh_adapt']
        exps_thresh_adapt2 +=data[i]['exps_thresh_adapt']**2
        exps_thresh_adapt_offline +=data[i]['exps_thresh_adapt_offline']
        exps_thresh_adapt_offline2 +=data[i]['exps_thresh_adapt_offline']**2
        exps_half += data[i]['exps_half']
        exps_half2 += data[i]['exps_half']**2
        exps_half_constrained += data[i]['exps_half_constrained']
        exps_half_constrained2 += data[i]['exps_half_constrained']**2
        exps_mixed += data[i]['exps_mixed']
        exps_mixed2 += data[i]['exps_mixed']**2
        accs_thresh_adapt += data[i]['accs_thresh_adapt']
        accs_thresh2 += data[i]['accs_thresh_adapt']**2
        accs_mixed += data[i]['accs_mixed']
        accs_mixed2 += data[i]['accs_mixed'] **2
        accs_order += data[i]['accs_order']
        accs_order2 += data[i]['accs_order'] **2
        accs_thresh_adapt_offline += data[i]['accs_thresh_adapt_offline']
        accs_thresh_adapt_offline2 += data[i]['accs_thresh_adapt_offline'] **2
        accs_half += data[i]['accs_half']
        accs_half2 += data[i]['accs_half'] **2
        accs_half_constrained += data[i]['accs_half_constrained']
        accs_half_constrained2 += data[i]['accs_half_constrained'] **2
        rnd_mixed += data[i]['rnd_mixed']
        rnd_mixed2 += data[i]['rnd_mixed'] **2
        Tb += data[i]['Tb']
        Tb2 += data[i]['Tb'] **2
        Cb += data[i]['Cb']
        Cb2 += data[i]['Cb'] **2
        marg += data[i]['marg']
        marg2 += data[i]['marg'] **2
        accs_mixed_val += data[i]['accs_mixed_val']
        accs_mixed_val2 += data[i]['accs_mixed_val'] **2
        exps_mixed_val += data[i]['exps_mixed_val']
        exps_mixed_val2 += data[i]['exps_mixed_val'] **2
        rnd_mixed_val += data[i]['rnd_mixed_val']
        rnd_mixed_val2 += data[i]['rnd_mixed_val'] **2
        Tb_val += data[i]['Tb_val']
        Tb_val2 += data[i]['Tb_val'] **2
        Cb_val += data[i]['Cb_val']
        Cb_val2 += data[i]['Cb_val'] **2
        marg_val += data[i]['marg_val']
        marg_val2 += data[i]['marg_val'] **2

        for j in range(len(threshs)):
            print(data[i]['exps_thresh'])
            exps_threshs[j][ :]+=data[i]['exps_thresh'][:, j]
            exps_threshs2[j][ :]+=data[i]['exps_thresh'][:, j]**2
            accs_threshs[j][ :]+=data[i]['accs_thresh'][:, j]
            accs_threshs2[j][ :]+=data[i]['accs_thresh'][:, j]**2

    for i in range(len(threshs)):
        exps_threshs[i][ :]/=seed_size
        exps_threshs2[i][ :]/=seed_size
        accs_threshs[i][ :]/=seed_size
        accs_threshs2[i][ :]/=seed_size
        exps_threshs_std[i][ :] = np.sqrt((exps_threshs2[i][ :]-exps_threshs[i][ :]**2))
        accs_threshs_std[i][ :] = np.sqrt((accs_threshs2[i][ :]-accs_threshs[i][ :]**2))
    exps/=seed_size
    accs/=seed_size

    exps_thresh_adapt/=seed_size
    exps_thresh_adapt_offline/=seed_size
    exps_half/=seed_size
    exps_half_constrained/=seed_size
    exps_mixed/=seed_size
    accs_thresh_adapt /=seed_size
    accs_thresh_std = np.sqrt((accs_thresh2/seed_size-accs_thresh_adapt**2))
    accs_mixed /= seed_size
    accs_mixed_std = np.sqrt((accs_mixed2/seed_size-accs_mixed**2))
    exps_std = np.sqrt((exps2/seed_size-exps**2))
    accs_std = np.sqrt((accs2/seed_size-accs**2))
    exps_thresh_adapt_std = np.sqrt((exps_thresh_adapt2/seed_size-exps_thresh_adapt**2))
    exps_thresh_adapt_offline_std = np.sqrt((exps_thresh_adapt_offline2/seed_size-exps_thresh_adapt_offline**2))
    exps_half_std = np.sqrt((exps_half2/seed_size-exps_half**2))
    exps_half_constrained_std = np.sqrt((exps_half_constrained2/seed_size-exps_half_constrained**2))
    exps_mixed_std = np.sqrt((exps_mixed2/seed_size-exps_mixed**2))
    accs_order /= seed_size
    accs_order_std = np.sqrt((accs_order2/seed_size-accs_order**2))
    accs_thresh_adapt_offline /= seed_size
    accs_thresh_adapt_offline_std = np.sqrt((accs_thresh_adapt_offline2/seed_size-accs_thresh_adapt_offline**2))
    accs_half /= seed_size
    accs_half_std = np.sqrt((accs_half2/seed_size-accs_half**2))
    accs_half_constrained /= seed_size
    accs_half_constrained_std = np.sqrt((accs_half_constrained2/seed_size-accs_half_constrained**2))
    rnd_mixed /= seed_size
    rnd_mixed_std = np.sqrt((rnd_mixed2/seed_size-rnd_mixed**2))
    Tb /= seed_size
    Tb_std = np.sqrt((Tb2/seed_size-Tb**2))
    Cb /= seed_size
    Cb_std = np.sqrt((Cb2/seed_size-Cb**2))
    marg /= seed_size
    marg_std = np.sqrt((marg2/seed_size-marg**2))
    accs_mixed_val /= seed_size
    accs_mixed_val_std = np.sqrt((accs_mixed_val2/seed_size-accs_mixed_val**2))
    exps_mixed_val /= seed_size
    exps_mixed_val_std = np.sqrt((exps_mixed_val2/seed_size-exps_mixed_val**2))
    rnd_mixed_val /= seed_size
    rnd_mixed_val_std = np.sqrt((rnd_mixed_val2/seed_size-rnd_mixed_val**2))
    Tb_val /= seed_size
    Tb_val_std = np.sqrt((Tb_val2/seed_size-Tb_val**2))
    Cb_val /= seed_size
    Cb_val_std = np.sqrt((Cb_val2/seed_size-Cb_val**2))
    marg_val /= seed_size
    marg_val_std = np.sqrt((marg_val2/seed_size-marg_val**2))


    #save data
    np.savez(save_folder+'/avg.npz', b=b, exps=exps, exps_thresh_adapt=exps_thresh_adapt, exps_thresh_adapt_offline=exps_thresh_adapt_offline, exps_half=exps_half, exps_half_constrained=exps_half_constrained, exps_mixed=exps_mixed, accs_thresh_adapt=accs_thresh_adapt, accs_thresh_std=accs_thresh_std, accs_mixed=accs_mixed, accs_mixed_std=accs_mixed_std)
    # plt.plot(b, exps_mixed, label="Expected Deferral for the mixed strategy")
    # #plt.plot(b, exps, label="Expected Deferral")
    # #plt.plot(b, b, linestyle='--', label="Maximum Expected Deferral")
    # plt.plot(b, exps_thresh_adapt, label="Expected Deferral for adaptive thresholding")
    #plt.plot(b, exps_thresh_adapt_offline, label="Expected Deferral for offline adaptive thresholding")
    #plt.plot(b, exps_half, label="Expected Deferral for >0.5 rejection")
    #plt.plot(b, exps_half_constrained, label="Expected Deferral for the constrained >0.5 rejection")
    plt.figure(figsize=(8, 7))
    # set the line width
    plt.rc('lines', linewidth=6)
    # set fontsize to 40
    plt.rc('font', size=40)
    ax = plt.subplot(2, 1, 1)
    ax.errorbar(b, exps_mixed, yerr = exps_mixed_std, label="Our method")
    ax.errorbar(b, exps_thresh_adapt, yerr = exps_thresh_adapt_std, label="Adaptive thresholding")
    ax.errorbar(b, exps_half, yerr = exps_half_std, label="Okati et al. with T = 0.5")
    plt.xlabel('Budget')
    plt.ylabel('Expected Deferral')

    plt.legend()
    plt.savefig(save_folder+'/Res.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(b, accs_mixed, yerr = accs_mixed_std, label="Our method")
    ax.errorbar(b, accs_thresh_adapt, yerr = accs_thresh_std, label="Adaptive thresholding")
    ax.errorbar(b, accs_half, yerr = accs_half_std, label="Okati et al. with T = 0.5")
    plt.xlabel('Budget')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_folder+'/Res_acc.pdf', bbox_inches='tight')

    # plot offline methods
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(b, exps_thresh_adapt_offline, yerr = exps_thresh_adapt_offline_std, label="Offline adaptive thresholding")
    ax.errorbar(b, exps_half, yerr = exps_half_std, label="Okati et al. with T = 0.5")
    ax.errorbar(b, exps_half_constrained, yerr = exps_half_constrained_std, label="Okati et al. with T = 0.5 and bounded budget")
    ax.errorbar(b, b, yerr = 0, label="Ordering-based strategy")

    plt.xlabel('Budget')
    plt.ylabel('Expected Deferral')
    plt.legend()
    plt.savefig(save_folder+'/Res_offline.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(b, accs_thresh_adapt_offline, yerr = accs_thresh_adapt_offline_std, label="Offline adaptive thresholding")
    ax.errorbar(b, accs_half, yerr = accs_half_std, label="Okati et al. with T = 0.5")
    ax.errorbar(b, accs_half_constrained, yerr = accs_half_constrained_std, label="Okati et al. with T = 0.5 and bounded budget")
    ax.errorbar(b, accs_order, yerr = accs_order_std, label="Ordering-based strategy")
    plt.xlabel('Budget')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_folder+'/Res_acc_offline.pdf', bbox_inches='tight')

    # plot randomized method vs mixed method
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(b, exps_mixed, yerr = exps_mixed_std, label="Our method")
    ax.errorbar(b, exps, yerr = exps_std, label="Randomized strategy")
    plt.xlabel('Budget')
    plt.ylabel('Expected Deferral')
    plt.legend()
    plt.savefig(save_folder+'/Res_rand_vs_mixed.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(b, accs_mixed, yerr = accs_mixed_std, label="Our method")
    ax.errorbar(b, accs, yerr = accs_std, label="Randomized strategy")
    plt.xlabel('Budget')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_folder+'/Res_acc_rand_vs_mixed.pdf', bbox_inches='tight')


    # plot randomness of the mixed method
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(b, rnd_mixed, yerr = rnd_mixed_std, label="Our method")
    plt.xlabel('Budget')
    plt.ylabel('Randomness')
    plt.legend()
    plt.savefig(save_folder+'/Res_mixed_randomness.pdf', bbox_inches='tight')

    # plot threshold of the mixed method
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(b, Tb, yerr = Tb_std, label="Our method")
    plt.xlabel('Budget')
    plt.ylabel('Threshold')
    plt.legend()
    plt.savefig(save_folder+'/Res_mixed_threshold.pdf', bbox_inches='tight')

    # plot amplitude of randomness of the mixed method
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(b, Cb, yerr = Cb_std, label=r"Amplitude of randomness ($C_b$) of the mixed strategy")
    plt.xlabel('Budget')
    plt.ylabel(r"Amplitude of randomness ($C_b$)")
    plt.legend()
    plt.savefig(save_folder+'/Res_mixed_amplitude.pdf', box_inches='tight')

    #plot threshs

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    for i in range(0, len(threshs)):
        ax.errorbar(b, accs_threshs[i][ :], yerr = accs_threshs_std[i][:], label="T= " + "{:.1f}".format(threshs[i]))
    plt.xlabel('Budget')
    plt.ylabel('Accuracy')
    plt.legend( bbox_to_anchor=(0, 1), loc='upper left', ncol=5)
    plt.savefig(save_folder+'/Res_threshs.pdf', box_inches='tight')

    #plot exps_threshs

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(2, 1, 1)
    for i in range(0, len(threshs)):
        ax.errorbar(b, exps_threshs[i][ :], yerr = exps_threshs_std[i][:], label="T= " + "{:.1f}".format(threshs[i]))
    plt.xlabel('Budget')
    plt.ylabel('Expected Deferral')
    plt.legend( bbox_to_anchor=(0, 1), loc='upper left', ncol=5)
    plt.savefig(save_folder+'/Res_exps_threshs.pdf', box_inches='tight')



if (__name__ == "__main__"):
    run()
