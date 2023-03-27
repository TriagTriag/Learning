#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:44:07 2022
"""
import warnings
warnings.filterwarnings("ignore") 
import torch as pt
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog 
device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
folder='./'
def print_out(st):
    print(st)
    # f = open(folder+"out.txt", "a")
    # f.write(str(st)+'\n')
    # #print('s')
    # f.close()

class LN(nn.Module):
    def __init__(self, dim_hidden, dim_in):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden).to(device)
        pt.nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(dim_hidden, 1).to(device)
        pt.nn.init.xavier_uniform(self.fc2.weight)
        self.relu = nn.ReLU().to(device);
        self.logsoftmax = nn.Sigmoid().to(device)
    def forward(self, in_):
        fc_out = self.relu(self.fc1(in_)).to(device)
#         print_out(fc_out)
        out_1 = self.fc2(fc_out).to(device)
        out = self.logsoftmax(out_1).to(device)
#         print_out(out)
        return out
    
def train(model, X, Y, Y_h, b):
#     print_out(Y)
    X = X.detach().cpu()
    Y = Y.detach().cpu()
    Y_h = Y_h.detach().cpu()
#     print_out(model.parameters())
    len_X = X.shape[0]
    minibatch_len = np.floor(len_X/10).astype(int)
    num_epochs = 20
    num_batches = int(X.shape[0] / minibatch_len)
    optimizer = pt.optim.Adam(model.parameters(), lr=0.03)
    loss_func = nn.BCELoss(reduction='none')
    train_losses = []
    best_val_loss = 1000
    eps = 1e-4
    
    for epoch in range(num_epochs):
#         print_out('----- epoch:',epoch, '-----')
        train_loss = 0
        machine_loss = []
        for i in range(num_batches):
            X_batch = (X[i * minibatch_len: (i + 1) * minibatch_len]).to(device)
            Y_batch = (Y[i * minibatch_len: (i + 1) * minibatch_len]).to(device)
            Y_h_batch =  (Y_h[i * minibatch_len: (i + 1) * minibatch_len]).to(device)
            diff = Y_h_batch.detach().cpu().numpy()-Y_batch.detach().cpu().numpy()
            sol = linprog(diff,np.ones([1, minibatch_len]), b*minibatch_len, bounds = (0, 1))
            lab = sol.x
            lab = pt.tensor(lab, dtype = pt.float).unsqueeze(1).to(device)
            optimizer.zero_grad()
#             print_out(lab, X_batch)
            loss = loss_func(model(X_batch),lab)
            loss.sum().backward()
            optimizer.step()
            train_loss += float(loss.mean())
        train_losses.append(train_loss / num_batches)
#         print_out('machine_loss:', train_loss/num_batches)

def test(model, X, Y, Y_H):
    h = model(X).detach().cpu().numpy()[:, 0, 0]
    y = Y.detach().cpu().numpy()[:, 0, 0]
    y_h = Y_H.detach().cpu().numpy()[:, 0, 0]
    # print("h.shape:"+str(h.shape)+", y.shape:"+str(y.shape)+", y_h.shape:"+str(y_h.shape))
    # print(h, y, y_h)
    acc = 1-np.sum(h*y_h[:]+(1-h)*y[:])/len(Y)
    exp = np.sum(h)/len(Y)
    print_out("Accuracy = "+str( acc))
#     print_out("Expected Deferral = ", exp)
    return acc, exp

def find_machine_samples(machine_loss, hloss,constraint):
    
    diff = machine_loss - hloss
    argsorted_diff = pt.clone(pt.argsort(diff))
#     print_out(argsorted_diff)
    num_outsource = int(constraint * machine_loss.shape[0])
#     print_out(constraint * machine_loss.shape[0])
    index = -num_outsource

    while (index < -1 and diff[argsorted_diff[index]] <= 0):
        index += 1

    
    if index==0:
        index = -1
    if index == -diff.shape[0]:
        index = 1
        
    machine_list = argsorted_diff[:index]

    return machine_list

def train_sorted(model, X, Y, Y_h, b):
#     print_out(Y)
    X = X.detach().cpu()
    Y = Y.detach().cpu()
    Y_h = Y_h.detach().cpu()
    length_X = X.shape[0]
    valid_perc = 0.2
    valid_num = int(np.floor(valid_perc*length_X))
    X_val = X[:valid_num]
    Y_val = Y[:valid_num].detach().cpu().numpy()
    Y_h_val = Y_h[:valid_num].detach().cpu().numpy()
    X = X[valid_num:].to(device)
    Y = Y[valid_num:].to(device)
    Y_h = Y_h[valid_num:].to(device)
    
    
#     print_out(model.parameters())
    len_X = X.shape[0]
    minibatch_len = np.floor(len_X/10).astype(int)
    num_epochs = 20
    num_batches = int(X.shape[0] / minibatch_len)
    optimizer = pt.optim.Adam(model.parameters(), lr=0.03)
    loss_func = nn.BCELoss(reduction='none')
    train_losses = []
    best_val_loss = 1000
    eps = 1e-4
    
    for epoch in range(num_epochs):
#         print_out('----- epoch:',epoch, '-----')
        train_loss = 0
#         with pt.no_grad():
#             mprim = copy.deepcopy(model)
        machine_loss = []
        for i in range(num_batches):
            X_batch = X[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            Y_batch = Y[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            Y_h_batch = Y_h[i * minibatch_len: (i + 1) * minibatch_len].to(device)
#             diff = Y_h_batch.detach().cpu().numpy()-Y_batch.detach().cpu().numpy()
#             print_out(diff)
#             sol = linprog(diff,np.ones([1, minibatch_len]), b*minibatch_len, bounds = (0, 1))
#             lab = sol.x
# #             print_out(lab)
#             lab = pt.tensor(lab, dtype = pt.float).unsqueeze(1)
#             loss_temp = loss_func(model(X_batch), lab)
#             print_out(Y_batch, model(X_batch), loss_temp)
            machine_indices = find_machine_samples(Y_batch[:, 0, 0], Y_h_batch[:, 0, 0], b)
            # print_out(machine_indices)
            lab = pt.tensor([0 if j in machine_indices else 1 for j in range(Y_h_batch.shape[0])], dtype=pt.float).to(device)
            # print(lab)
            optimizer.zero_grad()
            loss = loss_func(model(X_batch),lab.unsqueeze(1).unsqueeze(1))
#             print_out(Y_batch)
#             print_out(pt.autograd.grad(loss_temp.sum(), model.parameters()))
#             print_out(model(X_batch), Y_batch, loss)
            loss.sum().backward()
            optimizer.step()
            train_loss += float(loss.mean())
            # print_out(loss.mean())
        train_losses.append(train_loss / num_batches)
        print('machine_loss:', train_loss/num_batches)

    
    num_machine = int(np.floor((1-b)*len(Y_val)))
    # find size of model(pt.tensor(X_val).to(device))
    print_out("size of model(pt.tensor(X_val).to(device))="+str(model(pt.tensor(X_val).to(device)).shape))
    pred_val = model(pt.tensor(X_val).to(device))[ :, 0, 0]
    human_candidate = (pt.argsort(pred_val)[num_machine:])
    val_loss = np.zeros([valid_num-num_machine,])
    # find size of num_machine and pred_val and human_candidate
    print_out("num_machine="+str(num_machine)+", size_pred_val="+str(pred_val.shape)+", size_human_candidate="+str(human_candidate.shape))
    
    if (b!=0):
        for i in range(valid_num-num_machine):
            val_loss[i] = np.sum(Y_val[human_candidate[:i+1].cpu()]-Y_h_val[human_candidate[:i+1].cpu()])
        print_out("b="+str(b))
        thresh = pred_val[human_candidate[np.argmin(val_loss)]]
        optim_i = np.argmin(val_loss)
    else:
        thresh = pt.tensor([1.0])
        optim_i = valid_num
    print_out("Threshold="+str(thresh))
    print_out("optimal i is "+str(optim_i))
#    if (valid_num-num_machine!=0):
#        plt.plot(np.arange(valid_num-num_machine)/(valid_num-num_machine), val_loss/valid_num, label='b = '+str(b))
    return float(thresh.detach().cpu()),  np.arange(valid_num-num_machine)+num_machine, val_loss/valid_num

def test_by_ordering(model, X, Y, Y_H, b, model_thresh):
    num_machine = int(np.floor((1-b)*len(Y)))
    thresh = np.arange(0.1, 1.1, 0.1)
    h = model(X).detach().cpu().numpy()
    y = Y.detach().cpu().numpy()
    y_h = Y_H.detach().cpu().numpy()
    h_to_sort = model(X)[:, 0, 0]
#     print_out(h_to_sort.shape)
    to_machine = pt.argsort(h_to_sort)[:num_machine].cpu().data.numpy()
    # print("num_machine="+str(num_machine))
    # print_out("to_machine"+str(y[to_machine]))
    to_human = np.array([i for i in range(X.shape[0]) if i not in to_machine])
    if (len(to_human)==0):
        to_human = []
        
#     machine_candidates_adapt_thresh = pt.argsort(h_to_sort)[:num_machine].cpu().data.numpy()
#     print_out("to_machine"+str(y[to_machine]))
    to_human_adapt_thresh = np.array([i for i in range(X.shape[0]) if ( h_to_sort[i]>model_thresh)])
    to_machine_adapt_thresh = np.array([i for i in range(X.shape[0]) if i not in to_human_adapt_thresh])
    if (len(to_human_adapt_thresh)==0):
        to_human_adapt_thresh = []
    if (len(to_machine_adapt_thresh)==0):
        to_machine_adapt_thresh = []
        
        
        
    machine_candidates_adapt_thresh_offline = pt.argsort(h_to_sort)[:num_machine].cpu().data.numpy()
#     print_out("to_machine"+str(y[to_machine]))
    to_human_adapt_thresh_offline = np.array([i for i in range(X.shape[0]) if (i not in machine_candidates_adapt_thresh_offline and h_to_sort[i]>model_thresh)])
    to_machine_adapt_thresh_offline = np.array([i for i in range(X.shape[0]) if i not in to_human_adapt_thresh])
    if (len(to_human_adapt_thresh_offline)==0):
        to_human_adapt_thresh_offline = []
    if (len(to_machine_adapt_thresh_offline)==0):
        to_machine_adapt_thresh_offline = []
#     print_out(to_human, to_machine, to_human_adapt_thresh, to_machine_adapt_thresh)
#     print_out(to_human, to_machine)
    to_machine_half = pt.nonzero(h_to_sort<0.5)[:, 0].cpu().data.numpy()
    print_out("to_machine "+str(to_machine_half))
    print(h_to_sort)
    to_human_half = np.array([i for i in range(X.shape[0]) if i not in to_machine_half])
    if (len(to_human_half)==0):
        to_human_half = []
        
    to_machine_thresh = []
    to_human_thresh = []
    acc_thresh = []
    exp_thresh = []
    for i in range(len(thresh)):
        to_machine_thresh.append(pt.nonzero(h_to_sort<1-thresh[i])[:, 0].cpu().data.numpy())
        to_human_thresh.append(np.array([j for j in range(X.shape[0]) if j not in to_machine_thresh[i]]))
        if (len(to_human_thresh[i])==0):
            to_human_thresh[i] = []
        acc_thresh.append(1-np.sum(y_h[to_human_thresh[i]])/len(Y)-np.sum(y[to_machine_thresh[i]])/len(Y))
        exp_thresh.append(len(to_human_thresh[i])/len(Y))
        
        
    

    if len(Y)-num_machine<len(to_human_half):
        to_human_half_constrained = pt.argsort(h_to_sort[to_human_half])[num_machine-len(Y):].cpu().data.numpy()
    else:
        to_human_half_constrained = to_human_half
#     print_out(len(to_human_half_constrained))

#     print_out(h_to_sort[pt.argsort(h_to_sort)])
#     print_out(to_machine, to_machine_half)
    to_machine_half_constrained = np.array([i for i in range(X.shape[0]) if i not in to_human_half_constrained])
    print("to_machine_half_constrained: "+str(to_machine_half_constrained))
    if (len(to_machine_half_constrained)==0):
            to_machine_half_constrained = []
#     if (len(to_human_half_constrained)==0):
#         to_human_half_constrained = []
#     print_out("to_human_half"+str(y_h[to_human_half]))
#     print_out(h, y, y_h)
    acc = 1-np.sum(y_h[to_human])/len(Y)-np.sum(y[to_machine])/len(Y)
    acc_thresh_adapt = 1-np.sum(y_h[to_human_adapt_thresh])/len(Y)-np.sum(y[to_machine_adapt_thresh])/len(Y)
    acc_thresh_adapt_offline = 1-np.sum(y_h[to_human_adapt_thresh_offline])/len(Y)-np.sum(y[to_machine_adapt_thresh_offline])/len(Y)
    acc_half = 1-np.sum(y_h[to_human_half])/len(Y)-np.sum(y[to_machine_half])/len(Y)
    acc_half_constrained = 1-np.sum(y_h[to_human_half_constrained])/len(Y)-np.sum(y[to_machine_half_constrained])/len(Y)
#     print_out("Accuracy = ", acc, acc_half)
    exp_thresh_adapt = len(to_human_adapt_thresh)/len(Y)
    exp_thresh_adapt_offline = len(to_human_adapt_thresh_offline)/len(Y)
    exp_half = len(to_human_half)/len(Y)
    exp_half_constrained = len(to_human_half_constrained)/len(Y)
    return acc, acc_thresh_adapt, acc_thresh_adapt_offline, acc_half, acc_half_constrained, exp_thresh_adapt, exp_thresh_adapt_offline, exp_half, exp_half_constrained, acc_thresh, exp_thresh

def validation_test(X, Y, Y_h, X_val, Y_val, Y_h_val, N_iter, bi, thr, dim_hidden, dim_in):
    accs_mixed_val = 0
#     print_out("thr="+str(thr))
    for j in range(N_iter):
        model1 = LN(dim_hidden, dim_in)
        model2 = LN(dim_hidden, dim_in)
        #     model_ = LN(10)
        Tb_val, Cb_val, marg_val = train_losses(model1, model2, X, Y, Y_h, bi, thr, dim_hidden, dim_in, False)
        accs, exps_mixed_val, rnd_mixed_val = test_losses(model1, model2, X_val, Y_val, Y_h_val, Tb_val, Cb_val, marg_val)
        accs_mixed_val += accs
        if (thr<1e-8):
            print_out(accs)
    accs_mixed_val /=N_iter
#     if (thr<1e-8):
#         print_out("b="+str(bi)+", Acc="+str(accs_mixed_val))
    return accs_mixed_val

def point1(a, b, alpha):
    ret = (1+alpha)/2*a+(1-alpha)/2*b
    if ((ret - a)*( b-a)>np.abs(b-a)**2+1e-6):
        raise NameError('Err!')
    return (1+alpha)/2*a+(1-alpha)/2*b
def point2(a, b, alpha):
    ret = (1-alpha)/2*a+(1+alpha)/2*b
    if ((ret - a)*( b-a)>np.abs(b-a)**2+1e-6):
        raise NameError('Err!')
    return (1-alpha)/2*a+(1+alpha)/2*b

def golden (func, x0, v, MAX_ITER):
    class solution:
        def __init__(self, point, func, err, step, vector, threshold):
            self.maximizer = point
            self.eval_max = func
            self.err = err
            self.step = step
            self.opt_vec = vector
            self.threshold = threshold
#     if (pt.is_tensor(x0) ==0 or pt.is_tensor(v) == 0):
#         raise NameError('All the input points of function golden must be tensor.')

    #parameters
    alpha = np.sqrt(5, dtype = np.double)-2
    ABSTOL = 1e-3
    
    #Defs
    a = np.zeros([MAX_ITER,])
    b = np.zeros([MAX_ITER,])
    x1 = np.zeros([MAX_ITER,])
    x2 = np.zeros([MAX_ITER,])
    prec_threshold = False
    
    #Initialization of a and b
    a[0] = x0
    b[0] = x0+v
    #Main loop
    for i in range(MAX_ITER-1):
#         print_out('Iteration '+str(i))
        if (np.abs(a[i]- b[i])<=np.abs(a[0]- b[0])/100):
            prec_threshold = True
            break;
        if (i==0):
            x1[i] = point1(a[i], b[i], alpha)
            x2[i] = point2(a[i], b[i], alpha)
        f1 = func(x1[i])
        f2 = func(x2[i])
#         print_out(a[i], x1[i], x2[i], b[i])
#         print_out('f1, f2, f3, f4 =', func(a[i, :]), func(x1[i, :]), func(x2[i, :]), func(b[i, :]))
#         print_out('Err =', pt.abs(func(x1[i, :])-func(x2[i, :])).detach().cpu().numpy()<ep_diff)
        if (np.abs(f1-f2)<ABSTOL):
            # print_out('Case I')
            a[i+1] = x1[i]
            b[i+1] = x2[i]
            x1[i+1] = point1(a[i+1], b[i+1], alpha)
            x2[i+1] = point2(a[i+1], b[i+1], alpha)
        elif (f1<f2):
            # print_out('Case II')
            a[i+1] = x1[i]
            b[i+1] = b[i]
            x1[i+1] = x2[i]
            x2[i+1] = point2(a[i+1], b[i+1], alpha)
        elif (f1>f2):
            # print_out('Case III')
            a[i+1] = a[i]
            b[i+1] = x2[i]
            x2[i+1] = x1[i]
            x1[i+1] = point1(a[i+1], b[i+1], alpha)
        if (np.abs(a[i]-b[i])-np.abs(a[0]-b[0])>0):
                raise NameError('Golden Search is going outside of the interval! ||a-b||_2 = '+str(np.abs(a[0]-b[0]))+' ||a_k - b_k||_2 = '+str(np.abs(a[i]-b[i]).detach().cpu().numpy()))
    maximizer = (a[i]+b[i])/2 # i takes MAX_ITER - 1 in normal loop without break
#     print_out(pt.matmul(minimizer-x0, v.T))
#     print_out('v = ',v)
    err = b[i]-a[i]
    step = np.abs(maximizer-x0)/np.abs(v)
    print_out('step =' +str(step))
#             print_out('Grad_f_new.p = '+str(pt.matmul(grad_func(x0+v*step), v.T).detach().cpu().numpy()))
    print_out(func)
    fzero = func(x0)
    fmax = func(maximizer)
    if ((fmax-fzero)<-ABSTOL):
        print_out('Returned '+str(maximizer)+' but the result wasn\'t increasing. Indeed, the evaluation of function at the obtained point was '+ str(fmax)+' whereas the x0 function was '+str(fzero)+'.')
        print_out('||v||_2 = '+ str(np.abs(v)))
        print_out('||maximizer - x0||_2 = '+ str(np.abs(maximizer-x0)))
        return golden (func, x0, v/2, MAX_ITER)
    else:
        print_out('Max='+str(fmax))
        return solution(maximizer, fmax, err, step, v, prec_threshold)
def train_losses(model1, model2, X, Y, Y_h, b, margin, dim_hidden, dim_in, adaptive_margin=True):
    TOL =  1e-5
    
#     margin = 0.01#NeurIPS2022
#     print_out(Y)
    X = X.detach().cpu()
    Y = Y.detach().cpu()
    Y_h = Y_h.detach().cpu()
    if (adaptive_margin==True):
        length_X = X.shape[0]
        valid_perc = 0.2
        valid_num = int(np.floor(valid_perc*length_X))
        X_val = X[:valid_num]
        Y_val = Y[:valid_num]
        Y_h_val = Y_h[:valid_num]
        X = X[valid_num:]
        Y = Y[valid_num:]
        Y_h = Y_h[valid_num:]
        valid_fun = lambda tau: validation_test(X, Y, Y_h, X_val, Y_val, Y_h_val, 3, b, tau, dim_hidden, dim_in)
        # print_out(b)
        sol = golden(valid_fun, 0.0, margin, 3)
        margin = sol.maximizer
#     print_out(model.parameters())
    len_X = X.shape[0]
    minibatch_len = np.floor(len_X/10).astype(int)
    num_epochs = 20
    num_batches = int(X.shape[0] / minibatch_len)
    optimizer1 = pt.optim.Adam(model1.parameters(), lr=0.03)
    optimizer2 = pt.optim.Adam(model2.parameters(), lr=0.03)
    loss_func = nn.BCELoss(reduction='none')
    train_losses = []
    best_val_loss = 1000
    eps = 1e-4
    
    for epoch in range(num_epochs):

#         print_out('----- epoch:',epoch, '-----')
        train_loss = 0
#         with pt.no_grad():
#             mprim = copy.deepcopy(model)
        machine_loss = []
        for i in range(num_batches):
            X_batch = X[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            Y_batch = Y[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            optimizer1.zero_grad()
            loss1 = loss_func(model1(X_batch),Y_batch)
#             print_out(Y_batch)
#             print_out(pt.autograd.grad(loss_temp.sum(), model.parameters()))
#             print_out(model(X_batch), Y_batch, loss)
            loss1.sum().backward()
            optimizer1.step()
            train_loss += float(loss1.mean())
#             print_out(loss.mean())
        train_losses.append(train_loss / num_batches)
#         print_out('machine_loss:', train_loss/num_batches)

    for epoch in range(num_epochs):
#         print_out('----- epoch:',epoch, '-----')
        train_loss = 0
#         with pt.no_grad():
#             mprim = copy.deepcopy(model)
        machine_loss = []
        for i in range(num_batches):
            X_batch = X[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            Y_h_batch = Y_h[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            optimizer2.zero_grad()
            loss2 = loss_func(model2(X_batch),Y_h_batch)
#             print_out(Y_batch)
#             print_out(pt.autograd.grad(loss_temp.sum(), model.parameters()))
#             print_out(model(X_batch), Y_batch, loss)
            loss2.sum().backward()
            optimizer2.step()
            train_loss += float(loss2.mean())
#             print_out(loss.mean())
        train_losses.append(train_loss / num_batches)
#         print_out('machine_loss:', train_loss/num_batches)
    l_AI = model1(X.to(device))
    l_H  = model2(X.to(device))
    d = (l_H-l_AI).detach().cpu().numpy()
    neg_num = np.sum(d[:, 0, 0]<=TOL)
    if (b>=neg_num/X.shape[0]):
#         print_out('After cut-off!')
        Tb = 0
        Cb = 0
    else:
#         print_out('Before cut-off!')
#         print_out(int(np.floor(b*X.shape[0])))
        Tb = np.sort(d[:,0, 0])[int(np.floor(b*X.shape[0]))]
        u = np.argsort(X[:, 0, 0])
#         plt.plot(X[u].detach().cpu().numpy(), d[u])
        ind_in_marg = np.logical_and(d[:,0, 0]>=(Tb-margin), d[:, 0, 0]<=(Tb+margin))
#         print_out(ind_in_marg.shape)
        ind_more = d[:, 0, 0]>(Tb+margin)
        ind_less = d[:, 0, 0]<(Tb-margin)
        num_ds_in_marg = np.sum(ind_in_marg)
        num_ind_less = np.sum(ind_less)
#         print_out(num_ds_in_marg, num_ind_less)
        Cb = np.abs((b*X.shape[0]-num_ind_less)/num_ds_in_marg)
#     print_out('Finished training!')
#     print_out(b, Tb, Cb, margin)
    return Tb, Cb, margin

def test_losses(model1, model2, X, Y, Y_H, Tb, Cb, margin):
    TOL=1e-5
    l_AI = model1(X.to(device)).detach().cpu().numpy()
    l_H = model2(X.to(device)).detach().cpu().numpy()
    d = (l_H-l_AI)
    h1 = (d[:, 0]<(Tb-margin))*1.0
    h2 = (np.logical_and(d[:,0]>=(Tb-margin), d[:, 0]<=(Tb+margin)))*Cb
    h = h1+h2
#     if (margin<1e-8):
        
#         print_out("X.shape = "+str(X.shape))
#         print_out("Margin is "+str(margin))
#         print_out("sum(h1)="+str(sum(h1))+" Tb="+str(Tb))
#         plt.plot(np.sort(d), np.zeros_like(d), 'x')
#     print_out(h)
    y = Y.detach().cpu().numpy()
    y_h = Y_H.detach().cpu().numpy()
#     print_out(h, y, y_h)
    acc = 1-np.sum(h*y_h[:,0]+(1-h)*y[:,0])/len(Y)
    
    if (acc>1.0):
        print("acc="+str(acc))
    exp = np.sum(h)/len(Y)
    rnd = np.sum((np.logical_and(d[:,0]>=(Tb-margin), d[:, 0]<=(Tb+margin)))*1.0)/(len(Y))
#     print_out("Accuracy = ", acc)
#     print_out("Expected Deferral = ", exp)
    return acc, exp, rnd
