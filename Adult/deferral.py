# Generate data from a Gaussian distribution in torch

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_out(*args):
    # print_out(st)
    print(args)
    f = open("out"+str(seed_global)+".txt", "a")
    for st in args:
        f.write(str(st))
        print(st)


    f.write('\n')
    #print_out('s')
    f.close()
# Generate data from a Gaussian distribution
def find_machine_samples(machine_loss, hloss,constraint):
    if (len(hloss.shape)==2 and hloss.shape[1]==1):
        hloss = hloss.squeeze(1)
    if (len(machine_loss.shape)==2 and machine_loss.shape[1]==1):
        machine_loss = machine_loss.squeeze(1)
    diff = machine_loss - hloss
    # print_out("machine_loss: ", machine_loss)
    # print_out("hloss: ", hloss)
    # print_out("diff.shape: ", diff.shape)
    # print_out("machine_loss.shape: ", machine_loss.shape)
    # print_out("hloss.shape: ", hloss.shape)
    argsorted_diff = torch.clone(torch.argsort(diff))
    num_outsource = int(constraint * machine_loss.shape[0])
    index = -num_outsource
    # print_out("diffs: ", diff[argsorted_diff])
    while (index < -1 and diff[argsorted_diff[index]] <= 0):
        index += 1
    if index==0:
        index = -1
    if index == -diff.shape[0]:
        index = 1
    # print_out("index: ", index)
    machine_list = argsorted_diff[:index]

    return machine_list
def train_sorted(model, loss_h, loss_h_val, loss_classifier_in, X, Y, Y_h, X_val, Y_val, Y_h_val, b):
    valid_num = Y_val.shape[0]
    len_X = X.shape[0]
    # loss_h = loss_human_in(X, Y)
    loss_c = loss_classifier_in(X)
    # loss_h_val = loss_human_in(X_val, Y_val)
    loss_c_val = loss_classifier_in(X_val)
    minibatch_len = 100 #np.floor(len_X/10).astype(int)
    num_epochs = 200
    num_batches = int(X.shape[0] / minibatch_len)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_func = nn.BCELoss(reduction='none')
    train_losses = []
    best_val_loss = 1000
    eps = 1e-4
    print_out("Learning to predict deferral...")
    for epoch in range(num_epochs):
        train_loss = 0
        machine_loss = []
        for i in range(num_batches):
            X_batch = X[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            Y_batch = Y[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            Y_h_batch = Y_h[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            loss_c_batch = loss_c[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            loss_h_batch = loss_h[i * minibatch_len: (i + 1) * minibatch_len].to(device)
            # print_out("Y_batch.shape="+str(Y_batch.shape)+", Y_h_batch.shape="+str(Y_h_batch.shape))
            machine_indices = find_machine_samples(loss_c_batch, loss_h_batch, b)
            # print_out("len of machine_indices="+str(len(machine_indices)))
            lab = torch.tensor([0 if j in machine_indices else 1 for j in range(Y_h_batch.shape[0])], dtype=torch.float).to(device)
            optimizer.zero_grad()
            loss = loss_func(model(X_batch),lab.unsqueeze(1))
            loss.sum().backward()
            optimizer.step()
            train_loss += float(loss.mean())
        # print progress
        if epoch % 10 == 0:
            print_out('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / num_batches))
            print_out("num_machine: ", len(machine_indices))
        train_losses.append(train_loss / num_batches)

    
    num_machine = int(np.floor((1-b)*len(Y_val)))
    # print_out("size of model(torch.tensor(X_val).to(device))="+str(model(torch.tensor(X_val).to(device)).shape))
    pred_val = model(torch.tensor(X_val).to(device))[ :, 0]
    human_candidate = (torch.argsort(pred_val)[num_machine:])
    val_loss = np.zeros([valid_num-num_machine,])
    # print_out("num_machine="+str(num_machine)+", size_pred_val="+str(pred_val.shape)+", size_human_candidate="+str(human_candidate.shape))
    
    if (b!=0):
        for i in range(valid_num-num_machine):
            hyv = list(human_candidate[:i+1].cpu().numpy())
            yv = loss_c_val[hyv].detach().cpu().numpy()
            hyhv = list(human_candidate[:i+1].cpu().numpy())
            yhv = loss_h_val[hyhv].detach().cpu().numpy()
            val_loss[i] = np.sum(yhv-yv)
        print_out("b="+str(b))
        print_out("val_loss="+str(val_loss))
        thresh = pred_val[human_candidate[np.argmax(val_loss)]]
        optim_i = np.argmax(val_loss)
    else:
        thresh = torch.tensor([0.5])
        optim_i = valid_num
    print_out("Threshold="+str(thresh))
    print_out("optimal i is "+str(optim_i))
    # if (valid_num-num_machine!=0):
    # plt.figure()
    # plt.plot(np.arange(valid_num-num_machine)/(valid_num-num_machine), val_loss/valid_num, label='b = '+str(b))
    return float(thresh.detach().cpu())

def test_by_ordering(model, X, Y, Y_H, y_pred, b, model_thresh):
    y = Y.detach().cpu().numpy()
    y_h = Y_H.detach().cpu().numpy()
    # find the correct predictions of Y_pred
    y_pred = y_pred.detach().cpu().numpy()
    # a boolean that shows true whenever the prediction is correct
    print_out("shape of y_pred="+str(y_pred.shape)+", shape of y="+str(y.shape)+", shape of y_h="+str(y_h.shape))
    correct_c = (y[:, 0]==y_pred[:, 0])*1.0
    # correct_c = np.array([i for i in range(X.shape[0]) if (y[i]==y_pred[i])])
    # find the correct predictions of Y_H
    # correct_h = np.array([i for i in range(X.shape[0]) if (y[i]==y_h[i])])
    correct_h = (y[:, 0]==y_h[:, 0])*1.0
    h_to_sort = model(X)
    to_human_adapt_thresh = np.array([i for i in range(X.shape[0]) if ( h_to_sort[i]>model_thresh)])
    to_machine_adapt_thresh = np.array([i for i in range(X.shape[0]) if i not in to_human_adapt_thresh])
    if (len(to_human_adapt_thresh)==0):
        to_human_adapt_thresh = []
    if (len(to_machine_adapt_thresh)==0):
        to_machine_adapt_thresh = []

    # argsort h_to_sort decreasingly
    sort_idx = torch.argsort(h_to_sort, axis=0)
    # find the number that we defer to human
    num_to_human = int(np.floor(b*X.shape[0]))
    to_human_adapt_thresh_bounded = np.array([i for i in range(num_to_human) if ( h_to_sort[sort_idx[X.shape[0]-1-i]]>model_thresh )])
    to_machine_adapt_thresh_bounded = np.array([i for i in range(X.shape[0]) if i not in to_human_adapt_thresh])
    if (len(to_human_adapt_thresh_bounded)==0):
        to_human_adapt_thresh_bounded = []
    if (len(to_machine_adapt_thresh)==0):
        to_machine_adapt_thresh_bounded = []

    
    acc_thresh_adapt = np.sum(correct_h[to_human_adapt_thresh])/len(Y)+np.sum(correct_c[to_machine_adapt_thresh])/len(Y)
    exp_thresh_adapt = len(to_human_adapt_thresh)/len(Y)
    acc_thresh_bounded = np.sum(correct_h[to_human_adapt_thresh_bounded])/len(Y)+np.sum(correct_c[to_machine_adapt_thresh_bounded])/len(Y)
    exp_thresh_bounded = len(to_human_adapt_thresh_bounded)/len(Y)
    return acc_thresh_adapt, exp_thresh_adapt, acc_thresh_bounded, exp_thresh_bounded




def run():
    # include mushroom.npz
    val_rate = 0.2
    dataset = "adult"
    if (dataset == "mushroom"):
        data = np.load("mushroom.npz")
    elif (dataset == "adult"):
        data = np.load("adult.npz")
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    y_pred_h_train = data['y_pred_h_train']
    y_pred_h_test = data['y_pred_h_test']
    y_pred_class_train = data['y_pred_class_train']
    y_pred_class_test = data['y_pred_class_test']

    val_num = int(np.floor(X_train.shape[0]*val_rate))

    # convert to torch tensor
    X_train = torch.tensor(X_train[val_num:]).float().to(device)
    y_train = torch.tensor(y_train[val_num:]).long().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    y_test = torch.tensor(y_test).long().to(device)
    y_pred_h_train = torch.tensor(y_pred_h_train[val_num:]).long().to(device)
    y_pred_h_test = torch.tensor(y_pred_h_test).long().to(device)
    y_pred_class_train = torch.tensor(y_pred_class_train[val_num:]).long().to(device)
    y_pred_class_test = torch.tensor(y_pred_class_test).long().to(device)
    X_val = torch.tensor(X_train[:val_num]).float().to(device)
    Y_val = torch.tensor(y_train[:val_num]).long().to(device)
    y_pred_h_val = torch.tensor(y_pred_h_train[:val_num]).long().to(device)
    y_pred_class_val = torch.tensor(y_pred_class_train[:val_num]).long().to(device)

    batch_size = 100
    num_batch = int(np.floor(X_train.shape[0]/batch_size))
    num_epoch = 100


    acc_classifier = torch.sum(y_pred_class_test==y_test)/y_test.shape[0]
    print_out("acc_classifier="+str(acc_classifier))

    # train a neural network with two layers and with output of correct/incorrect human_pred as label
    class Classifier2(torch.nn.Module):
        def __init__(self, size_input):
            super(Classifier2, self).__init__()
            self.linear1 = torch.nn.Linear(size_input, 20)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(20, 1)
        def forward(self, x):
            y_pred = torch.sigmoid(self.linear2(self.relu(self.linear1(x))))
            return y_pred
    class Classifier3(torch.nn.Module):
        def __init__(self, size_input):
            super(Classifier3, self).__init__()
            self.linear1 = torch.nn.Linear(size_input, 20)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(20, 20)
            self.relu2 = torch.nn.ReLU()
            self.linear3 = torch.nn.Linear(20, 1)
        def forward(self, x):
            y_pred = torch.sigmoid(self.linear3(self.relu2(self.linear2(self.relu(self.linear1(x))))))
            return y_pred
    model_conf_human = Classifier2(X_train.shape[1])
    model_conf_human.to(device)
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model_conf_human.parameters(), lr=1e-3)
    print_out("Learning to predict human confidence...")
    for epoch in range(num_epoch):
        for batch in range(num_batch):
            x_batch = X_train[batch*batch_size:(batch+1)*batch_size, :].detach().to(device)
            y_batch = y_pred_h_train[batch*batch_size:(batch+1)*batch_size, :].detach().to(device)
            # find a boolean that human is correct
            y_batch = ((y_batch == y_train[batch*batch_size:(batch+1)*batch_size, :].to(device))*1.0).detach()
            y_pred = model_conf_human(x_batch)
            # print_out("y_pred: ", y_pred)
            # print_out("y_batch: ", y_pred)
            # if (batch>300):
            #     print_out("here")
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print average loss
        if epoch % 10 == 0:
            print_out('Epoch: ', epoch,' Loss: ', loss.item())
            conf_h_val_pred = model_conf_human(X_val) > 0.5*torch.zeros([X_val.shape[0], 1 ]).to(device)
            conf_h_val_true = (y_pred_h_val[:, 0].to(device) == Y_val[:, 0])
            print_out("Accuracy on validation set: ", torch.sum(conf_h_val_pred[:, 0] == conf_h_val_true.to(device))*1.0/Y_val.shape[0])

    # train another neural network with two layers and with output of classifier as label
    model_class = Classifier2(X_train.shape[1])
    model_class.to(device)
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model_class.parameters(), lr=1e-3)
    print_out("Learning to predict classifier confidence...")
    for epoch in range(num_epoch):
        for batch in range(num_batch):
            x_batch = X_train[batch*batch_size:(batch+1)*batch_size, :].detach().to(device)
            y_batch = y_train[batch*batch_size:(batch+1)*batch_size, :].detach().to(device)
            model_batch = y_pred_class_train[batch*batch_size:(batch+1)*batch_size, :]
            y_pred = model_class(x_batch)
            yclass = (model_batch > 0.5)*1.0
            label = ((yclass == y_batch)*1.0).detach()
            loss = criterion(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print_out('Epoch: ', epoch,' Loss: ', loss.item())


    loss_human = lambda x: 1-model_conf_human(x)
    loss_model = lambda x: 1-model_class(x)

    def obtain_Cb_Tb(loss_human, loss_model, b, X_val, margin=0.05, TOL=0.05):
        # print_out("obtain_Cb_Tb")
        # print_out("X_val.shape: ", X_val.shape)
        l_AI = loss_model(X_val.to(device))
        l_H  = loss_human(X_val.to(device))
        d = (l_H-l_AI).detach().cpu().numpy()
        neg_num = np.sum(d[:, 0]<=TOL)
        if (b>=neg_num/X_val.shape[0]):
            Tb = 0
            Cb = 0
        else:
            Tb = np.sort(d[:,0])[int(np.floor(b*X_val.shape[0]))]
            ind_in_marg = np.logical_and(d[:,0]>=(Tb-margin), d[:, 0]<=(Tb+margin))
            ind_less = d[:, 0]<(Tb-margin)
            num_ds_in_marg = np.sum(ind_in_marg)
            num_ind_less = np.sum(ind_less)
            Cb = np.abs((b*X_val.shape[0]-num_ind_less)/num_ds_in_marg)
        return Tb, Cb, margin

    def obtain_Cb_Tb_adaptive(loss_human, loss_model, X_val, Y_val, Y_h_val, b):
        max_margin = 0.3
        valid_fun = lambda tau: validation_test(loss_human, loss_model, X_val, Y_val, Y_h_val, 10, b, tau)
        sol = golden(valid_fun, 0.0, max_margin, 3)
        margin = sol.maximizer
        return margin

    def validation_test(loss_human, loss_model, X_val, Y_val, Y_h_val, N_iter, bi, thr):
        accs_mixed_val = 0
        for j in range(N_iter):
            Tb_val, Cb_val, _ = obtain_Cb_Tb(loss_human, loss_model, bi, X_val, margin=thr)
            accs, _, _, _ = test_losses(loss_human, loss_model, y_pred_class_val ,X_val, Y_val, Y_h_val, Tb_val, Cb_val, thr)
            accs_mixed_val += accs
            # print_out("thr: ", thr)
            # if (thr<1e-8):

                # print_out("accs: ", accs)
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

    def golden(func, x0, v, MAX_ITER):
        class solution:
            def __init__(self, point, func, err, step, vector, threshold):
                self.maximizer = point
                self.eval_max = func
                self.err = err
                self.step = step
                self.opt_vec = vector
                self.threshold = threshold
    #     if (torch.is_tensor(x0) ==0 or torch.is_tensor(v) == 0):
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
    #         print_out('Err =', torch.abs(func(x1[i, :])-func(x2[i, :])).detach().cpu().numpy()<ep_diff)
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
    #     print_out(torch.matmul(minimizer-x0, v.T))
    #     print_out('v = ',v)
        err = b[i]-a[i]
        step = np.abs(maximizer-x0)/np.abs(v)
        # print_out('step =' +str(step))
    #             print_out('Grad_f_new.p = '+str(torch.matmul(grad_func(x0+v*step), v.T).detach().cpu().numpy()))
        # print_out(func)
        fzero = func(x0)
        fmax = func(maximizer)
        if ((fmax-fzero)<-ABSTOL):
            # print_out('Returned '+str(maximizer)+' but the result wasn\'t increasing. Indeed, the evaluation of function at the obtained point was '+ str(fmax)+' whereas the x0 function was '+str(fzero)+'.')
            # print_out('||v||_2 = '+ str(np.abs(v)))
            # print_out('||maximizer - x0||_2 = '+ str(np.abs(maximizer-x0)))
            return golden (func, x0, v/2, MAX_ITER)
        else:
            # print_out('Max='+str(fmax))
            return solution(maximizer, fmax, err, step, v, prec_threshold)


    def test_losses(loss_human, loss_model, pred_AI, X, Y, Y_H, Tb, Cb, margin):
        TOL=1e-5
        l_AI = loss_model(X.to(device)).detach().cpu().numpy()
        # find prediction of AI based on l_AI
        # pred_AI = (class_model(X.to(device)).detach().cpu().numpy()[:,0]>0.5)*1.0
        # find correctness of that prediction
        correct_AI = (pred_AI.cpu().numpy()[:,0] == Y.detach().cpu().numpy()[:,0])*1.0
        # find correctness of human
        correct_H = (Y_H.detach().cpu().numpy()[:,0] == Y.detach().cpu().numpy()[:,0])*1.0
        l_H = loss_human(X.to(device)).detach().cpu().numpy()
        d = (l_H-l_AI)
        h1 = (d[:, 0]<(Tb-margin))*1.0
        h2 = (np.logical_and(d[:,0]>=(Tb-margin), d[:, 0]<=(Tb+margin)))*Cb
        h = h1+h2
        acc = np.sum(h*correct_H+(1-h)*correct_AI)/len(Y)
        if (acc>1.0):
            print_out("acc="+str(acc))
        exp = np.sum(h)/len(Y)
        rnd = np.sum((np.logical_and(d[:,0]>=(Tb-margin), d[:, 0]<=(Tb+margin)))*1.0)/(len(Y))
        return acc, exp, rnd, h
    N = 40
    acc = np.zeros([N, ])
    exp = np.zeros([N, ])
    rnd = np.zeros([N, ])
    acc_okati = np.zeros([N, ])
    exp_okati = np.zeros([N, ])
    acc_okati_bounded = np.zeros([N, ])
    exp_okati_bounded = np.zeros([N, ])
    acc_okati_half = np.zeros([N, ])
    exp_okati_half = np.zeros([N, ])
    
    for i in range(N):
        b = 0.025*i 
        marg = obtain_Cb_Tb_adaptive(loss_human, loss_model, X_val, Y_val, y_pred_h_val, b)
        # print_out('marg = '+str(marg))
        Tb, Cb, _ = obtain_Cb_Tb(loss_human, loss_model, b, X_val, margin=marg)

        # test the solution
        acc[i], exp[i], rnd[i], _ = test_losses(loss_human, loss_model, y_pred_class_test, X_test, y_test, y_pred_h_test, Tb, Cb, marg)
        print_out('acc = '+str(acc))
        print_out('exp = '+str(exp))
        print_out('acc_classifier = '+str(acc_classifier))

        # Use Okati's method to reach a deferral system
        model_okati = Classifier3(X_train.shape[1])
        model_okati.to(device)
        # print_out("Y_human.shape = "+str(Y_human.shape))
        # Y_pred_class = (model(X_test.to(device))[:,0]>0.5)*1.0
        # Y_pred_class_train = (model(X.to(device))[:,0]>0.5)*1.0
        # loss_human_okati = lambda x, y: ((human_pred(x)!=y)*1.0).detach()
        loss_h = ((y_pred_h_train!=y_train)*1.0).detach()
        loss_h_val = ((y_pred_h_val!=Y_val)*1.0).detach()
        loss_model_okati = loss_model #lambda x: (((model(x.to(device))[:,0]>0.5)*1.0)*1.0).detach()
        thresh = train_sorted(model_okati, loss_h, loss_h_val, loss_model_okati, X_train, y_train, y_pred_h_train, X_val, Y_val, y_pred_h_val, b)
        acc_okati[i], exp_okati[i], acc_okati_bounded[i], exp_okati_bounded[i] = test_by_ordering(model_okati, X_test, y_test, y_pred_h_test, y_pred_class_test, b, thresh)
        acc_okati_half[i], exp_okati_half[i], _, _ = test_by_ordering(model_okati, X_test, y_test, y_pred_h_test, y_pred_class_test, b, 0.5)
        del model_okati

        # np.savez('data'+'b='+str(b)+'seed='+str(seed)+'.npz', acc=acc, exp=exp, rnd=rnd, Tb=Tb, Cb=Cb, acc_classifier=acc_classifier, acc_okati=acc_okati, exp_okati=exp_okati)

        print_out('acc_okati = '+str(acc_okati))
        print_out('exp_okati = '+str(exp_okati))
        
    return acc, exp, rnd, Tb, Cb, acc_classifier,  acc_okati, exp_okati, acc_okati_bounded, exp_okati_bounded, acc_okati_half, exp_okati_half



def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--b', default=0.05, type=float)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train', default = False, action='store_true')
    parser.add_argument('--num_seeds', default = 10)

    args = parser.parse_args()
    return args
def main():
    args = get_args()
    # b = args.b
    train = args.train
    seed = args.seed
    global seed_global 
    seed_global = seed
    num_seeds = args.num_seeds
    N = 40
    train = False
    if (train):
        acc, exp, rnd, _, _, _,  acc_okati, exp_okati, acc_okati_bounded, exp_okati_bounded, acc_okati_half, exp_okati_half = run()
        np.savez('accs_seed='+str(seed)+'.npz', acc = acc, exp = exp, rnd = rnd,  acc_okati = acc_okati, exp_okati = exp_okati, acc_okati_bounded = acc_okati_bounded, exp_okati_bounded = exp_okati_bounded, acc_okati_half = acc_okati_half, exp_okati_half = exp_okati_half)
    else:
        acc_mean = np.zeros([N, ])
        exp_mean = np.zeros([N, ])
        rnd_mean = np.zeros([N, ])
        acc_okati_mean = np.zeros([N, ])
        exp_okati_mean = np.zeros([N, ])
        acc_okati_bounded_mean = np.zeros([N, ])
        exp_okati_bounded_mean = np.zeros([N, ])
        acc_std = np.zeros([N, ])
        exp_std = np.zeros([N, ])
        rnd_std = np.zeros([N, ])
        acc_okati_std = np.zeros([N, ])
        exp_okati_std = np.zeros([N, ])
        acc_okati_bounded_std = np.zeros([N, ])
        exp_okati_bounded_std = np.zeros([N, ])
        acc_okati_half_mean = np.zeros([N, ])
        exp_okati_half_mean = np.zeros([N, ])
        acc_okati_half_std = np.zeros([N, ])
        exp_okati_half_std = np.zeros([N, ])
        for seed_load in range(num_seeds):
            data = np.load("accs_seed="+str(seed_load)+".npz")
            acc = data['acc']
            exp = data['exp']
            rnd = data['rnd']
            acc_okati = data['acc_okati']
            exp_okati = data['exp_okati']
            acc_okati_bounded = data['acc_okati_bounded']
            exp_okati_bounded = data['exp_okati_bounded']
            acc_okati_half = data['acc_okati_half']
            exp_okati_half = data['exp_okati_half']
            acc_mean += acc
            exp_mean += exp
            rnd_mean += rnd
            acc_okati_mean += acc_okati
            exp_okati_mean += exp_okati
            acc_okati_bounded_mean += acc_okati_bounded
            exp_okati_bounded_mean += exp_okati_bounded
            acc_std += acc**2
            exp_std += exp**2
            rnd_std += rnd**2
            acc_okati_std += acc_okati**2
            exp_okati_std += exp_okati**2
            acc_okati_bounded_std += acc_okati_bounded**2
            exp_okati_bounded_std += exp_okati_bounded**2
            acc_okati_half_mean += acc_okati_half
            exp_okati_half_mean += exp_okati_half
            acc_okati_half_std += acc_okati_half**2
            exp_okati_half_std += exp_okati_half**2
        acc_mean /= num_seeds
        exp_mean /= num_seeds
        rnd_mean /= num_seeds
        acc_okati_mean /= num_seeds
        exp_okati_mean /= num_seeds
        acc_okati_bounded_mean /= num_seeds
        exp_okati_bounded_mean /= num_seeds
        acc_std = np.sqrt(acc_std/num_seeds - acc_mean**2)
        exp_std = np.sqrt(exp_std/num_seeds - exp_mean**2)
        rnd_std = np.sqrt(rnd_std/num_seeds - rnd_mean**2)
        acc_okati_std = np.sqrt(acc_okati_std/num_seeds - acc_okati_mean**2)
        exp_okati_std = np.sqrt(exp_okati_std/num_seeds - exp_okati_mean**2)
        acc_okati_bounded_std = np.sqrt(acc_okati_bounded_std/num_seeds - acc_okati_bounded_mean**2)
        exp_okati_bounded_std = np.sqrt(exp_okati_bounded_std/num_seeds - exp_okati_bounded_mean**2)
        acc_okati_half_mean /= num_seeds
        exp_okati_half_mean /= num_seeds
        acc_okati_half_std = np.sqrt(acc_okati_half_std/num_seeds - acc_okati_half_mean**2)
        exp_okati_half_std = np.sqrt(exp_okati_half_std/num_seeds - exp_okati_half_mean**2)


        # set fontsize
        my_dpi = 96
        plt.rcParams.update({'font.size': 35})
        # set size of figure
        plt.figure(figsize=(1000/my_dpi, 800/my_dpi), dpi=my_dpi)
        # plt.rcParams["figure.figsize"] = (20,16)
        # make the lines thicker
        plt.rcParams['lines.linewidth'] = 4
        for i in range(len(acc_okati_std)):
            if (i%3 == 0):
                acc_okati_std[i] = 0
                acc_okati_half_std[i] = 0
                exp_okati_std[i] = 0
                exp_okati_half_std[i] = 0
            if (i%3 == 1):
                acc_std[i] = 0
                acc_okati_half_std[i] = 0
                exp_std[i] = 0
                exp_okati_half_std[i] = 0
            if (i%3 == 2):
                acc_std[i] = 0
                acc_okati_std[i] = 0
                exp_std[i] = 0
                exp_okati_std[i] = 0
        # make error bars
        plt.errorbar(np.arange(0, N, 1)*0.025, acc_mean, yerr=acc_std, label='Our method')
        plt.errorbar(np.arange(0, N, 1)*0.025, acc_okati_mean, yerr=acc_okati_std, label='Adaptive thresholding')
        plt.errorbar(np.arange(0, N, 1)*0.025, acc_okati_half_mean, yerr=acc_okati_half_std, label='Okati et al. with T=0.5')
        plt.xlabel("Budget")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('acc.pdf', bbox_inches='tight')
        plt.show()
        print_out()
        
        plt.figure(figsize=(1000/my_dpi, 800/my_dpi), dpi=my_dpi)
        # plt.rcParams["figure.figsize"] = (20,16)
        plt.rcParams['lines.linewidth'] = 4
        plt.errorbar(np.arange(0, N, 1)*0.025, exp_mean, yerr=exp_std, label='Our method')
        plt.errorbar(np.arange(0, N, 1)*0.025, exp_okati_mean, yerr=exp_okati_std, label='Adaptive thresholding')
        plt.errorbar(np.arange(0, N, 1)*0.025, exp_okati_half_mean, yerr=exp_okati_half_std, label='Okati et al. with T=0.5')
        plt.xlabel("Budget")
        plt.ylabel("Expected Deferral")
        plt.legend()
        plt.savefig('exp.pdf', bbox_inches='tight')
        plt.show()
        print_out(np.arange(0, N, 1)*0.025)
    # save data
    # np.savez('data'+'b='+str(b)+'seed='+str(seed)+'.npz', acc=acc, exp=exp, rnd=rnd, Tb=Tb, Cb=Cb, acc_classifier=acc_classifier, acc_ubermensch=acc_ubermensch, exp_ubermensch=exp_ubermensch, rnd_ubermensch=rnd_ubermensch, acc_okati=acc_okati, exp_okati=exp_okati, acc_okati_ubermensch=acc_okati_ubermensch, exp_okati_ubermensch=exp_okati_ubermensch)
if __name__ == "__main__":
    main()
