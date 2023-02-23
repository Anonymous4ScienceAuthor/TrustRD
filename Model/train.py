import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
import time
import copy
from tensorboardX import SummaryWriter
from Model.model import *
import pickle
import torch.distributions as dist


def classify(treeDic, x_test  , x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter, fold_count):
    # Load the saved model state dictionary

    # Create a new instance of the model and load the saved state dictionary
    unsup_model = Net(64, 3).to(device)
    loss_list = []
    epoch_list = []
    for unsup_epoch in range(25):

        optimizer = th.optim.Adam(unsup_model.parameters(), lr=lr, weight_decay=weight_decay)
        unsup_model.train()
        traindata_list, _ = loadBiData(dataname, treeDic, x_train+x_test, x_test, 0.2, 0.2)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        batch_idx = 0
        loss_all = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            optimizer.zero_grad()
            Batch_data = Batch_data.to(device)
            loss = unsup_model(Batch_data)
            loss_all += loss.item() * (max(Batch_data.batch) + 1)

            loss.backward()
            optimizer.step()
            batch_idx = batch_idx + 1
        # calculate loss for epoch and save
        loss_epoch = loss_all / len(train_loader)
        print('unsup_epoch [{}/{}], Loss: {:.4f}'.format(25, unsup_epoch, loss_epoch))
        loss_list.append(loss_epoch)
        epoch_list.append(unsup_epoch)
    name = "best_pre_"+dataname +"l_4unsup" + ".pkl"
    th.save(unsup_model.state_dict(), name)
    # print('Finished the unsuperivised training.', '  Loss:', loss)
    # print("Start classify!!!")
    unsup_model.eval()


    model = Classfier(64*3,64,4).to(device)
    prior_mean = th.zeros_like(model.linear_one.weights_mean)
    prior_log_var = th.ones_like(model.linear_one.weights_log_var) * -10
    opt = th.optim.Adam(model.parameters(), lr=0.0005, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        model.train()
        unsup_model.train()
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            _, Batch_embed = unsup_model.encoder(Batch_data.x, Batch_data.edge_index, Batch_data.batch)

            # Data perturbation
            sigma_m = 0.1
            eta = 0.4
            noise = th.randn_like(Batch_embed) * sigma_m
            noisy_embed = Batch_embed + eta * noise
            loss_data = F.mse_loss(model(noisy_embed, Batch_data)[0], model(Batch_embed, Batch_data)[0])

            # Parameter perturbation
            zeta = 0.02
            model_copy = copy.deepcopy(model)
            with th.no_grad():
                for param, param_copy in zip(model.parameters(), model_copy.parameters()):
                    noise = th.randn_like(param)
                    noise = zeta * noise / noise.norm(p=2)
                    param_copy.data.add_(noise)
            loss_para = F.mse_loss(model(Batch_embed, Batch_data)[0], model_copy(Batch_embed, Batch_data)[0])
            out_labels,pred_prob, kl_div = model(Batch_embed, Batch_data)
            finalloss=F.nll_loss(out_labels,Batch_data.y)



            # Add the KL divergence term to the final loss
            loss=finalloss+ kl_div * 0.5+0.2*loss_para+0.2*loss_data
            opt.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            opt.step()
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 ))
            batch_idx = batch_idx + 1
            

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        unsup_model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            Batch_embed = unsup_model.encoder.get_embeddings(Batch_data)
            val_out,pred_prob,_ = model(Batch_embed, Batch_data)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc1+Acc2+Acc3+Acc4/4), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(Acc1+Acc2+Acc3+Acc4/4)

        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} ".format(epoch))

        res = ['acc:{:.4f}'.format((np.mean(temp_val_Acc1)+np.mean(temp_val_Acc2)+np.mean(temp_val_Acc3)+np.mean(temp_val_Acc4))/4),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('unsup_epoch:', (epoch+1) ,'   results:', res)

        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'TrustRD_'+str(fold_count)+'_', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if epoch>=199:
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,val_accs,accs,F1,F2,F3,F4


if __name__ == '__main__':
    lr = 0.0005
    weight_decay = 1e-4
    patience = 10
    batchsize = 128
    TDdroprate = 0.4
    BUdroprate = 0.4
    datasetname = sys.argv[1]  # "Twitter15"、"Twitter16"
    iterations = int(sys.argv[2])
    model = "TrustRD"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # device = th.device( 'cpu')
    n_epochs = 200
    # for unsup_epoch in range(30):
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    for iter in range(iterations):
        fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train = load5foldData(datasetname)

        treeDic = loadTree(datasetname)
        t1 = time.time()
        train_losses, val_losses,  val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = classify(treeDic,
                                                                                                  fold0_x_test,
                                                                                                  fold0_x_train,
                                                                                                  TDdroprate,
                                                                                                  BUdroprate,
                                                                                                  lr, weight_decay,
                                                                                                  patience,
                                                                                                  n_epochs,
                                                                                                  batchsize,
                                                                                                  datasetname,
                                                                                                  iter, 0)
        train_losses, val_losses,val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = classify(treeDic,
                                                                                                  fold1_x_test,
                                                                                                  fold1_x_train,
                                                                                                  TDdroprate,
                                                                                                  BUdroprate, lr,
                                                                                                  weight_decay,
                                                                                                  patience,
                                                                                                  n_epochs,
                                                                                                  batchsize,
                                                                                                  datasetname,
                                                                                                  iter, 1)
        train_losses, val_losses, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = classify(treeDic,
                                                                                                  fold2_x_test,
                                                                                                  fold2_x_train,
                                                                                                  TDdroprate,
                                                                                                  BUdroprate, lr,
                                                                                                  weight_decay,
                                                                                                  patience,
                                                                                                  n_epochs,
                                                                                                  batchsize,
                                                                                                  datasetname,
                                                                                                  iter, 2)
        train_losses, val_losses, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = classify(treeDic,
                                                                                                  fold3_x_test,
                                                                                                  fold3_x_train,
                                                                                                  TDdroprate,
                                                                                                  BUdroprate, lr,
                                                                                                  weight_decay,
                                                                                                  patience,
                                                                                                  n_epochs,
                                                                                                  batchsize,
                                                                                                  datasetname,
                                                                                                  iter, 3)
        train_losses, val_losses,  val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = classify(treeDic,
                                                                                                  fold4_x_test,
                                                                                                  fold4_x_train,
                                                                                                  TDdroprate,
                                                                                                  BUdroprate, lr,
                                                                                                  weight_decay,
                                                                                                  patience,
                                                                                                  n_epochs,
                                                                                                  batchsize,
                                                                                                  datasetname,
                                                                                                  iter, 4)
        test_accs.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
        NR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
        FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
        print("check  iter: {:04d} | aaaaaccs: {:.4f}".format(iter, test_accs[iter]))
        t2 = time.time()
        print("total time:")
        print(t2 - t1)
    print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations,
        sum(UR_F1) / iterations))



