## import module
from loader import *
from model import MDSyn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

import random
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pylab as plt
import matplotlib.patches as patches
from sklearn import metrics
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score,f1_score


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write(','.join(map(str, AUCs)) + '\n')



SEED=0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

def train(model, device, drug1_loader_train, drug2_loader_train, linc, optimizer):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        lincs = linc.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output, weight = model(data1, data2, lincs)
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))

def predicting(model, device, drug1_loader_test, drug2_loader_test, linc):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            lincs = linc.to(device)
            output, weight = model(data1, data2, lincs)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten(), weight


modeling = MDSyn
print(modeling.__name__)


TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 5e-4
LOG_INTERVAL = 20
NUM_EPOCHS = 300

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

device = torch.device('cuda:0')

drug1_data = DrugcombDataset(root='./data', dataset='ONeil_Drug1')
drug2_data = DrugcombDataset(root='./data', dataset='ONeil_Drug2')
row_lincs = pd.read_csv("data/raw/gene_vector.csv", index_col=0, header=None)
lincs_array = row_lincs.to_numpy()
linc = torch.tensor(lincs_array, dtype=torch.float32)
lenth = len(drug1_data)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

best_auc = 0
best_fold_data = None
num_folds = 5
best_auc_per_fold = [0] * num_folds
best_epoch_per_fold = [0] * num_folds
random_num = random.sample(range(0, lenth), lenth)

for i in range(num_folds):
    test_num = random_num[pot*i:pot*(i+1)]
    train_num = random_num[:pot*i] + random_num[pot*(i+1):]

    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    file_result = './result/metric/MD-Syn_fold' + str(i) +  '.csv'
    AUCs = ('Epoch,AUC,PR_AUC,ACC,BACC,Precision,TPR,KAPPA,RECALL,F1')
    with open(file_result, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, linc, optimizer)
        T, S, Y, weight = predicting(model, device, drug1_loader_test, drug2_loader_test, linc)

        # caculate metrics
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        FPR = fp / (fp + tn)
        fpr, tpr, _ = roc_curve(T, S)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall = recall_score(T, Y)
        precision = precision_score(T, Y)
        F1 = f1_score(T, Y)
        AUCs = [epoch, AUC, PR_AUC, ACC, BACC, precision, TPR, KAPPA, recall, F1]

        # save metric data
        if best_auc < AUC:
            best_auc = AUC
            save_AUCs(AUCs, file_result)
            fpr, tpr, _ = roc_curve(T, S)
        print('best_auc', best_auc)

    # plot ROC image
    roc_auc = auc(fpr, tpr)
    tprs.append(interp(mean_fpr, fpr, tpr))
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, label=f'Fold {i + 1} (AUC = {roc_auc:.3f})')

# 定制ROC曲线图
plt.plot([0,1],[0,1],linestyle = '--', color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.3f )' % (mean_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('5-Fold Cross-Validation')
plt.legend(loc="lower right")
plt.savefig("./result/figure/MD-Syn_ROC.png", dpi=600)
plt.show()



