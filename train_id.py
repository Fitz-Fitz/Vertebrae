# -*- coding: utf-8 -*-
# @Author: wuhan
# @Date:   2022-08-13 16:45:23
# @Last Modified by:   wuhan
# @Last Modified time: 2022-10-13 19:40:43

import os 
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch.nn.functional as F
import sys
from torchvision.models.resnet import resnet50

from torchvision.ops.boxes import torchvision
sys.path.append(".")
from data.id_dataset import id_dataset


class log_writer():
    def __init__(self, txt_name = ""):
        super().__init__()
        self.name = txt_name
    def write(self,info):
        writer = open(self.name,"a+")
        data = info + "\n"
        writer.writelines(data)
        print(info)
        writer.close()
    
def caculate_acc(output, target):
    output = F.softmax(output, dim=1)
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    index = np.argmax(output, axis=1)
    num = np.sum(index == target)
    all = len(target)
    return num,all

if __name__=="__main__":
    log_path = "logs/identification/"
    log_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    writer = SummaryWriter(log_path+log_name)
    txt_name = log_path + log_name + ".txt"
    txt_writer = log_writer(txt_name)
    base_train_path = "D:/Data/VerSe/VerSe19/verse19train/enhance_ct_8/enhance_drr"
    base_test_path = "D:/Data/VerSe/VerSe19/verse19test/enhance_ct_8/enhance_drr"#"D:/Data/VerSe/VerSe19/verse19test/enhance_ct_8/enhance_drr"
    save_path = log_path #"models/localization/"
    pre_trained_path = "logs/localization/DRR_Localization/enhance_8/119.pth" 
    #"https://download.pytorch.org/models/resnet50-0676ba61.pth" save to C:\Users\DELL/.cache\torch\hub\checkpoints\resnet50-0676ba61.pth
    epochs = 120
    base_lr = 0.001
    eval_epoch = 1
    log_inter = 50
    save_inter = 10
    compare = []

    model = torchvision.models.resnet50()
    model.fc = nn.Sequential(nn.Linear(2048, 24))
    # model.load_state_dict(torch.load("logs/identification/resnet50/119.pth")['net'])
    # nn.ReLU(),
    # nn.Dropout(0.1),
    # nn.Linear(1024, 512),
    # nn.ReLU(),
    # nn.Dropout(0.1),
    # nn.Linear(512, 24),
# )
    # model = models.segmentation.deeplabv3_resnet50(num_classes=1)#, weights_backbone=True)
    # model.load_state_dict(torch.load(pre_trained_path)['net'])
    # model = drr_net().cuda()
    
    # resnet50 = models.resnet50()
    # pretrained_dict = torch.load(pre_trained_path)
    # resnet50.load_state_dict(state_dict)
    # pretrained_dict = resnet50.state_dict()
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # model.load_state_dict(torch.load(""))
    mdeol = model.cuda()
    model.train()

    loss_func = nn.CrossEntropyLoss() #BCELoss2d()#nn.NLLLoss()#
    loss_func = loss_func.cuda()
   
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    schedule = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    txt_writer.write("-"*8 + "reading data" + "-"*8)
    train = id_dataset(drr_path=base_train_path, mode="train")
    test = id_dataset(drr_path=base_test_path, mode="test")
    trainset = DataLoader(dataset=train, batch_size=128, shuffle = True)
    testset = DataLoader(dataset=test, batch_size=128, shuffle = False) # trainset#
    txt_writer.write("-"*8 + "start identification training" + "-"*8)
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_all = 0
        for i, (data, target) in enumerate(trainset):
            data = data.cuda()
            target = target.long().cuda()
            optimizer.zero_grad()
            output = model(data)# feature,
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            
            correct, all = caculate_acc(output, target)
            train_correct += correct
            train_all += all
            running_loss += loss.item()
            if i % log_inter == (log_inter-1): 
                txt_writer.write(f'[*]epoch {epoch+1} [{i}/{len(trainset)}] loss is {running_loss/(i+1):.8f}')
        writer.add_scalar('Train  Loss', running_loss / (i+1), epoch)
        writer.add_scalar('Train  Acc', train_correct/train_all, epoch)
        schedule.step()
        epoch_lr = optimizer.param_groups[0]['lr']
        txt_writer.write(f'[*]Training epoch {epoch+1} Loss is {running_loss/(i+1):.8f},train_acc is {100*train_correct/train_all:.3f}%, lr is {epoch_lr}.')
        running_loss = 0.0
        
        if epoch%eval_epoch == eval_epoch-1:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_all = 0
            with torch.no_grad():
                for (data, target) in testset:
                    data, target = data.float(), target.float()
                    data = data.cuda()
                    target = target.long().cuda()
                    output = model(data)# feature,
                    loss = loss_func(output, target)
                    correct, all = caculate_acc(output, target)
                    test_loss += loss.item()
                    test_correct += correct
                    test_all += all
            writer.add_scalar('Test Loss', test_loss / len(testset), epoch)
            writer.add_scalar('Test Acc', test_correct/test_all, epoch)
            now = time.time()
            period = str(datetime.timedelta(seconds=int(now-start_time)))
            txt_writer.write(f'[*]Test finish, test epoch {epoch+1} Loss is {test_loss / len(testset):.8f},test_acc is {100*test_correct/test_all:.3f}%, training time is {period}')
            model.train()
        if epoch%save_inter == save_inter - 1:
            txt_writer.write("Save params to " + save_path+str(epoch+1)+".pth")
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': schedule.state_dict()
                }
            torch.save(checkpoint,save_path+str(epoch)+".pth")
            compare.append(test_correct/test_all)
            index = compare.index(max(compare))
            txt_writer.write(f'Now best test model is {(index+1)*save_inter}.pth')