import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from time import strftime, localtime
import copy
from utils import r2_score


class Trainer:
    def __init__(self, args, logger, model, train_dataloader, test_dataloader, optimizer) -> None:
        self.args = args    
        self.model = model    
        self.train_dataloader = train_dataloader  
        self.test_dataloader = test_dataloader   
        self.logger = logger  
        self.optimizer = optimizer
        self.adj = torch.from_numpy(np.random.randint(2, size=(47, 47))).to(self.args.device)
  
        self.logger.info('training arguments:')  
        for arg in vars(self.args):  
            self.logger.info('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))  

    def train(self):  
        best_loss = np.inf  
        best_mse, best_rmse, best_mae = None, None, None  
        self.criterion_mse = nn.MSELoss()   
        self.criterion_smoothl1 = nn.SmoothL1Loss()    
          
        for epoch in range(self.args.epochs):  
            self.logger.info('>' * 60)  
            self.logger.info('epoch: {}'.format(epoch+1))    
            self.model.train()  
            loss = 0   
            
            train_loss_all = []
            

            # for index, (x, y) in enumerate(tqdm(self.train_dataloader, desc='Train')):  
            #     inputs = x.to(self.args.device).squeeze()
            #     outputs = self.model(inputs, self.adj)  
            #     target = y.to(self.args.device) 
            #     loss_mse = self.criterion_mse(outputs, target)  
            #     loss_l1 = self.criterion_smoothl1(outputs, target)  
            #     loss = loss_mse + loss_l1  
            #     loss.backward()  
            #     self.optimizer.step()    
            #     self.optimizer.zero_grad()   
            #     train_loss_all.append(loss.item())
            
            # train_loss = np.array(train_loss_all).mean()
            # print("train loss: {:.2f}".format(train_loss))
            self.model = torch.load("./best_model/GAT_ATCN_mse_692.67_rmse_20.81_mae_19.74")
              
            test_loss, mse, rmse, mae = self.evaluate()  
            if test_loss < best_loss:  
                best_loss = test_loss  
                best_mse = mse  
                best_rmse = rmse  
                best_mae = mae  
                # save best model  
                if not os.path.exists('./best_model'):  
                    os.mkdir('./best_model')    
                model_path = './best_model/{}_mse_{:.2f}_rmse_{:.2f}_mae_{:.2f}'.format(  
                                                                        self.args.model_name, mse, rmse, mae)  
                self.best_model = copy.deepcopy(self.model)  
                self.logger.info('>> saved:{}'.format(model_path))   
          
        self.logger.info('>' * 60)  
        self.logger.info('save best model')  
        torch.save(self.best_model, model_path)    
        self.logger.info('mse: {:.2f}, rmse: {:.2f}, mae: {:.2f}'.format(best_mse, best_rmse, best_mae))   

    def evaluate(self):

        loss_all, mse, rmse, mae = [], [], [], []
        criterion_mae = nn.L1Loss()
        self.model.eval() # 开验证模式

        with torch.no_grad():
            val_loss_all = []
            val_rmse_all = []
            val_mae_all = []
            val_mse_all = []
            val_label_all = []
            val_pred_all = []
            for index, (x, y) in enumerate(tqdm(self.test_dataloader, desc='Test')):
                inputs = x.to(self.args.device).squeeze()  
                outputs = self.model(inputs, self.adj)  
                target = y.to(self.args.device) 
                val_label_all.append(target)
                val_pred_all.append(outputs)
                loss_mse = self.criterion_mse(outputs, target)   
                loss_rmse = torch.sqrt(loss_mse)
                loss_mae = criterion_mae(outputs, target)
                loss_l1 = self.criterion_smoothl1(outputs, target)   
                loss = loss_mse + loss_l1 
                
                val_mse_all.append(loss_mse.item())
                val_loss_all.append(loss.item())
                val_rmse_all.append(loss_rmse.item())
                val_mae_all.append(loss_mae.item())
            r2_pred = torch.stack(val_pred_all)
            r2_label = torch.stack(val_label_all)
            self.model.train() 
        loss_all = np.array(val_loss_all).mean() 
        mse = np.array(val_mse_all).mean()
        rmse = np.array(val_rmse_all).mean()
        mae = np.array(val_mae_all).mean()
        r2 = np.array(r2_score(r2_label, r2_pred).item())
        print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}, test loss: {loss_all:.2f}")

        return loss_all, mse, rmse, mae