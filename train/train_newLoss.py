import os, sys
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
#from torch.nn import init
from torch.utils.data import Subset, ConcatDataset, TensorDataset

#models
#from torchvision.models import resnet18
#from torchvision.models import resnet50
from torchvision.datasets.vision import VisionDataset
#import torchvision.models as models_

#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.data.distributed import DistributedSampler
#import torch.multiprocessing as mp

import argparse, copy
import pickle, time, random
from PIL import Image
import numpy as np

#from math import log10, sqrt, log2, log

from models import *
from deup import DEUP_model, create_network, create_network2

from netcal.scaling import LogisticCalibration, TemperatureScaling, BetaCalibration
from netcal.binning import IsotonicRegression
from ensemble_fusion import FusionClassifier


imageNet_original = False

torch.manual_seed(0)

random.seed(10000)

LOSS_MIN_CROSSENT = 0 # minimize cross entropy loss
LOSS_MIN_CROSSENT_UNC  = 1 # minimize cross_entropy_loss + uncertainty
LOSS_MIN_CROSSENT_MAX_UNC = 2 # minimize cross_entropy_loss - uncertainty = minimize cross_entropy_loss + maximize uncertainty 
LOSS_MIN_UNC = 3 # minimize uncertainty
LOSS_MAX_UNC = 4 # minimize -uncertainty = maximize uncertainty 
LOSS_MIN_BINARYCROSSENT = 5 # minimize binary  cross entropy loss

UNCERTAINTY_MEASURE = "PE" 
#UNCERTAINTY_MEASURE = "MI" 


#LOSS_2nd_stage = LOSS_MAX_UNC
LOSS_2nd_stage_wrong = LOSS_MIN_CROSSENT_MAX_UNC
#LOSS_2nd_stage_correct = LOSS_MIN_CROSSENT
LOSS_2nd_stage_correct = LOSS_MIN_CROSSENT_UNC

#option_stage2 = 'mix'
#option_stage2 = 'batch_mix'
option_stage2 = 'batch_mix2'
#option_stage2 = 'batch_granularity'

Normalize_entropy =  False #   True # 
cycle_lr = True #False #

normalization = False #  True #
normalization_sum = False #   True #   
standardization = False # True # 

Weighted_Sum = False # True # 
Adaptive_Balancing =  False #  True #
dynamic_weights =    False # True #

#for calibration - used only in binary classification problems
PlattScaling_Flag =  False 
IsotonicRegression_Flag = False #True  
TemperatureScaling_Flag = False
BetaCalibration_Flag =  False

print("MEASURE OF UNCERTAINTY IS " + UNCERTAINTY_MEASURE + ' LOSS 2ND STAGE IS ' + str(LOSS_2nd_stage_correct) + ' & ' + str(LOSS_2nd_stage_wrong) + ' OPTION 2ND STAGE ' +  option_stage2)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def entropy(output):
    """Calculates the entropy loss of a probability distribution."""
    log_probs = F.log_softmax(output, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    # entropy of each prediction
    return entropy




class Uncertainty(nn.Module):
    def entropy_loss(self, model, X, y, num_samples=10, reduction='mean'):
        """Calculates the entropy loss of a probability distribution."""
        probs = None 
        for _ in range(num_samples):
            #MC dropout
            model.enable_dropout()
            softmax_output = F.softmax( model(X), dim=1)

            if probs is None: probs = torch.zeros_like(softmax_output)
            probs = probs+softmax_output

            del softmax_output
            torch.cuda.empty_cache()

        probs /= float(num_samples)
        mask = probs == 0  # Create a mask of zero values
        probs[mask] = 10e-20  # Replace zero values with small_value

        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        if Normalize_entropy:
            _num_clases = torch.Tensor([probs.size(dim=1)]).to(entropy.device) 
            entropy_max = torch.log(_num_clases)
            entropy = torch.div(entropy, entropy_max) # normalized predictive entropy


        if reduction=='mean':
            loss = torch.mean(entropy) #mean entropy
        elif reduction=='sum':
            loss = torch.sum(entropy) #total entropy
        else:
            loss = entropy

        #probs = np.mean(outputs,axis=0) #p(y|D,x) = 1/T sum_T p(y|w_i,x)
        #probs = self._check_logs_zero(probs)
        #entropy_vals = -np.sum(probs * log_probs, axis=1)# predictive entropy PE=H
        #loss = np.mean(entropy_vals) # should we calculate the average here ??
        return  loss


    def mutual_information_loss(self,model, X, y, num_samples=10, reduction='mean'):
        """Calculates the entropy loss of a probability distribution."""
        probs = None
        mean_entropy = None
        #entropy H
        for _ in range(num_samples):
            #MC dropout
            model.enable_dropout()
            softmax_output = F.softmax(model(X), dim=1)

            if probs is None: probs = torch.zeros_like(softmax_output)
            probs = probs + softmax_output

            _entropy = entropy(softmax_output)
            if mean_entropy is None: mean_entropy = torch.zeros_like(_entropy)
            mean_entropy = mean_entropy + _entropy

            #probs = (probs+F.softmax(output, dim=1)) if probs is not None else F.softmax(output, dim=1) #probs = torch.zeros(output.shape).cuda()
            #mean_entropy = (mean_entropy+entropy(output)) if mean_entropy is not None else entropy(output) #: mean_entropy = torch.zeros(_entropy.shape).cuda()
            
            del softmax_output
            torch.cuda.empty_cache()

        probs /= float(num_samples)
        mask = probs == 0  # Create a mask of zero values
        probs[mask] = 10e-20  # Replace zero values with small_value

        log_probs = torch.log(probs)
        entropy_vals = -torch.sum(probs * log_probs, dim=1)# predictive entropy PE=H

        mean_entropy /= float(num_samples)
        mutual_information_vals = entropy_vals - mean_entropy

        if Normalize_entropy:
            num_clases = torch.Tensor([probs.size(dim=1)]).to(mutual_information_vals.device) 
            entropy_max = torch.log(num_clases)
            mutual_information_vals = torch.div(mutual_information_vals, entropy_max) # normalized mutual information 
        #mutual_information is maximum when the second term is 0 and the first is maxinum entropy (uniform distirbution)

        if reduction=='mean':
            loss = torch.mean(mutual_information_vals)  #mean mutial information
        elif reduction=='sum':
            loss = torch.sum(mutual_information_vals)  #total mutial information
        else:
            loss = mutual_information_vals


        
        return  loss


class CrossEntropy_Uncertainty_Loss(Uncertainty):
    # LOSS_MIN_CROSSENT_UNC  = 1 # minimizes cross_entropy_loss and uncertaint
    def __init__(self):
        super(CrossEntropy_Uncertainty_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=10):

        if standardization: 
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce = loss_ce/torch.std(loss_ce)
        
            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            #loss_unc = loss_unc/torch.std(loss_unc)

            loss_val = loss_ce + loss_unc
            loss_val = loss_val/torch.std(loss_val)

            return torch.mean(loss_val)
            #return torch.mean(loss_ce)-torch.mean(loss_unc)

        elif normalization:
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce_max, loss_ce_min = loss_ce.max(), loss_ce.min() 
            #loss_ce = (loss_ce-loss_ce_min)/(loss_ce_max-loss_ce_min)

            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            #loss_unc_max, loss_unc_min = loss_unc.max(), loss_unc.min() 
            #loss_unc = (loss_unc-loss_unc_min)/(loss_unc_max-loss_unc_min)

            loss_val = loss_ce + loss_unc
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)

            return torch.mean(loss_val) 
            #return torch.mean(loss_ce)-torch.mean(loss_unc) 
        
        elif normalization_sum:
        
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce = loss_ce/torch.sum(loss_ce)

            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            #loss_unc = loss_unc/torch.sum(loss_unc)
            
            loss_val = loss_ce + loss_unc
            loss_val = loss_val/torch.sum(loss_val)

            return torch.mean(loss_val)
            #return torch.mean(loss_ce)-torch.mean(loss_unc) 

        else:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = nn.CrossEntropyLoss()(y_pred,y)+self.entropy_loss(model, X, y, num_samples)
            else:
                loss_val = nn.CrossEntropyLoss()(y_pred,y)+self.mutual_information_loss(model, X, y, num_samples)

        return loss_val


class CrossEntropy_Certainty_Loss(Uncertainty):
    #LOSS_MIN_CROSSENT_MAX_UNC = 2 # minimize cross_entropy_loss - uncertainty = 
    # minimize cross_entropy_loss and  maximize uncertainty
    #  minimize cross_entropy_loss and  certainty  
    def __init__(self):
        super(CrossEntropy_Certainty_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=10):

        if standardization: 
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce = loss_ce/torch.std(loss_ce)
        
            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            #loss_unc = loss_unc/torch.std(loss_unc)

            loss_val = loss_ce - loss_unc
            loss_val = loss_val/torch.std(loss_val)

            return torch.mean(loss_val)
            #return torch.mean(loss_ce)-torch.mean(loss_unc)

        elif normalization:
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce_max, loss_ce_min = loss_ce.max(), loss_ce.min() 
            #loss_ce = (loss_ce-loss_ce_min)/(loss_ce_max-loss_ce_min)

            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            #loss_unc_max, loss_unc_min = loss_unc.max(), loss_unc.min() 
            #loss_unc = (loss_unc-loss_unc_min)/(loss_unc_max-loss_unc_min)

            loss_val = loss_ce - loss_unc
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)

            return torch.mean(loss_val) 
            #return torch.mean(loss_ce)-torch.mean(loss_unc) 
        
        elif normalization_sum:
        
            loss_ce = nn.CrossEntropyLoss(reduction='none')(y_pred, y)
            #loss_ce = loss_ce/torch.sum(loss_ce)

            if UNCERTAINTY_MEASURE == "PE":
                loss_unc = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_unc = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            #loss_unc = loss_unc/torch.sum(loss_unc)
            
            loss_val = loss_ce - loss_unc
            loss_val = loss_val/torch.sum(loss_val)

            return torch.mean(loss_val)
            #return torch.mean(loss_ce)-torch.mean(loss_unc) 

        else:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = nn.CrossEntropyLoss()(y_pred,y)-self.entropy_loss(model, X, y, num_samples)
            else:
                loss_val = nn.CrossEntropyLoss()(y_pred,y)-self.mutual_information_loss(model, X, y, num_samples)

        return loss_val


class Certainty_Loss(Uncertainty):
    #LOSS_MAX_UNC = 4 # minimize -uncertainty = 
    #  = maximize uncertainty = minimize certainty  
    def __init__(self):
        super(Certainty_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=10):
        
        if standardization: 
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            loss_val = loss_val/torch.std(loss_val)
            return -torch.mean(loss_val)

        elif normalization:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)

            return -torch.mean(loss_val) 
        
        elif normalization_sum:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            
            #print(loss_val)
            loss_val = loss_val/torch.sum(loss_val)
            return -torch.mean(loss_val) 

        else:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples)
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples)

        return -loss_val


class Uncertainty_Loss(Uncertainty):
    #LOSS_MIN_UNC = 3 # minimize -uncertainty = maximize uncertainty 
    def __init__(self):
        super(Uncertainty_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=10):
        if standardization: 
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')

            loss_val = loss_val/torch.std(loss_val)
            return torch.mean(loss_val)

        elif normalization:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)

            return torch.mean(loss_val) 
        
        elif normalization_sum:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples, reduction='none')
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples, reduction='none')
            
            #print(loss_val)
            loss_val = loss_val/torch.sum(loss_val)
            return torch.mean(loss_val) 

        else:
            if UNCERTAINTY_MEASURE == "PE":
                loss_val = self.entropy_loss(model, X, y, num_samples)
            else:
                loss_val = self.mutual_information_loss(model, X, y, num_samples)

        return loss_val


class CrossEntropy_Loss(nn.Module):
    #LOSS_MIN_CROSSENT= 0 # minimize cross_entropy_loss - uncertainty = 
    # minimize cross_entropy_loss 
    def __init__(self):
        super(CrossEntropy_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=0):
        if standardization:  
            _loss = nn.CrossEntropyLoss(reduction='none')
            loss_val = _loss(y_pred, y)
            loss_val = loss_val/torch.std(loss_val)
            return torch.mean(loss_val)

        elif normalization:
            _loss = nn.CrossEntropyLoss(reduction='none')
            loss_val = _loss(y_pred, y)
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)
            return torch.mean(loss_val)    

        elif normalization_sum:
            _loss = nn.CrossEntropyLoss(reduction='none')
            loss_val = _loss(y_pred, y)
            #print(loss_val)

            loss_val = loss_val/torch.sum(loss_val)
            return torch.mean(loss_val)          

        #if dynamic_weights:
        #    _loss = nn.CrossEntropyLoss(reduction='none')
        #   return _loss(y_pred, y)
             
        return nn.CrossEntropyLoss()(y_pred, y)
    

class BinaryCrossEntropy_Loss(nn.Module):
    #LOSS_MIN_CROSSENT= 0 # minimize cross_entropy_loss - uncertainty = 
    # minimize cross_entropy_loss 
    def __init__(self):
        super(BinaryCrossEntropy_Loss, self).__init__()
        
    def forward(self, model, X, y, y_pred, num_samples=0):
        if standardization:  
            _loss = nn.BCELoss(reduction='none')
            loss_val = _loss(y_pred, y)
            loss_val = loss_val/torch.std(loss_val)
            return torch.mean(loss_val)

        elif normalization:
            _loss = nn.BCELoss(reduction='none')
            loss_val = _loss(y_pred, y)
            loss_val_max, loss_val_min = loss_val.max(), loss_val.min() 
            loss_val = (loss_val-loss_val_min)/(loss_val_max-loss_val_min)
            return torch.mean(loss_val)    

        elif normalization_sum:
            _loss = nn.BCELoss(reduction='none')
            loss_val = _loss(y_pred, y)
            #print(loss_val)

            loss_val = loss_val/torch.sum(loss_val)
            return torch.mean(loss_val)          

        #if dynamic_weights:
        #    _loss = nn.CrossEntropyLoss(reduction='none')
        #   return _loss(y_pred, y)
             
        return nn.BCELoss()(y_pred, y)


class ImageNetLoader(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        #print(self.data)
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            #print("tensor")

        x = self.data['x'][idx] 
        y = self.data['y'][idx] 

        if self.transform:
            x = self.transform(x)

        return x, y
        
    def __len__(self):
        #return self.data['x'].shape[0]
        return len(self.data['x'])


class SmallImagenet(VisionDataset):
    # code taken from https://github.com/landskape-ai/ImageNet-Downsampled
    train_list = ['train_data_batch_{}'.format(i + 1) for i in range(10)]
    #train_list = ['train_data_batch_{}'.format(i + 1) for i in range(1)]
    #print("change imageNet dataset batches")
    val_list = ['val_data']

    def __init__(self, root="data", size=32, train=True, transform=None, classes=None, shuffle=False):
        super().__init__(root, transform=transform)
        file_list = self.train_list if train else self.val_list
        self.data = []
        self.targets = []
        for filename in file_list:
            filename = os.path.join(self.root, filename)
            with open(filename, 'rb') as f:
                entry = pickle.load(f)
            self.data.append(entry['data'].reshape(-1, 3, size, size))
            self.targets.append(entry['labels'])

        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.concatenate(self.targets).astype(int) - 1

        if classes is not None:
            classes = np.array(classes)
            filtered_data = []
            filtered_targets = []

            for l in classes:
                idxs = self.targets == l
                filtered_data.append(self.data[idxs])
                filtered_targets.append(self.targets[idxs])

            self.data = np.vstack(filtered_data)
            self.targets = np.concatenate(filtered_targets)

        if shuffle:
            list_val = np.arange(len(self.data))
            random.shuffle(list_val)
            shuffled_data = np.array([self.data[key] for key in list_val])
            shuffled_target = np.array([self.targets[key] for key in list_val])
            self.data = shuffled_data
            self.targets = shuffled_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class BinaryCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, positive_class='cat', negative_class=None):
        # Class labels for CIFAR-10
        class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Choose two classes for binary classification
        #positive_class = positive_class
     
        self.cifar10 = datasets.CIFAR10(root, train=train, download=True, transform=transform)
        self.positive_class = class_labels.index(positive_class)
        self.negative_class = class_labels.index(negative_class)
        self.data = []
        self.targets = []
        
        counter_positive_class = 0
        counter_negative_class = 0
        if negative_class is None:
            for data, target in self.cifar10:
                self.data.append(data)
                if target == self.positive_class:
                    counter_positive_class +=1
                    self.targets.append(1)  # 1 for positive class
                else:
                    counter_negative_class +=1
                    self.targets.append(0)  # 0 for negative class
        else:
            for data, target in self.cifar10:
                if target == self.positive_class:
                    counter_positive_class +=1
                    self.data.append(data)
                    self.targets.append(1)  # 1 for positive class
                elif target == self.negative_class:
                    self.data.append(data)
                    counter_negative_class +=1
                    self.targets.append(0)  # 0 for negative class

        print(positive_class + " class has size of " + str(counter_positive_class) + " and negtive class has size of " + str(counter_negative_class))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class DEUP():
    def __init__(self, trainloader, f_model, f_optim, loss_fn, device):
        self.trainloader = trainloader
        self.device=device

        self.f_predictor = f_model
        self.f_optimizer = f_optim
        self.e_predictor = ResNet(BasicBlock, [2, 2, 2, 2], 1, 18, 0.0).to(self.device) #create_network2(1, 1, 1024, 'relu', False, 5).to(self.device)
        self.e_optimizer =  optim.SGD(self.e_predictor.parameters(), lr=0.001, momentum=0.9)

        self.loss_fn=nn.CrossEntropyLoss(reduction='none')
        self.e_loss_fn=nn.MSELoss()

        self.percentage_valSet = 0.2


    def train(self, ):
        
        #calidation set for calibration
        size = int(len(self.trainloader.data_test))
        #ids = random.choices(list(range(size)), k=int(size*0.1))
        ids = random.sample(range(size), int(size*self.percentage_valSet))
        subset_test = Subset(self.trainloader.data_test, ids)

        size_train = int(len(self.trainloader.data_train))
        ids_train = random.sample(range(size_train), int(size*self.percentage_valSet))
        subset_train = Subset(self.trainloader.data_train, ids_train)

        subset = ConcatDataset([subset_test, subset_train])
        #sub_loader = DataLoader(subset, batch_size=32 , shuffle=True) 
        
        #self.model_deup.fit_uncertainty_estimator_dataloader(self.trainloader.data_test, epochs=200, batch_size=self.trainloader.batch_size)
        self.fit_uncertainty_estimator_dataloader(subset, epochs=100, batch_size=self.trainloader.batch_size)
        #self.fit_uncertainty_estimator_dataloader(self.trainloader.data_train, epochs=100, batch_size=self.trainloader.batch_size)



    def fit_uncertainty_estimator_dataloader(self, data, epochs=None, batch_size=128, data_test=None):

        train_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
        if data_test is not None: test_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

        #building the dataset
        self.f_predictor.eval()
        loss_predictions_list = [] #torch.tensor([], device=self.device)
        features_list = [] #torch.tensor([], device=self.device)
        for features, target in train_loader: # target here is the real class
            features = features.to(self.device)
            target = target.to(self.device)

            target_pred = self.f_predictor(features)
            loss_base_model = self.loss_fn(target_pred, target) #.cpu()
            features_list += features.tolist()
            loss_predictions_list += loss_base_model.tolist()


        loader = DataLoader(TensorDataset(torch.tensor(features_list), torch.tensor(loss_predictions_list).unsqueeze(-1)), shuffle=True, batch_size=batch_size)

        #training the error predictor
        self.e_predictor.train()
        train_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            e_loss_sum, j = 0, 0
            for features, target in loader: # target here is the loss of the base model
                features, target = features.to(self.device), target.to(self.device)
                self.e_optimizer.zero_grad()

                predicted_uncertainties = self.e_predictor(features)
                e_loss = self.e_loss_fn(predicted_uncertainties, target)

                e_loss.backward()
                self.e_optimizer.step()

                epoch_losses.append(e_loss.item())
                e_loss_sum += e_loss.item()
                j+=1


            e_loss_t_sum, i = 0, 0
            if data_test is not None: 
                for features_t, target_t in test_loader:
                    features_t = features_t.to(self.device)
                    target_t = target_t.to(self.device)

                    target_pred_t = self.f_predictor(features_t)
                    loss_base_model_t = self.loss_fn(target_pred_t, target_t).unsqueeze(-1)
                    
                    pred = self.e_predictor(features_t)
                    e_loss_t = self.e_loss_fn(pred, loss_base_model_t)
                    e_loss_t_sum += e_loss_t.item()
                    i+=1
                
                print("train," + str(e_loss_sum/j) + ",test," + str(e_loss_t_sum/i))
            train_losses.append(np.mean(epoch_losses))

        return train_losses


    def predict(self, X):
        self.e_predictor.eval()
        return self.e_predictor(X) #model_deup._uncertainty(features=X) #.get_prediction_with_uncertainty(X)


class dataset():
    def __init__(self, dataset_name="mnist", batch_size = 100,  batch_size_adv = 100):
        batch_size_test = 32
        # the shuffle needs to be false for the DataLoader to more easily store the IDs of wrong classified inputs 
        self.batch_size = batch_size
        self.batch_size_adv = batch_size_adv
        
        if dataset_name == "mnist":
            self.num_classes = 10
            self.data_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
            self.data_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())


            self.train_loader = DataLoader(self.data_train, batch_size = batch_size, shuffle=False) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train, batch_size = batch_size_adv, shuffle=False) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)


        elif dataset_name ==  "cifar10":
            self.num_classes = 10
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2471, 0.2435, 0.2616)

            #pre-processing 
            train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std), ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),])

            self.data_train = datasets.CIFAR10("../data", train=True, download=True, transform=train_transform)
            self.data_test = datasets.CIFAR10("../data", train=False, download=True, transform=test_transform)
            
            self.train_loader = DataLoader(self.data_train,    batch_size = batch_size, shuffle=False) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train,    batch_size = batch_size_adv, shuffle=False) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)


        elif dataset_name ==  "binaryCifar10":
            self.num_classes = 2
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2471, 0.2435, 0.2616)

            #pre-processing 
            train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std), ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),])

            positive_class='cat'
            negative_class='dog'
            self.data_train = BinaryCIFAR10("../data", train=True, transform=train_transform, positive_class=positive_class, negative_class=negative_class)
            self.data_test = BinaryCIFAR10("../data", train=False, transform=test_transform, positive_class=positive_class, negative_class=negative_class)
            
            self.train_loader = DataLoader(self.data_train,    batch_size = batch_size, shuffle=False) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train,    batch_size = batch_size_adv, shuffle=False) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)



        elif dataset_name ==  "cifar100":
            self.num_classes = 100
            cifar100_mean = (0.5071, 0.4867, 0.4408)
            cifar100_std = (0.2675, 0.2565, 0.2761)

            #pre-processing 
            train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar100_mean, cifar100_std), ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar100_mean, cifar100_std),])

            self.data_train = datasets.CIFAR100("../data", train=True, download=True, transform=train_transform)
            self.data_test = datasets.CIFAR100("../data", train=False, download=True, transform=test_transform)
            
            self.train_loader = DataLoader(self.data_train,    batch_size = batch_size, shuffle=False) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train,    batch_size = batch_size_adv, shuffle=False) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)


        elif dataset_name ==  "imageNet":
            self.num_classes = 1000
            if imageNet_original:
                print("Original dataset")
                traindir = '../data/imageNet/train'                
                valdir = '../data/imageNet/val'                
                crop_size = 224

                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                self.data_train = datasets.ImageFolder( traindir, transforms.Compose([
                        transforms.RandomResizedCrop(crop_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),normalize,
                    ]))

                self.data_test = datasets.ImageFolder(valdir, transforms.Compose([
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(),normalize,
                    ]))

            else:
                print("Downsampled dataset")
                root_dir = '../data/imageNet/'

                resolution=32 
                classes=1000
                
                normalize = transforms.Normalize(mean=[0.4810,0.4574,0.4078], std=[0.2146,0.2104,0.2138])

                tf_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
                tf_test = transforms.Compose([transforms.ToTensor(),normalize,])

                self.data_train = SmallImagenet(root=root_dir, size=resolution, train=True, transform=tf_train, classes=range(classes), shuffle=True) 
                self.data_test = SmallImagenet(root=root_dir, size=resolution, train=False, transform=tf_test, classes=range(classes)) 


            self.train_loader = DataLoader(self.data_train, batch_size = batch_size, shuffle=False, pin_memory=True,) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train, batch_size = batch_size_adv, shuffle=False, pin_memory=True,) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False, pin_memory=True,)


        elif dataset_name ==  "svhn":
            self.num_classes = 10

            self.data_train = datasets.SVHN("../data", split='train', download=True, transform=transforms.ToTensor())
            self.data_test = datasets.SVHN("../data", split='test', download=True, transform=transforms.ToTensor())

            self.train_loader = DataLoader(self.data_train, batch_size = batch_size, shuffle=False, pin_memory=True,) if batch_size > 0 else None
            self.trainAvd_loader = DataLoader(self.data_train, batch_size = batch_size_adv, shuffle=False, pin_memory=True,) if batch_size_adv > 0 else None
            self.test_loader = DataLoader(self.data_test, batch_size = batch_size_test, shuffle=False)

        else:
            raise("dataset not implementex")
        

    def load_data(self, idx, train=True, dir='../data'):
        if train:
            input_file = '/imageNet/train_data_batch_'
            #input_file = '/imageNet/Imagenet32_train/train_data_batch_'
            d = unpickle(dir+input_file+str(idx))
        else:
            input_file = '/imageNet/val_data'
            d = unpickle(dir+input_file)

        x = d['data']
        y = d['labels']

        y = [i-1 for i in y]

        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3))
        return x, y


    def update_trainLoader(self):
        self.train_loader = DataLoader(self.data_train, batch_size = self.batch_size, shuffle=False, pin_memory=True,) if self.batch_size > 0 else None
        self.trainAvd_loader = DataLoader(self.data_train, batch_size = self.batch_size_adv, shuffle=False, pin_memory=True,) if self.batch_size_adv > 0 else None


class trainModel():
    def __init__(self, device, half_prec=False, variants=None):
        self.device = device
        self.half_prec=half_prec

        #calibration FLAGS
        self.isCalibrated = False
        self.ToCalibrate = True if variants == 'calibration' else False

        # DEUP FLAGS
        self.deup = True if variants == 'deup' else False

        #ENSEMBLE FLAGS
        self.deep_ensemble = True if variants == 'ensemble' else False


        if self.ToCalibrate and self.deup and self.deep_ensemble:
            print("Calibration, DEUP and ensemble can not be used at the same time. Disabling deup.")
            self.ToCalibrate = False
            self.deup = False
            self.deep_ensemble = False

        elif self.ToCalibrate and self.deup:
            print("Calibration and DEUP can not be used at the same time. Disabling deup.")
            self.ToCalibrate = False
            self.deup = False
            self.deep_ensemble = False

        elif self.ToCalibrate and self.deep_ensemble:
            print("Calibration and ensemble can not be used at the same time. Disabling deup.")
            self.ToCalibrate = False
            self.deup = False
            self.deep_ensemble = False

        elif self.deup and self.deep_ensemble:
            print("ensemble and DEUP can not be used at the same time. Disabling deup.")
            self.ToCalibrate = False
            self.deup = False
            self.deep_ensemble = False


        self.LossInUse = LOSS_MIN_CROSSENT # LOSS_MIN_CROSSENT_UNC # 
        self.new_iterations = 0 #10
        self.printTimes = False #True

        if self.half_prec:
            # we only used mixed precision in imageNet
            #self.model, self.opt = amp.initialize(self.model, self.opt, opt_level="O1")
            # Creates once at the beginning of training
            self.scaler = torch.cuda.amp.GradScaler()

        return


    def writeResult(self, filename, data):
        trainTime, train_err, train_loss = data[0]
        testTime, test_err, test_loss = data[1]
        advTestTime_pgd, adv_err_pgd, adv_loss_pgd = data[2]

        str1 = "total_time=" + str(trainTime) + ";"
        str1 += "\ntraining_error=" + str(train_err) + ";"
        str1 += "\ntraining_loss=" + str(train_loss) + ";"

        str1 += "\ntesting_time=" + str(testTime) + ";"
        str1 += "\ntest_error=" + str(test_err) + ";"
        str1 += "\ntest_loss=" + str(test_loss) + ";"

        str1 += "\navd_test_time_pgd=" + str(advTestTime_pgd) + ";"
        str1 += "\nadversarial_error_pgd=" + str(adv_err_pgd) + ";"
        str1 += "\nadversarial_loss_pgd=" + str(adv_loss_pgd) + ";"

        if len(data) == 4:
            advTestTime_fgsm, adv_err_fgsm, adv_loss_fgsm = data[3]
            str1 += "\navd_test_time_fgsm=" + str(advTestTime_fgsm) + ";"
            str1 += "\nadversarial_error_fgsm=" + str(adv_err_fgsm) + ";"
            str1 += "\nadversarial_loss_fgsm=" + str(adv_loss_fgsm) + ";"

        f = open(filename, "w")
        f.write(str1)
        f.close()


    def saveModel(self, saveModel, state, filename, epoch):
        if not saveModel: return

        #torch.save(model.state_dict(), name)
        torch.save(state,  "../models/" + filename + "_epoch" + str(epoch) + ".pt")
        

    def LoadModel(self, model, opt, filename):
        filename = "../models/" + filename + ".pt"
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            training_time = checkpoint['training_time']
            train_err = checkpoint['error']
            train_loss = checkpoint['loss']

            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']+1))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            model, opt, training_time, train_err, train_loss, start_epoch = None, None, None, None, None, None

        return model, opt, training_time, train_err, train_loss, start_epoch


    def updateLogs(self, oldName, NewName, alg, ratio ,epsilon, numIt, alpha, ratioADV, epochMax):

        with open("../logs1/logs_" + oldName + '.txt', 'r') as f_read:
            with open("./logs/logs_" + NewName + '.txt', 'w') as f_write:
                for i, line in enumerate(f_read): 
                    if i == 0 :
                        f_write.write(line)
                    else:
                        #it,alg,ratio,epsilon,numIt,alpha,ratioAdv,algTest,epsilonTest,numItTest,alphaTest,Error,Loss,testingTime,trainTime
                        aux = line.split(",")
                        iteration = aux[0]
                        algTest = aux[7]
                        eps_test = aux[8]
                        num_iterTest = aux[9]
                        alpha_test = aux[10]
                        adv_err = aux[11]
                        adv_loss = aux[12]
                        advTestTime = aux[13]
                        trainTime = aux[14]
                        test_entropy = aux[15]
                        test_MI = aux[16]

                        if int(iteration) > epochMax: break

                        str_write = str(iteration) + "," + alg + "," + str(ratio) + "," + str(epsilon) + "," + str(numIt) + "," + str(alpha) + "," + str(ratioADV) + \
                                        "," + algTest +"," + str(eps_test) + "," + str(num_iterTest) + "," + str(alpha_test) + "," + \
                                        str(adv_err) + "," + str(adv_loss) + "," + str(advTestTime) + "," + str(trainTime) + "," + str(test_entropy) + "," + str(test_MI)    
                        f_write.write(str_write)
 

    def standard_train(self, model, modelName, loader, dataset, opt, iterations=10):
        '''training a standard model with checkpoint and saving the model.''' 
        if self.deep_ensemble:
            return self.standard_train_deep_ensemble(model, modelName, loader, dataset, opt, iterations)
             
        write_pred_logs = True
        lr = opt.param_groups[0]['lr']
        savePrevModel = False
        num_samples = 5

        
        for it in range(iterations, 0, -1):
            model_name = modelName + "_epoch" + str(it)
            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, model_name) # load model
            if _model is not None:
                # if models exists, load logs
                self.updateLogs(modelName, modelName, 'standard', 0, 0, 0, 0, 0, it)
                t1 = time.time()-trainTime
                model = _model
                opt = _opt
                counter += 1
                break
        print("Standard training")

        if _model is None: 
            it=0
            t1 = time.time()

        for counter in range(it+1, iterations+1):
            if counter == iterations: write_pred_logs = True
            print("epoch number " + str(counter))

            train_err, train_loss, misclassified_ids = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples)

            if counter % 5 == 0: 
                print("saving model on epoch " + str(counter))

                self.saveModel(True, {
                    'epoch': counter,
                    'state_dict': model.state_dict(),
                    'training_time': time.time() - t1,
                    'error': train_err,
                    'loss': train_loss,
                    'optimizer' : opt.state_dict(),}, modelName, counter)


        #opt = optim.Adam(model.parameters(), lr=lr/10.0)

        for param_group in opt.param_groups:
            param_group["lr"] = lr if not cycle_lr else  10e-4 #lr/10.0

        write_pred_logs = True

        
        if option_stage2 == 'batch_mix2':
            #separetes into different batchs the wrong and correct predicitons and uses differetn loss functions
            # min error for correct batch
            # max unc for wrong batchs
            # interleve batches

            if Adaptive_Balancing: 
                alpha = torch.tensor(0.9)
                alpha_delta = 0.05
                test_err_prev, test_unc_prev = None, None

            elif dynamic_weights:
                # Define the loss weights as nn.Parameter objects
                weight_loss1 = torch.tensor(0.5, requires_grad=True)
                weight_loss2 = torch.tensor(0.5, requires_grad=True)
                weight_optimizer = optim.Adam([{'params':[weight_loss1, weight_loss2]}],lr=0.01)
                alpha=[weight_optimizer, weight_loss1, weight_loss2]

            else:
                alpha=None
                
            if savePrevModel: 
                prev_model = copy.deepcopy(model)
                prev_test_err = None
                prev_test_unc = None

            epoch_dataSize = len(loader.data_train)
            #epoch_counter = 0
            epoch_counter = 1

            counter_dataSize = 0
            counter_repeat = 0
            half_batch_size=int(loader.batch_size/2)

            print("epoch number " + str(epoch_counter+iterations) + " and epoch size of " + str(epoch_dataSize))
            while epoch_counter < self.new_iterations+1:
                if self.printTimes: t1_init = time.time()
                if counter_repeat==0:
                    _, _, misclassified_ids = self.epoch(loader.train_loader, model) #test with training data

                    aux_correctclassified_ids = [i for i in range(epoch_dataSize) if i not in misclassified_ids]
                    size2include = len(misclassified_ids) if len(misclassified_ids) > half_batch_size else loader.batch_size-len(misclassified_ids)

                    # update loss fucntion and train with wrong predicitons
                    _subset_wrong = Subset(loader.data_train, misclassified_ids)
                    _train_loader_wrong = DataLoader(_subset_wrong, batch_size=half_batch_size , shuffle=True) 

                correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)
                _subset_correct = Subset(loader.data_train, correctclassified_ids)
                _train_loader_correct = DataLoader(_subset_correct, batch_size=half_batch_size, shuffle=True)
                #print("size correct " + str(len(_subset_correct)) + " size wrong " + str(len(_subset_wrong))) 
                if self.printTimes: print('time correct/wrong sets ' + str( time.time() - t1_init )) 

                # update loss fucntion and train with correct predicitons
                if self.printTimes: t1_init = time.time()
                self.epoch_interleave_batches(_train_loader_wrong,_train_loader_correct,  model, opt, num_samples=num_samples, weight_loss=alpha) # train
                if self.printTimes: print('time train ' + str( time.time() - t1_init )) 
                counter_dataSize += len(misclassified_ids)+len(correctclassified_ids)

                counter_repeat += 1
                #if counter_repeat==10: counter_repeat=0
                if counter_repeat>0: counter_repeat=0

                #test
                if counter_dataSize > epoch_dataSize:
                    if self.printTimes: t1_init = time.time()

                    test_err, _, test_entropy, test_MI, _, _, _, _ = self.testModel_logs(dataset, modelName, epoch_counter+iterations, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples)
                    test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                    counter_dataSize = 0
                    epoch_counter +=1

                    if savePrevModel: 
                        if prev_test_err is None or (test_err < 2.0*prev_test_err):
                            prev_test_err = test_err
                            prev_test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                            prev_model = copy.deepcopy(model)
                        else:
                            print("Rolling out the model")
                            model = prev_model


                    if Adaptive_Balancing:
                        if test_err_prev is None:
                            test_err_prev = test_err
                            test_unc_prev = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI

                        if test_err > test_err_prev and test_unc > test_unc_prev:
                            # error and uncertainty increseas
                            # increase alpha to give more weight to min Err Correct and reduce the weight of max Unc wrong
                            alpha +=alpha_delta

                        elif test_err > test_err_prev and test_unc < test_unc_prev:
                            # error increseas and uncertainty decreseas
                            # increase alpha to give more weight to min Err Correct and reduce the weight of max Unc wrong
                            alpha +=alpha_delta

                        elif test_err < test_err_prev and test_unc > test_unc_prev:
                            # error decreseas and uncertainty increases
                            alpha -=alpha_delta
                        
                        #elif test_err < test_err_prev and test_unc < test_unc_prev:
                        #    # error and uncertainty decreseas
                        #    #keep alpha

                        test_err_prev = test_err
                        test_unc_prev = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                        alpha = torch.clamp(alpha, 0.0, 1.0)


                    print("epoch number " + str(epoch_counter+iterations))
                    if self.printTimes: print('time testing ' + str( time.time() - t1_init )) 

                #del _subset_wrong,_subset_correct, _train_loader_wrong, _train_loader_correct
                #torch.cuda.empty_cache()


        elif option_stage2 == 'batch_mix':
            #separetes into different batchs the wrong and correct predicitons and uses differetn loss functions
            # min error for correct batch
            # max unc for wrong batchs
            # first batches of wrong and then batches of correcrt
            for t in range(1, self.new_iterations+1):
                #if t == self.new_iterations: write_pred_logs = True
                print("epoch number " + str(t+iterations))

                _, _, misclassified_ids = self.epoch(loader.train_loader, model) #test with training data

                aux_correctclassified_ids = [i for i in range(len(loader.data_train)) if i not in misclassified_ids]
                size2include = len(misclassified_ids) if len(misclassified_ids) > loader.batch_size else loader.batch_size-len(misclassified_ids)
                correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)

                # update loss fucntion and train with wrong predicitons
                self.LossInUse = LOSS_2nd_stage_wrong
                _subset = Subset(loader.data_train, misclassified_ids)
                _train_loader = DataLoader(_subset, batch_size = loader.batch_size, shuffle=False) 
                self.epoch(_train_loader, model, opt) # train

                self.testModel_logs(dataset, modelName, t+iterations, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs=2, num_samples=num_samples)

                # update loss fucntion and train with correct predicitons
                self.LossInUse = LOSS_MIN_CROSSENT 
                _subset = Subset(loader.data_train, correctclassified_ids)
                _train_loader = DataLoader(_subset, batch_size = loader.batch_size, shuffle=False) 
                self.epoch(_train_loader, model, opt) # train

                #test
                self.testModel_logs(dataset, modelName, t+iterations, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples)

                del _subset, _train_loader
                torch.cuda.empty_cache()


        elif option_stage2 == 'batch_granularity':
            epoch_dataSize = len(loader.data_train)
            #epoch_counter = 0
            epoch_counter = 1

            counter_dataSize = 0
            half_batch_size=int(loader.batch_size/2)

            print("epoch number " + str(epoch_counter+iterations) + " and epoch size of " + str(epoch_dataSize))

            while epoch_counter < self.new_iterations+1:

                _, _, misclassified_ids = self.epoch(loader.train_loader, model) #test with training data
                aux_correctclassified_ids = [i for i in range(epoch_dataSize) if i not in misclassified_ids]
                size2include = len(misclassified_ids) if len(misclassified_ids) > half_batch_size else loader.batch_size-len(misclassified_ids)

                _subset_wrong = Subset(loader.data_train, misclassified_ids)
                _train_loader_wrong = DataLoader(_subset_wrong, batch_size=half_batch_size , shuffle=True) 

                correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)
                _subset_correct = Subset(loader.data_train, correctclassified_ids)
                _train_loader_correct = DataLoader(_subset_correct, batch_size=half_batch_size, shuffle=True)

                # update loss fucntion and train with correct predicitons
                dataloader_iterator_wrong = iter(_train_loader_wrong)
                dataloader_iterator_correct = iter(_train_loader_correct)
        
                X_wrong,y_wrong = next(dataloader_iterator_wrong)
                X_correct,y_correct = next(dataloader_iterator_correct)

                X_wrong,y_wrong = X_wrong.to(self.device), y_wrong.to(self.device) # len of bacth size
                X_correct,y_correct = X_correct.to(self.device), y_correct.to(self.device) # len of bacth size

                if self.half_prec: 
                    # Runs the forward pass with autocasting.
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        y_pred_wrong = model(X_wrong)
                        y_pred_correct = model(X_correct)

                    if opt:   # backpropagation
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            self.LossInUse = LOSS_2nd_stage_wrong
                            loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=5)

                            self.LossInUse = LOSS_2nd_stage_correct
                            loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=5)

                            loss = loss_wrong + loss_correct

                        opt.zero_grad()
                        self.scaler.scale(loss).backward() # Scales the loss, and calls backward() . to create scaled gradients
                        self.scaler.step(opt) # Unscales gradients and calls or skips optimizer.step()
                        self.scaler.update()  # Updates the scale for next iteration


                else:
                    y_pred_wrong = model(X_wrong)
                    y_pred_correct = model(X_correct)

                    if opt:  # backpropagation
                        self.LossInUse = LOSS_2nd_stage_wrong
                        loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=5)

                        self.LossInUse = LOSS_2nd_stage_correct 
                        loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=5)

                        # Calculate the total loss with dynamic weights
                        loss = loss_wrong + loss_correct

                        opt.zero_grad()
                        loss.backward()
                        opt.step()



                counter_dataSize += half_batch_size*2.0 #len(misclassified_ids)+len(correctclassified_ids)
                #test
                if counter_dataSize > epoch_dataSize:
                    test_err, _, test_entropy, test_MI, _, _, _, _ = self.testModel_logs(dataset, modelName, epoch_counter+iterations, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples)
                    test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                    counter_dataSize = 0
                    epoch_counter +=1

                    print("epoch number " + str(epoch_counter+iterations))

                    #del _subset_wrong,_subset_correct, _train_loader_wrong, _train_loader_correct
                    #torch.cuda.empty_cache()


        else:
            self.LossInUse = LOSS_2nd_stage_wrong
            # mix correct and wrong prediciton in one batch and uses the same loss function
            for t in range(1, self.new_iterations+1):
                #if t == self.new_iterations: write_pred_logs = True
                print("epoch number " + str(t+iterations))

                _, _, misclassified_ids = self.epoch(loader.train_loader, model) #test with training data

                aux_correctclassified_ids = [i for i in range(len(loader.data_train)) if i not in misclassified_ids]
                size2include = len(misclassified_ids) if len(misclassified_ids) > loader.batch_size else loader.batch_size-len(misclassified_ids)
                correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)

                _subset = Subset(loader.data_train, misclassified_ids+correctclassified_ids)
                _train_loader = DataLoader(_subset, batch_size = loader.batch_size, shuffle=False) 
                _, _, misclassified_ids = self.epoch(_train_loader, model, opt) # train

                self.testModel_logs(dataset, modelName, t+iterations, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples)

                del _subset, _train_loader
                torch.cuda.empty_cache()



        if self.ToCalibrate:
            print("Model Calibration")
            self.calibrate(loader.data_test, model, num_samples=1)
            self.testModel_logs(dataset, modelName, counter, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples, calibration=True)


        elif self.deup:
            print("DEUP")
            if self.LossInUse == LOSS_MIN_CROSSENT_UNC: loss_fn = CrossEntropy_Uncertainty_Loss()
            elif self.LossInUse == LOSS_MIN_CROSSENT_MAX_UNC: loss_fn = CrossEntropy_Certainty_Loss()
            elif self.LossInUse == LOSS_MIN_UNC: loss_fn =  Uncertainty_Loss()
            elif self.LossInUse == LOSS_MAX_UNC: loss_fn = Certainty_Loss() 
            elif self.LossInUse == LOSS_MIN_BINARYCROSSENT: loss_fn = BinaryCrossEntropy_Loss()
            else: loss_fn = CrossEntropy_Loss()
            
            self.deup_model = DEUP(loader, model, opt, loss_fn, self.device)
            self.deup_model.train()

            self.testModel_logs(dataset, modelName, counter, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, write_pred_logs, num_samples=num_samples, calibration=False)

        trainTime = time.time() - t1

        return (trainTime, train_err, train_loss)


    def standard_pgd_train(self, model, modelName, loader, dataset, opt, iterations=10, ratio=1, num_iterTrain=20, eps_train=0.1, alpha_train=0.01, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        
        #load the model
        #logs_modelimageNet_std_train_lr0.1_momentum0.9_batch512_lrAdv0.0_momentumAdv0.0_batchAdv0_ratio0.3
        #logs_modelimageNet_robust_PGD_eps4.0_numIt20_alpha0.01_ratio1.0_ratioadv1.0_lr0.1_momentum0.9_batch512_lrAdv0.1_momentumAdv0.9_batchAdv512

        #loading the model
        write_pred_logs = True

        #update hyper-parameter for adversarial training
        lr = opt.param_groups[0]['lr']
        momentum = opt.param_groups[0]["momentum"]

        # load final models
        for it in range(iterations, 0, -1):
            model_name = modelName + "_epoch" + str(it) #str(iterations)
            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, model_name)
            if _model is not None:
                self.updateLogs(modelName, modelName, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, it) #iterations)
                t1 = time.time()-trainTime
                model = _model
                opt = _opt
                counter += 1
                break


        if _model is None: 
            # but check if we have a clean data model first
            auxName = modelName.split("_")
            ratioStd = 1 - ratio
            it = int(iterations * ratioStd)

            old_name = auxName[0] + "_std_train_" + auxName[8] + "_" + auxName[9] + "_" + auxName[10] + "_lrAdv0.0_momentumAdv0.0_batchAdv0"
            old_model_name = old_name + "_epoch" + str(it)

            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, old_model_name)
            if _model is not None:
                self.updateLogs(old_name, modelName, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, it)
                t1 = time.time()-trainTime
                model = _model
                opt = _opt
                counter += 1

        if _model is None: 
            t1 = time.time()
            counter = 1

        ST_it = int(iterations*(1-ratio))
        AT_it = int(iterations*ratio)

        for _ in range(counter, ST_it+1):
            print("epoch number (ST) " + str(counter))
            train_err, train_loss, _ = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1, write_pred_logs)

            if counter % 5 == 0: 
                print("saving model on epoch " + str(counter))

                self.saveModel(True, {
                            'epoch': counter,
                            'state_dict': model.state_dict(),
                            'training_time': time.time() - t1,
                            'error': train_err,
                            'loss': train_loss,
                            'optimizer' : opt.state_dict(),}, old_name, counter-1)
            counter += 1


        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv


        for t in range(counter, AT_it+1):
            print("epoch number  (AT)" + str(counter))
            #if counter == iterations: write_pred_logs=True

            train_err, train_loss, misclassified_ids_adv = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd", dataset, epsilon=eps_train, num_iter=num_iterTrain, \
                                                                                        alpha=alpha_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset,modelName,counter,'std_pgd',ratio,eps_train,num_iterTrain,alpha_train,ratio_adv,time.time()-t1,write_pred_logs)
            if counter % 5 == 0: 
                print("saving model on epoch " + str(counter))
                self.saveModel(True, {
                            'epoch': counter,
                            'state_dict': model.state_dict(),
                            'training_time': time.time() - t1,
                            'error': train_err,
                            'loss': train_loss,
                            'optimizer' : opt.state_dict(),}, modelName, counter)

            counter += 1

            
        #
        # here we end the normal training (ST + AT)
        #


        for param_group in opt.param_groups:
            param_group["lr"] = lr if not cycle_lr else  10e-4 #lr/10.0
            param_group["momentum"] = momentum                  


        #write_pred_logs = True
        #separetes into different batchs the wrong and correct predicitons and uses differetn loss functions
        # min error+unc for correct batch
        # min error + max unc for wrong batchs
        # interleve batches
        epoch_dataSize = len(loader.data_train)
        epoch_counter = 1
        counter_dataSize = 0
        half_batch_size=int(loader.batch_size/2)

        itST = int(self.new_iterations*(1-ratio)) 
        itAT = int(self.new_iterations*ratio)

        print("epoch number " + str(epoch_counter+iterations) + " and epoch size of " + str(epoch_dataSize))
        while epoch_counter < itST+1:
            _, _, misclassified_ids = self.epoch(loader.train_loader, model) # to determine wrong inputs

            aux_correctclassified_ids = [i for i in range(epoch_dataSize) if i not in misclassified_ids]
            size2include = len(misclassified_ids) if len(misclassified_ids) > half_batch_size else loader.batch_size-len(misclassified_ids)
            correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)

            # update loss fucntion and train with  wrong predicitons
            _subset_wrong = Subset(loader.data_train, misclassified_ids)
            _train_loader_wrong = DataLoader(_subset_wrong, batch_size = half_batch_size, shuffle=False) 

            # update loss fucntion and train with wrong predicitons
            _subset_correct = Subset(loader.data_train, correctclassified_ids)
            _train_loader_correct = DataLoader(_subset_correct, batch_size = half_batch_size, shuffle=False) 

            self.epoch_interleave_batches(_train_loader_wrong,_train_loader_correct,  model, opt) # train
            counter_dataSize += len(misclassified_ids)+len(correctclassified_ids)


            if counter_dataSize > epoch_dataSize:
                test_err, _, test_entropy, test_MI, _, _, _, _ =self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1, write_pred_logs)
                test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                counter_dataSize = 0
                epoch_counter +=1
                counter += 1

                print("epoch number " + str(epoch_counter+iterations))

            del _subset_correct, _train_loader_correct, _subset_wrong, _train_loader_wrong
            torch.cuda.empty_cache()

        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv if not cycle_lr else 10e-4 # lr_adv/10.0
            param_group["momentum"] = momentum_adv
        half_batch_size_adv=int(loader.batch_size_adv/2)
        epoch_counter = 1
        counter_dataSize = 0

        while epoch_counter < itAT+1:
            
            _, _, misclassified_ids_adv = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd", dataset, epsilon=eps_train, num_iter=num_iterTrain, \
                                                                                        alpha=alpha_train, ratio=ratio_adv, opt=None)

            aux_correctclassified_ids_adv = [i for i in range(epoch_dataSize) if i not in misclassified_ids_adv]
            size2include = len(misclassified_ids_adv) if len(misclassified_ids_adv) > half_batch_size_adv else loader.batch_size_adv-len(misclassified_ids_adv)
            correctclassified_ids_adv = random.choices(aux_correctclassified_ids_adv, k=size2include)

            # update loss fucntion and train with  wrong predicitons
            _subset_wrong = Subset(loader.data_train, misclassified_ids_adv)
            _train_loader_wrong = DataLoader(_subset_wrong, batch_size = loader.batch_size_adv, shuffle=False)

            # update loss fucntion and train with wrong predicitons
            _subset_correct = Subset(loader.data_train, correctclassified_ids_adv)
            _train_loader_correct = DataLoader(_subset_correct, batch_size = loader.batch_size_adv, shuffle=False)
            
            self.epoch_adversarial_interleave_batches(_train_loader_wrong,_train_loader_correct,model,"pgd",\
                                                        dataset,epsilon=eps_train,num_iter=num_iterTrain,alpha=alpha_train,ratio=ratio_adv,opt=opt)
            counter_dataSize += len(misclassified_ids_adv)+len(correctclassified_ids_adv)

            if counter_dataSize > epoch_dataSize:
                test_err, _, test_entropy, test_MI, _, _, _, _ =self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1, write_pred_logs)
                test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                counter_dataSize = 0
                epoch_counter +=1
                counter += 1

                print("epoch number " + str(epoch_counter+iterations))

            del _subset_wrong,_subset_correct,_train_loader_wrong, _train_loader_correct
            torch.cuda.empty_cache()

        trainTime = time.time() - t1

        return (trainTime, train_err, train_loss)


    def standard_fgsm_train(self, model, modelName, loader, dataset, opt, iterations=10, ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        #loading the model
        write_pred_logs = True

        #update hyper-parameter for adversarial training
        lr = opt.param_groups[0]['lr']
        momentum = opt.param_groups[0]["momentum"]

        # load final models
        for it in range(iterations, 0, -1):
            model_name = modelName + "_epoch" + str(it) #+ str(iterations)
            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, model_name)
            if _model is not None:
                self.updateLogs(modelName, modelName, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, iterations)
                t1 = time.time()-trainTime
                model = _model
                opt = _opt
                counter += 1
                break


        if _model is None: 
            # but check if we have a clean data model first
            auxName = modelName.split("_")
            ratioStd = 1.0 - ratio
            epoch = int(iterations * ratioStd)

            old_name = auxName[0] + "_std_train_" + auxName[6] + "_" + auxName[7] + "_" + auxName[8] + "_lrAdv0.0_momentumAdv0.0_batchAdv0"
            old_model_name = old_name + "_epoch" + str(epoch)

            _model, _opt, trainTime, train_err, train_loss, counter = self.LoadModel(model, opt, old_model_name)
            if _model is not None:
                self.updateLogs(old_name, modelName, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, epoch)
                t1 = time.time()-trainTime
                opt = _opt
                model = _model
                counter += 1
         
        if _model is None: 
            t1 = time.time()
            counter = 1

        ST_it = int(iterations*(1-ratio))
        AT_it = int(iterations*ratio)


        for t in range(counter, ST_it+1):
            print("epoch number (ST) " + str(counter))
            train_err, train_loss, _ = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)

            if counter % 5 == 0: 
                print("saving model on epoch " + str(counter))

                self.saveModel(True, {
                    'epoch': counter,
                    'state_dict': model.state_dict(),
                    'training_time': time.time() - t1,
                    'error': train_err,
                    'loss': train_loss,
                    'optimizer' : opt.state_dict(),}, old_name, counter)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        for _ in range(counter, AT_it+1):
            #if counter == iterations: write_pred_logs = True
            print("epoch number (AT) " + str(counter))
            train_err, train_loss, _ = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset,modelName,counter,'std_fgsm',ratio,eps_train,0,0,ratio_adv,time.time()-t1,write_pred_logs)  

            if counter % 5 == 0: 
                print("saving model on epoch " + str(counter))
                self.saveModel(True, {
                            'epoch': counter,
                            'state_dict': model.state_dict(),
                            'training_time': time.time() - t1,
                            'error': train_err,
                            'loss': train_loss,
                            'optimizer' : opt.state_dict(),}, modelName, counter)
            counter += 1

        #
        # here we end the normal training (ST + AT)
        #

        for param_group in opt.param_groups: # restore the ST HP values
            param_group["lr"] = lr if not cycle_lr else 10e-4 #  lr/10.0
            param_group["momentum"] = momentum     


        #separetes into different batchs the wrong and correct predicitons and uses differetn loss functions
        # min error+unc for correct batch
        # min error + max unc for wrong batchs
        # interleve batches
        epoch_dataSize = len(loader.data_train)
        epoch_counter = 1
        counter_dataSize = 0
        half_batch_size=int(loader.batch_size/2)

        itST = int(self.new_iterations*(1-ratio)) 
        itAT = int(self.new_iterations*ratio)

        print("epoch number " + str(epoch_counter+iterations) + " and epoch size of " + str(epoch_dataSize))
        while epoch_counter < itST+1:
            _, _, misclassified_ids = self.epoch(loader.train_loader, model) # to determine wrong inputs

            aux_correctclassified_ids = [i for i in range(epoch_dataSize) if i not in misclassified_ids]
            size2include = len(misclassified_ids) if len(misclassified_ids) > half_batch_size else loader.batch_size-len(misclassified_ids)
            correctclassified_ids = random.choices(aux_correctclassified_ids, k=size2include)

            # update loss fucntion and train with  wrong predicitons
            _subset_wrong = Subset(loader.data_train, misclassified_ids)
            _train_loader_wrong = DataLoader(_subset_wrong, batch_size = half_batch_size, shuffle=False) 

            # update loss fucntion and train with wrong predicitons
            _subset_correct = Subset(loader.data_train, correctclassified_ids)
            _train_loader_correct = DataLoader(_subset_correct, batch_size = half_batch_size, shuffle=False) 

            self.epoch_interleave_batches(_train_loader_wrong,_train_loader_correct,  model, opt) # train
            counter_dataSize += len(misclassified_ids)+len(correctclassified_ids)

            if counter_dataSize > epoch_dataSize:
                test_err, _, test_entropy, test_MI, _, _, _, _ =self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1, write_pred_logs)
                test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                counter_dataSize = 0
                epoch_counter +=1
                counter += 1

                print("epoch number " + str(epoch_counter+iterations))

            del _subset_correct, _train_loader_correct, _subset_wrong, _train_loader_wrong
            torch.cuda.empty_cache()

        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv if not cycle_lr else 10e-4 # lr_adv/10.0
            param_group["momentum"] = momentum_adv
        half_batch_size_adv=int(loader.batch_size_adv/2)
        epoch_counter = 1
        counter_dataSize = 0

        while epoch_counter < itAT+1:
            
            _, _, misclassified_ids_adv = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm", dataset, epsilon=eps_train, ratio=ratio_adv)

            aux_correctclassified_ids_adv = [i for i in range(epoch_dataSize) if i not in misclassified_ids_adv]
            size2include = len(misclassified_ids_adv) if len(misclassified_ids_adv) > half_batch_size_adv else loader.batch_size_adv-len(misclassified_ids_adv)
            correctclassified_ids_adv = random.choices(aux_correctclassified_ids_adv, k=size2include)

            # update loss fucntion and train with  wrong predicitons
            _subset_wrong = Subset(loader.data_train, misclassified_ids_adv)
            _train_loader_wrong = DataLoader(_subset_wrong, batch_size = loader.batch_size_adv, shuffle=False)

            # update loss fucntion and train with wrong predicitons
            _subset_correct = Subset(loader.data_train, correctclassified_ids_adv)
            _train_loader_correct = DataLoader(_subset_correct, batch_size = loader.batch_size_adv, shuffle=False)
            
            self.epoch_adversarial_interleave_batches(_train_loader_wrong,_train_loader_correct,model,"fgsm",dataset,epsilon=eps_train,ratio=ratio_adv,opt=opt)
            counter_dataSize += len(misclassified_ids_adv)+len(correctclassified_ids_adv)

            if counter_dataSize > epoch_dataSize:
                test_err, _, test_entropy, test_MI, _, _, _, _ =self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1, write_pred_logs)
                test_unc = test_entropy if UNCERTAINTY_MEASURE == 'PE' else test_MI
                counter_dataSize = 0
                epoch_counter +=1
                counter += 1

                print("epoch number " + str(epoch_counter+iterations))

            del _subset_wrong,_subset_correct,_train_loader_wrong, _train_loader_correct
            torch.cuda.empty_cache()


        trainTime = time.time() - t1

        return (trainTime, train_err, train_loss)


    def standard_train_deep_ensemble(self, model, modelName, loader, dataset, opt, iterations=10):
        t1 = time.time()
        write_pred_logs = True
        num_samples = 1

        cuda = False if 'cpu' in str(self.device) else True

        _model = copy.deepcopy(model)

        self.model = FusionClassifier(
            estimator=_model,
            n_estimators=3,
            cuda=cuda,
            #save_model=False
        )

        criterion = nn.CrossEntropyLoss()
        self.model.set_criterion(criterion)

        for param_group in opt.param_groups:
            lr = param_group['lr']
            momentum = param_group['momentum']
            dampening = param_group['dampening']
            weight_decay = param_group['weight_decay']
            nesterov = param_group['nesterov']
        
        self.model.set_optimizer('SGD',lr=lr, weight_decay=weight_decay, momentum=momentum, dampening=dampening, nesterov=nesterov) 

        self.model.fit(train_loader=loader.train_loader,epochs=iterations, save_model=False, test_loader=loader.test_loader, obj_test=self, dataset=dataset, modelName=modelName, write_pred_logs=write_pred_logs,num_samples=num_samples, t_init=t1)  

        return (0, 0, 0)


    def LossFunction(self, model, X, y, y_pred, num_samples=5, CrossEntropyFunction=False):
        # LOSS_MIN_CROSSENT = 0 # minimize cross entropy loss
        # LOSS_MIN_CROSSENT_UNC  = 1 # minimize cross_entropy_loss + uncertainty
        # LOSS_MIN_CROSSENT_MAX_UNC = 2 # minimize cross_entropy_loss - uncertainty = minimize cross_entropy_loss + maximize uncertainty 
        # LOSS_MIN_UNC = 3 # minimize -uncertainty = maximize uncertainty 
        
        if CrossEntropyFunction:
            # to test the model
            return nn.CrossEntropyLoss()(y_pred, y) #CrossEntropy_Loss()(model, X, y, y_pred, num_samples)

        if self.LossInUse == LOSS_MIN_CROSSENT_UNC:
            return CrossEntropy_Uncertainty_Loss()(model, X, y, y_pred, num_samples)
        elif self.LossInUse == LOSS_MIN_CROSSENT_MAX_UNC:
            return CrossEntropy_Certainty_Loss()(model, X, y, y_pred, num_samples)
        elif self.LossInUse == LOSS_MIN_UNC:
            return Uncertainty_Loss()(model, X, y, y_pred, num_samples)
        elif self.LossInUse == LOSS_MAX_UNC:
            return Certainty_Loss()(model, X, y, y_pred, num_samples)     
        elif self.LossInUse == LOSS_MIN_BINARYCROSSENT:
            return BinaryCrossEntropy_Loss()(model, X, y, y_pred, num_samples)    
        
        #if arrieves here , it means that the loss function is the cross entropy loss
        #else LOSS_MIN_CROSSENT = 0 # minimize cross entropy loss
        return CrossEntropy_Loss()(model, X, y, y_pred, num_samples)


    def epoch(self, loader, model, opt=None, num_samples=5):
        """Standard training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.
        misclassified_ids = []


        for ct, (X,y) in enumerate(loader):
            X,y = X.to(self.device), y.to(self.device) # len of bacth size
            if self.half_prec: 
                # Runs the forward pass with autocasting.
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    y_pred = model(X)
                    #loss = nn.CrossEntropyLoss().cuda()(yp,y)

                if opt:   # backpropagation
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        loss = self.LossFunction(model, X, y, y_pred, num_samples=num_samples)

                    opt.zero_grad()
                    self.scaler.scale(loss).backward() # Scales the loss, and calls backward() . to create scaled gradients
                    self.scaler.step(opt) # Unscales gradients and calls or skips optimizer.step()
                    self.scaler.update()  # Updates the scale for next iteration

            else:
                #loss = nn.CrossEntropyLoss()(yp,y)
                y_pred = model(X)

                if opt:  # backpropagation
                    loss = self.LossFunction(model, X, y, y_pred, num_samples=num_samples)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            
            misclassified = y_pred.max(dim=1)[1] != y
            # Calculate the IDs of misclassified inputs
            batch_misclassified_ids = (ct * loader.batch_size) + torch.nonzero(misclassified).view(-1)
            misclassified_ids.extend(batch_misclassified_ids.tolist())

            total_err += misclassified.sum().item()
            if opt:  # backpropagation
                total_loss += loss.item() * X.shape[0]

            del X, y, misclassified, batch_misclassified_ids
            torch.cuda.empty_cache()


        return total_err / len(loader.dataset), total_loss / len(loader.dataset), misclassified_ids


    def epoch_interleave_batches(self, loader_wrong, loader_correct, model, opt=None, num_samples=5, weight_loss=None):
        """Standard training/evaluation epoch over the dataset"""
        #total_loss, total_err = 0.,0.
        #misclassified_ids = []
        # ct mod 2 = 0 -> wrong batch
        # ct mod 2 = 1 -> correct batch
        dataloader_iterator_wrong = iter(loader_wrong)
        dataloader_iterator_correct = iter(loader_correct)
        
        no_batches = len(loader_wrong) #+len(loader_correct) # number of batches
        #datasetSize =  len(loader_wrong.dataset) +  len(loader_correct.dataset) # number of inputs in the dataset
        #batch_size = loader_wrong.batch_size

        prints = False
        if dynamic_weights:
            # Define the loss weights as nn.Parameter objects
            weight_loss1 = weight_loss[1] #torch.tensor(0.5, requires_grad=True)
            weight_loss2 = weight_loss[2] #torch.tensor(0.5, requires_grad=True)
            weight_optimizer = weight_loss[0] #optim.Adam([{'params':[weight_loss1, weight_loss2]}],lr=0.01)

        elif Weighted_Sum:
            weight_loss1 = torch.tensor(0.1)
            weight_loss2 = torch.tensor(0.9)

        elif Adaptive_Balancing:
            weight_loss1 = 1-weight_loss
            weight_loss2 = weight_loss



        for ct in range(no_batches):

            X_wrong,y_wrong = next(dataloader_iterator_wrong)
            X_correct,y_correct = next(dataloader_iterator_correct)

            X_wrong,y_wrong = X_wrong.to(self.device), y_wrong.to(self.device) # len of bacth size
            X_correct,y_correct = X_correct.to(self.device), y_correct.to(self.device) # len of bacth size

            if self.half_prec: 
                # Runs the forward pass with autocasting.
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    y_pred_wrong = model(X_wrong)
                    y_pred_correct = model(X_correct)
                    #loss = nn.CrossEntropyLoss().cuda()(yp,y)

                if opt:   # backpropagation
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        self.LossInUse = LOSS_2nd_stage_wrong
                        loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=num_samples)

                        self.LossInUse = LOSS_2nd_stage_correct
                        loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=num_samples)

                        # Calculate the total loss with dynamic weights
                        if Weighted_Sum or Adaptive_Balancing or dynamic_weights:
                            loss = weight_loss1 * loss_wrong + weight_loss2 * loss_correct
                        else:
                            loss = loss_wrong + loss_correct

                    opt.zero_grad()
                    if dynamic_weights: weight_optimizer.zero_grad() 

                    self.scaler.scale(loss).backward() # Scales the loss, and calls backward() . to create scaled gradients
                    self.scaler.step(opt) # Unscales gradients and calls or skips optimizer.step()
                    self.scaler.update()  # Updates the scale for next iteration

                    if dynamic_weights: weight_optimizer.step()


            else:
                #loss = nn.CrossEntropyLoss()(yp,y)
                #torch.autograd.set_detect_anomaly(True)
                y_pred_wrong = model(X_wrong)
                y_pred_correct = model(X_correct)

                if opt:  # backpropagation
                    self.LossInUse = LOSS_2nd_stage_wrong
                    loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=num_samples)
                    if prints:
                        print(self.LossInUse)
                        print(loss_wrong)

                    self.LossInUse = LOSS_2nd_stage_correct 
                    loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=num_samples)
                    if prints:
                        print(self.LossInUse)
                        print(loss_correct)

                    # Calculate the total loss with dynamic weights
                    if Weighted_Sum or Adaptive_Balancing or dynamic_weights:
                        loss = weight_loss1 * loss_wrong + weight_loss2 * loss_correct
                    else:
                        loss = loss_wrong + loss_correct

                    # self.LossInUse = LOSS_MIN_CROSSENT
                    # ce_loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=num_samples)
                    # self.LossInUse = LOSS_MAX_UNC
                    # pe_loss_wrong = self.LossFunction(model, X_wrong, y_wrong, y_pred_wrong, num_samples=num_samples)



                    # self.LossInUse = LOSS_MIN_CROSSENT 
                    # ce_loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=num_samples)
                    # self.LossInUse = LOSS_MIN_UNC 
                    # pe_loss_correct = self.LossFunction(model, X_correct, y_correct, y_pred_correct, num_samples=num_samples)

                    # loss = ce_loss_wrong + pe_loss_wrong + ce_loss_correct + pe_loss_correct
                    if prints:
                        print(loss)
                        print()

                    opt.zero_grad()
                    if dynamic_weights: weight_optimizer.zero_grad() 

                    loss.backward()
                    opt.step()
                    if dynamic_weights: weight_optimizer.step()
                        
        return #total_err /datasetSize, total_loss/datasetSize, misclassified_ids


    def epoch_adversarial(self, loader, model, attack, dataset, epsilon=0.1, num_iter=20, alpha=0.01, ratio=1, opt=None, num_samples=5, **kwargs):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.

        #ratio - some have adversarial example and other no
        no_adv = int(ratio * len(loader.dataset))
        no_clean = len(loader.dataset) - no_adv

        l_adv = [True] * no_adv 
        l_clean = [False] * no_clean 
        decision = l_adv + l_clean
        random.shuffle(decision)
        
        grad = 0 #for fgsm with gradient alignment

        # for fgsm free
        delta_real = None
        minibatch_replay = 4 
        counter_minibatch_replay = 0
        X_prev, y_prev =  None, None

        # for fgsm grad alignment
        #we use λ = 0.1 for the CIFAR-10 and λ = 0.5
        if dataset == "cifar10" or dataset == "binaryCifar10" or dataset == "cifar100" :
            grad_align_cos_lambda = 0.1 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "mnist":
            grad_align_cos_lambda = 0.5 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "imageNet":
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer
        else: #svhn
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer

        ct = 0
        misclassified_ids = []
        for X,y in loader:
            if attack == "fgsm_free" and X_prev is not None and counter_minibatch_replay % minibatch_replay != 0:  
                # take new inputs only each `minibatch_replay` iterations
                X, y = X_prev, y_prev # this way we ensure the same total number of images/batches/epcohs for free fgsm

            X,y = X.to(self.device), y.to(self.device)

            if decision[ct]:
                #adversarial example
                if attack == "fgsm": #adversarial examples fgsm
                    delta = self.fgsm(model, X, y, epsilon=epsilon, num_samples=num_samples, **kwargs) 
                elif attack == "fgsm_rs": #adversarial examples fgsm with random initialization of deltas
                    delta = self.fgsm_rs(model, X, y, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                elif attack == "fgsm_free": #adversarial examples fgsm with random initialization of deltas
                    delta, delta_real = self.fgsm_free(model, X, y, delta_real, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    counter_minibatch_replay += 1

                    if counter_minibatch_replay % minibatch_replay == 0:
                        counter_minibatch_replay = 0
                        X_prev = X.clone()
                        y_prev = y.clone()

                elif attack == "pgd":  #adversarial examples pgd_linf
                    delta = self.pgd_linf(model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                elif attack == "pgd_rs":#adversarial examples pgd_linf with random initialization of deltas
                    delta = self.pgd_linf_rs(model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                elif attack == "fgsm_grad_align": #adversarial examples fgsm with gradient alignment 
                    delta, grad = self.fgsm_grad_align(model, X, y, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                else: 
                    print("wrong attack")
                    return -1

                X_input = X + delta 
            else:
                #clean data example
                X_input = X 


            if self.half_prec: 
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    y_pred = model(X_input)
            else:
                y_pred = model(X_input)

            # gradient alignment 
            if decision[ct] and  attack == "fgsm_grad_align":
                loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

                # runs only if it's a adversarial exmaple and the attack is fgsm with gradient alignment 
                reg = torch.zeros(1).cuda(self.device)[0]  # for .item() to run correctly

                grad2 = self.get_input_grad(model, X, y, epsilon, delta_init='random_uniform', backprop=True)
                grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
                grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
                grad1_norms, grad2_norms = self.l2_norm_batch(grad1), self.l2_norm_batch(grad2)
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
                reg += grad_align_cos_lambda * (1.0 - cos.mean())
                loss += reg 
            


            if opt: # to train - backpropagation
                if self.half_prec: 
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

                    opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()  
                else:
                    loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()


            misclassified = y_pred.max(dim=1)[1]  != y

            # Calculate the IDs of misclassified inputs
            batch_misclassified_ids = (ct * loader.batch_size) + torch.nonzero(misclassified).view(-1)
            misclassified_ids.extend(batch_misclassified_ids.tolist())

            total_err += misclassified.sum().item() #error
            if opt: # to train - backpropagation
                total_loss += loss.item() * X.shape[0]  #loss
            ct += 1

            del X, y, misclassified, batch_misclassified_ids
            torch.cuda.empty_cache()

        return total_err / len(loader.dataset), total_loss / len(loader.dataset), misclassified_ids
    

    def epoch_adversarial_interleave_batches(self, loader_wrong, loader_correct, model, attack, dataset, epsilon=0.1, num_iter=20, alpha=0.01, ratio=1, opt=None, num_samples=5, **kwargs):
        """Adversarial training/evaluation epoch over the dataset"""
        #total_loss, total_err = 0.,0.

        dataloader_iterator_wrong = iter(loader_wrong)
        dataloader_iterator_correct = iter(loader_correct)
        
        no_batches = len(loader_wrong) #+len(loader_correct)
        datasetSize =  len(loader_wrong.dataset) +  len(loader_correct.dataset)

        #batch_size = loader_wrong.batch_size

        #ratio - some have adversarial example and other no
        no_adv = int(ratio * datasetSize)
        no_clean = datasetSize - no_adv

        l_adv = [True] * no_adv 
        l_clean = [False] * no_clean 
        decision = l_adv + l_clean
        random.shuffle(decision)
        
        grad = 0 #for fgsm with gradient alignment

        # for fgsm free
        delta_real_wrong = None
        delta_real_correct = None
        minibatch_replay = 4 
        counter_minibatch_replay = 0
        X_wrong_prev, y_wrong_prev =  None, None
        X_correct_prev, y_correct_prev =  None, None

        # for fgsm grad alignment
        #we use λ = 0.1 for the CIFAR-10 and λ = 0.5
        if dataset == "cifar10" or dataset == "binaryCifar10" or dataset == "cifar100" :
            grad_align_cos_lambda = 0.1 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "mnist":
            grad_align_cos_lambda = 0.5 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "imageNet":
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer
        else: #svhn
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer

        ct = 0
        #misclassified_ids = []
        for ct in range(no_batches):
            X_wrong,y_wrong = next(dataloader_iterator_wrong)
            X_correct,y_correct = next(dataloader_iterator_correct)

            if attack == "fgsm_free" and X_wrong_prev is not None and counter_minibatch_replay % minibatch_replay != 0:  
                # take new inputs only each `minibatch_replay` iterations
                X_correct, y_correct = X_correct_prev, y_correct_prev # this way we ensure the same total number of images/batches/epcohs for free fgsm
                X_wrong, y_wrong = X_wrong_prev, y_wrong_prev # this way we ensure the same total number of images/batches/epcohs for free fgsm

            X_wrong,y_wrong = X_wrong.to(self.device), y_wrong.to(self.device) # len of bacth size
            X_correct,y_correct = X_correct.to(self.device), y_correct.to(self.device) # len of bacth size
        

            if decision[ct]:
                #adversarial example
                if attack == "fgsm": #adversarial examples fgsm
                    delta_wrong = self.fgsm(model, X_wrong, y_wrong, epsilon=epsilon, num_samples=num_samples, **kwargs) 
                    delta_correct = self.fgsm(model, X_correct, y_correct, epsilon=epsilon, num_samples=num_samples, **kwargs) 

                elif attack == "fgsm_rs": #adversarial examples fgsm with random initialization of deltas
                    delta_wrong = self.fgsm_rs(model, X_wrong, y_wrong, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    delta_correct = self.fgsm_rs(model, X_correct, y_correct, epsilon=epsilon, num_samples=num_samples,  **kwargs) 

                elif attack == "fgsm_free": #adversarial examples fgsm with random initialization of deltas
                    delta_wrong, delta_real_wrong = self.fgsm_free(model, X_wrong, y_wrong, delta_real_wrong, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    delta_correct, delta_real_correct = self.fgsm_free(model, X_correct, y_correct, delta_real_correct, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    counter_minibatch_replay += 1

                    if counter_minibatch_replay % minibatch_replay == 0:
                        counter_minibatch_replay = 0
                        X_correct_prev = X_correct.clone()
                        y_correct_prev = y_correct.clone()
                        X_wrong_prev = X_wrong.clone()
                        y_wrong_prev = y_wrong.clone()

                elif attack == "pgd":  #adversarial examples pgd_linf
                    delta_wrong = self.pgd_linf(model, X_wrong, y_wrong, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                    delta_correct = self.pgd_linf(model, X_correct, y_correct, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                    
                elif attack == "pgd_rs":#adversarial examples pgd_linf with random initialization of deltas
                    delta_wrong = self.pgd_linf_rs(model, X_wrong, y_wrong, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs) 
                    delta_correct = self.pgd_linf_rs(model, X_correct, y_correct, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples,  **kwargs)

                elif attack == "fgsm_grad_align": #adversarial examples fgsm with gradient alignment 
                    delta_wrong, grad_wrong = self.fgsm_grad_align(model, X_wrong, y_wrong, epsilon=epsilon, num_samples=num_samples,  **kwargs) 
                    delta_correct, grad_correct = self.fgsm_grad_align(model, X_correct, y_correct, epsilon=epsilon, num_samples=num_samples,  **kwargs) 

                else: 
                    print("wrong attack")
                    return -1

                X_correct_input = X_correct + delta_correct 
                X_wrong_input = X_wrong + delta_wrong 
            else:
                #clean data example
                X_correct_input = X_correct 
                X_wrong_input = X_wrong 


            if self.half_prec: 
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    y_pred_wrong = model(X_wrong_input)
                    y_pred_correct = model(X_correct_input)
            else:
                y_pred_wrong = model(X_wrong_input)
                y_pred_correct = model(X_correct_input)

            # gradient alignment 
            if decision[ct] and  attack == "fgsm_grad_align":
                loss_wrong = self.LossFunction(model, X_wrong_input, y_wrong, y_pred_wrong, num_samples=num_samples)
                loss_correct = self.LossFunction(model, X_correct_input, y_correct, y_pred_correct, num_samples=num_samples)
                loss = loss_wrong + loss_correct
                X=X_wrong_input+X_correct_input
                y=y_wrong+y_correct

                # runs only if it's a adversarial exmaple and the attack is fgsm with gradient alignment 
                reg = torch.zeros(1).cuda(self.device)[0]  # for .item() to run correctly

                grad2 = self.get_input_grad(model, X, y, epsilon, delta_init='random_uniform', backprop=True)
                grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
                grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
                grad1_norms, grad2_norms = self.l2_norm_batch(grad1), self.l2_norm_batch(grad2)
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
                reg += grad_align_cos_lambda * (1.0 - cos.mean())
                loss += reg 
            

            if opt: # to train - backpropagation
                if self.half_prec: 
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        self.LossInUse = LOSS_2nd_stage_wrong
                        loss_wrong = self.LossFunction(model, X_wrong_input, y_wrong, y_pred_wrong, num_samples=num_samples)

                        self.LossInUse = LOSS_2nd_stage_correct
                        loss_correct = self.LossFunction(model, X_correct_input, y_correct, y_pred_correct, num_samples=num_samples)
                                                
                        loss = loss_wrong + loss_correct

                    opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()  
                else:
                    self.LossInUse = LOSS_2nd_stage_wrong
                    loss_wrong = self.LossFunction(model, X_wrong_input, y_wrong, y_pred_wrong, num_samples=num_samples)

                    self.LossInUse = LOSS_2nd_stage_correct 
                    loss_correct = self.LossFunction(model, X_correct_input, y_correct, y_pred_correct, num_samples=num_samples)
                    
                    loss = loss_wrong + loss_correct

                    opt.zero_grad()
                    loss.backward()
                    opt.step()


    def fgsm(self, model, X, y, epsilon=0.1, num_samples=10):
        """ Construct FGSM adversarial examples on the examples X"""
        
        if self.half_prec: 
            delta = torch.zeros_like(X, requires_grad=True).cuda()
            delta.requires_grad = True
            X_input = X + delta
            y_pred = model(X_input)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

            self.scaler.scale(loss).backward()  

            del X_input, y_pred
            torch.cuda.empty_cache()

        else:
            delta = torch.zeros_like(X, requires_grad=True)
            X_input = X + delta
            y_pred = model(X_input)

            loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)
            loss.backward()

            del X_input, y_pred
            torch.cuda.empty_cache()
            
            #print(loss)
            
        return epsilon * delta.grad.detach().sign()


    def fgsm_rs(self, model, X, y, epsilon=0.1, alpha=0.375, num_samples=10):
        """ Construct FGSM adversarial examples on the examples X with unform random initialization"""
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)# .cuda()
        delta.requires_grad = True
        X_input = X + delta
        y_pred = model(X_input)
        
        #loss = F.cross_entropy(output, y) # WHERE SHOUDL WE USE THE CROSS ENTROPY or  CrossEntropyLoss???
        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

        #loss = nn.CrossEntropyLoss()(output, y)# + entropy_loss(output)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
        delta = delta.detach()
        
        return delta


    def fgsm_free(self, model, X, y, delta, epsilon=0.1, alpha=0.375, num_samples=10):
        """ Construct FGSM adversarial examples on the examples X with unform random initialization"""
        if delta is None:
            delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)# .cuda()
    
        delta.requires_grad = True
        X_input = X + delta[:X.size(0)]
        y_pred = model(X_input)

        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)

        #loss = F.cross_entropy(output, y) # WHERE SHOUDL WE USE THE CROSS ENTROPY or  CrossEntropyLoss???
        #loss = nn.CrossEntropyLoss()(output, y) #+ entropy_loss(output)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[:X.size(0)] = torch.max(torch.min(1-X, delta.data[:X.size(0)]), 0-X)
        delta_return = delta.detach()
        
        return delta_return[:X.size(0)], delta


    def fgsm_grad_align(self, model, X, y, epsilon=0.1, alpha=0.375, num_samples=10):
        """ Construct FGSM adversarial examples on the examples X with unform random initialization"""
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)# .cuda()
        delta.requires_grad = True
        X_input = X +  delta
        y_pred = model(X_input)

        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)
        #loss = F.cross_entropy(output, y) # WHERE SHOUDL WE USE THE CROSS ENTROPY or  CrossEntropyLoss???
        #loss = nn.CrossEntropyLoss()(output, y) #+ entropy_loss(output)

        grad = torch.autograd.grad(loss, delta, create_graph=True)[0]
        loss.backward()

        #grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
        
        delta = delta.detach()
        grad = grad.detach()

        
        return delta, grad


    def pgd_linf(self, model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, num_samples=10, randomize=False, CrossEntropyFunction=False):
        """ Construct FGSM adversarial examples on the examples X"""
        #print(epsilon)
        if self.half_prec: 
            if randomize:
                delta = torch.rand_like(X, requires_grad=True).cuda()
                delta.data = delta.data * 2 * epsilon - epsilon
            else:
                delta = torch.zeros_like(X, requires_grad=True).cuda()
            delta.requires_grad = True

            for t in range(num_iter):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    X_input = X +  delta
                    y_pred = model(X_input)

                    loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples, CrossEntropyFunction=CrossEntropyFunction)
                    #loss = nn.CrossEntropyLoss().cuda()(output, y) #+ entropy_loss(output)

                self.scaler.scale(loss).backward()  
                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
                delta.grad.zero_()    

                del X_input, y_pred
                torch.cuda.empty_cache()

        else:

            if randomize:
                delta = torch.rand_like(X, requires_grad=True)
                delta.data = delta.data * 2 * epsilon - epsilon
            else:
                delta = torch.zeros_like(X, requires_grad=True)
            delta.requires_grad = True
            
            for t in range(num_iter):
                X_input = X + delta
                y_pred = model(X_input)
                
                loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples, CrossEntropyFunction=CrossEntropyFunction)
                loss.backward()

                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
                delta.grad.zero_()

                #del X_input, y_pred
                #torch.cuda.empty_cache()

        return delta.detach()


    def pgd_linf_rs(self, model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, num_samples=10, randomize=False):
        """ Construct FGSM adversarial examples on the examples X"""

        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)

        for _ in range(num_iter):
            delta.requires_grad = True
            X_input = X +  delta
            y_pred = model(X_input)
                
            loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)
            #loss = nn.CrossEntropyLoss()(output, y) #+ entropy_loss(output)

            #loss =  nn.CrossEntropyLoss()(output, y)
            loss.backward()

            grad = delta.grad.detach()
            I = y_pred.max(1)[1] == y
            delta.data[I] = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)[I]
            delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]

        return delta.detach()


    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


    def pgd_l2(self, model, X, y, epsilon, alpha, num_iter):
        delta = torch.zeros_like(X, requires_grad=True)

        for t in range(num_iter):
            output = model(X + delta)
            loss = nn.CrossEntropyLoss()(output, y) #+ entropy_loss(output)
            #loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data += alpha*delta.grad.detach() / self.norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.data *= epsilon / self.norms(delta.detach()).clamp(min=epsilon)
            delta.grad.zero_()
            
        return delta.detach()


    def get_input_grad(self, model, X, y, eps, delta_init='none', num_samples=10, backprop=False):
        if delta_init == 'none':
            delta = torch.zeros_like(X, requires_grad=True)
        elif delta_init == 'random_uniform':
            delta = torch.zeros_like(X).uniform_(-eps, eps)
            delta.requires_grad = True
            #delta = self.get_uniform_delta(X.shape, eps, requires_grad=True)
        elif delta_init == 'random_corner':
            delta = torch.zeros_like(X).uniform_(-eps, eps)
            delta.requires_grad = True
            #delta = self.get_uniform_delta(X.shape, eps, requires_grad=True)
            delta = eps * torch.sign(delta)
        else:
            raise ValueError('wrong delta init')

        X_input = X +  delta
        y_pred = model(X_input)
                
        loss = self.LossFunction(model, X_input, y, y_pred, num_samples=num_samples)
        grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]

        if not backprop:
            grad, delta = grad.detach(), delta.detach()
        return grad


    def get_uniform_delta(self, shape, eps, requires_grad=True):
        delta = torch.zeros(shape) #.cuda()
        delta.uniform_(-eps, eps)
        delta.requires_grad = requires_grad
        return delta


    def l2_norm_batch(self, v):
        norms = (v ** 2).sum([1, 2, 3]) ** 0.5
        return norms


    def testModel_logs(self, dataset_name, models_name, iteration, alg, ratio ,epsilon, numIt, alpha, ratioADV, trainTime, write_pred_logs=False, num_samples=5, calibration=False):

        self.model.eval() # evaluate the model
        
        list_to_write = []

        #test the accuracy of the model
        t3 = time.time()
        test_err, test_loss, test_entropy, test_MI = self.test_epoch(self.loader.test_loader, self.model, num_samples=num_samples, models_name=models_name, write_pred_logs=write_pred_logs, iteration=iteration, calibration=calibration)
        #test_err, test_loss = self.epoch(self.loader.test_loader, self.model)
        testTime = time.time() - t3

        _ratio = 0.0 if isinstance(ratio, list) else ratio # only used for standard training 
            

        str_write = str(iteration) + "," + alg + "," + str(_ratio) + "," + str(epsilon) + "," + str(numIt) + "," + str(alpha) + "," + str(ratioADV) +  \
                            ",std,0,0,0," + str(test_err) + "," + str(test_loss) + "," + str(testTime) + "," + str(trainTime) + "," + \
                            str(test_entropy) + "," + str(test_MI) + "\n"
        
        list_to_write.append(str_write)


        #test the adversarial accuracy of the model
        if dataset_name == "cifar10":
            #eps_test_list =  [2, 4, 8, 12, 16]
            eps_test_list =  [4]
        elif dataset_name == "binaryCifar10":
            eps_test_list =  [4]
        elif dataset_name == "cifar100":
            eps_test_list =  [4]            
        elif dataset_name ==   "mnist": #mnist
            #eps_test_list =  [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            eps_test_list =  [0.3]
        elif dataset_name == "imageNet":
            #eps_test_list =  [2, 4, 8]
            eps_test_list =  [4]
        else: #if dataset_name ==   "svhn":
            #eps_test_list =  [2, 4, 8, 12]
            eps_test_list =  [4]


        num_iterTest_list = [20]
        alpha_test_list = [0.01]
        #adv_err, adv_loss, uncertainty = 0., 0., 0.
        # testing the final model using PGD
        for i, eps_test in enumerate(eps_test_list):
            for num_iterTest in num_iterTest_list:
                for alpha_test in alpha_test_list:
                    if dataset_name == "mnist": #mnist
                        _eps_test = eps_test
                    else: #svhn or cifar10 or imageNet
                        _eps_test = eps_test/255.0

                    t4 = time.time()
                    #adv_err, adv_loss = self.epoch_adversarial(self.loader.test_loader, self.model, "pgd", "", _eps_test, num_iterTest, alpha_test, 1, newLoss=False)
                    adv_err, adv_loss, adv_entropy, adv_MI = self.test_epoch_adversarial(self.loader.test_loader, self.model, epsilon=_eps_test, num_iter=num_iterTest, alpha=alpha_test, num_samples=num_samples, models_name=models_name, write_pred_logs=write_pred_logs, iteration=iteration, calibration=calibration)
                    #if _eps_test == epsilon:
                    #    adv_err, adv_loss, adv_entropy, adv_MI = _adv_err, _adv_loss, _adv_entropy, _adv_MI

                    advTestTime = time.time() - t4
                    str_write = str(iteration) + "," + alg + "," + str(_ratio) + "," + str(epsilon) + "," + str(numIt) + "," + str(alpha) + "," + str(ratioADV) + \
                                    ",pgd," + str(_eps_test) + "," + str(num_iterTest) + "," + str(alpha_test) + "," + \
                                    str(adv_err) + "," + str(adv_loss) + "," + str(advTestTime) + "," + str(trainTime) + "," + str(adv_entropy) + "," + str(adv_MI) + "\n"

                    list_to_write.append(str_write)


        #write logs
        if isinstance(ratio, list):
            for _ratio_2_test in ratio:
                if _ratio_2_test < 1.0:
                    filename = "./logs1/logs_" + models_name + "_ratio" + str(_ratio_2_test) + ".txt"
                else:
                    filename = "./logs/logs_" + models_name + ".txt"

                if iteration == 1:
                    f = open(filename, "w")
                    f.write("it,alg,ratio,epsilon,numIt,alpha,ratioAdv,algTest,epsilonTest,numItTest,alphaTest,Error,Loss,testingTime,trainTime,Entropy,MI\n")
                else:
                    f = open(filename, "a")
        

                for str_write in list_to_write:
                    f.write(str_write)

        else:

            filename = "./logs/logs_" + models_name + ".txt"
            if iteration == 1:
                f = open(filename, "w")
                f.write("it,alg,ratio,epsilon,numIt,alpha,ratioAdv,algTest,epsilonTest,numItTest,alphaTest,Error,Loss,testingTime,trainTime,Entropy,MI\n")
            else:
                f = open(filename, "a")


            for str_write in list_to_write:
                f.write(str_write)

        f.close()
        self.model.train() # go back to train mode


        return test_err, test_loss, test_entropy, test_MI, adv_err, adv_loss, adv_entropy, adv_MI


    def MCdropout(self, model, X, y, num_samples=10, calibration=False):
    #def MCdropout(self, model, X, y, num_samples=10, adversarial=False, epsilon=0.1, num_iter=20, alpha=0.01, **kwargs):
        #MC DROPOUT - UNCERTAINTY
        probs = None
        mean_entropy = None
        num_clases = None

        for n in range(num_samples):
            #need to enable dropout
            model.eval() # evaluate the model

            if self.deep_ensemble: # ensemble as already the softmax applied
                softmax_output = model(X)
                
            else: 
                model.enable_dropout()
                softmax_output = F.softmax(model(X), dim=1)
            
                if calibration and self.isCalibrated:
                    softmax_output = self.predict_proba(softmax_output)


            if num_clases is None: num_clases = len(softmax_output[0])
            if probs is None: probs = torch.zeros_like(softmax_output)
            probs = probs + softmax_output
            #probs = (probs+F.softmax(output, dim=1)) if probs is not None else F.softmax(output, dim=1) 

            _entropy = entropy(softmax_output)
            if mean_entropy is None: mean_entropy = torch.zeros_like(_entropy)
            mean_entropy = mean_entropy + _entropy
            #mean_entropy = (mean_entropy+entropy(output)) if mean_entropy is not None else entropy(output)

            del softmax_output
            torch.cuda.empty_cache()
            #print(torch.cuda.memory_summary())
            #print("\n")


        probs /= float(num_samples) 
        mask = probs == 0  # Create a mask of zero values
        probs[mask] = 10e-20  # Replace zero values with small_value

        log_probs = torch.log(probs)
        entropy_vals = -torch.sum(probs * log_probs, dim=-1)# predictive entropy PE=H

        mean_entropy /= float(num_samples)
        mutual_information_vals = entropy_vals - mean_entropy
        #print(mean_entropy)
        #print(mutual_information_vals)

        #max entropy - uniform distrbution
        #https://math.stackexchange.com/questions/1156404/entropy-of-a-uniform-distribution
        entropy_max = np.log(num_clases)
        normalized_entropy = entropy_vals / entropy_max # normalized predictive entropy
        normalized_mutual_information = mutual_information_vals / entropy_max # normalized mutual information
        #mutual_information is maximum when the second term is 0 and the first is maxinum entropy (uniform distirbution)

        return normalized_entropy, normalized_mutual_information


    def test_epoch(self, loader, model, num_samples=10, models_name=None, write_pred_logs=False, iteration=-1, calibration=False):
        """Standard training/evaluation epoch over the dataset"""
        total_loss, total_err, total_entropy, total_mutual_information, counter_inputs = 0.,0.,0.,0.,0.
        write_scores = True if 'binary' in models_name else False
        lossfunc = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for X,y in loader:
                _data = []

                X,y = X.to(self.device), y.to(self.device) # len of bacth size
                counter_inputs += len(y)

                y_pred = model(X)
                # when using ensembles the output is probablities of each class

                if self.deup:
                    if write_scores: probs = F.softmax(y_pred, dim=1)
                    total_err += (y_pred.max(dim=1)[1] != y).sum().item()

                    normalized_entropy =  self.deup_model.predict(X).t()[0]
                    normalized_mutual_information = lossfunc(y_pred, y)
                    # normalized_mutual_information is not use -  we just print the same

                elif self.deep_ensemble:
                    if write_scores: probs = y_pred
                    total_err += (y_pred.max(dim=1)[1] != y).sum().item()
                    normalized_entropy, normalized_mutual_information = self.MCdropout(model, X, y, num_samples=1, calibration=calibration) #, adversarial=False)


                elif calibration and self.isCalibrated:
                    probs = self.predict_proba(F.softmax(y_pred, dim=1))
                    total_err += (probs.max(dim=1)[1] != y).sum().item()
                    normalized_entropy, normalized_mutual_information = self.MCdropout(model, X, y, num_samples=num_samples, calibration=calibration) #, adversarial=False)

                else:
                    if write_scores: probs = F.softmax(y_pred, dim=1)
                    total_err += (y_pred.max(dim=1)[1] != y).sum().item()
                    normalized_entropy, normalized_mutual_information = self.MCdropout(model, X, y, num_samples=num_samples, calibration=calibration) #, adversarial=False)
                    #normalized_entropy, normalized_mutual_information = self.MCdropout(model, X, y, num_samples=num_samples) #, adversarial=False)
                    #this prev line determines the uncertainty of a batch of inputs (but returns the uncertainties for all inputs/predictions in a batch)
                    # so later we need to average them
                    # but we deterime the cumulative uncertainty give N batches, so then we need to sum all entropies and average then                
                    
                total_entropy += (normalized_entropy.sum().item())
                total_mutual_information += (normalized_mutual_information.sum().item())
            

                if write_pred_logs and models_name is not None:
                    _entropy_normalized = normalized_entropy.tolist()
                    _normalized_mutual_information = normalized_mutual_information.tolist()
                    _predictions = y_pred.max(dim=1)[1].tolist()
                    _y = y.tolist()
                    if write_scores: _probs = probs.tolist()
                    for i in range(len(y)):
                        if write_scores: 
                            _probs_str = ""
                            for probs_ in _probs[i]:
                                _probs_str += str(probs_) + ':'

                            _data.append((iteration, _y[i], _predictions[i], _y[i]==_predictions[i], total_err/counter_inputs, total_loss/counter_inputs, \
                                        _entropy_normalized[i], total_entropy/counter_inputs,  _normalized_mutual_information[i], total_mutual_information/counter_inputs, \
                                        _probs_str[:-1]))
                        else:
                            _data.append((iteration, _y[i], _predictions[i], _y[i]==_predictions[i], total_err/counter_inputs, total_loss/counter_inputs, \
                                        _entropy_normalized[i], total_entropy/counter_inputs,  _normalized_mutual_information[i], total_mutual_information/counter_inputs))

                    if write_pred_logs==2:
                        self.write_logs_prediction(_data, 'STD1' + models_name)
                    else:
                        self.write_logs_prediction(_data, 'STD' + models_name)

                del X, y
                torch.cuda.empty_cache()

        return total_err/len(loader.dataset), total_loss/len(loader.dataset), total_entropy/len(loader.dataset), total_mutual_information/len(loader.dataset)


    def test_epoch_adversarial(self, loader, model, epsilon=0.1, num_iter=20, alpha=0.01, num_samples=10, models_name=None, write_pred_logs=False, iteration=-1, calibration=False,  **kwargs):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss, total_err, total_entropy, total_mutual_information, counter_inputs = 0.,0.,0.,0.,0.
        write_scores = True if 'binary' in models_name else False
        lossfunc = nn.CrossEntropyLoss(reduction='none')
        
        #with torch.no_grad():
        for X,y in loader:
            _data = []
            X, y = X.to(self.device), y.to(self.device)
            counter_inputs += len(y)

            #LOSS and ACCURACY
            #adversarial examples pgd_linf
            delta = self.pgd_linf(model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha, num_samples=num_samples, CrossEntropyFunction=True, **kwargs) 
            X_input = X + delta
            y_pred = model(X_input)

            if self.deup:
                if write_scores: probs = F.softmax(y_pred, dim=1)
                total_err += (y_pred.max(dim=1)[1] != y).sum().item()
                
                normalized_entropy =  self.deup_model.predict(X).t()[0]
                normalized_mutual_information = lossfunc(y_pred, y)
                
            elif self.deep_ensemble:
                if write_scores: probs = y_pred
                total_err += (y_pred.max(dim=1)[1] != y).sum().item()
                normalized_entropy, normalized_mutual_information = self.MCdropout(model, X, y, num_samples=1, calibration=calibration) #, adversarial=False)

            elif calibration and self.isCalibrated:
                probs = self.predict_proba(F.softmax(y_pred, dim=1))
                total_err += (probs.max(dim=1)[1] != y).sum().item()
                normalized_entropy, normalized_mutual_information = self.MCdropout(model, X_input, y, num_samples=num_samples, calibration=calibration)

            else:
                if write_scores: probs = F.softmax(y_pred, dim=1)
                total_err += (y_pred.max(dim=1)[1] != y).sum().item()

                normalized_entropy, normalized_mutual_information = self.MCdropout(model, X_input, y, num_samples=num_samples, calibration=calibration)
            #normalized_entropy, normalized_mutual_information = self.MCdropout(model, X_input, y, num_samples=num_samples) ,\
            #                                                                adversarial=True, epsilon=epsilon, num_iter=num_iter, alpha=alpha, **kwargs)
            
            total_entropy += (normalized_entropy.sum().item())
            total_mutual_information += (normalized_mutual_information.sum().item())


            if write_pred_logs and models_name is not None:
                _entropy_normalized = normalized_entropy.tolist()
                _normalized_mutual_information = normalized_mutual_information.tolist()
                _predictions = y_pred.max(dim=1)[1].tolist()
                _y = y.tolist()
                if write_scores: _probs = probs.tolist()
                for i in range(len(y)):
                    if write_scores: 
                        _probs_str = ""
                        for probs_ in _probs[i]:
                            _probs_str += str(probs_) + ':'

                        _data.append((iteration, _y[i], _predictions[i], _y[i]==_predictions[i], total_err/counter_inputs, total_loss/counter_inputs, \
                                    _entropy_normalized[i], total_entropy/counter_inputs,  _normalized_mutual_information[i], total_mutual_information/counter_inputs, \
                                     _probs_str[:-1]))
                
                    else:
                        _data.append((iteration, _y[i], _predictions[i], _y[i]==_predictions[i], total_err/counter_inputs, total_loss/counter_inputs, \
                                    _entropy_normalized[i], total_entropy/counter_inputs,  _normalized_mutual_information[i], total_mutual_information/counter_inputs))
                
                if write_pred_logs==2:
                    self.write_logs_prediction(_data, 'ADV1' + models_name)
                else:
                    self.write_logs_prediction(_data, 'ADV' + models_name)
                    
            del X, y, delta, X_input, normalized_mutual_information, normalized_entropy
            torch.cuda.empty_cache()

        return total_err/len(loader.dataset), total_loss/len(loader.dataset), total_entropy/len(loader.dataset), total_mutual_information/len(loader.dataset)


    def write_logs_prediction(self, data, models_name):

        filename = "./logs/predictions_" + models_name + ".txt"
        f = open(filename, "a")
        for pred_data in data:
            str_write = ''
            for dd in pred_data:
                str_write += str(dd) + ','
            f.write(str_write[:-1] + '\n')

        f.close()


    def calibrate(self, loader_data, model, num_samples=5):
        """calibrate probablities training/evaluation after training"""
        if not PlattScaling_Flag  and not IsotonicRegression_Flag and not TemperatureScaling_Flag:
            return
        
        #calidation set for calibration
        size = int(len(loader_data)*0.1)
        ids = random.sample(range(int(len(loader_data))), size)
        subset = Subset(loader_data, ids)
        sub_loader = DataLoader(subset, batch_size=32 , shuffle=True) 

        #y_pred_list, y_list = torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        y_pred_list, y_list = np.array([]), np.array([])

        with torch.no_grad():
            for X,y in sub_loader:
                scores = None
                X,y = X.to(self.device), y.to(self.device) # len of bacth size
                for n in range(num_samples):
                    #need to enable dropout
                    model.eval() # evaluate the model
                    model.enable_dropout()

                    if self.half_prec: 
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            softmax_output = F.softmax(model(X), dim=1)
                    else:
                        softmax_output = F.softmax(model(X), dim=1)
                    
                    if scores is None: scores = torch.zeros_like(softmax_output)
                    scores += softmax_output

                scores /= float(num_samples) 

                y_pred_list = np.concatenate((y_pred_list, scores.cpu().numpy()), axis=0) if len(y_pred_list)>0 else scores.cpu().numpy()
                y_list = np.concatenate((y_list, y.cpu().numpy()), axis=0) if len(y_list)>0 else y.cpu().numpy()


        # method (str, default: "mle") – 
        # ‘mle’: Maximum likelihood estimate without uncertainty using a convex optimizer. 
        # ‘momentum’: MLE estimate using Momentum optimizer for non-convex optimization. 
        # ‘variational’: Variational Inference with uncertainty. 
        # ‘mcmc’: Markov-Chain Monte-Carlo sampling with uncertainty.
        method = 'mle'
        #method = 'momentum'
        #method = 'variational'
        #method = 'mcmc'

        if TemperatureScaling_Flag:
            self.calibration = TemperatureScaling(detection=False, use_cuda=self.device, method=method)
            self.calibration.fit(y_pred_list, y_list)

        elif BetaCalibration_Flag:
            self.calibration = BetaCalibration(detection=False, use_cuda=self.device, method=method)
            self.calibration.fit(y_pred_list, y_list)

        elif IsotonicRegression_Flag:
            self.calibration = IsotonicRegression(detection=False)
            self.calibration.fit(y_pred_list, y_list)
        
        else:
            self.calibration = LogisticCalibration(detection=False, use_cuda=self.device, method=method)
            self.calibration.fit(y_pred_list, y_list)

        self.isCalibrated = True
        return 
        

    def predict_proba(self, scores):

        if not self.ToCalibrate and not self.isCalibrated:
            softmax_output = F.softmax(scores, dim=1)

        else:
            if PlattScaling_Flag or IsotonicRegression_Flag or TemperatureScaling_Flag or BetaCalibration_Flag:
                with torch.no_grad():
                    out = self.calibration.transform(scores.cpu().numpy())
                    softmax_output = torch.tensor(out, device=self.device)
                    
                    if len(softmax_output.shape)==1:
                        # binary case - compute the other probablity 
                        softmax_output = torch.cat((torch.zeros_like(softmax_output.unsqueeze(-1)), softmax_output.unsqueeze(-1)), dim=1)
                        softmax_output[:,0] = 1.0-softmax_output[:,1]
            else:
                softmax_output = F.softmax(scores, dim=1)
        return softmax_output


class model(trainModel):
    def __init__(self, dataset, dataset_name, device, devices_id, lr=0.1, momentum=0, lr_adv=0.1, momentum_adv=0, batch_adv=100, half_prec=False, variants=None):
        self.loader = dataset
        self.dataset_name = dataset_name 
        self.lr = lr 
        self.momentum = momentum 

        self.lr_adv = lr_adv 
        self.momentum_adv = momentum_adv 
        self.batch_adv = batch_adv

        self.devices_id = devices_id

        super().__init__(device, half_prec=half_prec, variants=variants) #initialize the datasets
    
        self.model = self.resetModel() #initialize model

        dampening=0
        weight_decay=0 if self.dataset_name != "imageNet" else 0.0001
        nesterov=False

        self.opt = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

        # hyper-parameters clean data: learning rate, momentum, dampening, weight_decay, nesterov
        # hyper-parameters FGSM: epsilon, ratio
        # hyper-parameters PGD: epsilon, num_iter, alpha, ratio

        #if self.dataset_name == "svhn" or self.dataset_name == "imageNet":
        #    # we only parallelize these 2 datasets
        #    self.model = self.parallelizeModel(self.model)


    def run(self, modelName, iterations=10, stop="epochs"):    
        
        self.model.train()
        return self._train_epochs(modelName, iterations=iterations)


    def _train_epochs(self, modelName, iterations=10):
        ''' uploads the models or trains it from scratch'''

        path = "./models/" + modelName
        if "binaryCifar10" in modelName:
            _dataset = "binaryCifar10"
        elif "cifar10" in modelName:
            _dataset = "cifar10"
        elif "cifar100" in modelName:
            _dataset = "cifar100"
        elif "mnist" in modelName:
            _dataset = "mnist"
        elif "imageNet" in modelName:
            _dataset = "imageNet"
        else: #svhn
            _dataset = "svhn"

        aux = modelName.split("_")

        if  "robust" in aux[1] and aux[2] == "FGSM":
            _ratio =  float(aux[4][5:])
            ratio_adv = float(aux[5][8:])

            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            else: #svhn or cifar10 or cifar100 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            print('train with clean data and then with fgsm (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])))

            trainTime, train_err, train_loss = self.standard_fgsm_train(self.model, modelName, self.loader, _dataset, self.opt, ratio=_ratio, 
                        eps_train=_eps_train, iterations=iterations, ratio_adv=ratio_adv, lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)
                

        elif "robust" in aux[1] and aux[2] == "PGD":
            _ratio = float(aux[6][5:])
            ratio_adv = float(aux[7][8:])

            _num_iterTrain = int(aux[4][5:])
            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            else: #svhn or cifar10 ro cifar100 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            _alpha_train =float( aux[5][5:])

            print('train with clean data and then with pgd (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])) + ', no_ite=' + str(_num_iterTrain) + ', alpha=' + str(_alpha_train))
            
            trainTime, train_err, train_loss = self.standard_pgd_train(self.model, modelName, self.loader, _dataset, self.opt, ratio=_ratio, 
                        num_iterTrain=_num_iterTrain, eps_train=_eps_train, alpha_train=_alpha_train, iterations=iterations, ratio_adv=ratio_adv,
                        lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)


        else:
            #train all model with checkpointing and early stopping (only used fo standard training to differentiate the full standard training and pre-training) 
            trainTime, train_err, train_loss = self.standard_train(self.model, modelName, self.loader, _dataset, self.opt, iterations=iterations)

        return trainTime, train_err, train_loss
        

    def testModel(self):
        '''test the model at the end to evaluate standard accuracy'''
        # switch to evaluate mode
        self.model.eval()

        t3 = time.time()
        test_err, test_loss,_ = self.epoch(self.loader.test_loader, self.model)
        testTime = time.time() - t3

        return (testTime, test_err, test_loss)


    def testModel_adversarial_pgd_all(self):
        '''test the model with adversarial examples at the end to evaluate adversarial accuracy
            using different sets of hyperparameter'''
        # switch to evaluate mode
        self.model.eval()

        eps_test_list =  [0.01, 0.05, 0.1, 0.2]
        num_iterTest_list = [10, 20]
        alpha_test_list = [0.01]

        advTestTime_pgd = []
        adv_loss_pgd = []
        adv_err_pgd = []

        # testing the final model using PGS
        for eps_test in eps_test_list:
            for num_iterTest in num_iterTest_list:
                for alpha_test in alpha_test_list:
                    print("PGD test epsilon test= " + str(eps_test) + "  num_iterTest = " + str(num_iterTest)  + "  alpha_test = " + str(alpha_test))

                    t4 = time.time()
                    adv_err, adv_loss,_ = self.epoch_adversarial(self.loader.test_loader, self.model, "pgd", "", eps_test, num_iterTest, alpha_test, 1)
                    advTestTime = time.time() - t4

                    advTestTime_pgd.append(advTestTime)
                    adv_loss_pgd.append(adv_loss)
                    adv_err_pgd.append(adv_err)

        return (advTestTime_pgd, adv_err_pgd, adv_loss_pgd)


    def testModel_adversarial_pgd(self, eps_test, num_iterTest, alpha_test):
        '''test the model with adversarial examples at the end to evaluate adversarial accuracy
            using fixing hyperparameter''' 
        # switch to evaluate mode
        self.model.eval()

        #print("PGD test epsilon test= " + str(eps_test) + "  num_iterTest = " + str(num_iterTest)  + "  alpha_test = " + str(alpha_test)
        t4 = time.time()

        adv_err, adv_loss, _data = self.test_epoch_adversarial(self.loader.test_loader, self.model, eps_test, num_iterTest, alpha_test)
        advTestTime = time.time() - t4

        return advTestTime, adv_err, adv_loss, _data


    def saveResults(self, pathFile, data):
        self.writeResult(pathFile, data)


    def parallelizeModel(self, model):
        if self.devices_id is not None: # parallelize the job 
            model = nn.DataParallel(model, device_ids = self.devices_id)
        return model.to(self.device)
            

    def resetModel(self, ):
        dropout_rate=0.5

        if self.dataset_name == "imageNet":
            num_classes=1000
            depth = 50
            dropout_rate=0.3


            # Create an instance of the ResNet-50 model with dropout
            #model = ResNet50WithDropout(num_classes, dropout_prob=dropout_rate)
            #model = CustomResNet50(models_.resnet.Bottleneck, [3, 4, 6, 3])

            if depth == 18:
                model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, depth, dropout_rate)
            elif depth == 34:
                model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, depth, dropout_rate)
            elif depth == 50:
                model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, depth, dropout_rate)
            elif depth == 101:
                model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, depth, dropout_rate)
            else:
                #model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, 18, dropout_rate)

                depth = 28
                widen_factor=10
                dropout_rate=0.3
            
                model = Wide_ResNet(depth, widen_factor, dropout_rate, num_classes)


        elif self.dataset_name == "svhn":
            num_classes=10
            depth = 18

            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, depth, dropout_rate)

        elif self.dataset_name == "cifar10":
            num_classes=10
            depth = 18

            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, depth, dropout_rate)
#            return PreActResNet(PreActBlock, [2,2,2,2], num_classes, dropout_rate)

        elif self.dataset_name == "binaryCifar10":
            num_classes=2
            depth = 18

            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, depth, dropout_rate)
#            return PreActResNet(PreActBlock, [2,2,2,2], num_classes, dropout_rate)

        elif self.dataset_name == "cifar100":
            num_classes=100
            depth = 28
            widen_factor=10
            dropout_rate=0.3
        
            model = Wide_ResNet(depth, widen_factor, dropout_rate, num_classes)

        else:
            print("CHECK the models firts")
        


        #model = resnet18(num_classes=num_classes)
        return model.to(self.device)
                    
        return model.to(self.device)


def main(modelName, dataset_name, iterations=10, stop="epochs", device=None, devices_id=None, lr=0.1, momentum=0, batch=100, lr_adv=0.1, momentum_adv=0, batch_adv=100, half_prec=False, variants=False):
    
    dataset_loader = dataset(dataset_name=dataset_name, batch_size = batch, batch_size_adv = batch_adv)
    model_cnn = model(dataset_loader, dataset_name, device, devices_id, lr, momentum, lr_adv, momentum_adv, batch_adv, half_prec=half_prec, variants=variants)
    trainTime, train_err, train_loss = model_cnn.run(modelName, iterations=iterations, stop=stop) # train

    return


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('True', 'yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--type', type=str, help='type of traing', default="std", choices=['std', 'robust'])
    parser.add_argument('--alg', type=str, help='path to store the model', default="pgd", choices=['pgd', 'fgsm', 'pgd_rs', 'fgsm_rs', 'fgsm_free', 'fgsm_grad_align'])

    parser.add_argument('--ratio_adv', type=float, help='percentage of data to train adversarial just for std+adv', default=1.0)
    parser.add_argument('--ratio', type=float, help='percentage of data to train adversarial', default=1.0)
    parser.add_argument('--epsilon', type=float, help='epsilon bound', default=0.1)

    parser.add_argument('--num_iter', type=int, help='number of iterations for pgd ', default=10)
    parser.add_argument('--alpha', type=float, help='alpha', default=0.01)
    
    parser.add_argument('--dataset', type=str, help='dataset', default="mnist", choices=['mnist', 'cifar10', 'binaryCifar10', 'cifar100', 'imageNet', 'svhn'])

    parser.add_argument('--stop', type=str, help='stop condition', default="epochs", choices=['epochs', 'time'])
    parser.add_argument('--stop_val', type=int, help='number of epochs or training time', default=10)

    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--momentum', type=float, help='momentum', default=0.0)
    parser.add_argument('--batch', type=int, help='batch', default=100)

    parser.add_argument('--lr_adv', type=float, help='learning rate adv training', default=0.01)
    parser.add_argument('--momentum_adv', type=float, help='momentum adv training', default=0.0)
    parser.add_argument('--batch_adv', type=int, help='batch adv training', default=100)

    parser.add_argument('--workers', type=str, help='GPU workers', default="0")
    parser.add_argument('--half_prec', type=str2bool, help='half precision', default=False)

    parser.add_argument('--variants', type=str, help='calibration, deup, ensemble', default=None)

    args = parser.parse_args()

    dataset_name = args.dataset
    if args.type == "std":
        args.ratio = 0.0
        args.ratio_adv = 0.0

    modelName = "model" + dataset_name

    if args.type == "std":
        modelName += "_std_train"

    else: #robust
        if "fgsm" in args.alg:
            modelName += "_robust_FGSM"
            if "rs" in args.alg:
                modelName += "rs"
            elif 'free' in args.alg:
                modelName += "free"
            elif 'grad_align' in args.alg:
                modelName += "gradAlign"

            modelName += "_eps" + str(args.epsilon) + "_ratio" + str(args.ratio) + "_ratioadv" + str(args.ratio_adv)

        else: #pgd
            modelName += "_robust_PGD"
            if "rs" in args.alg:
                modelName += "rs"
            elif 'free' in args.alg:
                modelName += "free"
            elif 'grad_align' in args.alg:
                modelName += "gradAlign"

            modelName += "_eps" + str(args.epsilon) + "_numIt" + str(args.num_iter) + "_alpha" + str(args.alpha) + "_ratio" + str(args.ratio) + "_ratioadv" + str(args.ratio_adv)

    modelName += "_lr" + str(args.lr) + "_momentum" + str(args.momentum) + "_batch" + str(args.batch)
    modelName += "_lrAdv" + str(args.lr_adv) + "_momentumAdv" + str(args.momentum_adv) + "_batchAdv" + str(args.batch_adv)

    #device = torch.device("cuda:" + str(args.workers) if torch.cuda.is_available() else "cpu")
    
    devices_id = [0]
    if args.dataset == 'imageNet' or args.dataset == 'svhn':
        aux_workers = args.workers.split(",")
        if len(aux_workers) > 1:
            devices_id = []
            for aux_w in  aux_workers:
                devices_id.append(int(aux_w))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    toRun= True if not os.path.isfile("./logs/logs_" + modelName + ".txt") else False 
    if toRun: # config doesnt exist in the VM
        if os.path.isfile("configs_run.txt"):
            with open("configs_run.txt", "r") as fl:
                for line in fl:
                    if "logs_" + modelName + ".txt" == line[:-1]: 
                        toRun= False
                        print("in file")
                        break


    if toRun:
        main(modelName, dataset_name, args.stop_val, args.stop, device, devices_id, args.lr, args.momentum, args.batch,  args.lr_adv, args.momentum_adv, 
                            args.batch_adv, args.half_prec, args.variants)  

        if os.path.isfile("configs_run.txt"):
            with open("configs_run.txt", "a") as fl:
                fl.write("logs_" + modelName + ".txt\n")

    else:
        print("config exists (no checkpointing is needed)")
