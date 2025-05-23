#### Metrics for use in various models

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

class SensitivityAndSpecificity(nn.Module):
    def __init__(self, threshold: float = 0.5):
        '''
        This is a metric that will calculate the sensitivity of a model.
        
        Args:
            threshold (float): The threshold that will be used to determine if a prediction is positive or negative.
        '''
        super(SensitivityAndSpecificity, self).__init__()
        self.threshold = threshold
        
    def forward(self, y_hat, y):
        '''
        This function will calculate the sensitivity of a model.
        
        Args:
            y_hat (torch.Tensor): The predictions of the model.
            y (torch.Tensor): The true labels of the model.
            
        Returns:
            torch.Tensor: The sensitivity of the model.
        '''
        y_hat = (y_hat > self.threshold)

        #Calculate TP, FP, TN, FN
        true_positives = torch.sum((y_hat == 1) & (y == 1))
        false_negatives = torch.sum((y_hat == 0) & (y == 1))
        true_negatives = torch.sum((y_hat == 0) & (y == 0))
        false_positives = torch.sum((y_hat == 1) & (y == 0))

        # Apply Haldane-Anscombe correction
        true_positives = true_positives.type(torch.float32) + 0.5
        false_negatives = false_negatives.type(torch.float32) + 0.5
        true_negatives = true_negatives.type(torch.float32) + 0.5
        false_positives = false_positives.type(torch.float32) + 0.5

        #Calculate Sensitivity and Specificity
        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)

        return sensitivity, specificity
    