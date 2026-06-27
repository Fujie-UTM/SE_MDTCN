# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 21:54:29 2025

@author: Fujie
"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:

    random_state=99
    num_sub=15   #被试数目
    sub_list=np.arange(num_sub) #被试编号列表
    num_chs = 64
    sfreq = 256
    
    NF_freq_list= [60]
    BG_name_list=['Delta','Theta','Alpha','Beta','Gamma', 'Full']
    BG_list=[[0.5, 4], [4, 8], [8, 16], [16, 30], [30, 125], [0.5, 125]]
    
    timewin_name_list=['-500_0','0_2000','-500_2000']
    timewin_list=[[-0.5, 0], [0, 2], [-0.5, 2]]
    
    prob_critera_ic=0.9 # Automatic rejection based on ICLabel classification probabilities
    artifact_type=['muscle artifact','eye blink','heart beat','line noise',
                   'channel noise'] # Types of artifacts to remove (updated on 2024-12-03)
    
    path_list = ['0_raw_data','1_epoch_data','2_CARrefered', '3_ica_train','4_time_extract',
                 '5_filtered', '6_ica_apply', '7_h5py', '8_EMD']
    
    data_folder_list=['Training set', 'Validation set', 'Test set']
    event_id={'Hello': 0, 'Help me': 1, 'Stop': 2, 'Thank you': 3, 'Yes': 4}
    
    electrode_dict = {
    1:  "Fp1", 2: "Fp2",  3: "F7",   4: "F3",    5: "Fz",   6: "F4", 
    7: "F8",   8: "FC5",  9: "FC1",  10: "FC2",  11: "FC6", 12: "T7",  
    13: "C3",  14: "Cz",  15: "C4",  16: "T8",   17: "TP9", 18: "CP5", 
    19: "CP1", 20: "CP2", 21: "CP6", 22: "TP10", 23: "P7",  24: "P3", 
    25: "Pz",  26: "P4",  27: "P8",  28: "PO9",  29: "O1",  30: "Oz",  
    31: "O2",  32: "PO10",33: "AF7", 34: "AF3",  35: "AF4", 36: "AF8",
    37: "F5",  38: "F1",  39: "F2",  40: "F6",   41: "FT9", 42: "FT7",
    43: "FC3", 44: "FC4", 45: "FT8", 46: "FT10", 47: "C5",  48: "C1",  
    49: "C2",  50: "C6",  51: "TP7", 52: "CP3",  53: "CPz", 54: "CP4",
    55: "TP8", 56: "P5",  57: "P1",  58: "P2",   59: "P6",  60: "PO7",
    61: "PO3", 62: "POz", 63: "PO4", 64: "PO8"
    }


    
