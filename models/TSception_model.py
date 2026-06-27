# -*- coding: utf-8 -*-
"""TSception model implementation for EEG classification.

Modified from:
https://github.com/yi-ding-cs/TSception

Reference:
    Ding, Y., Robinson, N., Zhang, S., Zeng, Q., and Guan, C. (2023).
    "TSception: Capturing Temporal Dynamics and Spatial Asymmetry From EEG
    for Emotion Recognition." IEEE Transactions on Affective Computing,
    14(3), 2238-2250. doi: 10.1109/TAFFC.2022.3169001.

Last modification: 2026-06-21
Author:
    Fujie
"""
import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
from typing import Optional, Sequence, Tuple, List
#%%
electrode_dict = {
    1:  "Fp1", 2: "Fp2",  3: "F7",   4: "F3",    5: "Fz",   6: "F4", 
    7:  "F8",  8: "FC5",  9: "FC1",  10: "FC2",  11: "FC6", 12: "T7",  
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

electrode_list = [electrode_dict[key] for key in sorted(electrode_dict.keys())]
TS_electrode_list_L=['Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'F7', 'FC1','FC3', 
                     'FC5', 'FT7', 'FT9', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3',
                     'CP5', 'TP7', 'TP9', 'P1', 'P3', 'P5', 'P7', 'PO3', 'O1', 
                     'PO7','PO9',]
TS_electrode_list_R=['Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2','FC4', 
                     'FC6', 'FT8', 'FT10', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4',
                     'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'PO4', 'O2', 
                     'PO8','PO10',]
TS_electrode_list=TS_electrode_list_L+TS_electrode_list_R
result = list(set(electrode_list) - set(TS_electrode_list))




class TSception(nn.Module):
    """Temporal-spatial convolutional network for EEG classification.

    TSception uses multi-scale temporal convolution branches, asymmetric
    spatial convolution branches, a high-level fusion layer, and a fully
    connected classifier to encode EEG signals.

    Attributes:
        Tception1 (nn.Sequential): First temporal convolution branch using the
            longest temporal window.
        Tception2 (nn.Sequential): Second temporal convolution branch using an
            intermediate temporal window.
        Tception3 (nn.Sequential): Third temporal convolution branch using the
            shortest temporal window.
        Sception1 (nn.Sequential): Spatial branch with a global kernel covering
            all selected electrodes.
        Sception2 (nn.Sequential): Spatial branch with a hemisphere kernel
            covering half of the selected electrodes.
        fusion_layer (nn.Sequential): Fusion convolution layer that aggregates
            global and hemisphere-level spatial features.
        BN_t (nn.BatchNorm2d): Batch normalization after temporal feature
            concatenation.
        BN_s (nn.BatchNorm2d): Batch normalization after spatial feature
            concatenation.
        BN_fusion (nn.BatchNorm2d): Batch normalization after the fusion layer.
        fc (nn.Sequential): Fully connected classifier containing a hidden
            layer, activation, dropout, and output layer.
    """

    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        """Builds a basic convolution block.

        The block contains a 2D convolution layer, a LeakyReLU activation, and
        an average pooling layer applied along the temporal dimension.

        Args:
            in_chan (int): Number of input channels for the convolution layer.
            out_chan (int): Number of output channels for the convolution layer.
            kernel (tuple[int, int]): Convolution kernel size as
                ``(height, width)``.
            step (int | tuple[int, int]): Convolution stride.
            pool (int): Temporal pooling window width.

        Returns:
            nn.Sequential: A sequential container with Conv2d, LeakyReLU, and
            AvgPool2d layers.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,            
                out_channels=out_chan,          
                kernel_size=kernel,           
                stride=step                    
            ),
            nn.LeakyReLU(),                    
            nn.AvgPool2d(                      
                kernel_size=(1, pool),          
                stride=(1, pool)               
            )
        )

    def __init__(self, num_classes, input_size, sampling_rate,
                 num_T, num_S, hidden, dropout_rate,inception_window=(0.25, 0.125, 0.0625)):
        """Initializes the TSception network.

        Args:
            num_classes (int): Number of output classes.
            input_size (tuple[int, int, int]): Input tensor shape as
                ``(input_channels, electrodes, time_points)``.
            sampling_rate (int): EEG sampling rate in Hz.
            num_T (int): Number of output channels in each temporal branch.
            num_S (int): Number of output channels in each spatial branch.
            hidden (int): Number of hidden units in the first fully connected
                layer.
            dropout_rate (float): Dropout probability used in the classifier.
            inception_window (tuple[float, float, float]): Ratios used to
                compute temporal convolution kernel lengths from
                ``sampling_rate``.

        Attributes:
            Tception1 (nn.Sequential): First temporal convolution branch.
            Tception2 (nn.Sequential): Second temporal convolution branch.
            Tception3 (nn.Sequential): Third temporal convolution branch.
            Sception1 (nn.Sequential): Global spatial convolution branch.
            Sception2 (nn.Sequential): Hemisphere spatial convolution branch.
            fusion_layer (nn.Sequential): High-level spatial fusion layer.
            BN_t (nn.BatchNorm2d): Temporal batch normalization layer.
            BN_s (nn.BatchNorm2d): Spatial batch normalization layer.
            BN_fusion (nn.BatchNorm2d): Fusion batch normalization layer.
            fc (nn.Sequential): Fully connected classifier.
        """
        super(TSception, self).__init__()     


        self.inception_window = inception_window

        self.pool = 8

        # Temporal convolution branches with three different kernel lengths.
        self.Tception1 = self.conv_block(
            1,                               # Input has one feature channel.
            num_T,                           # Outputs num_T channels.
            (1, int(self.inception_window[0] * sampling_rate)),  # First temporal kernel.
            1,                               # Stride is 1.
            self.pool                        # Temporal pooling window width.
        )
        self.Tception2 = self.conv_block(
            1,
            num_T,
            (1, int(self.inception_window[1] * sampling_rate)),  # Second temporal kernel.
            1,
            self.pool
        )
        self.Tception3 = self.conv_block(
            1,
            num_T,
            (1, int(self.inception_window[2] * sampling_rate)),  # Third temporal kernel.
            1,
            self.pool
        )

        # Spatial convolution branches for global and hemisphere representations.
        self.Sception1 = self.conv_block(
            num_T,                           # Input comes from temporal branches.
            num_S,                           # Outputs num_S channels.
            (int(input_size[1]), 1),        # Kernel covers all selected electrodes.
            1,                               # Stride is 1.
            int(self.pool * 0.25)            # Pooling window is pool * 0.25.
        )
        self.Sception2 = self.conv_block(
            num_T,
            num_S,
            (int(input_size[1] * 0.5), 1),   # Kernel covers half of the electrodes.
            (int(input_size[1] * 0.5), 1),   # Spatial stride covers half the electrodes.
            int(self.pool * 0.25)
        )

        # Fusion convolution branch for integrating spatial features.
        self.fusion_layer = self.conv_block(
            num_S,                           # Input channels after spatial branches.
            num_S,                           # Keeps the same channel count.
            (3, 1),                          # Small fusion kernel along space.
            1,                               # Stride is 1.
            4                                # Pooling window width is 4.
        )

        # Batch normalization layers.
        self.BN_t = nn.BatchNorm2d(num_T)    # After temporal feature concatenation.
        self.BN_s = nn.BatchNorm2d(num_S)    # After spatial feature concatenation.
        self.BN_fusion = nn.BatchNorm2d(num_S)  # After the fusion convolution layer.

        # Fully connected classifier.
        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),       # Maps num_S features to hidden units.
            nn.ReLU(),                      # ReLU activation.
            nn.Dropout(dropout_rate),       # Dropout for regularization.
            nn.Linear(hidden, num_classes)  # Maps hidden units to class logits.
        )

    def forward(self, x):
        """Runs the forward pass of TSception.

        Args:
            x (torch.Tensor): Input EEG tensor with shape
                ``(batch_size, 1, electrodes, time_points)``.

        Returns:
            torch.Tensor: Output logits with shape
            ``(batch_size, num_classes)``.
        """
        # Computes three temporal branches in parallel.
        y = self.Tception1(x)               # Output of the first temporal branch.
        out = y                             # Initializes the temporal concatenation output.
        y = self.Tception2(x)               # Output of the second temporal branch.
        out = torch.cat((out, y), dim=-1)   # Concatenates along the temporal dimension.
        y = self.Tception3(x)               # Output of the third temporal branch.
        out = torch.cat((out, y), dim=-1)   # Appends the third branch output.

        out = self.BN_t(out)                # Normalizes concatenated temporal features.

        # Computes two spatial branches in parallel.
        z = self.Sception1(out)             # Output of the global spatial branch.
        out_ = z                            # Initializes the spatial concatenation output.
        z = self.Sception2(out)             # Output of the hemisphere spatial branch.
        out_ = torch.cat((out_, z), dim=2)  # Concatenates along the spatial height dimension.

        out = self.BN_s(out_)               # Normalizes concatenated spatial features.

        out = self.fusion_layer(out)        # Applies the high-level fusion convolution.
        out = self.BN_fusion(out)           # Normalizes fused features.

        # Applies global average pooling over time and removes the singleton dimension.
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)

        out = self.fc(out)                  # Applies the fully connected classifier.

        return out                          # Returns the final logits.
    
#%% Full model



class TSception_Full(nn.Module):
    """Wraps TSception with electrode selection and reordering.

    This wrapper receives EEG tensors in the original electrode order, selects
    the electrodes required by TSception, reorders them according to
    ``ts_electrodes_list``, and forwards the selected tensor to the TSception
    backbone.

    Attributes:
        indices (list[int]): Indices used to select and reorder electrodes from
            the original EEG input tensor.
        input_size (tuple[int, int, int]): Input size passed to the TSception
            backbone as ``(input_channels, input_electrodes, input_times)``.
        tsception (TSception): TSception backbone model.
    """

    def __init__(
        self,
        all_electrodes_list: Optional[Sequence[str]] = None,
        ts_electrodes_list: Optional[Sequence[str]] = None,
        num_classes: int = 5,
        fs: int = 256,
        num_T: int = 15,
        num_S: int = 15,
        hidden: int = 32,
        dropout_rate: float = 0.5,
        input_channels: int = 1,
        input_electrodes: int = 58,
        input_times: int = 641,
        inception_window: Tuple[float, float, float] = (0.25, 0.125, 0.0625),
    ) -> None:
        """Initializes the full TSception wrapper.

        Args:
            all_electrodes_list (Optional[Sequence[str]]): Names of all
                electrodes in the original EEG input, ordered exactly as the
                electrode dimension of the input tensor. Defaults to None.
            ts_electrodes_list (Optional[Sequence[str]]): Electrode names used
                by TSception, usually ordered as left-hemisphere electrodes
                followed by right-hemisphere electrodes. Defaults to None.
            num_classes (int): Number of output classes. Defaults to 5.
            fs (int): EEG sampling rate in Hz. Defaults to 256.
            num_T (int): Number of temporal kernels per temporal branch.
                Defaults to 15.
            num_S (int): Number of spatial kernels per spatial branch.
                Defaults to 15.
            hidden (int): Number of hidden units in the first fully connected
                layer. Defaults to 32.
            dropout_rate (float): Dropout probability used in the classifier.
                Defaults to 0.5.
            input_channels (int): Number of input feature channels. Defaults to
                1.
            input_electrodes (int): Number of electrodes actually used by
                TSception. This value must equal ``len(ts_electrodes_list)``.
                Defaults to 58.
            input_times (int): Number of time points in each EEG sample.
                Defaults to 641.
            inception_window (Tuple[float, float, float]): Temporal kernel
                ratios used to compute kernel lengths from ``fs``. Defaults to
                ``(0.25, 0.125, 0.0625)``.

        Raises:
            ValueError: If ``all_electrodes_list`` is not provided.
            ValueError: If ``ts_electrodes_list`` is not provided.
            ValueError: If any electrode in ``ts_electrodes_list`` is missing
                from ``all_electrodes_list``.
            AssertionError: If ``input_electrodes`` does not equal
                ``len(ts_electrodes_list)``.
            AssertionError: If ``input_electrodes`` is not even.
        """

        super().__init__()

        # ------------------------------------------------------------
        # 0. Checks whether electrode lists are explicitly provided.
        # ------------------------------------------------------------
        # TSception_Full must know:
        # 1. The electrode order in the original EEG input: all_electrodes_list.
        # 2. The electrode order actually used by TSception: ts_electrodes_list.
        #
        # Therefore, this wrapper does not fall back to global default variables.
        # Users must provide both lists explicitly.
        if all_electrodes_list is None:
            raise ValueError(
                "all_electrodes_list must be provided explicitly. "
                "This list contains all electrode names and their order in the original EEG input."
            )

        if ts_electrodes_list is None:
            raise ValueError(
                "ts_electrodes_list must be provided explicitly. "
                "This list contains the paired left/right hemisphere electrodes and their order used by TSception."
            )

        # ------------------------------------------------------------
        # 1. Checks whether all TSception electrodes exist in all_electrodes_list.
        # ------------------------------------------------------------
        missing: List[str] = [
            item for item in ts_electrodes_list
            if item not in all_electrodes_list
        ]

        if len(missing) > 0:
            raise ValueError(
                f"These electrodes are not in all_electrodes_list: {missing}"
            )

        # ------------------------------------------------------------
        # 2. Gets TSception electrode indices in the original EEG input.
        # ------------------------------------------------------------
        self.indices: List[int] = [
            all_electrodes_list.index(item)
            for item in ts_electrodes_list
        ]

        # ------------------------------------------------------------
        # 3. Checks whether the selected electrode count equals input_electrodes.
        # ------------------------------------------------------------
        assert len(self.indices) == input_electrodes, (
            f"input_electrodes={input_electrodes}, "
            f"but len(ts_electrodes_list)={len(self.indices)}."
            "For TSception, input_electrodes must equal the number of selected electrodes."
        )

        # ------------------------------------------------------------
        # 4. Checks whether the selected electrode count is even.
        # ------------------------------------------------------------
        assert input_electrodes % 2 == 0, (
            "TSception hemisphere convolution requires an even number of input electrodes "
            "so that the electrodes can be split into equal left and right hemisphere groups."
        )

        # ------------------------------------------------------------
        # 5. Defines the input size for the TSception backbone.
        # ------------------------------------------------------------
        self.input_size: Tuple[int, int, int] = (
            input_channels,
            input_electrodes,
            input_times,
        )

        # ------------------------------------------------------------
        # 6. Builds the TSception backbone.
        # ------------------------------------------------------------
        self.tsception: TSception = TSception(
            num_classes=num_classes,
            input_size=self.input_size,
            sampling_rate=fs,
            num_T=num_T,
            num_S=num_S,
            hidden=hidden,
            dropout_rate=dropout_rate,
            inception_window=inception_window,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs electrode selection and the TSception forward pass.
    
        Args:
            x (torch.Tensor): Original EEG input tensor, usually with shape
                ``(batch_size, 1, 64, 641)``.
    
        Returns:
            torch.Tensor: Classification logits with shape
            ``(batch_size, num_classes)``.
        """
    
        # Selects the electrodes required by TSception from the original input.
        # For example, this can select 58 paired left/right hemisphere electrodes
        # from a 64-electrode EEG tensor.
        x = x[:, :, self.indices, :]
    
        # Forwards the selected and reordered EEG tensor to the TSception backbone.
        return self.tsception(x)
            

#%% test code

if __name__ == "__main__":

    electrode_dict = {
        1:  "Fp1", 2: "Fp2",  3: "F7",   4: "F3",    5: "Fz",   6: "F4", 
        7:  "F8",  8: "FC5",  9: "FC1",  10: "FC2",  11: "FC6", 12: "T7",  
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

    electrode_list = [electrode_dict[key] for key in sorted(electrode_dict.keys())]
    TS_electrode_list_L=['Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'F7', 'FC1','FC3', 
                         'FC5', 'FT7', 'FT9', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3',
                         'CP5', 'TP7', 'TP9', 'P1', 'P3', 'P5', 'P7', 'PO3', 'O1', 
                         'PO7','PO9',]
    TS_electrode_list_R=['Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2','FC4', 
                         'FC6', 'FT8', 'FT10', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4',
                         'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'PO4', 'O2', 
                         'PO8','PO10',]
    
    TS_electrode_list=TS_electrode_list_L+TS_electrode_list_R
    
    device = 'cpu'
    

    model = TSception_Full(                 
        all_electrodes_list = electrode_list,
        ts_electrodes_list = TS_electrode_list,
        input_channels = 1,
        input_electrodes = len(TS_electrode_list),
        input_times = 512,
        num_classes=5,
        fs=256,
        num_T = 15,
        num_S = 15, 
        hidden = 32, 
        dropout_rate = 0.5,
        inception_window=(0.25, 0.125, 0.0625),
        ).to(device)

    
    
    x = torch.randn(8, 1, len(electrode_list), 512, device=device)
    print(model(x).shape)



















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
