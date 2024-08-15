# **MD-Syn: Synergistic drug combination prediction based on the multidimensional feature fusion method and attention mechanisms**

## Abstract

Drug combination therapies have shown promising therapeutic efficacy in complex diseases and have had the potential to reduce drug resistance. However, the huge number of possible drug combination pairs makes it difficult to screen them all for synergistic drug combinations in traditional experiments. In this study, we proposed MD-Syn, a computational framework, based on the multidimensional feature fusion method and multi-head attention mechanisms. Given drug pair-cell line triplets, MD-Syn considers one-dimensional and two-dimensional feature spaces simultaneously. It consists of a one-dimensional feature embedding module (1D-FEM), a two-dimensional feature embedding module (2D-FEM), and a deep neural network-based classifier for synergistic drug combination pair prediction. MD-Syn achieved the AUC of 0.92 in 5-fold cross-validation, outperforming the state-of-the-art methods. Further, MD-Syn showed comparable results over two independent datasets. In addition, the multi-head mechanisms not only learn embeddings from different feature aspects but also focus on essential interactive feature elements, improving the interpretability ability of MD-Syn. In summary, MD-Syn is an interpretable framework to prioritize synergistic drug pairs with chemicals and cancer cell line gene expression profiles.

Keywords: drug combination, multidimensional feature fusion, graph neural network, attention mechanism

## Overview

- `data/` contains raw data files and processing data  files;

- `result/` contains 5-fold MD-Syn metric score files and auc metric score figure;

- `model.py` contains the code of MD-Syn model;

- `attention.py` contains the code of attention mechanism;

- `muti_head_attention.py` contains the code of muti-head-attention mechanism;

- `transformer.py` contains the code of transformer model;

- `loader.py` contains the code of data processing;

- `main.py` main function for MD-Syn.

  ## Requirements

  The MD-Syn model is built using PyTorch and PyTorch Geometric. You can use following commands to create conda env with related dependencies.

  ```linux
  conda create -n MDSyn python=3.11
  conda activate MDSyn
  pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
  conda install pyg -c pyg
  conda install pandas
  conda install rdkit
  conda install scipy==1.11.4
  conda install tqdm
  ```

  ## Dataset

   Due to the size of certain data files, direct uploading to GitHub is not feasible.These files can be accessed for download [here](https://drive.google.com/drive/folders/10M9KRnyQR-XR1VStL3RdayFSr81paEI2?usp=drive_link). If you require these data, please download it and place it in the appropriate folder.`data`

  ## Model Training

  Run the following commands to train MD-Syn.

  ```python
  python main.py
  ```

  ## Acknowledgement

  The backbone of the code is inherited from Pytorch.

  

  
