## import modules
import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


def atom_features(atom):
    """extract atomic features"""
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    """set max atom number equals to 100"""
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    mask = [1] * 100 ###
    features = torch.zeros([100, 78])
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum == 0:
            return None

        feature = atom_features(atom)
        features[i, :] = torch.tensor(feature / sum(feature))
        mask[i] = 0 ####

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=features, edge_index=edge_index)

    return data, mask

class DrugcombDataset(InMemoryDataset):
    def __init__(self, root='./data', dataset='drug1', transform=None, pre_transform=None):
        super(DrugcombDataset, self).__init__(root, transform, pre_transform)
        self.transform, self.pre_transform = transform, pre_transform
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        drug_embedding = pd.read_csv("./data/raw/all_drug_embedding.csv")
        ddi = pd.read_csv("./data/raw/all_ddi.csv")
        label = ddi['label']
        smile1 = ddi['drug1']
        smile2 = ddi['drug2']
        cell = ddi.iloc[:, 9:]
        molformer_embedding1 = drug_embedding.iloc[:, 0:768]
        molformer_embedding2 = drug_embedding.iloc[:, 768:]

        data_list = []
        data_len = len(smile1)
        print('number of data', data_len)

        with tqdm(total=data_len, desc="Converting SMILES to graph") as pbar:
            for i in range(data_len):
                smiles = smile1.iloc[i,]
                labels = label.iloc[i,]
                smile_embedding = molformer_embedding1.iloc[i,]
                cell_line = cell.iloc[i,]

                data, mask = smile_to_graph(smiles)
                boolean_list = [bool(value) for value in mask]
                data.mask = torch.tensor(boolean_list)
                data.y = torch.tensor(labels)
                embedding = [float(x) for x in smile_embedding]
                data.smiles_embedding = torch.FloatTensor([embedding])
                ccle_embedding = [float(x) for x in cell_line]
                data.ccle_embedding = torch.FloatTensor([ccle_embedding])

                data_list.append(data)
                pbar.update(1)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])


#test DrugcombDataset object
if __name__ == "__main__":
    print("accept the paper !")
    dataset = DrugcombDataset(root='./data', dataset='ONeil_Drug1')
    # dataset = DrugcombDataset(root='./data', dataset='ONeil_Drug2')










