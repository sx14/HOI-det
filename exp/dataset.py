import torch
from torch.utils.data import Dataset



class HICODatasetSpa(Dataset):

    def __init__(self, hoi_db):

        self.obj2vec = torch.from_numpy(hoi_db['obj2vec']).float()
        self.hboxes = torch.from_numpy(hoi_db['hboxes']).float()
        self.oboxes = torch.from_numpy(hoi_db['oboxes']).float()
        self.obj_classes = hoi_db['obj_classes']
        self.hoi_classes = torch.from_numpy(hoi_db['hoi_classes']).float()
        self.bin_classes = torch.from_numpy(hoi_db['bin_classes']).long()
        self.spa_feats = torch.from_numpy(hoi_db['spa_feats']).float()

    def __len__(self):
        return len(self.hboxes)

    def __getitem__(self, item):
        return torch.cat([self.spa_feats[item], self.obj2vec[self.obj_classes[item]]]), \
               self.hoi_classes[item], self.bin_classes[item], self.obj_classes[item]
