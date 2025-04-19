import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset, data_utils
from copy import deepcopy
import pickle


class SampleClusterDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        cluster_path,
    ):
        self.dataset = dataset
        self.seed = seed
        self.cluster = pickle.load(open(cluster_path, "rb"))
        self.cnt = len(self.cluster)
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
        # if epoch is not None:
        self._init_cluster(epoch)

    def _init_cluster(self, epoch):
        if epoch is None:
            epoch = 0
        with data_utils.numpy_seed(self.seed, epoch):
            self.cluster2id = {}
            key_list = list(self.cluster.keys())
            for i in range(len(key_list)):
                item = key_list[i]
                select_id = np.random.choice(len(self.cluster[item]), 1, replace=False)[
                    0
                ]
                select_id = self.cluster[item][select_id]
                select_key = select_id.to_bytes(4, byteorder="big")
                self.cluster2id[i] = select_id

    def __len__(self):
        return self.cnt

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        index = self.cluster2id[index]
        data = deepcopy(self.dataset[index])
        if "uid" not in data:
            data["uid"] = index
        return data

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
