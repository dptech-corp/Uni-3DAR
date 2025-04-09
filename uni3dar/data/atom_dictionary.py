import numpy as np
import math
from functools import lru_cache
from .atom_data import atom_list, protein_res_dict


class RawAtomFeature:
    def __init__(
        self,
        data_type,
        atom_pos,
        atom_type,
        atom_feats=None,
        mol_feats=None,
    ):
        self.data_type = data_type
        self.atom_pos = np.array(atom_pos, dtype=np.float64).reshape(-1, 3)
        self.atom_type = np.array(atom_type, dtype="<U15").reshape(-1)
        self.atom_feats = atom_feats if atom_feats is not None else None
        self.mol_feats = mol_feats if mol_feats is not None else None

    def subset(self, indices):
        atom_feats = None
        if self.atom_feats is not None:
            atom_feats = {}
            for key in self.atom_feats:
                atom_feats[key] = self.atom_feats[key][indices]

        return RawAtomFeature(
            self.data_type,
            self.atom_pos[indices],
            self.atom_type[indices],
            atom_feats=atom_feats,
            mol_feats=self.mol_feats,
        )

    def apply_rot_to_atom_target(self, key, rot_mat):
        if self.atom_feats is not None and key in self.atom_feats:
            # self.atom_feats[key] = self.atom_feats[key] @ rot_mat
            assert self.atom_feats[key].shape[1] == 3
            self.atom_feats["rot_T"] = (
                (rot_mat.T).reshape(1, 9).repeat(self.atom_feats[key].shape[0], axis=0)
            )

    def init_from_mol(atom_pos, atom_type, key="molecule"):
        return RawAtomFeature(
            key,
            atom_pos,
            atom_type,
        )

    def init_from_crystal(
        atom_pos, atom_type, lattice_matrix, is_train, crystal_random_shift_prob
    ):
        num_lattice = 8
        lattice_type = ["[LATTICE_O]"] * num_lattice
        lattice_pos = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        atom_type = lattice_type + atom_type
        atom_pos = np.concatenate([lattice_pos, atom_pos], axis=0)

        atom_type = np.array(atom_type, dtype="<U15")
        atom_pos = np.array(atom_pos, dtype=np.float64).reshape(-1, 3)
        if is_train:
            # here use rand(1), not rand(3).
            # the atoms will random shift in diagonal direction, not random direction
            shift_flag = np.random.rand()
            if shift_flag < crystal_random_shift_prob:
                atom_pos[num_lattice:] = atom_pos[num_lattice:] + np.random.rand()
        atom_pos[num_lattice:] = atom_pos[num_lattice:] % 1.0
        atom_pos = np.matmul(atom_pos, lattice_matrix)
        return RawAtomFeature("crystal", atom_pos, atom_type)

    def add_atom_and_mol_targets(self, mol_target=None, atom_target=None, mol_id=None):
        if mol_target is not None:
            mol_feats = {"target": mol_target, "index": np.array([mol_id]).reshape(1)}
            if self.mol_feats is not None:
                self.mol_feats.update(mol_feats)
            else:
                self.mol_feats = mol_feats
        else:
            mol_feats = None
        if atom_target is not None:
            atom_index = np.arange(atom_target.shape[0])
            atom_mol_id = np.full(atom_target.shape[0], mol_id)
            atom_feats = {
                "target": atom_target,
                "index": atom_index,
                "mol_index": atom_mol_id,
            }
            if self.atom_feats is not None:
                self.atom_feats.update(atom_feats)
            else:
                self.atom_feats = atom_feats


class AtomGridFeature:
    def __init__(
        self,
        atom_type,
        atom_pos,
        atom_xyz,
        dictionary,
        atom_feats=None,
        mol_feats=None,
        convert_atom_type=True,
    ):
        self.atom_pos = atom_pos
        self.atom_xyz = atom_xyz
        if convert_atom_type:
            self.atom_type = dictionary.get_token_batched(atom_type)
        else:
            self.atom_type = atom_type
        self.dictionary = dictionary
        self.atom_feats = atom_feats
        self.mol_feats = mol_feats

    def subset(self, indices):
        atom_feats = None
        if self.atom_feats is not None:
            atom_feats = {}
            for key in self.atom_feats:
                atom_feats[key] = self.atom_feats[key][indices]
        return AtomGridFeature(
            self.atom_type[indices],
            self.atom_pos[indices],
            self.atom_xyz[indices],
            self.dictionary,
            atom_feats=atom_feats,
            mol_feats=self.mol_feats,
            convert_atom_type=False,
        )

    def init_from_pos(atom_pos, type, xyz, dictionary):
        return AtomGridFeature(
            np.full(atom_pos.shape[0], type),
            atom_pos,
            np.full((atom_pos.shape[0], 3), xyz),
            dictionary,
            convert_atom_type=False,
        )

    def assign_with_index(self, index, other, other_index):
        assert (self.atom_pos[index] == other.atom_pos[other_index]).all()
        self.atom_xyz[index] = other.atom_xyz[other_index]
        self.atom_type[index] = other.atom_type[other_index]
        self.mol_feats = other.mol_feats
        if other.atom_feats is not None:
            if self.atom_feats is None:
                self.atom_feats = {}
            for key in other.atom_feats:
                if key not in self.atom_feats:
                    if len(other.atom_feats[key].shape) == 1:
                        self.atom_feats[key] = np.zeros(len(self.atom_type))
                    else:
                        feat_dim = other.atom_feats[key].shape[1]
                        self.atom_feats[key] = np.zeros((len(self.atom_type), feat_dim))
                self.atom_feats[key][index] = other.atom_feats[key][other_index]

    def extract_target_feat(self):
        feat = {}
        if self.mol_feats is not None:
            for key in self.mol_feats:
                if not isinstance(self.mol_feats[key], np.ndarray):
                    self.mol_feats[key] = np.array(self.mol_feats[key])
                feat["mol_" + key] = self.mol_feats[key].reshape(1, 1)
        if self.atom_feats is not None:
            for key in self.atom_feats:
                feat["atom_" + key] = self.atom_feats[key]
        return feat


class AtomFeatDict:
    def __init__(self, args):
        self.args = args
        self.atom_type = {}

        for res_type in protein_res_dict:
            for item in protein_res_dict[res_type]:
                atom_list.append(f"{res_type}_{item}")

        for j, atom_type in enumerate(atom_list):
            self.atom_type[atom_type] = j

        self.group_keys = ["[CLS]", "[UNK]", "[MASK]", "[NULL]"]
        for i in range(0, 256):
            self.group_keys.append(f"[TREE_{i}]")

        self.max_num_atom = self.args.max_num_atom
        if self.args.data_type == "crystal":
            self.group_keys += [
                "[COMPONENTS]",
                "[LATTICE_O]",
            ]
            for i in range(self.args.crystal_pxrd):
                self.group_keys.append(f"[PXRD_{i}]")
            # count tokens, inclusive for 0 and max_num_atom
            for i in range(self.max_num_atom + 1):
                self.group_keys.append(f"[CNT_{i}]")

        for key in self.group_keys:
            self.add_token(key)

    def add_token(self, token):
        if token not in self.atom_type:
            self.atom_type[token] = len(self.atom_type)

    @lru_cache(maxsize=16)
    def __getitem__(self, key):
        return self.atom_type[key]

    def __len__(self):
        return len(self.atom_type)

    def get_token_batched(self, tokens):
        return np.array([self.__getitem__(token) for token in tokens])
