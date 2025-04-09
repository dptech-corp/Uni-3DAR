from collections import Counter
import json
import os
import pandas as pd

# from ast import literal_eval
from tqdm import tqdm
from p_tqdm import p_map
from functools import partial

import numpy as np
import multiprocessing
from pathlib import Path
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure, Composition, Lattice
from pymatgen.io.cif import CifParser, CifBlock
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
import os
import sys
import contextlib
import tempfile
import time
import re
from io import StringIO
from absl import logging
from eval_utils import (
    smact_validity,
    structure_validity,
    get_fp_pdist,
    compute_cov,
    CompScaler,
)
import pickle


# TODO: AttributeError in CrystalNNFP
CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")

COV_Cutoffs = {
    "mp20": {"struc": 0.4, "comp": 10.0},
    "carbon": {"struc": 0.2, "comp": 4.0},
    "perovskite": {"struc": 0.2, "comp": 4},
}


def get_struct_from_raw(cif_content):
    try:
        cif_block = CifBlock.from_str(cif_content)
        a = float(cif_block.data["_cell_length_a"])
        b = float(cif_block.data["_cell_length_b"])
        c = float(cif_block.data["_cell_length_c"])
        alpha = float(cif_block.data["_cell_angle_alpha"])
        beta = float(cif_block.data["_cell_angle_beta"])
        gamma = float(cif_block.data["_cell_angle_gamma"])

        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        species = cif_block.data["_atom_site_type_symbol"]
        coords = list(
            zip(
                map(float, cif_block.data["_atom_site_fract_x"]),
                map(float, cif_block.data["_atom_site_fract_y"]),
                map(float, cif_block.data["_atom_site_fract_z"]),
            )
        )
        structure = Structure(lattice, species, coords)
        return structure
    except Exception as e:
        print("get_struct_from_raw | Error parsing block: ", e, "\n", cif_content)
        return None


def get_struct_from_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    blocks = re.split(r"(?=^[ \t]*data_)", content, flags=re.MULTILINE)
    if "data_" not in blocks[0]:
        blocks = blocks[1:]
    all_structs = []
    for block in blocks:
        block = block.strip()
        if not block:
            print("no block")
            continue
        try:
            parser = CifParser(StringIO(block))
            single_struct = parser.get_structures(primitive=True)[0]
            all_structs.append(single_struct)
        except Exception as e:
            print("get_struct_from_file | Error parsing block: ", e)
            struct = get_struct_from_raw(block)
            if struct is not None:
                all_structs.append(struct)

    return all_structs


@contextlib.contextmanager
def supress_stdout():
    in_memory_file = tempfile.SpooledTemporaryFile()
    # suppress stdout
    orig_stdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(in_memory_file.fileno(), 1)
    # suppress stderr
    orig_stderr_fno = os.dup(sys.stderr.fileno())
    os.dup2(in_memory_file.fileno(), 2)
    try:
        yield
    finally:
        os.fsync(in_memory_file)
        os.dup2(orig_stdout_fno, 1)  # restore stdout
        os.dup2(orig_stderr_fno, 2)  # restore stderr
        in_memory_file.seek(0)
        outputs = in_memory_file.read().decode("utf-8").strip()
        if outputs:
            logging.info(outputs)
        in_memory_file.close()


class Crystal(object):

    def __init__(self, crys_dict):
        self.crys_dict = crys_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        try:
            if type(self.crys_dict) is Structure:
                self.structure = self.crys_dict
                self.crys_dict = self.crys_dict.as_dict()
            elif type(self.crys_dict) is str:
                self.structure = Structure.from_str(self.crys_dict, fmt="cif")
            else:
                self.structure = Structure.from_dict(self.crys_dict)
            self.dict = {
                "frac_coords": self.structure.frac_coords,
                "atom_types": np.array([_.Z for _ in self.structure.species]),
                "lengths": np.array(self.structure.lattice.abc),
                "angles": np.array(self.structure.lattice.angles),
            }
            self.atom_types = [s.specie.number for s in self.structure]
            self.constructed = True
        except Exception as e:
            self.constructed = False
            self.invalid_reason = "construction_raises_exception"
            print("Construction failed", e)
        if self.structure.volume < 0.1:
            self.constructed = False
            self.invalid_reason = "unrealistically_small_lattice"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        with supress_stdout():
            try:
                site_fps = [
                    CrystalNNFP.featurize(self.structure, i)
                    for i in range(len(self.structure))
                ]
            except Exception:
                self.valid = False
                self.comp_fp = None
                self.struct_fp = None
                return
            self.struct_fp = np.array(site_fps).mean(axis=0)


def process_with_file_cache(func, cache_file):
    if not Path(cache_file).exists():
        res = func()
        with open(cache_file, "wb") as f:
            pickle.dump(res, f)
    with open(cache_file, "rb") as f:
        res = pickle.load(f)
    return res


class GenEval(object):

    def __init__(
        self,
        pred_crys,
        gt_crys,
        n_samples=1000,
        eval_model_name="mp20",
    ):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False
            )
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f"not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}"
            )

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {
            "comp_valid": comp_valid,
            "struct_valid": struct_valid,
            "valid": valid,
        }

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {"comp_div": comp_div}

    def get_struct_diversity(self):
        return {"struct_div": get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {"prop_stat_wdist_density": wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species)) for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {"prop_stat_wdist_num_elems": wdist_num_elems}

    def get_prop_wdist(self):
        pass

    def get_coverage(self):
        # in previous here uses self.crys, rather than self.valid_samples
        # the number of valid samples is about 1000, while the number of all samples is about 10000
        # It may be a bug, but we keep it consistent with the original code.
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        cov_metrics_dict = compute_cov(
            self.crys,
            self.gt_crys,
            struc_cutoff=cutoff_dict["struc"],
            comp_cutoff=cutoff_dict["comp"],
        )
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        print("Computing validity metrics ...")
        metrics.update(self.get_validity())
        print("Computing comp diversity metrics ...")
        metrics.update(self.get_comp_diversity())
        print("Computing struct diversity metrics ...")
        metrics.update(self.get_struct_diversity())
        print("Computing density wdist metrics ...")
        metrics.update(self.get_density_wdist())
        print("Computing num elem wdist metrics ...")
        metrics.update(self.get_num_elem_wdist())
        # print("Computing prop wdist metrics ...")
        # metrics.update(self.get_prop_wdist())
        print("Computing coverage metrics ...")
        metrics.update(self.get_coverage())
        print(metrics)
        return metrics


def process_pred_one(file_path, num_io_process=40):
    structure = get_struct_from_file(file_path)
    with multiprocessing.Pool(num_io_process) as p:
        pred_crys = list(tqdm(p.imap(Crystal, structure), total=len(structure)))
    return pred_crys


def process_gts(input_file, num_io_process=40):
    df = pd.read_csv(input_file)
    cifs = df["cif"]
    with multiprocessing.Pool(num_io_process) as p:
        gt_crys = list(tqdm(p.imap(Crystal, cifs), total=len(cifs)))
    return gt_crys


def main(input_file, output_file, dataset, gt_path, num_io_process=40, n_samples=1000):
    print("processing gt files ...")
    gt_crys = process_with_file_cache(
        lambda: process_gts(gt_path), gt_path.replace(".csv", ".pkl")
    )
    print("num of gt structures:", len(gt_crys))
    print("processing pred files ...")
    pred_crys = process_pred_one(input_file, num_io_process)
    print("num of used structures:", len(pred_crys))
    print("computing metrics ...")
    eval_m = GenEval(
        pred_crys,
        gt_crys,
        n_samples=n_samples,
        eval_model_name=dataset,
    )
    all_metrics = eval_m.get_metrics()
    all_metrics["pred_num"] = len(pred_crys)
    print("all_metrics:", all_metrics)

    # only overwrite metrics computed in the new run.
    if Path(output_file).exists():
        with open(output_file, "r") as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(output_file, "w") as f:
                    json.dump(all_metrics, f, indent=2)
        if isinstance(written_metrics, dict):
            with open(output_file, "w") as f:
                json.dump(written_metrics, f, indent=2)
    else:
        with open(output_file, "w") as f:
            json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    np.random.seed(0)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    dataset = sys.argv[3]
    gt_path = sys.argv[4]

    start = time.time()
    main(input_file, output_file, dataset, gt_path, num_io_process=40)
    end = time.time()
    print("Time taken: ", end - start)
