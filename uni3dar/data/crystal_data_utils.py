import numpy as np
import itertools
from collections import Counter
from scipy.interpolate import interp1d
from .atom_data import atom_list
from scipy.optimize import linear_sum_assignment
from itertools import combinations


def random_fill(
    known_peaks,
    fill_length,
    x_min,
    x_max,
    dist_margin=0.2,
    candidate_multiplier=50,
):
    known_peaks = np.array(known_peaks)
    num_known = len(known_peaks)
    num_to_fill = fill_length - num_known
    if num_to_fill <= 0:
        return np.zeros(0)
    candidate_count = num_to_fill * candidate_multiplier
    candidates = np.random.uniform(x_min, x_max, candidate_count)
    diffs = np.abs(candidates[:, None] - known_peaks[None, :])
    mask = np.all(diffs > dist_margin, axis=1)
    valid_candidates = candidates[mask]
    valid_candidates = valid_candidates[:num_to_fill]
    return valid_candidates


def interpolate_pxrd(
    pxrd_x_range_max, pxrd_x, pxrd_y, num_fill=128, threshold=5, step=0.1
):
    mask = pxrd_y > threshold
    pxrd_x = pxrd_x[mask]
    pxrd_y = pxrd_y[mask]
    pad_pxrd_x = random_fill(pxrd_x, num_fill, 0, pxrd_x_range_max)
    pxrd_x = np.concatenate([pxrd_x, pad_pxrd_x])
    pxrd_y = np.concatenate([pxrd_y, np.zeros(pad_pxrd_x.shape[0])])
    sorted_index = np.argsort(pxrd_x)
    pxrd_x = pxrd_x[sorted_index]
    pxrd_y = pxrd_y[sorted_index]
    if pxrd_x[0] > 0:
        pxrd_x = np.insert(pxrd_x, 0, 0)
        pxrd_y = np.insert(pxrd_y, 0, 0)
    if pxrd_x[-1] < pxrd_x_range_max:
        pxrd_x = np.append(pxrd_x, pxrd_x_range_max)
        pxrd_y = np.append(pxrd_y, 0)
    new_x = np.arange(0, pxrd_x_range_max, step)
    interpolator = interp1d(pxrd_x, pxrd_y, kind="linear", fill_value="extrapolate")
    new_y = interpolator(new_x)
    return new_x, new_y


def get_crystal_cond(
    args, cond_data, atom_type, is_train, dictionary, xyz_null_id, bsz=1
):
    assert args.data_type == "crystal"
    level = args.merge_level + 1

    if args.crystal_pxrd > 0:
        pxrd_x_range_max = 120
        assert pxrd_x_range_max % args.crystal_pxrd == 0
        pxrd_x = cond_data["pxrd_x"]
        pxrd_y = cond_data["pxrd_y"]
        ori_pxrd_x = np.array(pxrd_x)
        ori_pxrd_y = np.array(pxrd_y)
        num_prxd = int(pxrd_x_range_max / args.crystal_pxrd_step)
        pxrd_batches = []
        for _ in range(bsz):
            # TODO: try to not for loop with batch size
            pxrd_x, pxrd_y = interpolate_pxrd(
                pxrd_x_range_max,
                ori_pxrd_x,
                ori_pxrd_y,
                num_fill=args.crystal_pxrd_num_fill,
                threshold=args.crystal_pxrd_threshold,
                step=args.crystal_pxrd_step,
            )
            assert pxrd_y.shape[0] == num_prxd
            if is_train and args.crystal_pxrd_noise > 0:
                pxrd_y += np.random.randn(pxrd_y.shape[0]) * args.crystal_pxrd_noise
            if args.crystal_pxrd_sqrt:
                pxrd_y[pxrd_y < 0] = 0
                pxrd_y = np.sqrt(pxrd_y / 10)
            else:
                pxrd_y = pxrd_y / 50
            if is_train and np.random.rand() < args.crystal_cond_drop:
                pxrd_y[:] = 0
            num_pxrd_per_token = num_prxd // args.crystal_pxrd
            pxrd = pxrd_y.reshape(args.crystal_pxrd, num_pxrd_per_token)
            pxrd_batches.append(pxrd)
        pxrd = np.stack(pxrd_batches, axis=0)
    if args.crystal_component > 0:
        components = np.zeros((1, 128), dtype=np.int32)
        atom_type = [dictionary[i] for i in atom_type]
        for i in atom_type:
            assert i < len(atom_list)
            components[:, i] += 1
        components = components.astype(np.float32)
        if args.crystal_component_sqrt:
            components = np.sqrt(components)
        if is_train and args.crystal_component_noise > 0:
            components += (
                np.random.randn(1, components.shape[1]) * args.crystal_component_noise
            )
        if np.random.rand() < args.crystal_cond_drop and is_train:
            components[:, :] = 0

    tokens = []

    def add_token(token, token_list):
        token = np.array(token, dtype=np.int32)
        token_list.append(token)

    if args.crystal_pxrd > 0:
        add_token(
            [dictionary[f"[PXRD_{i}]"] for i in range(args.crystal_pxrd)],
            tokens,
        )
    if args.crystal_component > 0:
        assert dictionary.max_num_atom > 0
        count_atoms = len(atom_type)
        count_atoms = min(count_atoms, dictionary.max_num_atom)
        count_atoms = max(count_atoms, 1)
        add_token(
            [dictionary["[COMPONENTS]"], dictionary[f"[CNT_{count_atoms}]"]],
            tokens,
        )

    tokens = np.concatenate(tokens, axis=0) if tokens else np.zeros(0, dtype=np.int32)
    feat = {}
    feat["decoder_type"] = tokens
    feat["decoder_level"] = np.full(tokens.shape[0], level, dtype=np.int32)
    feat["decoder_xyz"] = np.full((tokens.shape[0], 3), xyz_null_id, dtype=np.int32)
    half_grid_size = 2 ** (args.merge_level) * 0.5 * args.grid_len
    feat["decoder_phy_pos"] = np.full((tokens.shape[0], 3), half_grid_size)
    feat["decoder_is_second_atom"] = np.full(tokens.shape[0], False, dtype=np.bool_)
    feat["decoder_remaining_atoms"] = np.full(tokens.shape[0], 0, dtype=np.int32)
    feat["decoder_remaining_tokens"] = np.full(tokens.shape[0], 0, dtype=np.int32)
    feat["decoder_count"] = np.full(tokens.shape[0], 0, dtype=np.int32)

    if bsz > 1:
        for key in feat:
            feat[key] = feat[key][None].repeat(bsz, axis=0)

    if args.crystal_pxrd:
        feat["pxrd"] = pxrd.astype(np.float32)
        if bsz <= 1:
            feat["pxrd"] = feat["pxrd"][0, :]
    if args.crystal_component:
        feat["components"] = components.astype(np.float32)
        if bsz > 1:
            feat["components"] = feat["components"][None].repeat(bsz, axis=0)
    return feat


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    import smact
    from smact.screening import pauling_test

    elem_symbols = tuple([atom_list[elem - 1] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def construct_from_ase_and_check_valid(ase_structure):
    from pymatgen.core import Lattice, Structure

    try:
        structure = Structure(
            Lattice(ase_structure.cell),
            ase_structure.get_chemical_symbols(),
            ase_structure.get_positions(),
            coords_are_cartesian=True,
        )
    except Exception:
        return None, False
    elem_counter = Counter(ase_structure.get_atomic_numbers())
    composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    elems, counts = list(zip(*composition))
    counts = np.array(counts)
    counts = counts / np.gcd.reduce(counts)
    comps = tuple(counts.astype("int").tolist())
    comp_valid = smact_validity(elems, comps)
    struct_valid = structure_validity(structure)
    return structure, comp_valid and struct_valid


# refer to https://github.com/jiaor17/DiffCSP/blob/7121d159826efa2ba9500bf299250d96da37f146/scripts/compute_metrics.py#L167-L197
def match_rate_at_k(gt, preds, k=20):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    gt, _ = construct_from_ase_and_check_valid(gt)
    # cannot construct gt, treating as matching failure
    if gt is None:
        return 0, 0
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3, primitive_cell=True)
    if len(preds) > k:
        preds = preds[:k]
    rmse_list_valid = []
    for i in range(len(preds)):
        pred, is_valid = construct_from_ase_and_check_valid(preds[i])
        if pred is None:
            continue
        rmse = matcher.get_rms_dist(gt, pred)
        if rmse is not None and is_valid:
            rmse_list_valid.append(rmse[0])
    is_match_valid = len(rmse_list_valid) > 0
    rmse_valid = np.min(rmse_list_valid) if is_match_valid else 0
    return is_match_valid, rmse_valid


def check_vertices_stable(
    origin, neighbors, others, cond_tol=10, rank_tol=0.2, round_tol=0.05
):
    v = neighbors - origin
    scale = np.linalg.norm(v, axis=1).mean() + 1e-6
    origin = origin / scale
    neighbors = neighbors / scale
    others = others / scale
    v = neighbors - origin
    if np.linalg.cond(v) > cond_tol:
        return False

    if np.linalg.matrix_rank(v, tol=rank_tol) != 3:
        return False

    vec = others - origin
    for idx in range(others.shape[0]):
        coeffs = np.linalg.lstsq(v.T, vec[idx], rcond=None)[0]
        # should be 0 or 1
        if not np.all(np.minimum(np.abs(coeffs), np.abs(coeffs - 1)) < round_tol):
            return False
        # should have more than 2 1s
        if np.sum(np.isclose(coeffs, 1, atol=round_tol)) < 2:
            return False
    return True


def to_right_handed_matrix(origin, neighbors):
    local_axes = neighbors - origin
    if np.linalg.det(local_axes) < 0:
        neighbors[[0, 1]] = neighbors[[1, 0]]
    return origin, neighbors


def find_adjacent_vertices(vertices, target_idx=0):
    if vertices.shape != (8, 3):
        raise ValueError("shape must be (8,3)")

    origin = vertices[target_idx]
    candidates = np.delete(np.arange(8), target_idx)
    for combo in combinations(candidates, 3):
        neighbors = np.stack([vertices[i] for i in combo])
        origin, neighbors = to_right_handed_matrix(origin, neighbors)
        others = np.stack([vertices[i] for i in candidates if i not in combo])
        if check_vertices_stable(origin, neighbors, others):
            return origin, neighbors
    raise ValueError("no valid vertices found")


def construct_symmetric_vertices(O, neighbors):
    A, B, C = neighbors[0], neighbors[1], neighbors[2]
    pts = []
    pts.append(O)
    pts.append(A)
    pts.append(B)
    pts.append(C)
    pts.append(A + B - O)
    pts.append(A + C - O)
    pts.append(B + C - O)
    pts.append(A + B + C - 2 * O)
    return np.array(pts)


def compute_error_hungarian(original_pts, constructed_pts):
    n = original_pts.shape[0]
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.linalg.norm(original_pts[i] - constructed_pts[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_error = cost_matrix[row_ind, col_ind].sum()
    average_error = total_error / n
    return average_error, (row_ind, col_ind)


def fit_parallelepiped_symmetric_right_hand(vertices):
    if vertices.shape != (8, 3):
        raise ValueError("shape should be (8, 3)")
    origin, neighbors = find_adjacent_vertices(vertices, 0)
    pred_vertices = construct_symmetric_vertices(origin, neighbors)
    error, _ = compute_error_hungarian(vertices, pred_vertices)
    if error > 1.0:
        raise ValueError("too large error to fit parallelepiped")
    return origin, neighbors
