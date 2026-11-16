"""Unit Test for Person Re-ID Evaluation Metrics (Rank-1, Rank-5, mAP, and mINP).

Verifies the mathematical correctness of CMC, mAP, and Mean Inverse Negative
Penalty (mINP) computations against known synthetic ground-truth rank configurations.
"""

import numpy as np


def eval_reid_metrics(
    dist_matrix: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 10,
    return_minp: bool = False,
):
    """Compute CMC Rank-k, mean Average Precision (mAP), and optional mINP.

    Standard Re-ID evaluation protocol (Market-1501 / AG-ReID):
    - Same identity and same camera images are treated as junk and ignored.
    - mINP (Ye et al., TPAMI 2021): INP_q = |G_q| / R_hardest^q.
    """
    num_q, num_g = dist_matrix.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(dist_matrix, axis=1)

    all_cmc = []
    all_ap = []
    all_inp = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_cam = q_cams[q_idx]

        order = indices[q_idx]

        # Find same-identity images
        same_id = g_pids[order] == q_pid
        # Find same-camera images
        same_cam = g_cams[order] == q_cam

        # Junk images: same identity AND same camera
        junk_mask = same_id & same_cam
        # Keep non-junk images
        keep = ~junk_mask

        # Filtered rankings
        filtered_same_id = same_id[keep]

        # Total relevant gallery images
        num_pos = int(np.sum(filtered_same_id))
        if num_pos == 0:
            continue

        num_valid_q += 1

        # Binary hit vector: 1 if match, 0 if non-match
        match_vec = filtered_same_id.astype(np.int32)

        # 1. CMC calculation
        cmc = match_vec.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        # 2. Average Precision (AP) calculation
        d_recall = 1.0 / num_pos
        pos_indices = np.where(filtered_same_id)[0]
        precision_at_k = [(idx + 1) / (pos + 1) for idx, pos in enumerate(pos_indices)]
        ap = sum(precision_at_k) * d_recall
        all_ap.append(ap)

        # 3. Inverse Negative Penalty (INP) calculation
        # R_hardest is the 1-indexed rank of the last (hardest) true match
        r_hardest = int(pos_indices[-1]) + 1
        inp = float(num_pos) / float(r_hardest)
        all_inp.append(inp)

    assert num_valid_q > 0, "No valid query samples found!"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    mean_cmc = all_cmc.sum(axis=0) / num_valid_q
    mean_ap = float(np.mean(all_ap))
    mean_inp = float(np.mean(all_inp))

    if return_minp:
        return mean_cmc, mean_ap, mean_inp
    return mean_cmc, mean_ap


def test_synthetic_metrics():
    """Test with known ground-truth distances."""
    print("Testing synthetic metric calculation (CMC, mAP, mINP)...")

    # Case 1: Query 0 matches Gallery 0 (Camera 0 vs Camera 1)
    # Query 1 matches Gallery 1 (Camera 0 vs Camera 1)
    dist_matrix = np.array([
        [0.1, 0.9, 0.8],  # Query 0: closest is Gallery 0 (Match!)
        [0.9, 0.2, 0.7],  # Query 1: closest is Gallery 1 (Match!)
    ])
    q_pids = np.array([1, 2])
    g_pids = np.array([1, 2, 3])
    q_cams = np.array([0, 0])
    g_cams = np.array([1, 1, 1])

    cmc, m_ap, m_inp = eval_reid_metrics(
        dist_matrix, q_pids, g_pids, q_cams, g_cams, max_rank=3, return_minp=True
    )
    print(f"  ✓ Perfect Match Rank-1: {cmc[0]*100:.1f}%, mAP: {m_ap*100:.1f}%, mINP: {m_inp*100:.1f}%")
    assert np.isclose(cmc[0], 1.0), "Rank-1 should be 100%"
    assert np.isclose(m_ap, 1.0), "mAP should be 100%"
    assert np.isclose(m_inp, 1.0), "mINP should be 100%"

    # Case 2: Query 0 matches Gallery 1 instead of Gallery 0 (Rank-2 hit)
    dist_matrix_suboptimal = np.array([
        [0.8, 0.2, 0.9],  # Query 0: closest is Gallery 1 (Wrong PID 2), 2nd is Gallery 0 (PID 1)
    ])
    cmc_sub, m_ap_sub, m_inp_sub = eval_reid_metrics(
        dist_matrix_suboptimal, np.array([1]), np.array([1, 2, 3]), np.array([0]), np.array([1, 1, 1]),
        max_rank=3, return_minp=True
    )
    print(f"  ✓ Suboptimal Match Rank-1: {cmc_sub[0]*100:.1f}%, Rank-2: {cmc_sub[1]*100:.1f}%, mAP: {m_ap_sub*100:.1f}%, mINP: {m_inp_sub*100:.1f}%")
    assert np.isclose(cmc_sub[0], 0.0), "Rank-1 should be 0%"
    assert np.isclose(cmc_sub[1], 1.0), "Rank-2 should be 100%"
    assert np.isclose(m_ap_sub, 0.5), "mAP should be 50% (1 match at position 2 = 1/2)"
    assert np.isclose(m_inp_sub, 0.5), "mINP should be 50% (1 positive / rank 2)"

    # Case 3: Multi-match query where 1st match is Rank-1 and hardest match is Rank-4
    # Gallery order by distance: [G0(hit), G1(miss), G2(miss), G3(hit)]
    dist_matrix_hardest = np.array([[0.1, 0.2, 0.3, 0.4]])
    cmc_h, m_ap_h, m_inp_h = eval_reid_metrics(
        dist_matrix_hardest,
        np.array([7]),
        np.array([7, 8, 9, 7]),
        np.array([0]),
        np.array([2, 2, 2, 3]),
        max_rank=4,
        return_minp=True,
    )
    print(f"  ✓ Hardest Match Rank-1: {cmc_h[0]*100:.1f}%, mAP: {m_ap_h*100:.1f}%, mINP: {m_inp_h*100:.1f}%")
    assert np.isclose(cmc_h[0], 1.0)
    assert np.isclose(m_ap_h, 0.75)  # (1/1 + 2/4) / 2 = 0.75
    assert np.isclose(m_inp_h, 0.50)  # 2 positives / rank 4 = 0.50

    print("All synthetic metric tests PASSED!")


if __name__ == "__main__":
    test_synthetic_metrics()

