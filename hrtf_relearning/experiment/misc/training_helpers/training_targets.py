"""1) Small geometry helpers (circular azimuth; mixed az/el distance)"""
import numpy
import logging

def _wrap180(a):
    """Wrap degrees to (-180, 180]."""
    return (numpy.asarray(a) + 180.0) % 360.0 - 180.0

def _wrap_diff_deg(a, b):
    """Smallest signed difference a-b on circle, in [-180,180)."""
    return (numpy.asarray(a) - numpy.asarray(b) + 180.0) % 360.0 - 180.0

def _az_el_distance_deg(p, q):
    """Euclidean distance with circular azimuth and linear elevation (degrees)."""
    daz = _wrap_diff_deg(p[0], q[0])
    delv = (p[1] - q[1])
    return float(numpy.hypot(daz, delv))


"""2) Build sector → candidate index lists (intersected with training ranges)"""
def _sources_in_sector_indices(sources_vp, center_az, center_el, az_size, el_size):
    """Return indices of sources within the rectangular sector around (az,el)."""
    src_az = sources_vp[:, 0]  # typically 0..360 for vertical_polar
    src_el = sources_vp[:, 1]
    half_az, half_el = az_size / 2.0, el_size / 2.0
    d_az = numpy.abs(_wrap_diff_deg(src_az, center_az))
    az_ok = d_az <= half_az
    el_ok = (src_el >= (center_el - half_el)) & (src_el <= (center_el + half_el))
    return numpy.nonzero(az_ok & el_ok)[0]

def _mask_training_ranges(sources_vp, az_range, ele_range):
    """Boolean mask for sources within training az/el ranges (az is circular)."""
    src_az = sources_vp[:, 0]
    src_el = sources_vp[:, 1]
    # normalize az_range to [0,360) and handle wrap
    az0 = az_range[0] % 360
    az1 = az_range[1] % 360
    if az0 <= az1:
        az_mask = (src_az >= az0) & (src_az <= az1)
    else:  # wrapped range
        az_mask = (src_az >= az0) | (src_az <= az1)
    el_mask = (src_el >= ele_range[0]) & (src_el <= ele_range[1])
    return az_mask & el_mask

def _build_sector_candidate_indices(hrir, sector_centers, sector_size, train_az_range, train_ele_range):
    """
    Precompute candidate indices per sector, limited to training ranges.
    Returns: list of 1D numpy arrays of indices into hrir.sources.vertical_polar.
    """
    sources = hrir.sources.vertical_polar  # (N, 3) with az, el, r
    in_train = _mask_training_ranges(sources, train_az_range, train_ele_range)
    candidates_per_sector = []
    for (caz, cel) in sector_centers:
        idx = _sources_in_sector_indices(sources, caz, cel, sector_size[0], sector_size[1])
        idx = idx[in_train[idx]]
        candidates_per_sector.append(idx)
    return candidates_per_sector

"""3) Probabilistic sector sampling with min-distance target picking"""
def _sample_sector_index(probs):
    """Draw a sector index from a probability vector; uniform fallback if needed."""
    p = numpy.asarray(probs, dtype=float)
    if p.ndim != 1 or p.size == 0 or not numpy.isfinite(p).all() or p.sum() <= 0:
        p = numpy.ones_like(p) / max(1, p.size)
    else:
        p = p / p.sum()
    return int(numpy.random.choice(len(p), p=p))

def _pick_target_from_sector(sources_vp, sector_indices, prev_target_deg, min_dist_deg, max_tries=200):
    """
    Choose a random source within a sector that is at least min_dist away from prev_target.
    prev_target_deg is expected in (-180,180] az, elevation in degrees.
    Returns: (az, el) in (-180,180] × (linear)
    """
    if len(sector_indices) == 0:
        return None
    for _ in range(max_tries):
        i = int(numpy.random.randint(len(sector_indices)))
        az0, el0 = sources_vp[sector_indices[i], :2]  # az in 0..360
        az = _wrap180(az0)  # convert to (-180,180] to be consistent with your pipeline
        el = el0
        if prev_target_deg is None or _az_el_distance_deg(prev_target_deg, (az, el)) >= min_dist_deg:
            return (float(az), float(el))
    return None  # couldn’t satisfy min distance within this sector

"4) New set_target that uses response_errors from target_p(...)"
def set_target_probabilistic(target, settings, sequence, hrir, max_sector_hops=10):
    """
    Pick next target using per-sector probabilities (response_errors[:,3]),
    constrained by training az/el ranges and min distance to the previous target.

    Writes the chosen (az, el) in (-180,180] × deg into the shared `target`.
    """
    min_dist = settings['min_dist']
    if not sequence:  # return default if no sequence was detected
        logging.debug('Could not load target probabilities. Returning random target.')
        return set_target(target, settings, hrir)

    sources = hrir.sources.vertical_polar
    try:
        sector_centers = sequence.response_errors[:, :2]
    except AttributeError:
        logging.debug('Could not load target probabilities. Returning random target.')
        return set_target(target, settings, hrir)

    sector_size = sequence.settings['sector_size']

    # training ranges
    train_az = settings['az_range']
    train_el = settings['ele_range']

    # build candidate indices per sector (filtered by training area)
    candidates_per_sector = _build_sector_candidate_indices(
        hrir, sector_centers, sector_size, train_az, train_el
    )

    # probabilities from response_errors
    probs = sequence.response_errors[:, 3]
    prev_tar = target[:]  # mp.Array('f', [az, el])
    prev = (float(prev_tar[0]), float(prev_tar[1])) if not (prev_tar[0] == 0 and prev_tar[1] == 0) else None

    # Try up to max_sector_hops sectors sampled by probability
    for _ in range(max_sector_hops):
        s = _sample_sector_index(probs)
        cand_idx = candidates_per_sector[s]
        picked = _pick_target_from_sector(sources, cand_idx, prev, min_dist)
        if picked is not None:
            # success
            target[:] = picked  # already (-180,180] az; el linear
            logging.info("Set Target (prob) to [%.1f, %.1f] in sector (%.1f, %.1f)",
                         picked[0], picked[1], sector_centers[s,0], sector_centers[s,1])
            return

        # soft backoff: temporarily zero out this sector prob and renormalize
        if probs.sum() > 0:
            probs = probs.copy()
            probs[s] = 0.0
            if probs.sum() == 0:
                break
            probs /= probs.sum()

    # Fallback: uniform over *all* candidates that meet min_dist
    logging.warning("Probabilistic sector selection failed to satisfy min_dist; falling back to uniform.")
    all_idx = numpy.concatenate([idx for idx in candidates_per_sector if len(idx) > 0]) \
              if any(len(idx) > 0 for idx in candidates_per_sector) else numpy.arange(sources.shape[0])
    numpy.random.shuffle(all_idx)
    for i in all_idx:
        az = _wrap180(sources[i, 0]); el = sources[i, 1]
        if prev is None or _az_el_distance_deg(prev, (az, el)) >= min_dist:
            target[:] = (float(az), float(el))
            logging.info("Fallback Set Target to [%.1f, %.1f]", az, el)
            return

    # Last resort: keep previous target (should be extremely rare)
    logging.error("No valid target found given min_dist; keeping previous target.")

""" Fallback if no sequence exists """
def set_target(target, settings, hrir):
    logging.debug(f'Setting target...')
    sources = hrir.sources.vertical_polar
    az_range = settings['az_range']
    ele_range = settings['ele_range']
    az_range = (az_range[0] % 360, az_range[1] % 360)
    az_mask = ((sources[:, 0] >= az_range[0]) & (sources[:, 0] <= az_range[1])) if az_range[0] <= az_range[1]\
        else ((sources[:, 0] >= az_range[0]) | (sources[:, 0] <= az_range[1]))
    el_mask = (sources[:, 1] >= ele_range[0]) & (sources[:, 1] <= ele_range[1])
    candidates = sources[az_mask & el_mask, :2]
    if candidates.shape[0] == 0:
        raise RuntimeError("No HRIR positions within the given ranges!")
    prev_tar = target[:]
    while True:
        next_tar = candidates[numpy.random.randint(len(candidates))]
        if numpy.linalg.norm(numpy.subtract(prev_tar, next_tar)) >= settings['min_dist']:
            break
    next_tar[0] = (next_tar[0] + 180) % 360 - 180
    target[:] = next_tar
    logging.info("Fallback Set Target to [%.1f, %.1f]" % (next_tar[0], next_tar[1]))
