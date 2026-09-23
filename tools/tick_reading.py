"""Portable tick-count decoder used by the local annotation UI."""

import numpy as np

N_THETA, N_RHO = 720, 96


def _runs(binary):
    delta = np.diff(np.concatenate([[0], binary, [0]]))
    starts = np.where(delta == 1)[0]
    ends = np.where(delta == -1)[0] - 1
    return [(start, end, (start + end) / 2) for start, end in zip(starts, ends)]


def decode(mask, min_ticks=5):
    tick_ys, tick_xs = np.where(mask == 1)
    if len(tick_xs) < 20 or not (mask == 2).any():
        return {"ok": False, "reason": "empty tick or pointer mask"}

    center_x, center_y = tick_xs.mean(), tick_ys.mean()
    outer_radius = np.percentile(
        np.hypot(tick_xs - center_x, tick_ys - center_y), 98
    ) * 1.15
    theta = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    radius = np.linspace(outer_radius, outer_radius * 0.15, N_RHO)
    xs = np.clip(
        (center_x + np.outer(radius, np.cos(theta)) + 0.5).astype(int),
        0,
        mask.shape[1] - 1,
    )
    ys = np.clip(
        (center_y + np.outer(radius, np.sin(theta)) + 0.5).astype(int),
        0,
        mask.shape[0] - 1,
    )
    polar = mask[ys, xs]
    tick_line = (polar == 1).sum(0).astype(float)
    pointer_line = (polar == 2).sum(0).astype(float)
    tick_binary = (tick_line > tick_line.mean()).astype(int)
    pointer_binary = (pointer_line > pointer_line.mean()).astype(int)

    if not np.where(tick_binary == 0)[0].size:
        return {"ok": False, "reason": "ticks everywhere"}
    gaps = _runs(1 - tick_binary)
    largest_gap = max(gaps, key=lambda run: run[1] - run[0])
    shift = -(largest_gap[1] + 1) % N_THETA
    tick_binary = np.roll(tick_binary, shift)
    pointer_binary = np.roll(pointer_binary, shift)

    ticks = [run[2] for run in _runs(tick_binary)]
    if len(ticks) < min_ticks:
        return {"ok": False, "reason": f"only {len(ticks)} ticks"}
    pointer_runs = _runs(pointer_binary)
    if not pointer_runs:
        return {"ok": False, "reason": "no pointer run"}
    pointer = max(pointer_runs, key=lambda run: run[1] - run[0])[2]

    if pointer < ticks[0]:
        pointed = 0.0
    elif pointer >= ticks[-1]:
        pointed = float(len(ticks) - 1)
    else:
        index = int(np.searchsorted(ticks, pointer)) - 1
        pointed = index + (pointer - ticks[index]) / (
            ticks[index + 1] - ticks[index] + 1e-9
        )
    return {
        "ok": True,
        "n_ticks": len(ticks),
        "pointed": pointed,
        "frac": pointed / (len(ticks) - 1),
    }