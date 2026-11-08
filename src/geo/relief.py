"""Mapa de relevo: hipsometria + sombreamento (hillshade)."""

import numpy as np

# Rampa hipsométrica: verde (baixo) -> amarelo -> marrom -> branco (alto).
COLOR_STOPS = [
    (0.00, (34, 102, 51)),
    (0.25, (120, 170, 90)),
    (0.50, (220, 210, 120)),
    (0.75, (170, 120, 80)),
    (1.00, (245, 245, 245)),
]


def _interpolate_ramp(position: np.ndarray) -> np.ndarray:
    """Interpola a rampa de cores para posições 0..1."""
    position = np.clip(position, 0.0, 1.0)
    output = np.zeros((*position.shape, 3), dtype=np.float32)

    for index in range(len(COLOR_STOPS) - 1):
        start_pos, start_color = COLOR_STOPS[index]
        end_pos, end_color = COLOR_STOPS[index + 1]
        mask = (position >= start_pos) & (position <= end_pos)
        if not mask.any():
            continue

        local = (position[mask] - start_pos) / max(end_pos - start_pos, 1e-6)
        for channel in range(3):
            output[..., channel][mask] = (
                np.array(start_color[channel], dtype=np.float32) * (1 - local)
                + np.array(end_color[channel], dtype=np.float32) * local
            )

    return output


def make_relief_rgb(
    elevation: np.ndarray,
    hillshade: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Gera RGB de relevo com hipsometria e sombreamento."""
    elevation = np.asarray(elevation, dtype=np.float32)
    vmin = float(np.nanmin(elevation)) if vmin is None else float(vmin)
    vmax = float(np.nanmax(elevation)) if vmax is None else float(vmax)
    span = max(vmax - vmin, 1e-6)

    position = (elevation - vmin) / span
    rgb = _interpolate_ramp(position)

    if hillshade is not None:
        shade = np.asarray(hillshade, dtype=np.float32)
        shade = np.clip(shade / 255.0, 0.0, 1.0)
        # Preserva a cor e aplica sombreamento suave (55%..100%).
        rgb = rgb * (0.55 + 0.45 * shade[..., None])

    return np.clip(rgb, 0, 255).astype(np.uint8)


def elevation_legend(
    vmin: float, vmax: float, n_bins: int = 5
) -> list[dict]:
    """Itens de legenda de elevação (cor + faixa em metros)."""
    if vmax <= vmin:
        vmax = vmin + 1.0
    step = (vmax - vmin) / n_bins
    legend = []
    for index in range(n_bins):
        low = vmin + index * step
        high = vmin + (index + 1) * step
        midpoint = (low + high) / 2
        position = (midpoint - vmin) / (vmax - vmin)
        color = _interpolate_ramp(np.array([position]))[0].astype(int)
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        legend.append(
            {
                "label": f"{low:.0f}–{high:.0f} m",
                "color": hex_color,
            }
        )
    return legend
