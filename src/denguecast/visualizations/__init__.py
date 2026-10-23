"""Visualization package for DengueCast."""

from denguecast.visualizations.real_map import (
    COLOR_MAP,
    DengueMapConfig,
    generate_high_res_static_map,
    generate_interactive_html_map,
    load_real_dengue_data,
)

__all__ = [
    "COLOR_MAP",
    "DengueMapConfig",
    "generate_high_res_static_map",
    "generate_interactive_html_map",
    "load_real_dengue_data",
]
