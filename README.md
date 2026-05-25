# li_2025_annotations

Convert manual pathology annotations from Li et al. 2025 (provided as SVG files
overlaid on H&E images) into per-spot CSV annotations aligned with 10x Visium
spaceranger output.

## Overview

Each SVG contains hand-drawn polygons (colored by morphological class) over an
embedded low-resolution H&E image. 

| Class         | Color       |
|---------------|-------------|
| tumor         | red         |
| immune        | yellow      |
| DCIS          | dark blue   |
| blood_vessel  | light blue  |
| necrosis      | black       |

This tool:
1. Parses polygons from the SVG and classifies them by stroke/fill color
   (tumor, immune, DCIS, blood vessel, necrosis).
1. Aligns Visium fullres spot coordinates into the SVG's coordinate space using
   the embedded image bounds and the SVG's transform matrix.
1. Assigns each spot a label based on which polygon it falls within, using a
   priority order to resolve overlaps.
1. Writes a per-sample CSV (`Barcode,Morphological Annotation`) and a spatial
   scatter plot.

## Installation

```shell
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```shell
download   # fetches spaceranger outputs and SVG annotations
process    # parses SVGs, assigns annotations, writes CSVs + plots
```

## Paper

> Li, T., Yang, Q., Acs, B. et al. Computational pathology annotation enhances
> the resolution and interpretation of breast cancer spatial transcriptomics
> data. *npj Precis. Oncol.* **9**, 310 (2025).
> https://doi.org/10.1038/s41698-025-01104-3
