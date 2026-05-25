"""Process spaceranger outputs + SVG annotations → per-sample CSVs."""
import base64, io, re

import squidpy
import numpy
import svgelements
import shapely
from pathlib import Path
from PIL import Image
import pandas

# Silence var_names_make_unique warning
import warnings

from anndata import AnnData

warnings.filterwarnings("ignore", message="Variable names are not unique")

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "samples" / "Li_2025"
SPACERANGER_DIR = DATA_ROOT / "spaceranger_output"
SVG_DIR = DATA_ROOT / "Images" / "Manual_annotation"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

CATEGORIES = ["tumor", "immune", "DCIS", "blood_vessel", "necrosis"]
PALETTE = {
    "tumor": "#e41a1c",
    "immune": "#ffc803",
    "DCIS": "#20009d",
    "blood_vessel": "#3ecbea",
    "necrosis": "#000000",
}

# sample_id -> annotation SVG filename
SAMPLES: dict[str, str] = {
    "V10F24-112_C1": "200616_BC-SA2_V10F24-112_KT.C1_annotated.svg",
    "V10F24-112_D1": "200616_BC-SA2_V10F24-112_KT.D1_annotated.svg",

    "V10F24-113_A1": "200715_BC_SA2_C_D_V10F24-113.A1_annotated.svg",
    "V10F24-114_C1": "200715_BC_SA2_D_E_V10F24-114.C1_annotated.svg",

    # V19T26-012: V1-V4 correspond to wells A1-D1
    "V19T26-012_A1": "200116_ST-AR_BC_A_B_V19T26-012_KT_V1_annotated.svg",
    "V19T26-012_B1": "200116_ST-AR_BC_A_B_V19T26-012_KT_V2_annotated.svg",
    "V19T26-012_C1": "200116_ST-AR_BC_A_B_V19T26-012_KT_V3_annotated.svg",
    "V19T26-012_D1": "200116_ST-AR_BC_A_B_V19T26-012_KT_V4_annotated.svg",

    # V19T26-031: only B1 and C1 exist (V2, V3)
    "V19T26-031_B1": "200219_BC_SA3_V19T26-031_CE_V2_annotated.svg",
    "V19T26-031_C1": "200219_BC_SA3_V19T26-031_CE_V3_annotated.svg",

    # V19T26-032: V1-V4 correspond to wells A1-D1
    "V19T26-032_A1": "200221_BC_SA3_C-D_V19T26-032_CE_V1_annotated.svg",
    "V19T26-032_B1": "200221_BC_SA3_C-D_V19T26-032_CE_V2_annotated.svg",
    "V19T26-032_C1": "200221_BC_SA3_C-D_V19T26-032_CE_V3_annotated.svg",
    "V19T26-032_D1": "200221_BC_SA3_C-D_V19T26-032_CE_V4_annotated.svg",
}

COLOR_TO_CLASS = {
    # Red
    "#ff0000": "tumor",
    "#c80000": "tumor",
    # Yellow
    "#ffc803": "immune",
    "#ffc809": "immune",
    "#ffc800": "immune",
    "#ffc802": "immune",
    # Dark Blue
    "#20009d": "DCIS",
    "#0001b3": "DCIS",
    "#0000b4": "DCIS",
    "#0000b3": "DCIS",
    # Black
    "#000000": "necrosis",
    # Light/Sky Blue
    "#3ecbea": "blood_vessel",
    "#42cbeb": "blood_vessel",
    "#00ffff": "blood_vessel",
}


def load_visium_raw(filename, library_id="sample") -> AnnData:
    adata = squidpy.read.visium(path=filename, library_id=library_id)
    adata.var_names_make_unique()
    return adata


def normalize_hex(c):
    """#f00 -> #ff0000, handles svgelements Color objects."""
    if c is None:
        return None
    s = str(svgelements.Color(c)).lower()
    return s


def img_to_svg(mt, x_img, y_img):
    return (mt.a * x_img + mt.c * y_img + mt.e,
            mt.b * x_img + mt.d * y_img + mt.f)


def path_to_polygons(element) -> list[shapely.geometry.Polygon]:
    """Robust path -> polygon list."""
    polys = []
    for subpath in element.as_subpaths():
        # subpath is itself iterable yielding segments
        pts = []
        for seg in subpath:
            # Move starts a ring; Line/Close contribute end points
            end = getattr(seg, "end", None)
            if end is not None:
                pts.append((end.x, end.y))
        if len(pts) < 3:
            continue
        poly = shapely.geometry.Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        # Explode MultiPolygon results
        if poly.is_empty:
            continue
        if isinstance(poly, shapely.geometry.MultiPolygon):
            polys.extend(list(poly.geoms))
        else:
            polys.append(poly)
    return polys


def parse_svg(svg_file: Path) -> tuple[
    list[tuple[str, shapely.geometry.Polygon]],
    svgelements.Matrix,
    float | int,
    float | int
]:
    svg = svgelements.SVG.parse(svg_file)
    annotations: list[tuple[str, shapely.geometry.Polygon]] = []
    matrix_transform: svgelements.Matrix | None = None
    element_height: float | int | None = None
    element_width: float | int | None = None
    for element in svg.elements():
        if isinstance(element, svgelements.Use):
            matrix_transform = element.transform
            element_height = element.height
            element_width = element.width
        if isinstance(element, svgelements.Path):
            stroke = normalize_hex(element.stroke) if element.stroke else None
            fill = normalize_hex(element.fill) if element.fill else None
            label = COLOR_TO_CLASS.get(stroke) or COLOR_TO_CLASS.get(fill)
            if label is None:
                continue
            for poly in path_to_polygons(element):
                annotations.append((label, poly))

    if matrix_transform is None or element_height is None or element_width is None:
        raise ValueError(f"No <use> element found in {svg_file}")
    return annotations, matrix_transform, element_height, element_width


def _content_bounds(svg_file: Path) -> tuple[int, int, int, int]:
    with open(svg_file) as f:
        svg_text = f.read()
    m = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', svg_text)
    if m is None:
        raise ValueError(f"No embedded PNG found in {svg_file}")
    embedded = numpy.array(Image.open(io.BytesIO(base64.b64decode(m.group(1)))))
    rgb = embedded[..., :3]
    col_std = rgb.std(axis=(0, 2))
    row_std = rgb.std(axis=(1, 2))
    min_x, max_x = numpy.where(col_std > 5)[0][[0, -1]]
    min_y, max_y = numpy.where(row_std > 5)[0][[0, -1]]
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    return height, width, min_x, min_y


def _get_spatial_metadata(anndata: AnnData) -> dict[str, float | int]:
    lib = list(anndata.uns['spatial'].keys())[0]
    lowres_scalef = anndata.uns['spatial'][lib]['scalefactors'][
        'tissue_lowres_scalef']
    lowres_img = anndata.uns['spatial'][lib]['images']['lowres']
    spot_diameter_fullres = anndata.uns['spatial'][lib]['scalefactors'][
        'spot_diameter_fullres']
    lowres_h, lowres_w = lowres_img.shape[:2]
    return {
        'lowres_scalef': lowres_scalef,
        'spot_diameter_fullres': spot_diameter_fullres,
        'lowres_w': lowres_w,
        'lowres_h': lowres_h,
    }


def _spots_to_svg(min_x: float | int, min_y: float | int,
                  matrix_transform: svgelements.Matrix,
                  scalef_x, scalef_y,
                  anndata: AnnData) -> numpy.ndarray:
    spots_full = numpy.asarray(anndata.obsm['spatial'])
    spots_in_content = spots_full * numpy.array([scalef_x, scalef_y])
    spots_in_embedded = spots_in_content + numpy.array([min_x, min_y])
    matrix = numpy.array([[matrix_transform.a, matrix_transform.c, matrix_transform.e],
                          [matrix_transform.b, matrix_transform.d, matrix_transform.f]])
    ones = numpy.ones((spots_in_embedded.shape[0], 1))
    return numpy.hstack([spots_in_embedded, ones]) @ matrix.T


def _assign_labels(annotations, priority, spot_radius, spots_svg) \
    -> numpy.ndarray:
    labels, polys = zip(*annotations) if annotations else ([], [])
    polys = list(polys)
    tree = shapely.strtree.STRtree(polys)
    rank = {lbl: i for i, lbl in enumerate(priority)}
    max_priority = len(priority)

    result = numpy.full(len(spots_svg), "", dtype=object)
    for i, (sx, sy) in enumerate(spots_svg):
        disc = shapely.geometry.Point(sx, sy).buffer(spot_radius)
        best_rank = max_priority + 1
        for j in tree.query(disc):
            j = int(j)
            if polys[j].intersects(disc):
                r = rank.get(labels[j], max_priority)
                if r < best_rank:
                    best_rank = r
                    result[i] = labels[j]
    return result


def assign_annotations(anndata: AnnData,
                       annotations: list[tuple[str, shapely.geometry.Polygon]],
                       matrix_transform: svgelements.Matrix,
                       svg_file: Path, priority=None) -> AnnData:
    if priority is None:
        priority = ["DCIS", "necrosis", "blood_vessel", "immune", "tumor"]

    # 1. Extract embedded PNG to find tissue content bounds
    height, width, start_x, start_y = _content_bounds(svg_file)

    # 2. Lowres image dimensions and scale factor
    sp_metadata = _get_spatial_metadata(anndata)

    # 3. fullres -> content-region pixels in embedded image
    scalef_x = sp_metadata["lowres_scalef"] * (width / sp_metadata["lowres_w"])
    scalef_y = sp_metadata["lowres_scalef"] * (height / sp_metadata["lowres_h"])
    fullres_to_svg_x = scalef_x * abs(matrix_transform.a)
    fullres_to_svg_y = scalef_y * abs(matrix_transform.d)
    fullres_to_svg = (fullres_to_svg_x + fullres_to_svg_y) / 2
    spot_radius = (sp_metadata["spot_diameter_fullres"] / 2) * fullres_to_svg

    # 4. Spot coords: fullres -> content-pixels -> embedded-pixels -> SVG space
    spots_svg = _spots_to_svg(start_x, start_y, matrix_transform, scalef_x, scalef_y,
                              anndata)

    # 5. Build spatial index + assign
    result = _assign_labels(annotations, priority, spot_radius, spots_svg)

    anndata.obs['annotation'] = pandas.Categorical(result)
    return anndata


def process_sample(sample_id: str, spaceranger_outs: Path,
                   svg_path: Path) -> AnnData:
    anndata = load_visium_raw(spaceranger_outs)
    annotations, matrix_transform, element_height, element_width = parse_svg(svg_path)
    print(f"Processing sample {sample_id} {spaceranger_outs} {svg_path}")
    return assign_annotations(anndata, annotations, matrix_transform, svg_path)


def write_out_csv(sample_id: str, anndata: AnnData, index_name: str = "Barcode",
                  annotation_column: str = "Morphological Annotation"):
    out = anndata.obs[["annotation"]].copy()
    out.index.name = index_name
    out = out.rename(columns={"annotation": annotation_column})
    out.to_csv(OUT_DIR / f"{sample_id}.csv", index=True, na_rep="")


def plot_result(sample_id: str, anndata: AnnData):
    ann = anndata.obs["annotation"].astype(str)
    ann = ann.where(ann.isin(CATEGORIES), other=numpy.nan)
    anndata.obs["annotation"] = pandas.Categorical(ann, categories=CATEGORIES)
    anndata.uns["annotation_colors"] = [PALETTE[c] for c in CATEGORIES]
    anndata = anndata[anndata.obs["annotation"].notna()].copy()

    squidpy.pl.spatial_scatter(
        anndata,
        color="annotation",
        size=1,
        alpha=0.95,
        legend_loc="right margin",
        legend_na=False,
        save=OUT_DIR / f"{sample_id}.png",
        dpi=100,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for sample_id, svg_name in SAMPLES.items():
        outs = SPACERANGER_DIR / sample_id / "outs"
        svg = SVG_DIR / svg_name
        if not outs.is_dir():
            missing.append(f"  missing spaceranger: {outs}")
            continue
        if not svg.is_file():
            missing.append(f"  missing svg: {svg}")
            continue
        anndata = process_sample(sample_id, outs, svg)
        write_out_csv(sample_id, anndata)
        plot_result(sample_id, anndata)

    if missing:
        print("\nIssues:")
        print("\n".join(missing))
        return 1
    print(f"\nProcessed {len(SAMPLES)} samples to directory {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
