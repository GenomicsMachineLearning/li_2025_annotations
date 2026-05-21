"""Process spaceranger outputs + SVG annotations → per-sample CSVs."""
import base64, io, re
import scanpy
import numpy
import svgelements
import shapely
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "samples" / "Li_2025"
SPACERANGER_DIR = DATA_ROOT / "spaceranger_output"
SVG_DIR = DATA_ROOT / "Images" / "Manual_annotation"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

CATEGORIES = ["tumor", "immune", "DCIS", "blood_vessel", "necrosis"]
PALETTE = {
    "tumor": "#e41a1c",
    "immune": "#377eb8",
    "DCIS": "#4daf4a",
    "blood_vessel": "#984ea3",
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

def load_visium_raw(filename, library_id="sample"):
    adata = scanpy.read_visium(path=filename, library_id=library_id)
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

def path_to_polygons(element):
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

def parse_svg(svg_file):
    svg = svgelements.SVG.parse(svg_file)
    annotations = []
    matrix_transform = None
    element_height = None
    element_width = None
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
    return annotations, matrix_transform, element_height, element_width

def assign_annotations(visium_file, annotations, matrix_transform,
                       svg_file, priority=None, default="none"):
    # 1. Extract embedded PNG to find tissue content bounds
    with open(svg_file) as f:
        svg_text = f.read()
    m = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', svg_text)
    embedded = numpy.array(Image.open(io.BytesIO(base64.b64decode(m.group(1)))))
    rgb = embedded[..., :3]
    col_std = rgb.std(axis=(0, 2))
    row_std = rgb.std(axis=(1, 2))
    cx0, cx1 = numpy.where(col_std > 5)[0][[0, -1]]
    cy0, cy1 = numpy.where(row_std > 5)[0][[0, -1]]
    content_w = cx1 - cx0 + 1
    content_h = cy1 - cy0 + 1

    # 2. Lowres image dimensions and scale factor
    lib = list(visium_file.uns['spatial'].keys())[0]
    lowres_scalef = visium_file.uns['spatial'][lib]['scalefactors']['tissue_lowres_scalef']
    lowres_img = visium_file.uns['spatial'][lib]['images']['lowres']
    spot_diameter_fullres = visium_file.uns['spatial'][lib]['scalefactors']['spot_diameter_fullres']
    lowres_h, lowres_w = lowres_img.shape[:2]

    # 3. fullres -> content-region pixels in embedded image
    scalef_x = lowres_scalef * (content_w / lowres_w)
    scalef_y = lowres_scalef * (content_h / lowres_h)

    # 4. Spot coords: fullres -> content-pixels -> embedded-pixels -> SVG space
    spots_full = numpy.asarray(visium_file.obsm['spatial'])
    spots_in_content = spots_full * numpy.array([scalef_x, scalef_y])
    spots_in_embedded = spots_in_content + numpy.array([cx0, cy0])

    M = numpy.array([[matrix_transform.a, matrix_transform.c, matrix_transform.e],
                  [matrix_transform.b, matrix_transform.d, matrix_transform.f]])
    ones = numpy.ones((spots_in_embedded.shape[0], 1))
    spots_svg = numpy.hstack([spots_in_embedded, ones]) @ M.T

    # 5. Build spatial index + assign
    labels = [lbl for lbl, _ in annotations]
    polys  = [poly for _, poly in annotations]
    tree = shapely.strtree.STRtree(polys)

    if priority is None:
        priority = ["DCIS", "necrosis", "blood_vessel", "immune", "tumor"]
    priority_rank = {lbl: i for i, lbl in enumerate(priority)}

    result = numpy.full(spots_svg.shape[0], "", dtype=object)  # default to empty string
    for i, (sx, sy) in enumerate(spots_svg):
        pt = shapely.geometry.Point(sx, sy)
        for j in tree.query(pt):
            if polys[j].contains(pt):
                current = result[i]
                if current == "" or priority_rank.get(labels[j],
                                                      1e9) < priority_rank.get(current,
                                                                               1e9):
                    result[i] = labels[j]

    visium_file.obs['annotation'] = pandas.Categorical(result)
    return visium_file

def process_sample(sample_id: str, spaceranger_outs: Path, svg_path: Path) -> pandas.DataFrame:
    adata = load_visium_raw(spaceranger_outs)
    annotations, matrix_transform, element_height, element_width = parse_svg(svg_path)
    adata = assign_annotations(adata, annotations, matrix_transform, svg_path)
    print(f"Processing sample {sample_id} {spaceranger_outs} {svg_path}")
    return adata

def write_out_csv(sample_id: str, adata, index_name="Barcode",
                  annotation_column="Morphological Annotation"):
    out = adata.obs[["annotation"]].copy()
    out.index.name = index_name
    out = out.rename(columns={"annotation": annotation_column})
    out.to_csv(OUT_DIR / f"{sample_id}.csv", index=True, na_rep="")

def plot_result(sample_id: str, adata):
    ann = adata.obs["annotation"].astype(str)
    ann = ann.where(ann.isin(CATEGORIES), other=numpy.nan)
    adata.obs["annotation"] = pandas.Categorical(ann, categories=CATEGORIES)
    adata.uns["annotation_colors"] = [PALETTE[c] for c in CATEGORIES]
    adata = adata[adata.obs["annotation"].notna()].copy()

    scanpy.pl.spatial(
        adata,
        color="annotation",
        spot_size=100,
        alpha=0.9,
        legend_loc="right margin",
        na_in_legend=False,
        show=False,
    )
    plt.savefig(OUT_DIR / f"{sample_id}.png", dpi=100, bbox_inches="tight")
    plt.close()

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
        adata = process_sample(sample_id, outs, svg)
        write_out_csv(sample_id, adata)
        plot_result(sample_id, adata)

    if missing:
        print("\nIssues:")
        print("\n".join(missing))
        return 1
    print(f"\nProcessed {len(SAMPLES)} samples to directory {OUT_DIR}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())