---
title: "COMPASS annotations"
subtitle: "Guide and reference for `compass.core.annot`"
author: "Vlad Popovici"
format:
  html:
    toc: true
    toc-depth: 3
    toc-location: left
    number-sections: true
    code-copy: true
    embed-resources: true
    theme: cosmo
---

Annotations are vector objects drawn on a slide: cells (points),
regions (polygons), tracks (polylines), Visium spots (circles), each with
optional scalar attributes such as a molecular expression profile. They are
organized in **layers** containing **groups**; an object belongs to one
layer, through one or more of its groups. All of it lives in
`compass.core.annot`, which depends only on core's own dependencies
(numpy, shapely, geopandas, polars, pandas, pydantic).

This document covers the API, the file format and the port from the legacy
modules. The general developer documentation is [`index.md`](index.md); the
*why* behind the storage design (SQLite + R-tree, vectorized access) is
[`architecture.md`](architecture.md), §4.

Every example below was run while writing this document; output blocks are
real output.

# Overview

## Three ways to work with annotations

| Level | Use for | API |
|---|---|---|
| **Objects** | building or inspecting a handful of objects | `Point`, `PointSet`, `PolyLine`, `Polygon`, `Circle` |
| **Frames** | bulk work in memory: thousands to millions of objects | geopandas GeoDataFrames + `compass.core.annot.transform` |
| **Store** | persistence, spatial queries, editing on disk | `AnnotationStore` (SQLite + R-tree) |

The store reads and writes frames directly. Objects are converted only when
asked for (`fetch_objects`, `objects_to_gdf`, `gdf_to_objects`), so bulk
paths never create one Python object per annotation.

The legacy API (`Annotation`, `save_annotation`, `load_annotation`,
`query_*`, `fetch_*`) is still available, built on the store; see
[Legacy API](#legacy-api).

## Modules

| Module | Contents |
|---|---|
| `annot.objects` | the object classes, `make_annotation_object`, `create_empty_annotation_object`, `objects_to_gdf`, `gdf_to_objects` |
| `annot.transform` | `translate`, `scale`, `resize`, `affine` over GeoDataFrames; matrix helpers |
| `annot.store` | `AnnotationStore` |
| `annot.schema` | `SQLITE_DDL`, `SCHEMA_VERSION`, `ANNOT_TYPE_CODE`, `ANNOT_CODE_TYPE`, `init_db`, `geometry_type_names` |
| `annot.compat` | the legacy `Annotation` class and module functions |
| `connector.asap` | `read_asap_xml`, `asap_to_store`, `AsapTarget`: import of ASAP XML annotations |

Everything is importable from `compass.core.annot`; the most used names
(`AnnotationStore`, `Annotation`, the object classes, the schema constants)
also from `compass.core`.

## Conventions

- **Coordinates are `(x, y)`** pixels, never (row, column). Store
  coordinates are those of the image recorded in the store's metadata
  (`image_shape`, `mpp`) — normally level 0 of the slide's pyramid.
- **Types**: `POINT`, `POINTSET` (multipoint), `POLYLINE` (linestring or
  multilinestring), `POLYGON` (polygon or multipolygon, holes allowed),
  `CIRCLE`.
- **Circles** are exact: center + radius. In memory a `Circle`'s `geom` is
  its polygonal disk (64 vertices); the store keeps the center point plus
  `circle_r`. Only similarity transforms keep a circle a circle — see
  [Circles under transforms](#circles-under-transforms).
- **Scalar attributes** are float32, sparse per object: an object stores
  only the attributes it has (e.g. ~3,000 of ~20,000 genes).

# Annotation objects

```python
from compass.core.annot import Point, PointSet, PolyLine, Polygon, Circle

p = Point((120.0, 80.0), name="cell 1", data={"CD3": 1.5, "area": 42})
roi = Polygon([(0, 0), (400, 0), (400, 300), (0, 300)], name="ROI A")
spot = Circle((250.0, 150.0), 55.0, name="spot 17")

print(p)                                 # POINT <cell 1>: POINT (120 80)
roi.bounding_box(), roi.size()           # ((0.0, 0.0, 400.0, 300.0), 5)
print(spot, spot.size())                 # CIRCLE <spot 17>: center=(250.0, 150.0), radius=55.0 65
```

Constructors take coordinates (`(x, y)` or `[(x0, y0), (x1, y1), ...]`)
or a ready shapely geometry of the right kind, e.g. a polygon with holes.
Without `name`, the type is used ("POINTSET" objects default to "POINTS").

| Member | Meaning |
|---|---|
| `geom` | the shapely geometry (a circle's disk) |
| `type`, `name`, `data` | type string, name (settable), scalar attributes (`dict`); `set_data(d)` replaces `data` |
| `id` | id in its store; -1 until stored |
| `x`, `y`, `xy()`, `size()` | coordinates (always vectors), (n, 2) array, number of coordinates |
| `bounding_box()` | `(min_x, min_y, max_x, max_y)` |
| `duplicate()` | independent copy (geometry, name, data), no id |
| `asdict()`, `fromdict(d)` | legacy dictionary form (geometry as WKT) |
| `Circle.center`, `.radius`, `.to_polygon()` | exact circle parameters; the disk as a `Polygon` |

## Transforming objects

Transforms act in place:

```python
roi.translate(100, 50)    # y_off defaults to x_off
roi.bounding_box()        # (100.0, 50.0, 500.0, 350.0)
roi.scale(0.5)            # about its own center (default origin="center")
roi.bounding_box()        # (200.0, 125.0, 400.0, 275.0)
roi.resize(2)             # uniform scale about (0, 0): a change of resolution
roi.bounding_box()        # (400.0, 250.0, 800.0, 550.0)

spot.affine([[0, -1, 1000], [1, 0, 0]])   # rotate 90°, shift
spot.center, spot.radius                  # ((850.0, 250.0), 55.0)
spot.scale(2, 1)
# ValueError: non-uniform scaling would deform the circle; use to_polygon() first
```

`scale(x, y=None, origin=...)` takes `"center"` (bounding-box center),
`"centroid"`, a shapely point or an `(x, y)` tuple. `affine(M)` takes a
2 × 3 matrix, row-major, acting on `(x, y, 1)`:
`x' = M[0,0]·x + M[0,1]·y + M[0,2]`, `y' = M[1,0]·x + M[1,1]·y + M[1,2]`
(a 3 × 3 homogeneous matrix works too).

## Between objects and frames

```python
from compass.core.annot import objects_to_gdf, gdf_to_objects

gdf = objects_to_gdf([p, roi, spot])     # columns: type, name, circle_r, geometry
objs = gdf_to_objects(gdf, data={0: {"CD3": 1.5}})
```

`objects_to_gdf` gives the same columns the store returns (circles as
disks unless `circles_as_disks=False`); it is indexed by object id when
every object has one. `gdf_to_objects` reads `type` (inferred from the
geometry if absent), `name` and `circle_r`; `data` maps index labels to
attribute dicts. `make_annotation_object(type, geom, name, data, circle_r)`
wraps one geometry.

# The annotation store

`AnnotationStore` is one SQLite database file per slide (or in memory).

| `mode` | Meaning |
|---|---|
| `"r"` (default) | read-only — what the viewer uses; writes raise `PermissionError` |
| `"r+"` | read and write an existing file |
| `"w"` | create a new file (replacing an existing one); `path=":memory:"` for an in-memory store |

A new store has the default layer `"base"` with group `"no_group"` (both
id 0), the same as the legacy model. Files are opened in WAL mode, so a
read-only store (the viewer) can read while another process writes. Use
the store as a context manager or call `close()`.

## Building a store

```python
import numpy as np, pandas as pd, geopandas as gpd, shapely
from compass.core.annot import AnnotationStore, Polygon, Circle

rng = np.random.default_rng(0)
xy = rng.uniform(0, 6000, size=(1000, 2))
cells = gpd.GeoDataFrame({"name": [f"cell{i}" for i in range(1, 1001)]},
                         geometry=shapely.points(xy))
markers = pd.DataFrame({                         # NaN = attribute absent
    "CD3":   np.where(np.arange(1000) % 2 == 0, rng.random(1000), np.nan),
    "CD8":   np.where(np.arange(1000) % 2 == 1, rng.random(1000), np.nan),
    "PanCK": np.where(np.arange(1000) % 2 == 0, rng.random(1000), np.nan),
})
left = xy[:, 0] < 3000

with AnnotationStore("demo.ann.sqlite", mode="w") as store:
    store.set_meta(name="demo", mpp=0.25, image_width=6000, image_height=6000)
    with store.batch():
        tumor = store.add_objects(cells[left], layer="cells", group="tumor",
                                  data=markers[left].reset_index(drop=True))
        stroma = store.add_objects(cells[~left], layer="cells", group="stroma",
                                   data=markers[~left].reset_index(drop=True))
        regions = store.add_objects(
            [Polygon([(500, 500), (2500, 500), (2500, 2000), (500, 2000)], name="roi A"),
             Circle((4000, 4000), 150.0, name="spot")],
            layer="regions", group="annotated")
    print(tumor[:3], stroma[:3], regions)
```

```
[0 1 2] [487 488 489] [1000 1001]
```

`add_objects(objects, layer="base", group="no_group", data=None)`:

- **`objects`**: object instances, or a GeoDataFrame / GeoSeries. A frame
  may have `type` (otherwise inferred from the geometry), `name` and
  `circle_r` columns. A `CIRCLE` row needs `circle_r` and may hold the
  center point or the disk — so a frame read from the store can be written
  back as it is.
- **`layer`, `group`**: names or ids; a new name is created. `group` can be
  a list: the objects join all of them (all in the same layer).
- **`data`**: per-object scalars aligned with the objects — a sequence of
  dicts (`None` for no values) or a wide pandas/polars frame with one column
  per attribute, where null/NaN means absent. Instances default to their
  own `data`. A value that isn't numeric is skipped with a logged warning.
- **Returns** the new ids, assigned in order. Instances get their `id` set.

`store.batch()` makes the writes inside it a single transaction — all
kept, or none if an exception escapes. Every write method is atomic on its
own; `batch()` matters for speed when there are many small writes.

Metadata: `store.name`, `store.mpp` and `store.image_shape` (an
`ImageShape`) are properties with setters; `set_meta(**kw)` writes any
key, `store.meta` returns them all as text.

## Querying

The read API returns **polars DataFrames** (tables, ids, scalars) or
**geopandas GeoDataFrames indexed by `object_id`** (geometry, decoded with
one `shapely.from_wkb` call per query).

```python
store = AnnotationStore("demo.ann.sqlite")          # read-only
store.get_layers()             # layer_id, name
store.get_groups(layer_id=1)   # group_id, layer_id, name
store.count_objects_per_group()
```

```
shape: (4, 4)
┌──────────┬──────────┬────────────┬───────────┐
│ group_id ┆ layer_id ┆ group_name ┆ n_objects │
│ ---      ┆ ---      ┆ ---        ┆ ---       │
│ i64      ┆ i64      ┆ str        ┆ i64       │
╞══════════╪══════════╪════════════╪═══════════╡
│ 0        ┆ 0        ┆ no_group   ┆ 0         │
│ 1        ┆ 1        ┆ tumor      ┆ 487       │
│ 2        ┆ 1        ┆ stroma     ┆ 513       │
│ 3        ┆ 2        ┆ annotated  ┆ 2         │
└──────────┴──────────┴────────────┴───────────┘
```

Names resolve to ids. Group names are unique only within a layer, so pass
the layer to disambiguate:

```python
cells = store.get_layer_id("cells")                     # 1
tumor = store.get_group_id("tumor", layer_id=cells)     # 1
store.get_layer_id("nuclei")
# ValueError: Unknown layer name 'nuclei'. Available layers: base, cells, regions
```

### Viewport queries

ROIs are `(x0, y0, x1, y1)` and match on bounding-box intersection via the
R-tree (a circle's box includes its radius).

```python
roi = (0, 0, 2000, 2000)
store.count_objects_in_roi(cells, roi)     # 102 — R-tree only, no geometry

gdf = store.get_objects_in_roi(cells, roi, group_ids=[tumor])
gdf.head(3)
```

```
            type    name  circle_r                  geometry
object_id
0          POINT   cell2       NaN    POINT (245.841 99.166)
2          POINT  cell11       NaN     POINT (169.918 745.7)
10         POINT  cell30       NaN  POINT (1383.853 312.128)
```

`group_ids` restricts the query to the visible groups. The columns are
always `type` (categorical), `name`, `circle_r` (NaN except for circles)
and `geometry`.

### Fetching by id

```python
store.get_object_ids_in_layer(cells)        # np.ndarray[int64], 1000 ids
store.get_object_ids_in_group(tumor)        # 487 ids
store.get_object_ids_in_roi(cells, roi)     # array([ 0,  2, 10, 27, 28, ...])

store.get_objects(store.get_object_ids_in_layer(2))
```

```
              type   name  circle_r                                       geometry
object_id
1000       POLYGON  roi A       NaN  POLYGON ((500 500, 2500 500, 2500 2000, 500 20...
1001        CIRCLE   spot     150.0  POLYGON ((4150 4000, 4147.118 3970.736, 4138.5...
```

Circles come back as disks by default, so every geometry draws the same
way. `circles_as_disks=False` keeps the center points (the radius stays in
`circle_r`). Id lists of any length work (queries are chunked below
SQLite's parameter limit).

For a small selection — say, the object under the mouse — `fetch_objects`
returns instances with their data:

```python
store.fetch_objects([0, 1001])
# [POINT <cell2>: POINT (245.84114361716814 99.16581317117456),
#  CIRCLE <spot>: center=(4000.0, 4000.0), radius=150.0]
```

### Bounding boxes and memberships

When geometry isn't needed (density maps, coarse hit-testing), skip
decoding:

```python
store.get_bboxes(layer_id=2)            # object_id, min_x, min_y, max_x, max_y
store.get_memberships([0, 1, 1000])     # object_id, group_id (all if omitted)
```

```
┌───────────┬────────┬────────┬────────┬────────┐
│ object_id ┆ min_x  ┆ min_y  ┆ max_x  ┆ max_y  │
╞═══════════╪════════╪════════╪════════╪════════╡
│ 1000      ┆ 500.0  ┆ 500.0  ┆ 2500.0 ┆ 2000.0 │
│ 1001      ┆ 3850.0 ┆ 3850.0 ┆ 4150.0 ┆ 4150.0 │
└───────────┴────────┴────────┴────────┴────────┘
```

### Scalar attributes

```python
store.get_scalar_attribute_names()      # attr_id, attr_name
store.get_scalar_data_long([0, 1])      # long (COO) form
```

```
┌───────────┬─────────┬──────────┐
│ object_id ┆ attr_id ┆ value    │
╞═══════════╪═════════╪══════════╡
│ 0         ┆ 2       ┆ 0.79497  │
│ 1         ┆ 2       ┆ 0.624738 │
└───────────┴─────────┴──────────┘
```

Wide form, restricted to the attributes needed (with ~20,000 genes, always
restrict):

```python
store.get_scalar_data(gdf.index.to_numpy()[:4], attributes=["CD3", "CD8"])
```

```
┌───────────┬──────────┬──────────┐
│ object_id ┆ CD8      ┆ CD3      │
╞═══════════╪══════════╪══════════╡
│ 0         ┆ 0.79497  ┆ null     │
│ 2         ┆ null     ┆ 0.311247 │
│ 10        ┆ 0.973053 ┆ null     │
│ 27        ┆ null     ┆ 0.162319 │
└───────────┴──────────┴──────────┘
```

Absent values are `null`. Columns follow the order the attributes appear
in the data, so select by name. Objects with no values get no row.
`get_scalar_dicts(ids)` returns `{object_id: {name: value}}`.

## Editing

Open with `mode="r+"`. Two rules hold for every edit:

- **An object without a group is deleted.** An object belongs to its layer
  only through its groups, so one left in none would be invisible to every
  query. `remove_group`, `remove_layer` and `remove_from_group` delete such
  objects and return how many.
- **Ids are never reused.** After deleting the newest objects, new ones
  still get fresh ids, so a stale selection can't point to a different
  object. The last ids handed out are kept in `meta`
  (`last_object_id`, `last_group_id`, `last_layer_id`, the keys the legacy
  writer used).

```python
store = AnnotationStore("demo.ann.sqlite", mode="r+")

store.add_to_group([0, 2, 10], "CD8-high", layer="cells")    # new group
store.remove_from_group([0], "tumor", layer="cells")          # 0 — still in CD8-high
store.remove_from_group([0], "CD8-high", layer="cells")       # 1 — in no group: deleted
store.remove_group("CD8-high", layer="cells")                 # 0 — 2 and 10 stay in tumor

store.remove_objects([1001])                                  # 1
store.add_object(Point((10, 10)), layer="regions", group="annotated")   # 1002, not 1001
```

An object can't join a group of another layer (`ValueError`, enforced by
a schema trigger). Layers and groups: `add_layer`, `add_group`,
`rename_layer`, `rename_group`, `remove_layer`, `remove_group`.

**Geometry and names**: `update_objects` takes a GeoDataFrame indexed by
`object_id` with a geometry and optionally `type`, `name`, `circle_r`.
Bounding boxes and the R-tree follow.

```python
g = store.get_objects([2, 10])
g["geometry"] = g.geometry.translate(5, 5)
g["name"] = ["moved A", "moved B"]
store.update_objects(g[["name", "geometry"]])
```

Without a `type` column, circles stay circles and other objects take the
type of their new geometry.

**Scalars**: `set_scalar_data(ids, data)` merges the given attributes into
what each object already has (`replace=True` replaces the whole vector);
`delete_scalar_data(ids, attributes=None)` removes all or the named ones.

```python
store.set_scalar_data([2, 10], [{"CD3": 0.9, "Ki67": 0.2}, {"Ki67": 0.7}])
store.get_scalar_dicts([2, 10])
# {2: {'CD3': 0.9, 'PanCK': 0.0809, 'Ki67': 0.2}, 10: {'CD8': 0.9731, 'Ki67': 0.7}}
store.set_scalar_data([2], [{"Ki67": 0.3}], replace=True)
store.delete_scalar_data([10], attributes=["Ki67"])
store.get_scalar_dicts([2, 10])
# {2: {'Ki67': 0.3}, 10: {'CD8': 0.9731}}
```

(Values are float32: `0.9` reads back as `0.8999999761581421`.)

## In-memory stores, saving and loading

```python
mem = AnnotationStore(":memory:", mode="w")
mem.add_objects(...)
mem.save_as("slide.ann.sqlite")                       # full copy (SQLite backup API)

work = AnnotationStore.load_in_memory("slide.ann.sqlite")   # edit a copy; file untouched
```

`save_as(path, overwrite=True)` raises `FileExistsError` when
`overwrite=False` and the file exists.

# Transforms

## On frames

`compass.core.annot.transform` returns transformed copies of a
GeoDataFrame, in one numpy pass over all coordinates. Circle radii are
scaled along.

```python
from compass.core.annot import transform

transform.translate(gdf, 100, 0)
transform.scale(gdf, 2)                         # about (0, 0)
transform.scale(gdf, 2, origin="center")        # each object about its own center
transform.resize(gdf, 0.5)                      # = scale about (0, 0)
transform.affine(gdf, [[0.5, 0, -10], [0, 0.5, -10]])
```

## On the store

The same operations run on the whole store (or on `object_ids=`), in
batches of 100,000 objects inside one transaction:

| Method | Effect |
|---|---|
| `translate(dx, dy=None, object_ids=None)` | shift |
| `scale(sx, sy=None, origin=(0, 0), object_ids=None, circles="raise")` | scale about a point, or each object's `"center"` / `"centroid"` |
| `affine(M, object_ids=None, circles="raise")` | 2 × 3 affine matrix |
| `resize(factor)` | scale everything about (0, 0); `mpp /= factor`, `image_shape *= factor` (rounded) |
| `set_resolution(mpp)` | `resize` to a new resolution |
| `match_image(raster, level=0)` | fit to a `PyramidalRaster` level: factor = mean of the width and height ratios, then `image_shape` and `mpp` exactly the raster's |

Only `resize`, `set_resolution` and `match_image` update the metadata.

```python
store.mpp, store.image_shape                 # (0.25, ImageShape(width=6000, height=6000))
store.resize(0.5)
store.mpp, store.image_shape                 # (0.5, ImageShape(width=3000, height=3000))
store.set_resolution(0.25)                   # back
```

## Annotations from the source slide onto an imported pyramid

Ingestion (`wsi2zarr`, `ometiff2zarr`) can crop the slide and start the
pyramid at a lower level. The import record stored with the pyramid
carries the source → output transform as a 2 × 3 matrix with the same
convention as `affine`, so annotations made in source-slide coordinates map
over in one call:

```python
from compass.core import ZarrRaster
from compass.core.annot import AnnotationStore, Point, Circle

raster = ZarrRaster("AP1064-1.zarr")
rec = raster.info["import"]
rec["transform"]
# {'type': 'affine', 'from': 'source_level0', 'to': 'output_level0',
#  'matrix': [[0.5, 0.0, -3120.0], [0.0, 0.5, -5328.0]]}

store = AnnotationStore(":memory:", mode="w")
x0, y0 = rec["crop"]["x0"], rec["crop"]["y0"]
store.add_objects([Point((x0 + 1000, y0 + 2000)), Circle((x0 + 5000, y0 + 5000), 80)],
                  layer="landmarks")
store.affine(rec["transform"]["matrix"])
store.image_shape = raster.shape(0)
store.mpp = raster.get_mpp_for_level(0)
store.get_objects([0, 1], circles_as_disks=False)
```

```
             type  circle_r           geometry
object_id
0           POINT       NaN   POINT (500 1000)
1          CIRCLE      40.0  POINT (2500 2500)
```

## Importing ASAP XML annotations

ASAP XML files are read by `compass.connector`
(standard-library XML only). `Dot` becomes `POINT`, `PointSet` becomes
`POINTSET`, `Polygon` becomes `POLYGON`; any other type, a bad root element or missing coordinates raise
`ValueError`. Coordinates are sorted by their `Order` attribute.

ASAP knows only a flat list of named groups, whereas the store has
layer → group → object. The mapping is explicit: `group_map` sends each ASAP
group name to an `AsapTarget(layer, group, as_type=None)`; `as_type` forces
the object type (e.g. `"POLYLINE"` for a `PointSet` that is really a line).

```python
from compass.core import ZarrRaster
from compass.connector import asap_to_store, AsapTarget

report = asap_to_store(
    "AP1064-1.xml", "AP1064-1.morphotypes.ann.sqlite", ZarrRaster("AP1064-1.zarr"),
    group_map={"Pattern CT": AsapTarget("morphology", "Pattern CT"),
               "Invasive front": AsapTarget("landmarks", "Invasive front", "POLYLINE")},
    on_unmapped="error",     # or "skip"
    prune_empty=True,        # drop empty groups/layers, incl. the default base/no_group
)
report["per_group"], report["unmapped"], report["repaired"], report["out_of_bounds"]
```

With a `ZarrRaster`, the source → output affine of the import record (see
the previous section) is applied to the store, and `image_shape` and `mpp`
are set from the pyramid; without one, pass `matrix`, `image_shape` and `mpp`
yourself, or none for coordinates already in level-0 pixels. Invalid
polygons are repaired with `make_valid` and degenerate ones skipped; both are
counted in the report, as are objects outside the image. Existing output is
replaced unless `overwrite=False`.

## Circles under transforms

Shift, rotation, uniform scale and reflection (similarities) keep a circle
a circle: the center is transformed exactly and the radius scaled. Any
other transform would turn it into an ellipse, so:

- `circles="raise"` (default): `ValueError`, nothing changed;
- `circles="to_polygon"`: the circles become `POLYGON` disks first, then
  are transformed.

A `Circle` object raises the same way; call `to_polygon()` first.

# File format

One SQLite database per slide. The schema (`SQLITE_DDL`) is the legacy
writer's, unchanged (a test checks it), so files written by the legacy code
and by `AnnotationStore` can be read by either. `init_db(path)` creates an
empty schema.

| Table | Contents |
|---|---|
| `meta` | key/value: `schema_version`, `name`, `mpp`, `image_width`, `image_height`, `last_layer_id`, `last_group_id`, `last_object_id`, plus free-form keys |
| `layers` | `layer_id`, unique `name` |
| `groups` | `group_id`, `layer_id`, `name` (unique within a layer) |
| `objects` | `object_id`, `type_code`, `name`, `wkb` geometry, `circle_r`, bbox `min_x/min_y/max_x/max_y` |
| `object_groups` | many-to-many object ↔ group; a trigger forbids groups of *different* layers for one object |
| `objects_rtree` | R-tree over the bboxes, kept in sync by triggers |
| `scalar_attributes` | attribute dictionary `attr_id` ↔ `attr_name` (e.g. gene names) |
| `object_scalar_sparse` | per object: packed `uint32` attribute ids (sorted) + `float32` values |

Type codes (`ANNOT_TYPE_CODE`): `POINT=1`, `POINTSET=2`, `POLYLINE=3`,
`POLYGON=4`, `CIRCLE=5`. A circle is stored as its center `POINT` plus
`circle_r`, with a bbox that includes the radius. Deleting an object
cascades to its memberships and scalars (foreign keys, enabled by the
store on every writable connection).

# Legacy API {#legacy-api}

`compass.core.annot` provides the legacy names, so scripts and notebooks
port by changing their imports:

```python
# before
import compass.annot as ann
from compass.annot_sqlite_storage import save_annotation, load_annotation, query_layer_id
# after
import compass.core.annot as ann
from compass.core.annot import save_annotation, load_annotation, query_layer_id
```

```python
from compass.core import ImageShape
import compass.core.annot as ann

a = ann.Annotation(name="Annotation 1", image_shape=ImageShape(width=2000, height=1500), mpp=1.0)
a.add_annotation_object(ann.Point((200, 200), name="PT1", data={"a": 5, "b": 10}))
a.add_annotations([ann.Point((600, 500), name="PT3"),
                   ann.Circle((200, 400), 50, name="Circ1", data={"a": 4.5})],
                  group="group two", layer="new layer")

[o.name for o in a.scan_objects("new layer", group=None)]     # ['PT3', 'Circ1']
a.set_resolution(0.5)
a.get_base_image_shape()                                      # width=4000 height=3000
list(a.scan_objects("new layer", None, "CIRCLE"))             # [CIRCLE <Circ1>: center=(400.0, 800.0), radius=100.0]

ann.save_annotation(a, "a1.sqlite")
b = ann.load_annotation("a1.sqlite")
ann.query_object_ids_in_layer("a1.sqlite", 1)                 # [1, 2]
ann.fetch_scalar_data("a1.sqlite", [0, 2])                    # {0: {'a': 5.0, 'b': 10.0}, 2: {'a': 4.5}}
```

`Annotation` keeps its objects in an in-memory store, `a.store`, which
gives it the whole vectorized API (`a.store.get_objects_in_roi(...)`).

| Legacy | Here |
|---|---|
| `Annotation(name, image_shape, mpp)` | same; also `Annotation.load(path)`, `a.save(path)`, `a.store` |
| `add_annotation_object`, `add_annotations` | same; `AnnotationStore.add_objects` |
| `scan_objects(layer, group, filter_object_type)` | same (yields in id order, in batches) |
| `resize`, `set_resolution`, `match_image` | same; store methods of the same names |
| `get_layer_names`, `get_base_image_shape`, `get_resolution`, `name` | same |
| `asdict()`, `fromdict(d)` | same dictionary structure |
| `save_annotation`, `load_annotation` | same |
| `query_layer_id`, `query_group_id`, `query_groups_in_layer` | same; `query_group_id` also takes `layer_id` |
| `query_object_ids_in_layer / _in_group / _in_layer_roi` | same (lists) |
| `fetch_objects`, `fetch_geometries`, `fetch_scalar_data` | same |
| `createEmptyAnnotationObject` | same; also `create_empty_annotation_object` |
| object classes and their methods | same constructors and methods |

## Behaviour changes

- **Objects are copied in.** The legacy `Annotation` kept references, so
  changing an object after adding it changed the annotation. Now it
  doesn't: edit through `a.store` (or re-add).
- **Group names are per layer**, as in the file format. Legacy treated
  them as global, so reusing a group name in a second layer produced an
  annotation that couldn't be saved.
- **`scan_objects(group=None)`** yields each object once, even when it is
  in several groups of the layer.
- **Circles** are 64-vertex disks in memory (legacy: an 8-vertex octagon)
  and keep their exact center and radius through every transform.
- **Errors** are `ValueError` (legacy: `RuntimeError`), and
  `PermissionError` for writes to a read-only store.
- `save_annotation(..., overwrite=False)` refuses an existing file
  (legacy inserted into the existing content and failed on clashing ids).

## Legacy bugs fixed in the port

`compass.legacy` itself is unchanged.

- `data` defaulted to `None` instead of `{}` (`data if not None else {}`).
- `duplicate()` crashed (passed a nonexistent `group` attribute).
- `Polygon` and `PolyLine` never got their default names.
- `Circle.translate(x)` without `y_off` crashed; `Circle.fromdict` lost the
  center and radius.
- `scan_objects` raised `RuntimeError` (a `StopIteration` inside a
  generator) on an empty selection, and with `group=None` looked group ids
  up as names and crashed with `KeyError`.
- A non-numeric attribute value was written anyway, with the previous
  attribute's value (or crashed, if it was the first).
- `fetch_scalar_data([])` returned `[]` instead of `{}`.

# Performance

Measured on this machine, file store, 1,000,000 points with 2 scalar
attributes each:

| Operation | Time |
|---|---|
| `add_objects` (one call) | 17 s |
| `affine` (whole store) | 23 s |
| `remove_group` (deletes all 1M) | 12 s |

Most of the time is SQLite maintaining the R-tree (rebuilding it after a
bulk update was measured and is slower). Writable stores use a 256 MB page
cache, allocated as needed, which is ~30% faster at this size than
SQLite's 2 MB default. Reads are unaffected by any of this.

For many small writes (e.g. one `add_object` per Visium spot), wrap the
loop in `store.batch()` or collect the objects and call `add_objects` once.

# Method summary

## `AnnotationStore` — reading

| Method | Returns |
|---|---|
| `meta`, `name`, `mpp`, `image_shape`, `n_objects` / `len()` | metadata |
| `get_layers()`, `get_layer_names()` | polars `layer_id`, `name`; list of names |
| `get_groups(layer_id=None)` | polars `group_id`, `layer_id`, `name` |
| `get_layer_id(name)`, `get_group_id(name, layer_id=None)` | `int` (`ValueError` if unknown) |
| `count_objects_per_group()` | polars `group_id`, `layer_id`, `group_name`, `n_objects` |
| `count_objects_in_roi(layer_id, roi)` | `int` |
| `get_object_ids_in_layer / _in_group / _in_roi(...)` | `np.ndarray[int64]` |
| `get_objects(ids, circles_as_disks=True)` | GeoDataFrame |
| `get_objects_in_roi(layer_id, roi, group_ids=None, circles_as_disks=True)` | GeoDataFrame |
| `get_bboxes(layer_id=None)` | polars `object_id`, `min_x`, `min_y`, `max_x`, `max_y` |
| `get_memberships(ids=None)` | polars `object_id`, `group_id` |
| `get_scalar_attribute_names()` | polars `attr_id`, `attr_name` |
| `get_scalar_data_long(ids)` | polars `object_id`, `attr_id`, `value` |
| `get_scalar_data(ids, attributes=None)` | polars, wide |
| `get_scalar_dicts(ids)` | `{object_id: {name: value}}` |
| `fetch_objects(ids, with_data=True)` | list of object instances |

## `AnnotationStore` — writing (`mode="r+"` / `"w"`)

| Method | Effect |
|---|---|
| `set_meta(**kw)`; `name`, `mpp`, `image_shape` setters | metadata |
| `batch()` | context manager: one transaction |
| `add_layer(name, exist_ok=True)`, `add_group(name, layer="base", exist_ok=True)` | id of the new (or existing) layer / group |
| `rename_layer(layer, new)`, `rename_group(group, new, layer=None)` | rename |
| `remove_layer(layer)`, `remove_group(group, layer=None)` | delete; returns the number of objects deleted |
| `add_objects(objects, layer, group, data=None)`, `add_object(obj, layer, group)` | new ids |
| `update_objects(gdf)` | replace geometry / type / name / radius |
| `remove_objects(ids)` | number deleted |
| `add_to_group(ids, group, layer=None)`, `remove_from_group(ids, group, layer=None)` | membership; the latter returns the number of objects deleted |
| `set_scalar_data(ids, data, replace=False)`, `delete_scalar_data(ids, attributes=None)` | scalars |
| `translate`, `scale`, `affine`, `resize`, `set_resolution`, `match_image` | transforms (see above) |
| `save_as(path, overwrite=True)`, `AnnotationStore.load_in_memory(path)` | copies |
