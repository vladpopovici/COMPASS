# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Annotations: object wrappers, vectorized transforms, and the SQLite/R-tree
store (read + write), plus the legacy API on top of them.

- ``objects``   Point, PointSet, PolyLine, Polygon, Circle (single objects)
- ``transform`` translate / scale / resize / affine over GeoDataFrames
- ``store``     AnnotationStore: queries -> geopandas/polars, edits, transforms
- ``schema``    the SQL schema (unchanged from legacy), type codes, init_db
- ``compat``    legacy ``Annotation`` + save/load/query/fetch functions

Full guide: docs/annotations.md.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from . import transform
from .compat import (Annotation, fetch_geometries, fetch_objects, fetch_scalar_data,
                     load_annotation, query_group_id, query_groups_in_layer,
                     query_layer_id, query_object_ids_in_group,
                     query_object_ids_in_layer, query_object_ids_in_layer_roi,
                     save_annotation)
from .objects import (CIRCLE_N_POINTS, AnnotationObject, Circle, Point, PointSet,
                      Polygon, PolyLine, create_empty_annotation_object,
                      createEmptyAnnotationObject, gdf_to_objects,
                      make_annotation_object, objects_to_gdf)
from .schema import (ANNOT_CODE_TYPE, ANNOT_TYPE_CODE, SCHEMA_VERSION, SQLITE_DDL,
                     geometry_type_names, init_db)
from .store import DEFAULT_GROUP, DEFAULT_LAYER, AnnotationStore

__all__ = [
    "ANNOT_CODE_TYPE", "ANNOT_TYPE_CODE", "CIRCLE_N_POINTS", "DEFAULT_GROUP",
    "DEFAULT_LAYER", "SCHEMA_VERSION", "SQLITE_DDL",
    "Annotation", "AnnotationObject", "AnnotationStore",
    "Circle", "Point", "PointSet", "PolyLine", "Polygon",
    "create_empty_annotation_object", "createEmptyAnnotationObject",
    "fetch_geometries", "fetch_objects", "fetch_scalar_data",
    "gdf_to_objects", "geometry_type_names", "init_db", "load_annotation",
    "make_annotation_object", "objects_to_gdf",
    "query_group_id", "query_groups_in_layer", "query_layer_id",
    "query_object_ids_in_group", "query_object_ids_in_layer",
    "query_object_ids_in_layer_roi", "save_annotation", "transform",
]
