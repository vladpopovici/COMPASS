# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

## This module handles whole slide image annotations for own algorithms as well
## as several import/export formats (HistomicsTK, Hamamatsu, ASAP, etc).
##
## The annotation objects belong to at least one group: "no_group".
## Other groups may be added, and objects may belong to several groups.
##
## All annotations share the same underlying mesh (= a raster of pixels with
## predefined extent and fixed resolution (microns-per-pixels)).
##
## The coordinates of the various objects (or parts of them) are specified
## as (X,Y) pairs (horizontal and vertical coordinates) and not as (row, column)
## coordinates.

__all__ = ["AnnotationObject", "Point", "Polygon", "PointSet", "Annotation", "Circle"]

import collections
import io
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import shapely

import shapely.affinity as sha
import shapely.geometry as shg
import orjson as json

from ._pyr import PyramidalImage
from .core import ImageShape

# ---------------------------------------------------------------------
# Type codes for geometry types (map your AnnotationObject.type strings)
# ---------------------------------------------------------------------

ANNOT_TYPE_CODE: dict[str, int] = {
    "POINT": 1,
    "POINTSET": 2,
    "POLYLINE": 3,
    "POLYGON": 4,
    "CIRCLE": 5,
}

ANNOT_CODE_TYPE: dict[int, str] = {v: k for k, v in ANNOT_TYPE_CODE.items()}


##-
class AnnotationObject(ABC):
    """Define the AnnotationObject minimal interface. This class is made
    abstract to force more meaningful names (e.g. Point, Polygon, etc.) in
    subclasses."""

    def __init__(
        self,
        coords: list[any] | tuple[any, ...] | np.ndarray,
        name: str | None = None,
        data: dict[str, np.float32] | None = None,
    ):
        # main geometrical object describing the annotation:
        self._geom = shg.Point()  # some empty geometry
        self._name: str = name
        self._annotation_type: str = None
        self._data: dict[str, np.float32] = data if not None else {}
        self._id: int = -1  # to be used internally

        return

    @abstractmethod
    def duplicate(self):
        pass

    @property
    def data(self):
        return self._data

    def set_data(self, data: dict[str, any] | None):
        self._data = data

    def __str__(self):
        """Return a string representation of the object."""
        return str(self.type) + " <" + str(self.name) + ">: " + str(self.geom)

    def bounding_box(self):
        """Compute the bounding box of the object."""
        return self.geom.bounds

    def translate(self, x_off, y_off=None):
        """Translate the object by a vector [x_off, y_off], i.e.
        the new coordinates will be x' = x + x_off, y' = y + y_off.
        If y_off is None, then the same value as in x_off will be
        used.

        :param x_off: (double) shift in thr X-direction
        :param y_off: (double) shift in the Y-direction; if None,
            y_off == x_off
        """
        if y_off is None:
            y_off = x_off
        self._geom = sha.translate(self.geom, x_off, y_off, zoff=0.0)

        return

    def scale(self, x_scale, y_scale=None, origin="center"):
        """Scale the object by a specified factor with respect to a specified
        origin of the transformation. See shapely.geometry.scale() for details.

        :param x_scale: (double) X-scale factor
        :param y_scale: (double) Y-scale factor; if None, y_scale == x_scale
        :param origin: reference point for scaling. Default: "center" (of the
            object). Alternatives: "centroid" or a shapely.geometry.Point object
            for arbitrary origin.
        """
        if y_scale is None:
            y_scale = x_scale
        self._geom = sha.scale(
            self.geom, xfact=x_scale, yfact=y_scale, zfact=1, origin=origin
        )

        return

    def resize(self, factor: float) -> None:
        """Resize an object with the specified factor. This is equivalent to
        scaling with the origin set to (0,0) and the same factor for both x and y
        coordinates.

        :param factor: (float) resizing factor.
        """
        self.scale(factor, origin=shg.Point((0.0, 0.0)))

        return

    def affine(self, M):
        """Apply an affine transformation to all points of the annotation.

        If M is the affine transformation matrix, the new coordinates
        (x', y') of a point (x, y) will be computed as

        x' = M[1,1] x + M[1,2] y + M[1,3]
        y' = M[2,1] x + M[2,2] y + M[2,3]

        In other words, if P is the 3 x n matrix of n points,
        P = [x; y; 1]
        then the new matrix Q is given by
        Q = M * P

        :param M: numpy array [2 x 3]

        :return:
            nothing
        """

        self._geom = sha.affine_transform(
            self.geom, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]]
        )

        return

    @property
    def geom(self):
        """The geometry of the object."""
        return self._geom

    @property
    def name(self) -> str:
        """Return the name of the annotation."""
        return self._name

    @property
    def type(self):
        """Return the annotation type as a string."""
        return self._annotation_type

    @property
    def x(self):
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return shapely.get_coordinates(self.geom)[:, 0]

    @property
    def y(self):
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return shapely.get_coordinates(self.geom)[:, 1]

    def xy(self) -> np.array:
        return shapely.get_coordinates(self.geom)

    def size(self) -> int:
        """Return the number of points defining the object."""
        raise NotImplementedError

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "id": self._id,
            "annotation_type": self._annotation_type,
            "name": self._name,
            "data": self._data,
            "geom": shapely.to_wkt(self.geom),  # text representation of the geometry
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Initialize the object from a dictionary."""
        self._id = d["id"]
        self._annotation_type = d["annotation_type"]
        self._name = d["name"]
        self._data = d["data"]
        self._geom = shapely.from_wkt(d["geom"])

        return


##-


##-
class Point(AnnotationObject):
    """Point: a single position in the image."""

    def __init__(
        self,
        coords: list[any] | tuple[any, ...] | np.ndarray = (0.0, 0.0),
        name: str | None = None,
        data: dict[str, np.float32] | None = None,
    ):
        """Initialize a POINT annotation, i.e. a single point in plane.

        Args:
            coords (list or vector or tuple): the (x,y) coordinates of the point
            name (str): the name of the annotation
        """
        super().__init__(coords, name, data)

        self._annotation_type = "POINT"
        self._name = "POINT" if name is None else name

        if not isinstance(coords, collections.abc.Iterable):
            raise RuntimeError("coords parameter cannot be interpreted as a 2D vector")

        self._geom = shg.Point(coords[:2])

        return

    def duplicate(self):
        return Point(
            shapely.get_coordinates(self.geom),
            name=self.name,
            group=self.group,
            data=self.data,
        )

    def size(self) -> int:
        """Return the number of points defining the object."""
        return 1


##-


##-
class PointSet(AnnotationObject):
    """PointSet: an ordered collection of points."""

    def __init__(
        self,
        coords: list[any] | tuple[any, ...] | np.ndarray,
        name: str | None = None,
        data: dict[str, np.float32] | None = None,
    ):
        """Initialize a POINTSET annotation, i.e. a collection
         of points in plane.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """

        super().__init__(coords, name, data)
        self._annotation_type = "POINTSET"
        self._name = "POINTS" if name is None else name

        # check whether coords is iterable and build the coords from it:
        if not isinstance(coords, collections.abc.Iterable):
            raise RuntimeError("coords parameter cannot be interpreted as a 2D array")

        self._geom = shg.MultiPoint(coords)

        return

    def duplicate(self):
        return PointSet(
            shapely.get_coordinates(self.geom),
            name=self.name,
            group=self.group,
            data=self.data,
        )

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.xy().shape[0]


##-


class PolyLine(AnnotationObject):
    """PolyLine: polygonal line (a sequence of segments)"""

    def __init__(
        self,
        coords: list[any] | tuple[any, ...] | np.ndarray,
        name: str | None = None,
        data: dict[str, np.float32] | None = None,
    ):
        """Initialize a POLYLINE object.

        Args:
            coords (list or tuple): coordinates of the points [(x0,y0), (x1,y1), ...]
                defining the segments (x0,y0)->(x1,y1); (x1,y1)->(x2,y2),...
            name (str): the name of the annotation
        """
        super().__init__(coords, name, data)
        self._annotation_type = "POLYLINE"

        # check whether x is iterable and build the coords from it:
        if not isinstance(coords, collections.abc.Iterable):
            raise RuntimeError("coords parameter cannot be interpreted as a 2D array")

        self._geom = shg.LineString(coords)

        return

    def duplicate(self):
        return PolyLine(
            shapely.get_coordinates(self.geom),
            name=self.name,
            group=self.group,
            data=self.data,
        )

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.xy().shape[0]


##-


##-
class Polygon(AnnotationObject):
    """Polygon: an ordered collection of points where the first and
    last points coincide."""

    def __init__(
        self,
        coords: list[any] | tuple[any, ...] | np.ndarray,
        name: str | None = None,
        data: dict[str, np.float32] | None = None,
    ):
        """Initialize a POINTSET annotation, i.e. a collection
         of points in plane.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """

        super().__init__(coords, name, data)
        self._annotation_type = "POLYGON"
        self._name = name if not None else "POLYGON"

        # check whether x is iterable and build the coords from it:
        if not isinstance(coords, collections.abc.Iterable):
            raise RuntimeError("coords parameter cannot be interpreted as a 2D array")

        self._geom = shg.Polygon(coords)

        return

    def duplicate(self):
        return Polygon(
            shapely.get_coordinates(self.geom),
            name=self.name,
            group=self.group,
            data=self.data,
        )

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.xy().shape[0]


##-


##
class Circle(Polygon):
    """Circle annotation is implemented as a polygon (octogon) to be compatible with GeoJSON specifications.
    The center coordinates and radius are computed properties, based on the polygon."""

    def __init__(
        self,
        center: list[any] | tuple[any, ...] | np.ndarray,
        radius: float,
        name: str | None = None,
        data: dict[str, np.float32] | None = None,
        n_points: int = 8,
    ):
        alpha = np.array([k * 2 * np.pi / n_points for k in range(n_points)])
        coords = np.vstack(
            (radius * np.sin(alpha) + center[0], radius * np.cos(alpha) + center[1])
        ).transpose()

        super().__init__(coords.tolist(), name, data)
        self._annotation_type = "CIRCLE"
        self._center = shg.Point(center[:2])
        self._radius = radius

    @property
    def center(self) -> tuple:
        # c = shapely.centroid(self._geom)
        return tuple(shapely.get_coordinates(self._center)[0])

    @property
    def radius(self) -> float:
        return self._radius  # shapely.minimum_bounding_radius(self._geom)

    def translate(self, x_off, y_off=None):
        super().translate(x_off, y_off)
        self._center = sha.translate(self._center, x_off, y_off, zoff=0.0)
        return

    def scale(self, x_scale, y_scale=None, origin="center"):
        if y_scale is not None and x_scale != y_scale:
            # non-homogeneous scaling: diforms the circle, do not accept
            raise RuntimeError(
                "Non homogeneous scaling is not accepted for circles. Use polygons instead."
            )
        super().scale(x_scale, y_scale, origin)
        self._center = sha.scale(
            self._center, xfact=x_scale, yfact=x_scale, zfact=1.0, origin=origin
        )
        self._radius *= x_scale
        return

    def affine(self, M):
        raise RuntimeError("Not implemented for circles, use polygons instead.`")
        return

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = super().asdict()
        d["radius"] = self.radius
        d["center"] = self.center

        return d

    def fromdict(self, d: dict) -> None:
        """Initialize the object from a dictionary."""
        super().fromdict(d)
        self._annotation_type = "CIRCLE"

        return


##-


def createEmptyAnnotationObject(annot_type: str) -> AnnotationObject:
    """Function to create an empty annotation object of a desired type.

    Args:
        annot_type (str):
            type of the annotation object:
            POINT
            POINTSET
            POLYLINE/LINESTRING
            POLYGON
            CIRCLE

    """

    obj = None
    tp = annot_type.upper()
    if tp == "POINT":
        obj = Point(coords=[0, 0])
    elif tp == "POINTSET":
        obj = PointSet([[0, 0]])
    elif tp == "LINESTRING" or tp == "POLYLINE":
        obj = PolyLine([[0, 0], [1, 1], [2, 2]])
    elif tp == "POLYGON":
        obj = Polygon([[0, 0], [1, 1], [2, 2]])
    elif tp == "CIRCLE":
        obj = Circle([0.0, 0.0], 1.0)
    else:
        raise RuntimeError("unknown annotation type: " + annot_type)
    return obj


##-


# reverse the mapping of the dictionary
def _reverse_dict(d: dict[any, any]) -> dict[any, any]:
    return {v: k for k, v in d.items()}


##
class Annotation(object):
    """
    An annotation is a collection of AnnotationObjects represented on the same
    coordinate system (mesh) and grouped into groups and layers. An object may
    belong to several groups within a layer, but a group may belong to one and
    only one layer. Thus, the objects themselves belong to a single layer. If
    no group and layer are specified, the objects are added to the default group
    ("no_group") and layer ("base").

    The objects are added via reference, not deep-copy, thus any change elsewhere
    will also be reflected here. It is the user's responsibility to create a new
    object before adding it to the annotation.
    """

    def __get_layer_id(self, layer: str, create_if_needed: bool = False) -> int:
        l2id = _reverse_dict(self._id_layer)
        if layer in l2id:
            return l2id[layer]

        # no such layer, add it?
        if create_if_needed:
            # new group:
            self.__id_register["layer"] = layer_id = self.__id_register["layer"] + 1
            self._id_layer[layer_id] = layer
            return layer_id

        return -1

    def __get_group_id(self, group: str, create_if_needed: bool = False) -> int:
        g2id = _reverse_dict(self._id_group)
        if group in g2id:
            return g2id[group]

        # no such group, add it?
        if create_if_needed:
            # new group:
            self.__id_register["group"] = grp_id = self.__id_register["group"] + 1
            self._id_group[grp_id] = group
            return grp_id

        return -1

    def _update_register(self, key: str, value: int) -> None:
        self.__id_register[key] = value

    def __init__(
        self, name: str = "", image_shape: ImageShape = None, mpp: float = 1.0
    ) -> None:
        """Initialize an Annotation for a slide.

        :param name: (str) name of the annotation
        :param image_shape: (dict) shape of the image corresponding to the annotation
            {'width':..., 'height':...}
        :param mpp: (float) slide resolution (microns-per-pixel) for the image
        """
        self._name: str = name
        self._image_shape: ImageShape = (
            image_shape if image_shape is not None else ImageShape(width=0, height=0)
        )
        self._mpp = mpp

        self._id_layer: dict[int, str] = {0: "base"}  # map ID -> layer name
        self._id_group: dict[int, str] = {0: "no_group"}  # map ID -> group name
        self.__id_register: dict[str, int] = {
            "layer": 0,  # last layer ID
            "group": 0,  # last group ID
            "object": -1,  # last object ID, no objects
        }

        # initially, there is a single layer with a single group:
        self._layers: dict[int, set[int]] = {
            0: {0} # layer_id -> group_ids
        }  # all layers, each with its unique groups
        self._annots: dict[
            int, list[AnnotationObject]
        ] = {0: []}  # all annots <group_id>:<list of annotation objects>, no obj in "no_group"

        return

    def get_layer_names(self, sorted_list: bool=False) -> list[str]:
        """
        Return the list of layer names, eventually sorted.

        :param sorted_list: (bool) should the list be sorted?
        """
        res = list(self._id_layer.values())
        if sorted_list:
            res.sort()
        return res

    def add_annotation_object(
        self,
        a: AnnotationObject,
        group: str | list[str] = "no_group",
        layer: str = "base",
    ) -> None:
        """
        Add a single annotation object to a group/layer. Create new group and layer if needed.

        :param a: An AnnotationObject
        :param group: The group (str) or the list of groups the object is added to.
        :param layer: The layer (str) the object is added to.
        """

        # First, get an ID for the new object:
        self.__id_register["object"] = a._id = self.__id_register["object"] + 1

        # then prepare the rest of the structures...
        l_id = self.__get_layer_id(layer, create_if_needed=True)

        if isinstance(group, str):
            group = [group]
        g_id = [
            self.__get_group_id(grp, create_if_needed=True) for grp in group
        ]  # all group ids

        # append group(s) to the set of groups of the layer l_id
        self._layers.setdefault(l_id, set()).update(g_id)

        # add the object to all the groups:
        for g in g_id:
            self._annots.setdefault(g, []).append(a)

        return

    def add_annotations(
        self,
        a_list: list[AnnotationObject],
        group: str | list[str] = "no_group",
        layer: str = "base",
    ) -> None:
        """
        Add a list of annotation objects to a group/layer. Create new group and layer if needed.

        :param a_list: list of annotation objects
        :param group: group/list of groups
        :param layer: the layer
        """

        # First, get an IDs for the new objects:
        for a in a_list:
            self.__id_register["object"] = a._id = self.__id_register["object"] + 1

        # then prepare the rest of the structures...
        l_id = self.__get_layer_id(layer, create_if_needed=True)

        if isinstance(group, str):
            group = [group]
        g_id = [
            self.__get_group_id(grp, create_if_needed=True) for grp in group
        ]  # all group ids

        # append group(s) to the set of groups of the layer l_id
        self._layers.setdefault(l_id, set()).update(g_id)

        # add the objects to all the groups:
        for g in g_id:
            self._annots.setdefault(g, []).extend(a_list)

        return

    def scan_objects(
            self,
            layer: str = "base",
            group: str|None = "no_group",
            filter_object_type: int|str|None = None
    ) -> Iterable[AnnotationObject]:
        """
        Browse all objects in a layer/group from an annotation collection. Optionally,
        returns only the annotation objects of a specific type.

        :param layer: (str) name of the layer
        :param group: (str) name of the group; if None, all groups are scanned
        :param filter_object_type: (int|str|list|None) the type of objects to return; if None, all objects
            are scanned

        Returns:
        an iterable of AnnotationObject
        """
        layer_id = self.__get_layer_id(layer, create_if_needed=False)
        if layer_id == -1:
            raise RuntimeError("Unknown layer")

        if filter_object_type is not None:
            if isinstance(filter_object_type, int):
                filter_object_type = ANNOT_CODE_TYPE[filter_object_type] # get the name of the obj type

        group_ids = []
        groups = list(self._layers[layer_id])
        if group is None:
            if len(groups) > 0:
                group_ids = [self.__get_group_id(gr, create_if_needed=False) for gr in groups]
        else:
            group_id = self.__get_group_id(group, create_if_needed=False)
            if group_id not in groups:
                raise RuntimeError(f"Unknown group in layer {layer}")
            group_ids = [group_id]

        if len(group_ids) == 0:
            raise StopIteration

        if filter_object_type is not None:  # and is already str
            for gr in group_ids:
                for obj in self._annots[gr]:
                    if obj._annotation_type == filter_object_type:
                        yield obj
        else:  # all obj types
            for gr in group_ids:
                for obj in self._annots[gr]:
                    yield obj

    def get_base_image_shape(self) -> ImageShape:
        """Return the basic space for the annotations."""
        return self._image_shape

    @property
    def name(self):
        """Return the name of the annotation object."""
        return self._name

    @property
    def type(self):
        """Return the annotation type as a string."""
        return "Annotation"

    def get_resolution(self) -> float:
        return self._mpp

    def resize(self, factor: float) -> None:
        """
        Re-scales all annotations (in all layers) by a factor f (multiply by f).
        """
        self._mpp /= factor
        nw, nh = factor*self._image_shape.width, factor*self._image_shape.height
        self._image_shape = ImageShape(
            width=int(np.round(nw)),
            height=int(np.round(nh))
        )

        for grp in self._annots:  # for all groups
            for obj in self._annots[grp]:  # for all objects in the current group
                obj.resize(factor)
        return

    def match_image(
            self,
            pyr_img: PyramidalImage,
            pyr_level: int = 0,
    ) -> None:
        """
        Transform the annotation (just scale) ti match the target image (at
        the specified level). The scaling might not be exactly the same on
        both axes, the function uses the average scaling factor.
        Args:
            pyr_img: PyramidalImage - the image providing the geometry to
                match
            pyr_level: target level of the pyramidal image
        Returns:
            None
        """
        target_shape = pyr_img.shape(pyr_level)
        fw = target_shape.width / self._image_shape.width
        fh = target_shape.height / self._image_shape.height

        factor = 0.5 * (fw + fh)
        self.resize(factor)

        # force new shape to match exactly the one requested
        self._image_shape = ImageShape(
            width=target_shape.width,
            height=target_shape.height,
        )
        self._mpp = pyr_img.get_mpp_for_level(pyr_level)

        return

    def set_resolution(self, mpp: float) -> None:
        """Scales the annotation to the desired mpp.

        :param mpp: (float) target mpp
        """
        if mpp != self._mpp:
            f = self._mpp / mpp
            self.resize(f)
            self._mpp = mpp

        return

    def asdict(self) -> dict:
        d = {
            "name": self._name,
            "image_shape": self._image_shape,
            "mpp": self._mpp,
            "id_layer": self._id_layer,
            "id_group": self._id_group,
            "id_reg": self.__id_register,
            "layers": {k: list(self._layers[k]) for k in self._layers},
            "annotations": {
                g: [a.asdict() for a in self._annots[g]] for g in self._annots
            },
        }

        return d

    def fromdict(self, d: dict) -> None:
        self._name = d["name"]
        self._image_shape = d["image_shape"]
        self._mpp = d["mpp"]
        self._id_layer = d["id_layer"]
        self._id_group = d["id_group"]
        self.__id_register = d["id_reg"]
        self._layers = {k: set(d["layers"][k]) for k in d["layers"]}

        self._annots = {}
        for grp in d["annotations"]:
            self._annots[grp] = []
            for a in d["annotations"][grp]:
                obj = createEmptyAnnotationObject(a["annotation_type"])
                obj.fromdict(a)
                self._annots[grp].append(obj)

        return


##-


