from functools import reduce
import logging
from abc import abstractmethod
import sys
if sys.version_info[:2] > (3, 8):
    from collections.abc import Iterable
else:
    from typing import Iterable
from pathlib import Path

from cadquery import Edge, Face, Shape, Shell, Solid, Vector, Wire, exporters, importers
from scipy.spatial import ConvexHull as sphull
from numpy import array, ndarray

from .engine import GeometryEngine, GeometryExporter, GeometryImporter


class _CadQueryStepImporter(GeometryImporter[Shape]):
    """
    Importer for cadquery that can import STEP files
    """

    @staticmethod
    def file_type() -> Iterable:
        return [".step"]

    @staticmethod
    def import_resource(path: Path) -> Shape:
        logging.info("Importing %s", path)
        return importers.importShape(exporters.ExportTypes.STEP, str(path))


class _CadQueryStepExporter(GeometryExporter[Shape]):
    """
    Exporter for cadquery that can export STEP files
    """

    @staticmethod
    def file_type() -> str:
        return ".step"

    @staticmethod
    def export_geometry(shape: Shape, path: Path):
        logging.info("Exporting to %s", path)
        exporters.exportShape(w=shape, fname=str(path), exportType=exporters.ExportTypes.STEP)


class CadQueryEngine(GeometryEngine[Shape]):
    """
    CadQuery geometry engine
    """

    @staticmethod
    def box(width: float, height: float, depth: float) -> Shape:
        return Solid.makeBox(width, height, depth)

    @staticmethod
    def cylinder(radius: float, height: float, _segments: int = 100) -> Shape:
        return Solid.makeCylinder(radius, height)

    @staticmethod
    def sphere(radius: float, _segments: int = 100) -> Shape:
        return Solid.makeSphere(radius)

    @staticmethod
    def cone(radius_bottom: float, radius_top: float, height: float, _segments: int = 100) -> Shape:
        return Solid.makeCone(radius1=radius_bottom, radius2=radius_top, height=height)

    @staticmethod
    def rotate(shape: Shape, euler_degrees: ndarray) -> Shape:
        origin = (0, 0, 0)
        shape = shape.rotate(startVector=origin, endVector=(1, 0, 0), angleDegrees=euler_degrees[0])
        shape = shape.rotate(startVector=origin, endVector=(0, 1, 0), angleDegrees=euler_degrees[1])
        shape = shape.rotate(startVector=origin, endVector=(0, 0, 1), angleDegrees=euler_degrees[2])
        return shape

    @staticmethod
    def translate(shape: Shape, vector: ndarray) -> Shape:
        return shape.translate(vector)

    @staticmethod
    def scale(shape: Shape, vector: ndarray) -> Shape:
        raise NotImplementedError

    @staticmethod
    def mirror(shape: Shape, vector: ndarray) -> Shape:
        return shape.mirror(vector)

    @staticmethod
    def union(shapes: Iterable[Shape]) -> Shape:
        logging.debug("union()")
        if not any(shapes):
            raise ValueError("shapes cannot be empty")

        if len(shapes) == 1:
            return shapes[0].copy()
        return reduce(lambda x, y: x.fuse(y), shapes)

    @staticmethod
    def difference(initial_shape: Shape, subtractions: Iterable[Shape]) -> Shape:
        logging.debug("difference()")
        if not any(subtractions):
            return initial_shape.copy()
        return reduce(lambda initial, to_remove: initial.cut(to_remove), subtractions, initial_shape)

    @staticmethod
    def intersect(shapes: Iterable[Shape]) -> Shape:
        logging.debug("intersect()")
        if not any(shapes):
            raise ValueError("shapes cannot be empty")

        if len(shapes) == 1:
            return shapes[0].copy()
        return reduce(lambda x, y: x.intersect(y), shapes)

    @staticmethod
    def convex_hull(shapes: Iterable[Shape]) -> Shape:
        if not any(shapes):
            raise ValueError("shapes cannot be empty")

        vertices = []
        for shape in shapes:
            vertices.extend(v.toTuple() for v in shape.Vertices())

        return CadQueryEngine._hull_from_points(vertices)

    @staticmethod
    def _face_from_points(points):
        edges = []
        num_pnts = len(points)
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % num_pnts]
            edges.append(Edge.makeLine(p1, p2))

        return Face.makeFromWires(Wire.assembleEdges(edges))

    @staticmethod
    def _hull_from_points(points):
        hull_calc = sphull(points)
        n_faces = len(hull_calc.simplices)

        faces = []
        for i in range(n_faces):
            face_items = hull_calc.simplices[i]
            fpnts = []
            for item in face_items:
                fpnts.append(points[item])
            faces.append(CadQueryEngine._face_from_points(fpnts))

        return Solid.makeSolid(Shell.makeShell(faces))

    @staticmethod
    def project_base(shapes: Iterable[Shape], _height: float = 1) -> Shape:
        logging.debug("bottom_hull()")
        shape = None
        for item in shapes:
            vertices = []
            verts = item.faces().vertices()
            for vert in verts.objects:
                v0 = vert.toTuple()
                v1 = [v0[0], v0[1], -10]
                vertices.append(array(v0))
                vertices.append(array(v1))

            t_shape = CadQueryEngine.hull_from_points(vertices)

            if shape is None:
                shape = t_shape
            else:
                shape = CadQueryEngine.union([shape, t_shape])

        return shape

    @staticmethod
    def importers() -> Iterable[GeometryImporter[Shape]]:
        return [_CadQueryStepImporter()]

    @staticmethod
    def exporters() -> Iterable[GeometryExporter[Shape]]:
        return [_CadQueryStepExporter]
