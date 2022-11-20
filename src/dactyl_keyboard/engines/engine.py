from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar
import sys
if sys.version_info[:2] > (3, 8):
    from collections.abc import Iterable
else:
    from typing import Iterable

from numpy import ndarray

TGeometry = TypeVar("TGeometry")


class GeometryImporter(ABC, Generic[TGeometry]):
    """
    A class that encapsulates the ability to import a resource.
    """
    @staticmethod
    @abstractmethod
    def file_type() -> Iterable[str]:
        """
        The file extension(s) this importer supports
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def import_resource(path: Path) -> TGeometry:
        """
        Import the given file, and return the imported geometry
        """
        raise NotImplementedError


class GeometryExporter(ABC, Generic[TGeometry]):
    """
    A class that encapsulates the ability to export geometry.
    """

    @staticmethod
    @abstractmethod
    def file_type() -> str:
        """
        The file extension this exporter supports
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def export_geometry(shape: TGeometry, path: Path):
        """
        Export the given shape to path.
        """
        raise NotImplementedError


class GeometryEngine(ABC, Generic[TGeometry]):
    """
    Engine base class.

    Dimensions are in millimeters.
    All operations that manipulate shapes return copies; no in-place manipulation is performed.
    """

    @staticmethod
    @abstractmethod
    def box(width: float, height: float, depth: float) -> TGeometry:
        """
        Create a box with the given dimensions.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cylinder(radius: float, height: float, segments: int = 100) -> TGeometry:
        """
        Create a cylinder with the given dimensions.

        The number of segments may be provided, but this may be ignored on some engines.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sphere(radius: float, segments: int = 100) -> TGeometry:
        """
        Create a sphere with the given radius.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cone(radius_bottom: float, radius_top: float, height: float, segments: int = 100) -> TGeometry:
        """
        Create a cone with the given radii and height.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def rotate(shape: TGeometry, euler_degrees: ndarray) -> TGeometry:
        """
        Rotate the shape by the given euler angles in degrees, and return a copy.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def translate(shape: TGeometry, vector: ndarray) -> TGeometry:
        """
        Translate the given shape, and return a copy.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def scale(shape: TGeometry, vector: ndarray) -> TGeometry:
        """
        Scale the given shape, and return a copy.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mirror(shape: TGeometry, vector: ndarray) -> TGeometry:
        """
        Mirror the given shape about the vector from the origin, and return a copy
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def union(shapes: Iterable[TGeometry]) -> TGeometry:
        """
        Create a new shape from the union of multiple other shapes
        It is an error to pass an empty collection.
        If `shapes` contains a single element, a copy is returned.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def difference(initial_shape: TGeometry, subtractions: Iterable[TGeometry]) -> TGeometry:
        """
        Create a new shape from the subtraction of multiple shapes from a starting shape.
        If `subtractions` is empty, a copy of `initial_shape` is returned.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def intersect(shapes: Iterable[TGeometry]) -> TGeometry:
        """
        Create a new shape from the intersection of multiple shapes.
        It is an error to pass an empty collection.
        If `shapes` contains a single element, a copy is returned.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convex_hull(shapes: Iterable[TGeometry]) -> TGeometry:
        """
        Construct a convex hull from the collection of multiple shapes.
        It is an error to pass an empty collection.y
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def project_base(shapes: Iterable[TGeometry], height: float = 1) -> TGeometry:
        """
        Seems to be a method to project geometry to create the base plate?
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def importers() -> Iterable[GeometryImporter[TGeometry]]:
        """
        Get the importers this engine supports.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def exporters() -> Iterable[GeometryExporter[TGeometry]]:
        """
        Get the exporters this engine supports.
        """
        raise NotImplementedError
