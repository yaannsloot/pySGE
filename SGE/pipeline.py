from abc import ABC, abstractmethod
import numpy as np
import pywavefront
from typing import Callable, TypeAlias
from .datatypes import Vector3, Vector2, Color, Transform

RenderFunction: TypeAlias = Callable[[Transform],None] # Must be a world transform
CPUPointBuffer: TypeAlias = np.ndarray[tuple[float, float], np.float32]
CPUVertexBuffer: TypeAlias = np.ndarray[tuple[float, float, float], np.float32]
CPUIFaceBuffer: TypeAlias = np.ndarray[tuple[int, int, int], np.uint32]

# Update this to include 
class Mesh(ABC):
    def __init__(self, 
                 vertices:CPUVertexBuffer,
                 normals:CPUVertexBuffer,
                 vertex_colors:CPUVertexBuffer,
                 faces:CPUIFaceBuffer,
                 uv: CPUPointBuffer):
        self._v_buf = vertices
        self._n_buf = normals
        self._c_buf = vertex_colors
        self._f_buf = faces # Should not be edited
        self._uv_buf = uv
        
    @property
    @abstractmethod
    def vertices(self) -> tuple[Vector3]:
        # Should return a pre-initialized tuple of Vector3s WITH views to internal CPU buffers.
        # A syncronization mechanism should also ideally be implemented to push partial updates to the GPU.
        pass

    @property
    @abstractmethod
    def normals(self) -> tuple[Vector3]:
        # Should return a pre-initialized tuple of Vector3s WITH views to internal CPU buffers.
        # A syncronization mechanism should also ideally be implemented to push partial updates to the GPU.
        pass

    @property
    @abstractmethod
    def colors(self) -> tuple[Color]:
        # Should return a pre-initialized tuple of Colors WITH views to internal CPU buffers.
        # A syncronization mechanism should also ideally be implemented to push partial updates to the GPU.
        pass

    @property
    @abstractmethod
    def uv(self) -> tuple[Vector2]:
        # Should return a pre-initialized tuple of Vector2s WITH views to internal CPU buffers.
        # A syncronization mechanism should also ideally be implemented to push partial updates to the GPU.
        pass

    @abstractmethod
    def draw_wire(self):
        pass

    @abstractmethod
    def draw_solid(self):
        pass

    @classmethod
    def from_obj_file(cls, path:str) -> "Mesh":
        scene = pywavefront.Wavefront(path, collect_faces=True)
        vertices = np.array(scene.vertices, dtype=np.float32)
        if scene.parser.tex_coords:
            uvs = np.array(scene.parser.tex_coords, dtype=np.float32)
        else:
            uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        faces = []
        for mesh in scene.mesh_list:
            faces.append(np.array(mesh.faces, np.uint32))
        faces = np.vstack(faces)
        colors = np.ones_like(vertices)
        fv = vertices[faces]
        p1 = fv[:,0]
        p2 = fv[:,1]
        p3 = fv[:,2]
        u = p2 - p1
        v = p3 - p1
        normals = np.zeros_like(vertices, dtype=np.float32)
        face_normals = np.cross(u, v)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        return cls(vertices, normals, colors, faces, uvs)

class RenderingPipeline(ABC):
    """Actual implementation should include type aliases and render function factories"""

    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def post(self):
        pass
