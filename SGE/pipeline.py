from abc import ABC, abstractmethod
import numpy as np
import pywavefront
import OpenImageIO as oiio
from typing import Callable, TypeAlias, Any, Union
from .datatypes import Vector3, Vector2, Color4, Color, RGBA, RGB, Transform
from enum import Enum

RenderFunction: TypeAlias = Callable[[Transform],None] # Must be a world transform
CPUPointBuffer: TypeAlias = np.ndarray[tuple[float, float], np.float32]
CPUVertexBuffer: TypeAlias = np.ndarray[tuple[float, float, float], np.float32]
CPUIFaceBuffer: TypeAlias = np.ndarray[tuple[int, int, int], np.uint32]
CPUPixelBuffer: TypeAlias = np.ndarray[Any, Union[np.uint8, np.uint16, np.float16, np.float32]]

class WrapMethod(Enum):
    REPEAT = 1
    MIRRORED = 2
    CLAMP_TO_EDGE = 3
    CLAMP_TO_BORDER = 4

class FilterMethod(Enum):
    NEAREST = 1
    LINEAR = 2

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

class Texture(ABC):
    def __init__(self, 
                 data: CPUPixelBuffer, 
                 wrap_method = WrapMethod.MIRRORED,
                 filter_method = FilterMethod.LINEAR,
                 border_color: RGBA = (1.0, 1.0, 1.0, 0.0)):
        if data.dtype.kind in ('i', 'u'):
            orig_t = data.dtype
            nb = orig_t.itemsize
            max_val = 2**nb - 1
            data = data.astype(np.float32)
            data /= max_val
        shape = data.shape
        if len(shape) == 2:
            data = data[:, :, np.newaxis]
            shape = data.shape
        channels = shape[2]
        if channels == 1:
            data = np.repeat(data, 3, axis=-1)
            channels = data.shape[2]
        if channels == 3:
            data = np.dstack((data, np.ones((shape[0], shape[1], 1), dtype=np.float32)))
        self._data = data
        self.wrap_method = wrap_method
        self.filter_method = filter_method
        self._border_color = border_color if isinstance(border_color, Color4) else Color4(*border_color)
    
    @property
    def border_color(self):
        return self._border_color
    
    @border_color.setter
    def border_color(self, value: RGBA):
        self._border_color.rgba = tuple(value)

    @property
    def height(self):
        return self._data.shape[0]
    
    @property
    def width(self):
        return self._data.shape[1]

    @property
    @abstractmethod
    def data(self) -> tuple[tuple[Color4]]:
        # Should return a pre-initialized tuple of tuples of Color4s WITH views to internal CPU buffers.
        # A syncronization mechanism should also ideally be implemented to push partial updates to the GPU.
        pass

    @classmethod
    def from_file(cls, path):
        img = oiio.ImageInput.open(path)
        if img:
            data = img.read_image()
            img.close()
            return cls(data)

class Material(ABC):
    def __init__(self,
                 base_color: Union[Texture, RGBA] = (1, 1, 1, 1),
                 roughness: Union[Texture, float] = 0.5,
                 specular: Union[Texture, RGB] = (1, 1, 1),
                 emission: Union[Texture, RGB] = (0, 0, 0),
                 normal: Union[Texture, Vector3] = Vector3.ones()):
        self._base_color = (base_color if isinstance(base_color, Texture) else 
                            base_color if isinstance(base_color, Color4) else Color4(*base_color))
        self._roughness = (roughness if isinstance(roughness, Texture) else min(1, max(0, roughness)))
        self._specular = (specular if isinstance(specular, Texture) else 
                          specular if isinstance(specular, Color) else Color(*specular))
        self._emission = (emission if isinstance(emission, Texture) else 
                          emission if isinstance(emission, Color) else Color(*emission))
        self._normal = (normal if isinstance(normal, Texture) else 
                        normal.normalized if isinstance(normal, Vector3) else Vector3(*normal).normalized)

    @property
    def base_color(self):
        return self._base_color
    
    @property
    def roughness(self):
        return self._roughness
    
    @property
    def specular(self):
        return self._specular

    @property
    def emission(self):
        return self._emission
    
    @property
    def normal(self):
        return self._normal

    @base_color.setter
    def base_color(self, value: Union[Texture, RGBA]):
        if isinstance(value, Texture):
            self._base_color = value
        elif isinstance(self._base_color, Color4):
            self._base_color.rgba = tuple(value)
        else:
            self._base_color = Color4(*value)

    @roughness.setter
    def roughness(self, value: Union[Texture, float]):
        if isinstance(value, Texture):
            self._roughness = value
        else:
            self._roughness = max(0, min(1, value))

    @specular.setter
    def specular(self, value: Union[Texture, RGB]):
        if isinstance(value, Texture):
            self._specular = value
        elif isinstance(self._specular, Color):
            self._specular.rgb = tuple(value)
        else:
            self._specular = Color(*value)

    @emission.setter
    def emission(self, value: Union[Texture, RGB]):
        if isinstance(value, Texture):
            self._emission = value
        elif isinstance(self._emission, Color):
            self._emission.rgb = tuple(value)
        else:
            self._emission = Color(*value)

    @normal.setter
    def normal(self, value: Union[Texture, Vector3]):
        if isinstance(value, Texture):
            self._normal = value
        elif isinstance(self._normal, Vector3):
            self._normal.xyz = Vector3(*value).normalized.xyz
        else:
            self._normal = Vector3(*value).normalized

class RenderingPipeline(ABC):
    """Actual implementation should include type aliases and render function factories"""

    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def post(self):
        pass
