from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import pywavefront
import OpenImageIO as oiio
from typing import Callable, TypeAlias, Annotated, Literal, Any, Union
from .datatypes import Vector3, Vector2, Color4, Color, RGBA, RGB, Transform
from enum import Enum

RenderFunction: TypeAlias = Callable[[Transform],None] # Must be a world transform
CPUPointBuffer: TypeAlias = Annotated[npt.NDArray[np.float32], Literal["N", 2]]
CPUVertexBuffer: TypeAlias = Annotated[npt.NDArray[np.float32], Literal["N", 3]]
CPUPixelBuffer: TypeAlias = Annotated[npt.NDArray[Union[np.uint8, np.uint16, np.float16, np.float32]], Literal["H", "W", "C"]]

class WrapMethod(Enum):
    REPEAT = 1
    MIRRORED = 2
    CLAMP_TO_EDGE = 3
    CLAMP_TO_BORDER = 4

class FilterMethod(Enum):
    NEAREST = 1
    LINEAR = 2

class Mesh(ABC):
    def __init__(self, 
                 vertices:list[CPUVertexBuffer],
                 normals:list[CPUVertexBuffer],
                 vertex_colors:list[CPUVertexBuffer],
                 uv: list[CPUPointBuffer]):
        self._v_buf = vertices
        self._n_buf = normals
        self._c_buf = vertex_colors
        self._uv_buf = uv
        
    @property
    @abstractmethod
    def vertices(self) -> tuple[tuple[Vector3]]:
        # Should return a pre-initialized tuple of Vector3s WITH views to internal CPU buffers.
        # A syncronization mechanism should also ideally be implemented to push partial updates to the GPU.
        pass

    @property
    @abstractmethod
    def normals(self) -> tuple[tuple[Vector3]]:
        # Should return a pre-initialized tuple of Vector3s WITH views to internal CPU buffers.
        # A syncronization mechanism should also ideally be implemented to push partial updates to the GPU.
        # Unlike the layout of vertices and vertex colors, this will return an array with the same
        # flattened shape as UVs, as normals are per vertex, per face.
        pass

    @property
    @abstractmethod
    def colors(self) -> tuple[tuple[Color]]:
        # Should return a pre-initialized tuple of Colors WITH views to internal CPU buffers.
        # A syncronization mechanism should also ideally be implemented to push partial updates to the GPU.
        pass

    @property
    @abstractmethod
    def uv(self) -> tuple[tuple[Vector2]]:
        # Should return a pre-initialized tuple of Vector2s WITH views to internal CPU buffers.
        # A syncronization mechanism should also ideally be implemented to push partial updates to the GPU.
        # Unlike the formatting expected by the initializer, this function should return 
        # a flattened tuple in the shape of 3*s, 2, 2 being the Vector2 reference.
        pass

    @abstractmethod
    def draw_wire(self):
        pass

    @abstractmethod
    def draw_solid(self, material_idx:int):
        pass

    @abstractmethod
    def _gpu_sync(self):
        # If any CPU buffers are dirty, this will reupload the buffer in bulk to the GPU.
        # Meant for cases where there is a significant amount of edits between frames.
        pass

    @property
    def num_materials(self):
        return len(self._v_buf)

    @staticmethod
    def _load_mesh_from_obj(path:str):
        scene = pywavefront.Wavefront(path, collect_faces=True)
        faces = []
        data = {
            "V3F": {
                "len": 3,
                "d": []
            },
            "N3F": {
                "len": 3,
                "d": []
            },
            "T2F": {
                "len": 2,
                "d": []
            },
            "C3F": {
                "len": 3,
                "d": []
            },
        }
        for mesh in scene.mesh_list:
            for m, mat in enumerate(mesh.materials):
                layout = mat.vertex_format.split('_')
                seg_n = sum(data[l]["len"] for l in layout)
                vertices = np.array(mat.vertices, np.float32)
                vertices = vertices.reshape((vertices.size // seg_n, seg_n))
                offset = 0
                for l in layout:
                    length = data[l]["len"]
                    s = vertices[:,offset:offset + length]
                    data[l]["d"].append(s)
                    offset += length
                if len(data["C3F"]["d"]) != len(data["V3F"]["d"]):
                    data["C3F"]["d"].append(np.ones_like(data["V3F"]["d"][m]))
                if len(data["T2F"]["d"]) != len(data["V3F"]["d"]):
                    data["T2F"]["d"].append(None)
        return data["V3F"]["d"], data["N3F"]["d"], data["C3F"]["d"], data["T2F"]["d"]

    @classmethod
    def from_obj_file(cls, path:str) -> "Mesh":
        return cls(*cls._load_mesh_from_obj(path))

class Texture(ABC):
    def __init__(self, 
                 data: CPUPixelBuffer, 
                 wrap_method = WrapMethod.REPEAT,
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

    @staticmethod
    def _load_img_file(path):
        img = oiio.ImageInput.open(path)
        if img:
            data = img.read_image()
            img.close()
            data = np.flipud(data)
            return data

    @classmethod
    def from_file(cls, path):
        img = cls._load_img_file(path)
        if img:
            return cls(img)

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
