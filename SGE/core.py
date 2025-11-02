import math
import uuid
from typing import Union, Optional, TypeAlias
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from .datatypes import Vector3, Vector2, Quaternion, Transform, Vec3, Vec2
from .pipeline import RenderFunction

ParentObj: TypeAlias = Union["Object", "Scene"]

@dataclass
class Object:
    local_transform: Transform = field(default_factory=Transform, compare=False)
    render_func: Optional[RenderFunction] = field(default=None, compare=False)
    priority: int = 0 # lower is higher
    _uuid: uuid.UUID = field(default_factory=uuid.uuid4, init=False, compare=True)
    _parent: Optional[ParentObj] = field(default=None, init=False)
    _children: set["Object"] = field(default_factory=set, init=False, compare=False)

    def set_parent(self, parent:ParentObj) -> None:
        if self._parent is not None:
            # Should ideally fix transforms here as well. TODO for later.
            if isinstance(self._parent, Scene):
                self._parent._objects.discard(self)
            else:
                self._parent._children.discard(self)
        self._parent = parent
        if isinstance(parent, Scene):
            parent._objects.add(self)
        else:
            parent._children.add(self)

    def render(self) -> None:
        if self.render_func is None:
            return
        self.render_func(self.world_transform)

    def rotate_local(self, axis:Vec3, angle:float):
        rot = self.local_transform.rot
        axis = Vector3(*axis).normalized
        r = math.radians(angle)
        rh = r * 0.5
        sh = math.sin(rh)
        ch = math.cos(rh)
        q = Quaternion(
            ch,
            axis.x * sh,
            axis.y * sh,
            axis.z * sh
        ).normalized
        self.local_transform.rot = (rot * q).normalized

    def rotate_world(self, axis:Vec3, angle:float):
        rot = self.local_transform.rot
        axis = Vector3(*axis).normalized
        r = math.radians(angle)
        rh = r * 0.5
        sh = math.sin(rh)
        ch = math.cos(rh)
        q = Quaternion(
            ch,
            axis.x * sh,
            axis.y * sh,
            axis.z * sh
        ).normalized
        if isinstance(self._parent, Object):
            r_parent = self._parent.world_transform.rot
            q = (r_parent.inv * q * r_parent).normalized
        self.local_transform.rot = (q * rot).normalized

    def __hash__(self):
        return hash(self._uuid)

    @property
    def world_transform(self) -> Transform:
        if isinstance(self._parent, Object):
            return self._parent.world_transform @ self.local_transform
        return self.local_transform

class Light(Object, ABC):
    def __init__(self, local_transform: Optional[Transform] = None):
        local_transform = Transform() if local_transform is None else local_transform
        super().__init__(local_transform, self._render)
        self.priority = -99999 # second
        
    @abstractmethod
    def _render(self, wt: Transform) -> None:
        pass

class Camera(Object, ABC):
    def __init__(self,
                 local_transform: Optional[Transform] = None,
                 viewport_dims: Vec2 = (1, 1),
                 fov: float = 60.0,
                 near: float = 1.0,
                 far: float = 100.0,
                 orthographic: bool = False,
                 ortho_size = 10.0):
        local_transform = Transform() if local_transform is None else local_transform
        self._v_dims = viewport_dims if isinstance(viewport_dims, Vector2) else Vector2(*viewport_dims)
        self.fov = fov
        self.near = near
        self.far = far
        self.orthographic = orthographic
        self.ortho_size = ortho_size
        super().__init__(local_transform, self._render)
        self.priority = -100000 # first

    @property
    def fov(self) -> float:
        return self._vfov
    
    @fov.setter
    def fov(self, value: float):
        self._vfov = min(180.0, max(0.0, value))

    @property
    def w(self) -> int:
        return self._v_dims.x
    
    @w.setter
    def w(self, value:int):
        self._v_dims.x = max(1, value)

    @property
    def h(self) -> int:
        return self._v_dims.y
    
    @h.setter
    def h(self, value:int):
        self._v_dims.y = max(1, value)

    @property
    def aspect(self):
        return self._v_dims.x / self._v_dims.y

    @abstractmethod
    def _render(self, wt: Transform) -> None:
        pass

@dataclass
class Scene:
    active_camera: Optional[Camera] = None
    _objects: set[Object] = field(default_factory=set, init=False, compare=False)
    _uuid: uuid.UUID = field(default_factory=uuid.uuid4, init=False, compare=True)

    def _recursive_render(self, obj: Object, parent_wt: Transform, rendered: set[Object], bins: dict[int, list[tuple[Object, Transform]]]) -> None:
        if obj in rendered:
            raise RuntimeError("Cycle detected in scene graph")
        rendered.add(obj)
        wt = parent_wt @ obj.local_transform
        for child in obj._children:
            self._recursive_render(child, wt, rendered, bins)
        if obj.render_func is None:
            return
        if isinstance(obj, Camera) and obj != self.active_camera:
            return
        bins.setdefault(obj.priority, []).append((obj, wt))

    def render(self) -> None:
        if self.active_camera is None:
            return
        rendered: set[Object] = set()
        identity_tf = Transform()
        priority_bins: dict[int, list[tuple[Object, Transform]]] = dict()
        for obj in self._objects:
            self._recursive_render(obj, identity_tf, rendered, priority_bins)
        bin_order = sorted(list(priority_bins.keys()))
        for bin in bin_order:
            for obj, wt in priority_bins[bin]:
                obj.render_func(wt)

    def __hash__(self):
        return hash(self._uuid)
