import math
import numpy as np
from array import array
from typing import final, Union, TypeAlias
from dataclasses import dataclass, field
from .utils import index

Number: TypeAlias = Union[int,float]
RGB: TypeAlias = Union["Color", tuple[float, float, float]]
RGBA: TypeAlias = Union["Color4", tuple[float, float, float, float]]
Vec3: TypeAlias = Union["Vector3", tuple[float, float, float]]
Vec2: TypeAlias = Union["Vector2", tuple[float, float]]

class Color4:
    def __init__(self, r:float=1.0, g:float=1.0, b:float=1.0, a=1.0, buffer=None):
        self._c = np.zeros(4, dtype=np.float32) if buffer is None else buffer
        if buffer is None:
            self.r = r
            self.g = g
            self.b = b
            self.a = a

    @property
    def r(self):
        return self._c[0]

    @property
    def g(self):
        return self._c[1]

    @property
    def b(self):
        return self._c[2]
    
    @property
    def a(self):
        return self._c[3]

    @r.setter
    def r(self, val:float):
        val = max(0, min(val, 1))
        self._c[0] = val

    @g.setter
    def g(self, val:float):
        val = max(0, min(val, 1))
        self._c[1] = val

    @b.setter
    def b(self, val:float):
        val = max(0, min(val, 1))
        self._c[2] = val

    @a.setter
    def a(self, val:float):
        val = max(0, min(val, 1))
        self._c[3] = val

    @property
    def rgb(self)-> tuple[float, float, float]:
        return (self.r, self.g, self.b)
    
    @rgb.setter
    def rgb(self, value: tuple[float, float, float]):
        self.r = value[0]
        self.g = value[1]
        self.b = value[2]

    @property
    def rgba(self)-> tuple[float, float, float, float]:
        return tuple(self)
    
    @rgba.setter
    def rgba(self, value: tuple[float, float, float, float]):
        self.r = value[0]
        self.g = value[1]
        self.b = value[2]
        self.a = value[3]

    def __iter__(self):
        yield self.r
        yield self.g
        yield self.b
        yield self.a

    def __eq__(self, value:"Color4"):
        return all(a == b for a, b in zip(self, value))
    
    def __str__(self):
        return f"Color4(r={self.r:.2f}, g={self.g:.2f}, b={self.b:.2f}, a={self.a:.2f})"
    
    def __repr__(self):
        return str(self)

class Color:
    def __init__(self, r:float=1.0, g:float=1.0, b:float=1.0, buffer=None):
        self._c = np.zeros(3, dtype=np.float32) if buffer is None else buffer
        if buffer is None:
            self.r = r
            self.g = g
            self.b = b

    @property
    def r(self):
        return self._c[0]

    @property
    def g(self):
        return self._c[1]

    @property
    def b(self):
        return self._c[2]

    @r.setter
    def r(self, val:float):
        val = max(0, min(val, 1))
        self._c[0] = val

    @g.setter
    def g(self, val:float):
        val = max(0, min(val, 1))
        self._c[1] = val

    @b.setter
    def b(self, val:float):
        val = max(0, min(val, 1))
        self._c[2] = val

    @property
    def rgb(self)-> tuple[float, float, float]:
        return tuple(self)
    
    @rgb.setter
    def rgb(self, value: tuple[float, float, float]):
        self.r = value[0]
        self.g = value[1]
        self.b = value[2]

    def __iter__(self):
        yield self.r
        yield self.g
        yield self.b

    def __eq__(self, value:"Color"):
        return all(a == b for a, b in zip(self, value))
    
    def __str__(self):
        return f"Color(r={self.r:.2f}, g={self.g:.2f}, b={self.b:.2f})"
    
    def __repr__(self):
        return str(self)

class Vector3:
    def __init__(self, 
                 x:float=0.0, 
                 y:float=0.0, 
                 z:float=0.0,
                 buffer=None):
        self._v = np.array([x, y, z], dtype=np.float32) if buffer is None else buffer

    @property
    def x(self):
        return self._v[0]

    @property
    def y(self):
        return self._v[1]

    @property
    def z(self):
        return self._v[2]

    @x.setter
    def x(self, val:Number):
        self._v[0] = val

    @y.setter
    def y(self, val:Number):
        self._v[1] = val

    @z.setter
    def z(self, val:Number):
        self._v[2] = val

    def __mul__(self, other:Union["Vector3",Number]) -> "Vector3":
        if isinstance(other, Vector3):
            return Vector3(
                self.x * other.x,
                self.y * other.y,
                self.z * other.z
            )
        return Vector3(
            self.x * other,
            self.y * other,
            self.z * other
        )
    
    def __truediv__(self, other:Union["Vector3",Number]) -> "Vector3":
        if isinstance(other, Vector3):
            return Vector3(
                self.x / other.x,
                self.y / other.y,
                self.z / other.z
            )
        return Vector3(
            self.x / other,
            self.y / other,
            self.z / other
        )

    def __add__(self, other:Union["Vector3",Number]) -> "Vector3":
        if isinstance(other, Vector3):
            return Vector3(
                self.x + other.x,
                self.y + other.y,
                self.z + other.z
            )
        return Vector3(
            self.x + other,
            self.y + other,
            self.z + other
        )

    def __sub__(self, other:Union["Vector3",Number]) -> "Vector3":
        if isinstance(other, Vector3):
            return Vector3(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z
            )
        return Vector3(
            self.x - other,
            self.y - other,
            self.z - other
        )

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __eq__(self, value:"Vector3"):
        return all(a == b for a, b in zip(self, value))

    def __str__(self):
        return f"Vector3(x={self.x:f.4}, y={self.y:f.4}, z={self.z:f.4})"
    
    def __repr__(self):
        return str(self)

    def __getitem__(self, idx:int):
        return self._v[idx]
    
    def __setitem__(self, idx:int, value:Number):
        self._v[idx] = value

    @property
    def inv(self) -> "Vector3":
        return Vector3(
            1 / self.x if self.x != 0 else 0,
            1 / self.y if self.y != 0 else 0,
            1 / self.z if self.z != 0 else 0,
        )

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def normalized(self) -> "Vector3":
        m = self.magnitude
        if m == 0:
            return Vector3()
        return Vector3(
            self.x / m,
            self.y / m,
            self.z / m
        )    
    
    @property
    def xyz(self):
        return tuple(self)
    
    @xyz.setter
    def xyz(self, value: tuple[float, float, float]):
        self.x = value[0]
        self.y = value[1]
        self.z = value[2]

    @classmethod
    def ones(cls) -> "Vector3":
        return cls(1, 1, 1)

class Vector2:
    def __init__(self, 
                 x:float=0.0, 
                 y:float=0.0, 
                 buffer=None):
        self._v = np.array([x, y], dtype=np.float32) if buffer is None else buffer

    @property
    def x(self):
        return self._v[0]

    @property
    def y(self):
        return self._v[1]

    @x.setter
    def x(self, val:Number):
        self._v[0] = val

    @y.setter
    def y(self, val:Number):
        self._v[1] = val

    def __mul__(self, other:Union["Vector2",Number]) -> "Vector2":
        if isinstance(other, Vector2):
            return Vector2(
                self.x * other.x,
                self.y * other.y
            )
        return Vector2(
            self.x * other,
            self.y * other
        )
    
    def __truediv__(self, other:Union["Vector2",Number]) -> "Vector2":
        if isinstance(other, Vector2):
            return Vector2(
                self.x / other.x,
                self.y / other.y
            )
        return Vector2(
            self.x / other,
            self.y / other
        )

    def __add__(self, other:Union["Vector2",Number]) -> "Vector2":
        if isinstance(other, Vector2):
            return Vector2(
                self.x + other.x,
                self.y + other.y
            )
        return Vector2(
            self.x + other,
            self.y + other
        )

    def __sub__(self, other:Union["Vector2",Number]) -> "Vector2":
        if isinstance(other, Vector2):
            return Vector2(
                self.x - other.x,
                self.y - other.y
            )
        return Vector2(
            self.x - other,
            self.y - other
        )

    def __neg__(self) -> "Vector2":
        return Vector2(-self.x, -self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __eq__(self, value:"Vector2"):
        return all(a == b for a, b in zip(self, value))

    def __str__(self):
        return f"Vector2(x={self.x:f.4}, y={self.y:f.4})"
    
    def __repr__(self):
        return str(self)

    def __getitem__(self, idx:int):
        return self._v[idx]
    
    def __setitem__(self, idx:int, value:Number):
        self._v[idx] = value

    @property
    def inv(self) -> "Vector2":
        return Vector2(
            1 / self.x if self.x != 0 else 0,
            1 / self.y if self.y != 0 else 0
        )

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def normalized(self) -> "Vector2":
        m = self.magnitude
        if m == 0:
            return Vector2()
        return Vector2(
            self.x / m,
            self.y / m
        )    
    
    @property
    def xy(self):
        return tuple(self)
    
    @xy.setter
    def xy(self, value: tuple[float, float]):
        self.x = value[0]
        self.y = value[1]

    @classmethod
    def ones(cls) -> "Vector2":
        return cls(1, 1)

@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    @property
    def normalized(self) -> "Quaternion":
        m = self.magnitude
        if m == 0:
            return Quaternion()
        return Quaternion(
            self.w / m,
            self.x / m,
            self.y / m,
            self.z / m
        )

    @property
    def inv(self) -> "Quaternion":
        return Quaternion(
            self.w,
            -self.x,
            -self.y,
            -self.z
        )
    
    def _mul_quat(self, other:"Quaternion") -> "Quaternion":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        )

    def _mul_vec(self, other:Vector3) -> Vector3:
        qv = Quaternion(0, other.x, other.y, other.z)
        qr = self._mul_quat(qv)._mul_quat(self.inv)
        return Vector3(qr.x, qr.y, qr.z)

    def __mul__(self, other:Union["Quaternion",Vector3]) -> Union["Quaternion",Vector3]:
        if isinstance(other, Vector3):
            return self._mul_vec(other)
        return self._mul_quat(other)

    def __rmul__(self, other:Vector3) -> Vector3:
        return self._mul_vec(other)

    def set_euler(self, x:float, y:float, z:float) -> None:
        """XYZ must be in degrees"""
        x, y, z = map(math.radians, [x, y, z])
        cx = math.cos(x * 0.5)
        sx = math.sin(x * 0.5)
        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cz = math.cos(z * 0.5)
        sz = math.sin(z * 0.5)
        self.w = cx * cy * cz + sx * sy * sz
        self.x = sx * cy * cz - cx * sy * sz
        self.y = cx * sy * cz + sx * cy * sz
        self.z = cx * cy * sz - sx * sy * cz
        m = self.magnitude
        self.w /= m
        self.x /= m
        self.y /= m
        self.z /= m

    def get_euler(self) -> tuple[float, float, float]:
        q = self.normalized
        sxcy = 2 * (q.w * q.x + q.y * q.z)
        cxcy = 1 - 2 * (q.x**2 + q.y**2)
        x = math.atan2(sxcy, cxcy)
        sy = math.sqrt(1 + 2 * (q.w * q.y - q.x * q.z))
        cy = math.sqrt(1 - 2 * (q.w * q.y - q.x * q.z))
        y = 2 * math.atan2(sy, cy) - math.pi / 2
        szcx = 2 * (q.w * q.z + q.x * q.y)
        czcx = 1 - 2 * (q.y**2 + q.z**2)
        z = math.atan2(szcx, czcx)
        x, y, z = map(lambda k: k * (180/math.pi), [x, y, z])
        return x, y, z

    @classmethod
    def look_rotation(cls, forward: Vector3, up: Vector3) -> "Quaternion":
        f = forward.normalized
        r = Vector3(
            up.y*f.z - up.z*f.y,
            up.z*f.x - up.x*f.z,
            up.x*f.y - up.y*f.x
        )
        r = r.normalized
        u = Vector3(
            f.y*r.z - f.z*r.y,
            f.z*r.x - f.x*r.z,
            f.x*r.y - f.y*r.x
        )
        m00, m01, m02 = r.x, u.x, f.x
        m10, m11, m12 = r.y, u.y, f.y
        m20, m21, m22 = r.z, u.z, f.z
        t = m00 + m11 + m22
        if t > 0:
            s = math.sqrt(t+1.0) * 2
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        return cls(w, x, y, z).normalized

    @classmethod
    def from_euler(cls, x:float, y:float, z:float) -> "Quaternion":
        out = cls()
        out.set_euler(x, y, z)
        return out

@final
class Mat4x4:
    def __init__(self) -> None:
        self.m = array('f', [0.0]*16)
        self[0,0] = 1
        self[1,1] = 1
        self[2,2] = 1
        self[3,3] = 1

    def __getitem__(self, item: tuple[int,int]) -> float:
        i = index(item[0], 4)
        j = index(item[1], 4)
        return self.m[i*4 + j]

    def __setitem__(self, item: tuple[int,int], value: float) -> None:
        i = index(item[0], 4)
        j = index(item[1], 4)
        self.m[i*4 + j] = value

    def __matmul__(self, other: "Mat4x4") -> "Mat4x4":
        output = Mat4x4()
        for i in range(4):
            for j in range(4):
                prod = 0
                for k in range(4):
                    prod += self[i, k] * other[k, j]
                output[i, j] = prod
        return output

    @property
    def T(self) -> "Mat4x4":
        out = Mat4x4()
        for i in range(4):
            for j in range(4):
                out[i, j] = self[j, i]
        return out

    def to_string(self, precision: int = 4, width: int = 9) -> str:
        fmt = f"{{:{width}.{precision}f}}"
        lines: list[str] = []
        for i in range(4):
            row = " ".join(fmt.format(self[i, j]) for j in range(4))
            lines.append(f"[ {row} ]")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        vals = ", ".join(f"{self[i,j]:.6g}" for i in range(4) for j in range(4))
        return f"Transform([{vals}])"

@dataclass
class Transform:
    loc: Vector3 = field(default_factory=Vector3)
    rot: Quaternion = field(default_factory=Quaternion)
    scale: Vector3 = field(default_factory=Vector3.ones)

    def __matmul__(self, other: "Transform") -> "Transform":
        scale = self.scale * other.scale
        rot = self.rot * other.rot
        loc = self.loc + self.rot * (self.scale * other.loc)
        return Transform(loc, rot, scale)

    @property
    def inv(self) -> "Transform":
        inv_scale = self.scale.inv
        inv_rot = self.rot.inv
        return Transform(
            inv_rot * (inv_scale*(-self.loc)),
            inv_rot,
            scale=inv_scale
        )

    @property
    def mat(self) -> Mat4x4:
        tx, ty, tz = self.loc.x, self.loc.y, self.loc.z
        sx, sy, sz = self.scale.x, self.scale.y, self.scale.z
        w, x, y, z = self.rot.w, self.rot.x, self.rot.y, self.rot.z
        xx, yy, zz = x**2, y**2, z**2
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        r00 = 1 - 2*(yy+zz)
        r01 = 2*(xy - wz)
        r02 = 2*(xz + wy)
        r10 = 2*(xy + wz)
        r11 = 1 - 2*(xx+zz)
        r12 = 2*(yz - wx)
        r20 = 2*(xz - wy)
        r21 = 2*(yz + wx)
        r22 = 1 - 2*(xx+yy)
        r00 *= sx; r10 *= sx; r20 *= sx
        r01 *= sy; r11 *= sy; r21 *= sy
        r02 *= sz; r12 *= sz; r22 *= sz
        m = Mat4x4()
        m.m = array("f", 
            [r00, r01, r02, tx,
            r10, r11, r12, ty,
            r20, r21, r22, tz,
            0, 0, 0, 1]
        )
        return m