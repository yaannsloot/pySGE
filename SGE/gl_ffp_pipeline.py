import math
from typing import Optional, TypeAlias, Union, Optional, Iterable
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from .core import Light, Camera
from .utils import SignalingProxyBuffer, group
from .datatypes import Vec3, RGBA, RGB, Vector3, Vector2, Color, Quaternion, Transform
from .pipeline import RenderingPipeline, RenderFunction, Mesh, Material
from .gl_common_types import glTexture

class glFFPMesh(Mesh):
    def __init__(self, vertices, normals, vertex_colors, uv, dynamic=True):
        super().__init__(vertices, normals, vertex_colors, uv)
        n_m = self.num_materials
        self._v_writes = [set() for _ in range(n_m)]
        self._n_writes = [set() for _ in range(n_m)]
        self._c_writes = [set() for _ in range(n_m)]
        self._uv_writes = [set() for _ in range(n_m)]
        self._v_refs = tuple([tuple([
                Vector3(buffer=SignalingProxyBuffer(
                    buffer=self._v_buf[m][i],
                    buffer_index=i,
                    modified_list=self._v_writes[m])
                ) for i in range(self._v_buf[m].shape[0])
            ]) for m in range(n_m)]) if dynamic else ()
        self._n_refs = tuple([tuple([
                Vector3(buffer=SignalingProxyBuffer(
                    buffer=self._n_buf[m][i],
                    buffer_index=i,
                    modified_list=self._n_writes[m])
                ) for i in range(self._n_buf[m].shape[0])
            ]) for m in range(n_m)]) if dynamic else ()
        self._c_refs = tuple([tuple([
                Color(buffer=SignalingProxyBuffer(
                    buffer=self._c_buf[m][i],
                    buffer_index=i,
                    modified_list=self._c_writes[m])
                ) for i in range(self._c_buf[m].shape[0])
            ]) for m in range(n_m)]) if dynamic else ()
        self._uv_refs = tuple([tuple([
                Vector2(buffer=SignalingProxyBuffer(
                    buffer=self._uv_buf[m][i],
                    buffer_index=i,
                    modified_list=self._uv_writes[m])
                ) for i in range(self._uv_buf[m].shape[0])
            ]) if self._uv_buf[m] is not None else None for m in range(n_m)]) if dynamic else ()
        self._vert_vbo = []
        self._normal_vbo = []
        self._color_vbo = []
        self._uv_vbo = []
        for m in range(n_m):
            self._vert_vbo.append(vbo.VBO(self._v_buf[m]))
            self._normal_vbo.append(vbo.VBO(self._n_buf[m]))
            self._color_vbo.append(vbo.VBO(self._c_buf[m]))
            self._uv_vbo.append(vbo.VBO(self._uv_buf[m]) if self._uv_buf[m] is not None else None)

    @property
    def vertices(self) -> tuple[tuple[Vector3]]:
        return self._v_refs
    
    @property
    def normals(self) -> tuple[tuple[Vector3]]:
        return self._n_refs
    
    @property
    def colors(self) -> tuple[tuple[Color]]:
        return self._c_refs
    
    @property
    def uv(self) -> tuple[tuple[Vector2]]:
        return self._uv_refs

    def _push_writes(self, write_list, buffer, vec_n=3):
        # For float32 buffers only
        # MUST BIND BEFORE CALLING
        if not write_list:
            return
        write_groups = group(write_list)
        for g in write_groups:
            first = g[0]
            last = g[-1]
            g_data = buffer[first:last+1]
            glBufferSubData(GL_ARRAY_BUFFER, first * vec_n * 4, g_data.nbytes, g_data)

    def _gpu_sync(self):
        for m in range(self.num_materials):
            if self._v_writes[m]:
                self._vert_vbo[m].bind()
                glBufferSubData(GL_ARRAY_BUFFER, 0, self._v_buf[m].nbytes, self._v_buf[m])
                self._vert_vbo[m].unbind()
                self._v_writes[m].clear()
            if self._n_writes[m]:
                self._normal_vbo[m].bind()
                glBufferSubData(GL_ARRAY_BUFFER, 0, self._n_buf[m].nbytes, self._n_buf[m])
                self._normal_vbo[m].unbind()
                self._n_writes[m].clear()
            if self._c_writes[m]:
                self._color_vbo[m].bind()
                glBufferSubData(GL_ARRAY_BUFFER, 0, self._c_buf[m].nbytes, self._c_buf[m])
                self._color_vbo[m].unbind()
                self._c_writes[m].clear()
            if self._uv_writes[m] and self._uv_vbo[m] is not None:
                self._uv_vbo[m].bind()
                glBufferSubData(GL_ARRAY_BUFFER, 0, self._uv_buf[m].nbytes, self._uv_buf[m])
                self._uv_vbo[m].unbind()
                self._uv_writes[m].clear()

    def _bind_and_update_v(self, material_idx):
        self._vert_vbo[material_idx].bind()
        self._push_writes(self._v_writes[material_idx], self._v_buf[material_idx])

    def _bind_and_update_n(self, material_idx):
        self._normal_vbo[material_idx].bind()
        self._push_writes(self._n_writes[material_idx], self._n_buf[material_idx])

    def _bind_and_update_c(self, material_idx):
        self._color_vbo[material_idx].bind()
        self._push_writes(self._c_writes[material_idx], self._c_buf[material_idx])

    def _bind_and_update_uv(self, material_idx):
        if self._uv_vbo[material_idx] is None:
            return
        self._uv_vbo[material_idx].bind()
        self._push_writes(self._uv_writes[material_idx], self._uv_buf[material_idx], vec_n=2)

    def _unbind_v(self, material_idx):
        self._vert_vbo[material_idx].unbind()

    def _unbind_n(self, material_idx):
        self._normal_vbo[material_idx].unbind()

    def _unbind_c(self, material_idx):
        self._color_vbo[material_idx].unbind()

    def _unbind_uv(self, material_idx):
        if self._uv_vbo[material_idx] is None:
            return
        self._uv_vbo[material_idx].unbind()

    def draw_wire(self):
        for m in range(self.num_materials):
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glEnableClientState(GL_VERTEX_ARRAY)
            self._bind_and_update_v(m)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glDrawElements(GL_TRIANGLES, self._v_buf[m].shape[0], GL_UNSIGNED_INT, None)
            self._unbind_v(m)
            glDisableClientState(GL_VERTEX_ARRAY)

    def draw_solid(self, material_idx):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnable(GL_NORMALIZE)
        self._bind_and_update_v(material_idx)
        glVertexPointer(3, GL_FLOAT, 0, None)
        self._bind_and_update_n(material_idx)
        glNormalPointer(GL_FLOAT, 0, None)
        self._bind_and_update_c(material_idx)
        glColorPointer(3, GL_FLOAT, 0, None)
        self._bind_and_update_uv(material_idx)
        glTexCoordPointer(2, GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, self._v_buf[material_idx].shape[0])
        self._unbind_uv(material_idx)
        self._unbind_c(material_idx)
        self._unbind_n(material_idx)
        self._unbind_v(material_idx)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    @classmethod
    def from_obj_file(cls, path, dynamic=True):
        return cls(*cls._load_mesh_from_obj(path), dynamic)

class glFFPLight(Light):
    def __init__(self, 
                 local_transform: Optional[Transform] = None,
                 color: RGB = (1.0, 1.0, 1.0),
                 ambient: RGB = (0.0, 0.0, 0.0),
                 attenuation: Vec3 = (0.5, 0.02, 0.01),
                 intensity: float = 1.0,
                 directional: bool = False,
                 index: int = 0):
        super().__init__(local_transform)
        self.color = Color(*color)
        self.ambient = Color(*ambient)
        self.attenuation = Vector3(*attenuation)
        self.intensity = intensity
        self.directional = directional
        self.index = index

    @property
    def index(self):
        return self._gl_light_idx
    
    @index.setter
    def index(self, value:int):
        if value < 0 or value > 7:
            raise ValueError("index must be (0 <= idx <= 7)")
        if value == 0:
            self._gl_light_idx = GL_LIGHT0
        elif value == 1:
            self._gl_light_idx = GL_LIGHT1
        elif value == 2:
            self._gl_light_idx = GL_LIGHT2
        elif value == 3:
            self._gl_light_idx = GL_LIGHT3
        elif value == 4:
            self._gl_light_idx = GL_LIGHT4
        elif value == 5:
            self._gl_light_idx = GL_LIGHT5
        elif value == 6:
            self._gl_light_idx = GL_LIGHT6
        else:
            self._gl_light_idx = GL_LIGHT7

    def _render(self, wt):
        glPushMatrix()
        glMultMatrixf(list(wt.mat.T.m))
        glLightfv(self._gl_light_idx, GL_POSITION, (0.0, 0.0, 
                                       1.0 if self.directional else 0.0, 
                                       0.0 if self.directional else 1.0))
        l = tuple(c * self.intensity for c in self.color)
        glLightfv(self._gl_light_idx, GL_DIFFUSE, (*l, 1.0))
        glLightfv(self._gl_light_idx, GL_SPECULAR, (*l, 1.0))
        glLightfv(self._gl_light_idx, GL_AMBIENT, (*self.ambient, 1.0))
        if not self.directional:
            glLightf(self._gl_light_idx, GL_CONSTANT_ATTENUATION,  self.attenuation[0])
            glLightf(self._gl_light_idx, GL_LINEAR_ATTENUATION,    self.attenuation[1])
            glLightf(self._gl_light_idx, GL_QUADRATIC_ATTENUATION, self.attenuation[2])
        else:
            glLightf(self._gl_light_idx, GL_CONSTANT_ATTENUATION,  1.0)
            glLightf(self._gl_light_idx, GL_LINEAR_ATTENUATION,    0.0)
            glLightf(self._gl_light_idx, GL_QUADRATIC_ATTENUATION, 0.0)
        glEnable(self._gl_light_idx)
        glPopMatrix()

class glFFPCamera(Camera):
    def _render(self, wt):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.aspect
        if self.orthographic:
            glOrtho(-self.ortho_size*aspect, 
                    self.ortho_size*aspect, 
                    -self.ortho_size, 
                    self.ortho_size, 
                    self.near, 
                    self.far)
        else:
            gluPerspective(self._vfov, aspect, self.near, self.far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(list(wt.inv.mat.T.m))

class glFFPMaterial(Material):
    def __init__(self, 
                 base_color: Union[glTexture, RGBA] = (1, 1, 1, 1), 
                 specular: Union[glTexture, RGB] = (1, 1, 1), 
                 emission: Union[glTexture, RGB] = (0, 0, 0),
                 shininess: float = 32.0):
        super().__init__(base_color=base_color,
                         specular=specular,
                         emission=emission)
        self.shininess = shininess

    def draw(self, mesh: glFFPMesh, mesh_mat_idx: int):
        lit = glIsEnabled(GL_LIGHTING)
        if isinstance(self.base_color, glTexture):
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        else:
            glDisable(GL_TEXTURE_2D)
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.base_color.rgba)
        if isinstance(self.specular, Color):
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (*self.specular.rgb, 1.0))
        else:
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        if isinstance(self.emission, Color):
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (*self.emission.rgb, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, self.shininess)
        glActiveTexture(GL_TEXTURE0)
        if isinstance(self.base_color, glTexture):
            glEnable(GL_TEXTURE_2D)
            self.base_color.bind()
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        else:
            glDisable(GL_TEXTURE_2D)
        mesh.draw_solid(mesh_mat_idx)
        glActiveTexture(GL_TEXTURE0)
        if isinstance(self.specular, glTexture):
            glDepthMask(GL_FALSE)
            glDepthFunc(GL_EQUAL)
            glEnable(GL_BLEND)
            glBlendFunc(GL_DST_COLOR, GL_ONE)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_TEXTURE_2D)
            self.specular.bind()
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            mesh.draw_solid(mesh_mat_idx)
            glDepthMask(GL_TRUE)
            glDepthFunc(GL_LESS)
            glDisable(GL_BLEND) 
        if isinstance(self.emission, glTexture):
            glDepthMask(GL_FALSE)
            glDepthFunc(GL_EQUAL)
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_TEXTURE_2D)
            self.emission.bind()
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            mesh.draw_solid(mesh_mat_idx)
            glDepthMask(GL_TRUE)
            glDepthFunc(GL_LESS)
            glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE0)
        glTexture.unbind()
        if lit:
            glEnable(GL_LIGHTING)

class glFFPRenderingPipeline(RenderingPipeline):
    Mesh: TypeAlias = glFFPMesh
    Texture: TypeAlias = glTexture
    Material: TypeAlias = glFFPMaterial
    Light: TypeAlias = glFFPLight
    Camera: TypeAlias = glFFPCamera

    def __init__(self):
        super().__init__()
        self._g_ambient = Color(0.05, 0.05, 0.05)
        self.lighting = True

    @property
    def global_ambient_light(self) -> Color:
        return self._g_ambient
    
    @global_ambient_light.setter
    def global_ambient_light(self, value:RGB):
        self._g_ambient.rgb = tuple(value)

    def pre(self):
        glEnable(GL_DEPTH_TEST) 
        if self.lighting:
            glEnable(GL_LIGHTING)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (*self._g_ambient, 1.0))
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    def post(self):
        return # No-op since this pipeline doesn't use post processing
    
    # Pipeline-specific render functions
    @staticmethod
    def Circle(radius:float=1.0, segments:int=36, color:tuple[float,float,float]=(1.0, 0.0, 1.0)) -> RenderFunction:
        def r_circle(wt: Transform) -> None:
            glPushMatrix()
            glMultMatrixf(list(wt.mat.T.m))
            glColor3f(*color)
            glBegin(GL_LINE_LOOP)
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                glVertex3f(x, y, 0.0)
            glEnd()
            glPopMatrix()
        return r_circle
    
    @staticmethod
    def FilledCircle(radius: float = 1.0, segments: int = 36,
                 color: tuple[float, float, float] = (1.0, 0.5, 0.0)) -> RenderFunction:
        def r_circle(wt: Transform) -> None:
            glPushMatrix()
            glMultMatrixf(list(wt.mat.T.m))
            glColor3f(*color)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0.0, 0.0, 0.0)
            for i in range(segments + 1):
                angle = 2.0 * math.pi * i / segments
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                glVertex3f(x, y, 0.0)
            glEnd()

            glPopMatrix()
        return r_circle
    
    @staticmethod
    def Billboarded(cam_pos: Vector3, render_func: RenderFunction) -> RenderFunction:
        def wrapper(wt: Transform) -> None:
            forward = (cam_pos - wt.loc).normalized
            up = Vector3(0, 0, 1)
            rot = Quaternion.look_rotation(forward, up)
            wt_facing = Transform(
                loc=wt.loc,
                rot=rot,
                scale=wt.scale
            )
            render_func(wt_facing)
        return wrapper
    
    @staticmethod
    def MeshRenderer(mesh: Mesh, 
                     materials: Optional[list[glFFPMaterial]] = None, 
                     solid=True, 
                     flat_shading=False, 
                     wire=False, 
                     wire_color:tuple[float,float,float]=(1.0, 1.0, 1.0)) -> RenderFunction:
        if not isinstance(materials, Iterable):
            materials = [materials]
        def r_mesh(wt: Transform):
            glPushMatrix()
            glMultMatrixf(list(wt.mat.T.m))
            if solid:
                glShadeModel(GL_FLAT if flat_shading else GL_SMOOTH)
                for m in range(mesh.num_materials):
                    if m < len(materials) and materials[m] is not None:
                        materials[m].draw(mesh, m)
                    else:
                        mesh.draw_solid(m)
            if wire:
                glDisable(GL_LIGHTING)
                glColor3f(*wire_color)
                mesh.draw_wire()
            glPopMatrix()
        return r_mesh