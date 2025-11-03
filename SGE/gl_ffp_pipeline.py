import math
from typing import Optional, TypeAlias, Union, Optional
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from .core import Light, Camera
from .utils import SignalingProxyBuffer, group
from .datatypes import Vec3, RGBA, RGB, Vector3, Vector2, Color, Quaternion, Transform
from .pipeline import RenderingPipeline, RenderFunction, Mesh, Material
from .gl_common_types import glTexture

class glFFPMesh(Mesh):
    def __init__(self, vertices, normals, vertex_colors, faces, uv, dynamic=True):
        super().__init__(vertices, normals, vertex_colors, faces, uv)
        f_indices = self._f_buf.flatten()
        v_bins = [[] for _ in range(self._v_buf.shape[0])]
        v_n = self._v_buf.shape[0]
        for i, v in enumerate(f_indices):
            v_bins[v].append(i)
        self._v_buf = self._v_buf[self._f_buf].reshape(-1, 3)
        if self._n_buf.shape != self._v_buf.shape:
            self._n_buf = self._n_buf[self._f_buf].reshape(-1, 3)
        self._c_buf = self._c_buf[self._f_buf].reshape(-1, 3)
        self._uv_buf = self._uv_buf.reshape(-1, 2)
        self._v_writes = set()
        self._n_writes = set()
        self._c_writes = set()
        self._uv_writes = set()
        self._v_refs = tuple([
            Vector3(buffer=SignalingProxyBuffer(
                buffer = [self._v_buf[j] for j in v_bins[i]],
                buffer_index = v_bins[i],
                modified_list = self._v_writes)) for i in range(v_n)]) if dynamic else ()
        self._n_refs = tuple([
            Vector3(buffer=SignalingProxyBuffer(
                buffer = self._n_buf[i],
                buffer_index = i,
                modified_list = self._n_writes)) for i in range(self._n_buf.shape[0])]) if dynamic else ()
        self._c_refs = tuple([
            Vector3(buffer=SignalingProxyBuffer(
                buffer = [self._c_buf[j] for j in v_bins[i]],
                buffer_index = v_bins[i],
                modified_list = self._c_writes)) for i in range(v_n)]) if dynamic else ()
        self._uv_refs = tuple([
            Vector2(buffer=SignalingProxyBuffer(self._uv_buf[i], buffer_index=i, modified_list=self._uv_writes))
            for i in range(self._uv_buf.shape[0])]) if dynamic else ()
        self._vert_vbo = vbo.VBO(self._v_buf)
        self._normal_vbo = vbo.VBO(self._n_buf)
        self._color_vbo = vbo.VBO(self._c_buf)
        self._uv_vbo = vbo.VBO(self._uv_buf)

    @property
    def vertices(self) -> tuple[Vector3]:
        return self._v_refs
    
    @property
    def normals(self) -> tuple[Vector3]:
        return self._n_refs
    
    @property
    def colors(self) -> tuple[Vector3]:
        return self._c_refs
    
    @property
    def uv(self) -> tuple[Vector2]:
        return self._uv_refs

    def recalculate_normals(self):
        super().recalculate_normals()
        self._n_writes.add(0) # mark buffer as dirty

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
        if self._v_writes:
            self._vert_vbo.bind()
            glBufferSubData(GL_ARRAY_BUFFER, 0, self._v_buf.nbytes, self._v_buf)
            self._vert_vbo.unbind()
            self._v_writes.clear()
        if self._n_writes:
            self._normal_vbo.bind()
            glBufferSubData(GL_ARRAY_BUFFER, 0, self._n_buf.nbytes, self._n_buf)
            self._normal_vbo.unbind()
            self._n_writes.clear()
        if self._c_writes:
            self._color_vbo.bind()
            glBufferSubData(GL_ARRAY_BUFFER, 0, self._c_buf.nbytes, self._c_buf)
            self._color_vbo.unbind()
            self._c_writes.clear()
        if self._uv_writes:
            self._uv_vbo.bind()
            glBufferSubData(GL_ARRAY_BUFFER, 0, self._uv_buf.nbytes, self._uv_buf)
            self._uv_vbo.unbind()
            self._uv_writes.clear()

    def _bind_and_update_v(self):
        self._vert_vbo.bind()
        self._push_writes(self._v_writes, self._v_buf)

    def _bind_and_update_n(self):
        self._normal_vbo.bind()
        self._push_writes(self._n_writes, self._n_buf)

    def _bind_and_update_c(self):
        self._color_vbo.bind()
        self._push_writes(self._c_writes, self._c_buf)

    def _bind_and_update_uv(self):
        self._uv_vbo.bind()
        self._push_writes(self._uv_writes, self._uv_buf, vec_n=2)

    def draw_wire(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glEnableClientState(GL_VERTEX_ARRAY)
        self._bind_and_update_v()
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawElements(GL_TRIANGLES, self._face_count, GL_UNSIGNED_INT, None)
        self._vert_vbo.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)

    def draw_solid(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnable(GL_NORMALIZE)
        self._bind_and_update_v()
        glVertexPointer(3, GL_FLOAT, 0, None)
        self._bind_and_update_n()
        glNormalPointer(GL_FLOAT, 0, None)
        self._bind_and_update_c()
        glColorPointer(3, GL_FLOAT, 0, None)
        self._bind_and_update_uv()
        glTexCoordPointer(2, GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, self._v_buf.shape[0])
        self._uv_vbo.unbind()
        self._color_vbo.unbind()
        self._normal_vbo.unbind()
        self._vert_vbo.unbind()
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

    def draw(self, mesh: glFFPMesh):
        lit = glIsEnabled(GL_LIGHTING)
        glActiveTexture(GL_TEXTURE0)
        if isinstance(self.base_color, glTexture):
            glEnable(GL_TEXTURE_2D)
            self.base_color.bind()
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        else:
            glDisable(GL_TEXTURE_2D)
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.base_color.rgba)
        if isinstance(self.specular, Color):
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (*self.specular.rgb, 1.0))
        else:
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        if isinstance(self.emission, Color):
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (*self.emission, 1.0))
        mesh.draw_solid()
        if isinstance(self.specular, glTexture):
            glDepthMask(GL_FALSE)
            glDepthFunc(GL_EQUAL)
            glEnable(GL_BLEND)
            glBlendFunc(GL_DST_COLOR, GL_ONE)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_TEXTURE_2D)
            self.specular.bind()
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            mesh.draw_solid()
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
            mesh.draw_solid()
            glDepthMask(GL_TRUE)
            glDepthFunc(GL_LESS)
            glDisable(GL_BLEND)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, self.shininess)
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
                     material: Optional[glFFPMaterial] = None, 
                     solid=True, 
                     flat_shading=False, 
                     wire=False, 
                     wire_color:tuple[float,float,float]=(1.0, 1.0, 1.0)) -> RenderFunction:
        def r_mesh(wt: Transform):
            glPushMatrix()
            glMultMatrixf(list(wt.mat.T.m))
            if solid:
                glShadeModel(GL_FLAT if flat_shading else GL_SMOOTH)
                if material:
                    material.draw(mesh)
                mesh.draw_solid()
            if wire:
                glDisable(GL_LIGHTING)
                glColor3f(*wire_color)
                mesh.draw_wire()
            glPopMatrix()
        return r_mesh