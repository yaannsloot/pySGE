import math
from typing import Optional, TypeAlias, Union, Optional
from OpenGL.GL import *
from OpenGL.GLU import *
from .core import Light, Camera
from .datatypes import Vec3, RGBA, RGB, Vector3, Color, Quaternion, Transform
from .pipeline import RenderingPipeline, RenderFunction, Material
from .gl_common_types import glMesh, glTexture

class glFFPMesh(glMesh):
    def draw_wire(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glEnableClientState(GL_VERTEX_ARRAY)
        self._bind_and_update_v()
        glVertexPointer(3, GL_FLOAT, 0, None)
        self._face_vbo.bind()
        glDrawElements(GL_TRIANGLES, self._face_count, GL_UNSIGNED_INT, None)
        self._face_vbo.unbind()
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
        self._face_vbo.bind()
        glDrawElements(GL_TRIANGLES, self._face_count, GL_UNSIGNED_INT, None)
        self._face_vbo.unbind()
        self._uv_vbo.unbind()
        self._color_vbo.unbind()
        self._normal_vbo.unbind()
        self._vert_vbo.unbind()
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

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
                 specular: Union[glTexture, float] = (1, 1, 1), 
                 emission: Union[glTexture, RGB] = (0, 0, 0),
                 shininess: float = 32.0):
        super().__init__(base_color=base_color,
                         specular=specular,
                         emission=emission)
        self.shininess = shininess

    def bind(self):
        lit = glIsEnabled(GL_LIGHTING)
        glActiveTexture(GL_TEXTURE0)
        if isinstance(self.base_color, glTexture):
            glEnable(GL_TEXTURE_2D)
            self.base_color.bind()
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        else:
            glDisable(GL_TEXTURE_2D)
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.base_color.rgba)
        glActiveTexture(GL_TEXTURE1)
        if isinstance(self.specular, glTexture):
            glEnable(GL_TEXTURE_2D)
            self.specular.bind()
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE)
            glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_ADD)
            glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_RGB, GL_PREVIOUS)
            glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE1_RGB, GL_TEXTURE)
        else:
            glDisable(GL_TEXTURE_2D)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (*self.specular.rgb, 1.0))
        if isinstance(self.emission, glTexture):
            glActiveTexture(GL_TEXTURE0)
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE)
            glEnable(GL_TEXTURE_2D)
            self.emission.bind()
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        else:
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (*self.emission, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, self.shininess)
        glActiveTexture(GL_TEXTURE0)
        if lit:
            glEnable(GL_LIGHTING)

    def unbind(self):
        for unit in (GL_TEXTURE2, GL_TEXTURE1, GL_TEXTURE0):
            glActiveTexture(unit)
            glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE0)
        glTexture.unbind()

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
                    material.bind()
                mesh.draw_solid()
                if material:
                    material.unbind()
            if wire:
                glDisable(GL_LIGHTING)
                glColor3f(*wire_color)
                mesh.draw_wire()
            glPopMatrix()
        return r_mesh