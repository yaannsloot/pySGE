from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from .pipeline import Mesh
from .datatypes import Vector3, Vector2, Color
from .utils import SignalingProxyBuffer, group

class glMesh(Mesh):
    def __init__(self, vertices, normals, vertex_colors, faces, uv):
        super().__init__(vertices, normals, vertex_colors, faces, uv)
        self._v_writes = set()
        self._n_writes = set()
        self._c_writes = set()
        self._uv_writes = set()
        self._v_refs = tuple([
            Vector3(buffer=SignalingProxyBuffer(self._v_buf[i], buffer_index=i, modified_list=self._v_writes))
            for i in range(self._v_buf.shape[0])])
        self._n_refs = tuple([
            Vector3(buffer=SignalingProxyBuffer(self._n_buf[i], buffer_index=i, modified_list=self._n_writes))
            for i in range(self._n_buf.shape[0])])
        self._c_refs = tuple([
            Color(buffer=SignalingProxyBuffer(self._c_buf[i], buffer_index=i, modified_list=self._c_writes))
            for i in range(self._c_buf.shape[0])])
        self._uv_refs = tuple([
            Vector2(buffer=SignalingProxyBuffer(self._uv_buf[i], buffer_index=i, modified_list=self._uv_writes))
            for i in range(self._uv_buf.shape[0])])
        self._vert_vbo = vbo.VBO(self._v_buf)
        self._normal_vbo = vbo.VBO(self._n_buf)
        self._color_vbo = vbo.VBO(self._c_buf)
        self._face_vbo = vbo.VBO(self._f_buf.ravel(), target=GL_ELEMENT_ARRAY_BUFFER)
        self._uv_vbo = vbo.VBO(self._uv_buf)
        self._face_count = self._f_buf.size

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
