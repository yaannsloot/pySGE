from OpenGL.GL import *
from OpenGL.GLU import *
from .pipeline import Texture, WrapMethod, FilterMethod
from .datatypes import Color4
from .utils import SignalingProxyBuffer, group

class glTexture(Texture):
    def __init__(self, data, dynamic=True):
        super().__init__(data)
        self._data_writes = set()
        w = self.width
        h = self.height
        self._data_refs = tuple([tuple([
                Color4(buffer=SignalingProxyBuffer(self._data[j][i], buffer_index=j * w + i, modified_list=self._data_writes))
                for i in range(w)]) for j in range(h)]) if dynamic else False
        self._data_gl_texid = glGenTextures(1)
        self.bind()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 
                     0, GL_RGBA, GL_FLOAT, self._data)
        self.unbind()
        
    @property
    def data(self):
        return self._data_refs

    def gpu_sync(self):
        self.bind()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 
                     0, GL_RGBA, GL_FLOAT, self._data)
        self.unbind()

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self._data_gl_texid)
        if self.wrap_method == WrapMethod.REPEAT:
            mode = GL_REPEAT
        elif self.wrap_method == WrapMethod.MIRRORED:
            mode = GL_MIRRORED_REPEAT
        elif self.wrap_method == WrapMethod.CLAMP_TO_EDGE:
            mode = GL_CLAMP_TO_EDGE
        else:
            mode = GL_CLAMP_TO_BORDER
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, self.border_color.rgba)
        if self.filter_method == FilterMethod.NEAREST:
            filter_mode = GL_NEAREST
        else:
            filter_mode = GL_LINEAR
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    @classmethod
    def from_file(cls, path, dynamic=True):
        img = cls._load_img_file(path)
        if img is not None:
            return cls(img, dynamic)

    @staticmethod
    def unbind():
        glBindTexture(GL_TEXTURE_2D, 0)
    
