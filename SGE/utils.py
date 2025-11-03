from typing import TypeVar, Generic, Union, Optional
from collections.abc import Iterable

def index(i: int, n: int) -> int:
    n = abs(n)
    if i >= n or i < -n:
        raise IndexError
    return i % n

def group(values):
    if not values:
        return None
    if not isinstance(values, set):
        values = set(values)
    values = sorted(list(values))
    groups = [[values[0]]]
    last = values[0]
    for i in range(1, len(values)):
        k = values[i]
        if k - last > 1:
            groups.append(list())
        groups[-1].append(k)
        last = k
    return groups

T = TypeVar('T')
class Mutable(Generic[T]):
    """One value container to keep references to values between objects"""
    def __init__(self, value:T):
        self.value = value

class SignalingProxyBuffer:
    def __init__(self, 
                 buffer,
                 flag:Union[Mutable[bool],None]=None, 
                 buffer_index:Optional[Union[set[int],list[int], int]]=None,
                 modified_list:Optional[Union[set[int],list[int]]]=None):
        self.buffer = buffer
        self.flag = flag
        self.buffer_index = buffer_index if isinstance(buffer_index, (list, set)) else [buffer_index]
        self.modified_list = modified_list

    def _notify_write(self):
        if self.flag is not None:
            self.flag.value = True
        if all(i is not None for i in (self.buffer_index, self.modified_list)):
            if isinstance(self.modified_list, list):
                self.modified_list.extend(self.buffer_index)
            else:
                self.modified_list.update(self.buffer_index)

    def __getitem__(self, idx:int):
        if self.buffer and isinstance(self.buffer, list) and isinstance(self.buffer[0], Iterable):
            return self.buffer[0][idx]
        return self.buffer[idx]
    
    def __setitem__(self, idx:int, value):
        self._notify_write()
        if self.buffer and isinstance(self.buffer, list) and isinstance(self.buffer[0], Iterable):
            for buf in self.buffer:
                buf[idx] = value
        else:
            self.buffer[idx] = value