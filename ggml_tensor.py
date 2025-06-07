
import torch # need torch to work

class GGMLTensor(torch.Tensor):
    def __init__(self, tensor_type, tensor_shape, patches=[]):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, 'tensor_type', None)
        new.tensor_shape = getattr(self, 'tensor_shape', new.data.shape)
        new.patches = getattr(self, 'patches', []).copy()
        return new
    def clone(self):
        return self
    def detach(self):
        return self
    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            print(f"ignoring 'copy_' on tensor: {e}")
    def empty_(self, size, *args, **kwargs):
        new_tensor = super().empty_(size, *args, **kwargs)
        return GGMLTensor(new_tensor, tensor_type=getattr(self,
            'tensor_type', None), tensor_shape=size, patches=getattr(self,
            'patches', []).copy())
    @property
    def shape(self):
        if not hasattr(self, 'tensor_shape'):
            self.tensor_shape = self.size()
        return self.tensor_shape
if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'disable'):
    torch_compiler_disable = torch.compiler.disable
else:
    def torch_compiler_disable():
        def noop(x):
            return x
        return noop
