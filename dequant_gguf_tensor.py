
import torch # need torch to work

from dequant_tensor import dequantize_functions
from gguf_connector.reader import GGML_QUANT_SIZES

SUPPORTED_GGUF_QUANT_TYPES = list(dequantize_functions.keys())

def _quant_shape_from_byte_shape(shape, type_size, block_size):
    return (*shape[:-1], shape[-1] // type_size * block_size)

def dequantize_gguf_tensor(tensor):
    if not hasattr(tensor, "quant_type"):
        return tensor
    quant_type = tensor.quant_type
    dequant_fn = dequantize_functions[quant_type]
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    tensor = tensor.view(torch.uint8)
    shape = _quant_shape_from_byte_shape(tensor.shape, type_size, block_size)
    n_blocks = tensor.numel() // type_size
    blocks = tensor.reshape((n_blocks, type_size))
    dequant = dequant_fn(blocks, block_size, type_size)
    dequant = dequant.reshape(shape)
    return dequant.as_tensor()

class GGUFParameter(torch.nn.Parameter):
    def __new__(cls, data, requires_grad=False, quant_type=None):
        data = data if data is not None else torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quant_type = quant_type
        block_size, type_size = GGML_QUANT_SIZES[quant_type]
        self.quant_shape = _quant_shape_from_byte_shape(self.shape, type_size, block_size)
        return self

    def as_tensor(self):
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)

    @staticmethod
    def _extract_quant_type(args):
        for arg in args:
            if isinstance(arg, list) and isinstance(arg[0], GGUFParameter):
                return arg[0].quant_type
            if isinstance(arg, GGUFParameter):
                return arg.quant_type
        return None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)
        if isinstance(result, torch.Tensor):
            quant_type = cls._extract_quant_type(args)
            return cls(result, quant_type=quant_type)
        elif type(result) in (list, tuple):
            quant_type = cls._extract_quant_type(args)
            wrapped = [cls(x, quant_type=quant_type) if isinstance(x, torch.Tensor) else x for x in result]
            return type(result)(wrapped)
        else:
            return result

class GGUFLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype

    def forward(self, inputs):
        weight = dequantize_gguf_tensor(self.weight)
        weight = weight.to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
        output = torch.nn.functional.linear(inputs, weight, bias)
        return output
