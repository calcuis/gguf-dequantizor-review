
import numpy as np

# original values
activation = np.array([1, 2, 3, 4])
weight = np.array([5, 6, 7, 8])

# quantization parameters
bit = 16  # Desired bit precision
min_val = min(np.min(activation), np.min(weight))
max_val = max(np.max(activation), np.max(weight))

# calculate scale factor
scale_factor = (2 ** (bit - 1) - 1) / max(abs(min_val), abs(max_val))

# quantize activation and weight values
quantized_activation = np.round(activation * scale_factor).astype(np.int16)
quantized_weight = np.round(weight * scale_factor).astype(np.int16)

# dequantize activation and weight values
dequantized_activation = quantized_activation / scale_factor
dequantized_weight = quantized_weight / scale_factor

# print values
print("Original activation:", activation)
print("Original weight:", weight)
print("Minimum value:", min_val)
print("Maximum value:", max_val)
print("Scale factor:", scale_factor)
print("Quantized activation:", quantized_activation)
print("Quantized weight:", quantized_weight)
print("Dequantized activation:", dequantized_activation)
print("Dequantized weight:", dequantized_weight)

# dummy output
output = np.sum(dequantized_activation * dequantized_weight)
print("Dequantized output:", output) # 70.00183110125477
