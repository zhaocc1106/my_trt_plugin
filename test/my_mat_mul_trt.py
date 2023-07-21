import ctypes
import os
import time
import typing

import numpy as np
import onnx
import pycuda.driver as cuda
import onnxruntime as ort
import tensorrt as trt
import torch
from torch.onnx import symbolic_helper

# called to init pyCUDA
import pycuda.autoinit

_registered_ops: typing.AbstractSet[str] = set()

_OPSET_VERSION = 11


def _reg(symbolic_fn: typing.Callable):
    name = "::%s" % symbolic_fn.__name__
    torch.onnx.register_custom_op_symbolic(name, symbolic_fn, _OPSET_VERSION)
    _registered_ops.add(name)


def register_ops():
    """Register ONNX Runtime's built-in contrib ops.
    Should be run before torch.onnx.export().
    """

    def matmul(g, input, other):
        return g.op(
            "com.microsoft::MatMulPluginDynamic",
            input,
            other,
        )

    _reg(matmul)


class MyMatMul(torch.nn.Module):
    def __init__(self):
        super(MyMatMul, self).__init__()

    def forward(self, inp1, inp2):
        return torch.matmul(inp1, inp2)


if __name__ == "__main__":
    inp1 = torch.rand([3, 4])
    inp2 = torch.rand([4, 5])
    print('inp1:')
    print(inp1)
    print('inp2:')
    print(inp2)

    model = MyMatMul()
    model.eval()
    model = model.cuda()
    out = model(inp1.cuda(), inp2.cuda())

    # 转换成ONNX
    model_onnx_path = "./my_mat_mul.onnx"
    register_ops()  # 注册自定义mat_mul op
    print('Exporting model to ONNX format...')
    with torch.no_grad():
        torch.onnx.export(model, (inp1, inp2), model_onnx_path,
                          verbose=True, input_names=['input', 'other'], output_names=['out'],
                          dynamic_axes={'input': [0, 1], 'other': [0, 1], 'out': [0, 1]},
                          opset_version=_OPSET_VERSION)
    print('Model exported to ' + model_onnx_path)
    model_onnx = onnx.load(model_onnx_path)
    onnx.checker.check_model(model_onnx)
    print('Onnx model graph:')
    print(onnx.helper.printable_graph(model_onnx.graph))

    # 转换成TRT
    cmd = 'trtexec --onnx={} --saveEngine={} ' \
          '--minShapes=input:1x1,other:1x1 ' \
          '--maxShapes=input:1024x1024,other:1024x1024 ' \
          '--optShapes=input:512x512,other:512x512 ' \
          '--exportLayerInfo=./generator_trt_layer_info.json ' \
          '--plugins=/usr/local/lib/libzcc_plugin.so --skipInference' \
        .format('my_mat_mul.onnx', './my_mat_mul.trt')
    os.system(cmd)

    # trt gpu推理
    # 加载引擎
    ctypes.CDLL("/usr/local/lib/libzcc_plugin.so")
    with open('./my_mat_mul.trt', 'rb') as f:
        trt_engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
        inspector = trt_engine.create_engine_inspector()
        print('trt_engine layer_info:\n{}'.format(
            inspector.get_engine_information(trt.LayerInformationFormat(1))
        ))
        trt_ctx = trt_engine.create_execution_context()

    # malloc
    d_input = cuda.mem_alloc(3 * 4 * 4)
    d_other = cuda.mem_alloc(4 * 5 * 4)
    d_output = cuda.mem_alloc(3 * 5 * 4)

    # create a stream in which to copy inputs/outputs and run inference
    stream = cuda.Stream()

    # set shape
    idx_input = trt_engine['input']
    idx_other = trt_engine['other']
    trt_ctx.set_binding_shape(idx_input, (3, 4))
    trt_ctx.set_binding_shape(idx_other, (4, 5))

    # 将数据从cpu拷贝到gpu
    input_ca = np.ascontiguousarray(inp1.numpy())
    other_ca = np.ascontiguousarray(inp2.numpy())
    cuda.memcpy_htod_async(d_input, input_ca, stream)
    cuda.memcpy_htod_async(d_other, other_ca, stream)

    # 执行推理
    trt_ctx.execute_async_v2(
        bindings=[int(d_input), int(d_other), int(d_output)], stream_handle=stream.handle)

    # 将结果从gpu拷贝到cpu
    h_output = cuda.pagelocked_empty((3, 5), dtype=np.float32)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    # stream sync
    stream.synchronize()

    # 验证结果
    print('out: \n{}'.format(out))
    print('trt_out: \n{}'.format(h_output))
