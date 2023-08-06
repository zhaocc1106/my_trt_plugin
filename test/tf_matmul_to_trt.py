import numpy as np
import os
import ctypes

import pycuda.driver as cuda
import tensorrt as trt
import tensorflow as tf
import tf2onnx
from onnx import helper

# called to init pyCUDA
import pycuda.autoinit

_TENSORFLOW_DOMAIN = "zcc"


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, inputs):
        return tf.matmul(inputs[0], inputs[1])


if __name__ == '__main__':
    model = MyModel()
    inp1 = np.random.normal(size=[3, 4]).astype(np.float32)
    inp2 = np.random.normal(size=[4, 5]).astype(np.float32)

    a = tf.convert_to_tensor(inp1)
    b = tf.convert_to_tensor(inp2)
    with tf.device('cpu:0'):
        c = model([a, b])

    # Save to onnx
    model_onnx_path = 'my_mat_mul.onnx'
    print('Exporting model to ONNX format...')
    tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                                       tf.TensorSpec(shape=[None, None], dtype=tf.float32)],
                               opset=12, output_path=model_onnx_path, custom_ops={'MatMul': _TENSORFLOW_DOMAIN})
    print('Model exported to ' + model_onnx_path)

    # 转换成TRT
    cmd = 'trtexec --onnx={} --saveEngine={} ' \
          '--minShapes=args_0:1x1,args_1:1x1 ' \
          '--maxShapes=args_0:1024x1024,args_1:1024x1024 ' \
          '--optShapes=args_0:512x512,args_1:512x512 ' \
          '--exportLayerInfo=./generator_trt_layer_info.json ' \
          '--plugins=/usr/local/lib/libzcc_plugin.so --skipInference' \
        .format(model_onnx_path, './my_mat_mul.trt')
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
    idx_input = trt_engine['args_0']
    idx_other = trt_engine['args_1']
    trt_ctx.set_binding_shape(idx_input, (3, 4))
    trt_ctx.set_binding_shape(idx_other, (4, 5))

    # 将数据从cpu拷贝到gpu
    input_ca = np.ascontiguousarray(inp1)
    other_ca = np.ascontiguousarray(inp2)
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
    print('tf output: {}'.format(c))
    print('trt_out: \n{}'.format(h_output))
