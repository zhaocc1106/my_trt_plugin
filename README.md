# 实现自己的tensorrt算子插件

## 环境
* CUDA: 11.6
* TENSORRT: TensorRT-8.6.1.6

## 构建
```bash
mkdir build
cd build
cmake ..
make -j8
make install
```

## 测试
### 自定义matmul算子
```bash
cd test
# 生成trt engine并对比trt算子结果与torch算子结果
python3 torch_matmul_to_trt.py
# 生成trt engine并对比trt算子结果与tf算子结果
python3 tf_matmul_to_trt.py
```
