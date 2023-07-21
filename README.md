# 实现自定义tensorrt插件

## 构建
```bash
mkdir build
cd build
cmake ..
make -j8
cp src/plugins/libzcc_plugin.so /usr/local/lib/
```

## 测试
### 自定义matmul算子
```bash
cd test
# 生成trt engine并对比trt算子结果与torch算子结果
python3 my_mat_mul_trt.py
```