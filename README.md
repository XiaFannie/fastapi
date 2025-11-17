# FastAPI 模型推理服务

一个支持模型预加载和并发推理的 FastAPI 服务。

## 特性

- ✅ 模型预加载：应用启动时自动加载模型
- ✅ 并发支持：支持多个客户端同时访问
- ✅ 单个预测：支持单个样本预测
- ✅ 批量预测：支持批量样本预测
- ✅ 异步批量预测：使用线程池处理 CPU 密集型任务
- ✅ 健康检查：提供健康检查端点

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务

```bash
python app.py
```

或者使用 uvicorn 直接运行：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API 端点

### 1. 根路径
```
GET /
```

### 2. 健康检查
```
GET /health
```

### 3. 单个预测
```
POST /predict
Content-Type: application/json

{
    "features": [0.5, 0.3, 0.2, 0.1]
}
```

### 4. 批量预测
```
POST /predict/batch
Content-Type: application/json

{
    "inputs": [
        [0.5, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.9, 0.8, 0.7, 0.6]
    ]
}
```

### 5. 异步批量预测
```
POST /predict/async-batch
Content-Type: application/json

{
    "inputs": [
        [0.5, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4]
    ]
}
```

## 自定义模型

修改 `load_model()` 函数以加载你的实际模型：

```python
def load_model():
    # PyTorch 示例
    import torch
    model = torch.load('model.pth')
    model.eval()
    return model
    
    # TensorFlow 示例
    # import tensorflow as tf
    # model = tf.keras.models.load_model('model.h5')
    # return model
    
    # ONNX 示例
    # import onnxruntime as ort
    # session = ort.InferenceSession('model.onnx')
    # return session
```

## 测试

使用 curl 测试：

```bash
# 单个预测
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.5, 0.3, 0.2, 0.1]}'

# 批量预测
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"inputs": [[0.5, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]}'
```

## 性能优化建议

1. **GPU 模型**：如果使用 GPU，建议 `workers=1`，避免多进程竞争 GPU
2. **CPU 模型**：可以增加 `workers` 数量以提高并发处理能力
3. **异步处理**：对于 CPU 密集型模型，使用 `/predict/async-batch` 端点
4. **批量处理**：尽量使用批量预测端点以提高吞吐量

