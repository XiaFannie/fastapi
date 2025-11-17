"""
FastAPI 模型推理服务
支持模型预加载和并发推理
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import logging
from contextlib import asynccontextmanager
import asyncio

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储模型
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    在启动时加载模型，关闭时清理资源
    """
    # 启动时加载模型
    global model
    logger.info("正在加载模型...")
    model = load_model()
    logger.info("模型加载完成！")
    
    yield
    
    # 关闭时清理
    logger.info("正在清理资源...")
    model = None
    logger.info("资源清理完成")


# 创建 FastAPI 应用
app = FastAPI(
    title="模型推理服务",
    description="支持预加载模型和并发推理的 FastAPI 服务",
    version="1.0.0",
    lifespan=lifespan
)


def load_model():
    """
    加载模型函数
    根据实际使用的模型框架修改此函数
    """
    # 示例：使用 scikit-learn 模型
    # 实际使用时，替换为你的模型加载逻辑
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        
        # 创建一个示例模型（实际使用时，应该从文件加载）
        logger.info("创建示例模型...")
        X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
        model = LogisticRegression()
        model.fit(X, y)
        logger.info("示例模型创建完成")
        return model
    except ImportError:
        logger.warning("scikit-learn 未安装，使用模拟模型")
        # 返回一个模拟模型对象
        class MockModel:
            def predict(self, X):
                return np.random.randint(0, 2, size=len(X))
        return MockModel()


# 定义请求模型
class SingleInput(BaseModel):
    """单个输入数据"""
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.2, 0.1]
            }
        }


class BatchInput(BaseModel):
    """批量输入数据"""
    inputs: List[List[float]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "inputs": [
                    [0.5, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4],
                    [0.9, 0.8, 0.7, 0.6]
                ]
            }
        }


class PredictionResponse(BaseModel):
    """预测响应"""
    prediction: List[int]
    probabilities: Optional[List[List[float]]] = None


class SinglePredictionResponse(BaseModel):
    """单个预测响应"""
    prediction: int
    probability: Optional[float] = None


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "模型推理服务",
        "status": "运行中",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=SinglePredictionResponse)
async def predict_single(input_data: SingleInput):
    """
    单个样本预测
    支持并发请求
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
    
    try:
        # 转换为 numpy 数组
        features = np.array([input_data.features])
        
        # 执行预测
        prediction = model.predict(features)[0]
        
        # 如果模型支持 predict_proba，获取概率
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            probability = float(proba[prediction])
        
        return SinglePredictionResponse(
            prediction=int(prediction),
            probability=probability
        )
    except Exception as e:
        logger.error(f"预测错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict/batch", response_model=PredictionResponse)
async def predict_batch(input_data: BatchInput):
    """
    批量预测
    支持并发请求
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
    
    try:
        # 转换为 numpy 数组
        features = np.array(input_data.inputs)
        
        # 执行批量预测
        predictions = model.predict(features).tolist()
        
        # 如果模型支持 predict_proba，获取概率
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            probabilities = proba.tolist()
        
        return PredictionResponse(
            prediction=[int(p) for p in predictions],
            probabilities=probabilities
        )
    except Exception as e:
        logger.error(f"批量预测错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


@app.post("/predict/async-batch")
async def predict_async_batch(input_data: BatchInput):
    """
    异步批量预测（适用于 CPU 密集型任务）
    使用线程池执行推理，避免阻塞事件循环
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
    
    try:
        import concurrent.futures
        
        # 在线程池中执行推理
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            features = np.array(input_data.inputs)
            predictions = await loop.run_in_executor(
                executor,
                model.predict,
                features
            )
            
            probabilities = None
            if hasattr(model, 'predict_proba'):
                proba = await loop.run_in_executor(
                    executor,
                    model.predict_proba,
                    features
                )
                probabilities = proba.tolist()
        
        return PredictionResponse(
            prediction=[int(p) for p in predictions.tolist()],
            probabilities=probabilities
        )
    except Exception as e:
        logger.error(f"异步批量预测错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"异步批量预测失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境建议设为 False
        workers=1  # 如果使用 GPU，建议设为 1；CPU 可以增加
    )

