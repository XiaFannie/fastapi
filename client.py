"""
使用 requests 访问 FastAPI 模型推理服务的客户端脚本
"""
import requests
import json


# 服务地址
BASE_URL = "http://localhost:8000"


def check_service():
    """检查服务是否运行"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✓ 服务状态: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到服务，请确保 app.py 正在运行")
        return False


def health_check():
    """健康检查"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✓ 健康检查: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False


def predict_single(features):
    """
    单个样本预测
    
    Args:
        features: 特征列表，例如 [0.5, 0.3, 0.2, 0.1]
    
    Returns:
        预测结果字典
    """
    url = f"{BASE_URL}/predict"
    data = {"features": features}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        print(f"✓ 预测结果: {result}")
        return result
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP 错误: {e}")
        print(f"  响应内容: {response.text}")
        return None
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return None


def predict_batch(inputs):
    """
    批量预测
    
    Args:
        inputs: 输入列表，例如 [[0.5, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]
    
    Returns:
        预测结果字典
    """
    url = f"{BASE_URL}/predict/batch"
    data = {"inputs": inputs}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        print(f"✓ 批量预测结果: {result}")
        return result
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP 错误: {e}")
        print(f"  响应内容: {response.text}")
        return None
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return None


def predict_async_batch(inputs):
    """
    异步批量预测
    
    Args:
        inputs: 输入列表，例如 [[0.5, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]
    
    Returns:
        预测结果字典
    """
    url = f"{BASE_URL}/predict/async-batch"
    data = {"inputs": inputs}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        print(f"✓ 异步批量预测结果: {result}")
        return result
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP 错误: {e}")
        print(f"  响应内容: {response.text}")
        return None
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return None


# 示例使用
if __name__ == "__main__":
    print("=" * 60)
    print("FastAPI 模型推理服务客户端")
    print("=" * 60)
    print()
    
    # 1. 检查服务
    print("1. 检查服务状态...")
    if not check_service():
        exit(1)
    print()
    
    # 2. 健康检查
    print("2. 健康检查...")
    health_check()
    print()
    
    # 3. 单个预测
    print("3. 单个样本预测...")
    result = predict_single([0.5, 0.3, 0.2, 0.1])
    print()
    
    # 4. 批量预测
    print("4. 批量预测...")
    batch_result = predict_batch([
        [0.5, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.9, 0.8, 0.7, 0.6]
    ])
    print()
    
    # 5. 异步批量预测
    print("5. 异步批量预测...")
    async_result = predict_async_batch([
        [0.5, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4]
    ])
    print()
    
    print("=" * 60)
    print("所有请求完成！")
    print("=" * 60)
    
    # 使用示例
    print("\n使用示例:")
    print("-" * 60)
    print("# 导入函数")
    print("from client import predict_single, predict_batch")
    print()
    print("# 单个预测")
    print("result = predict_single([0.5, 0.3, 0.2, 0.1])")
    print()
    print("# 批量预测")
    print("result = predict_batch([[0.5, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])")

