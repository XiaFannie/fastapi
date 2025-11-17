"""
测试客户端脚本
用于测试 FastAPI 模型推理服务
"""
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_URL = "http://localhost:8000"


def test_root():
    """测试根路径"""
    print("测试根路径...")
    response = requests.get(f"{BASE_URL}/")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_health():
    """测试健康检查"""
    print("测试健康检查...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_single_predict():
    """测试单个预测"""
    print("测试单个预测...")
    data = {
        "features": [0.5, 0.3, 0.2, 0.1]
    }
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_batch_predict():
    """测试批量预测"""
    print("测试批量预测...")
    data = {
        "inputs": [
            [0.5, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.8, 0.7, 0.6]
        ]
    }
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_concurrent_requests(num_requests=10):
    """测试并发请求"""
    print(f"测试并发请求 ({num_requests} 个请求)...")
    
    def make_request(i):
        data = {
            "features": [0.5 + i * 0.01, 0.3, 0.2, 0.1]
        }
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        elapsed = time.time() - start_time
        return {
            "request_id": i,
            "status_code": response.status_code,
            "response": response.json(),
            "elapsed": elapsed
        }
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        results = [future.result() for future in as_completed(futures)]
    
    total_time = time.time() - start_time
    
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个请求: {total_time/num_requests:.3f} 秒")
    print(f"QPS: {num_requests/total_time:.2f}")
    print(f"成功请求数: {sum(1 for r in results if r['status_code'] == 200)}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("FastAPI 模型推理服务测试")
    print("=" * 50)
    print()
    
    try:
        # 等待服务启动
        print("等待服务就绪...")
        time.sleep(2)
        
        test_root()
        test_health()
        test_single_predict()
        test_batch_predict()
        test_concurrent_requests(10)
        
        print("=" * 50)
        print("所有测试完成！")
        print("=" * 50)
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器，请确保服务正在运行")
        print("运行命令: python app.py")
    except Exception as e:
        print(f"错误: {str(e)}")

