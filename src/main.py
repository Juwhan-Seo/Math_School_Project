from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 입력 변수와 목표 변수 설정
vehicle_count = np.array([100, 150, 200, 250, 300]).reshape(-1, 1)  # 차량 수
signal_cycle = np.array([60, 50, 70, 80, 90]).reshape(-1, 1)  # 신호등 주기
congestion_level = np.array([3, 4, 6, 8, 7])  # 도로 교통 혼잡 수준

# 다항 회귀 모델 생성
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(np.concatenate((vehicle_count, signal_cycle), axis=1), congestion_level)

# 새로운 입력 데이터에 대한 도로 교통 혼잡 예측
new_vehicle_count = np.array([120]).reshape(-1, 1)
new_signal_cycle = np.array([55]).reshape(-1, 1)
predicted_congestion = model.predict(np.concatenate((new_vehicle_count, new_signal_cycle), axis=1))

print("예상 도로 교통 혼잡 수준: {:.2f}".format(predicted_congestion[0]))
