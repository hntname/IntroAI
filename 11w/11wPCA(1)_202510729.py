import numpy as np
np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

# 1. 원본 데이터: 각 행은 관측값, 각 열은 x1과 x2
data = np.array([
    [1.4, 0.2], [1.4, 0.2], [1.3, 0.2], [1.5, 0.2], [1.4, 0.2], [1.7, 0.4], [1.4, 0.3],
    [1.5, 0.2], [1.4, 0.2], [1.5, 0.1], ], dtype=float)

# 2. 각 열의 평균 계산
n = data.shape[0]
mean = data.mean(axis=0)

# 3. 각 열에서 평균 빼기
X = data - mean

# 4. 행렬 곱으로 공분산 행렬 계산
# X.T: 2×10, X: 10×2 → 결과: 2×2
C = (X.T @ X) / n

# 5. NumPy 함수로 검산
# rowvar=False: 열을 변수로 취급
# ddof=0: n으로 나눔 / ddof=1: n-1로 나눔
C_numpy = np.cov(data, rowvar=False, ddof=0)

print("각 열의 평균:")
print(mean)

print("\n평균을 뺀 데이터 X:")
print(X)

print("\n공분산 행렬 C:")
print(C)

print(f"\n분산(x1) = {C[0, 0]:.4f}")
print(f"분산(x2) = {C[1, 1]:.4f}")
print(f"공분산(x1, x2) = {C[0, 1]:.4f}")

print("\nNumPy 결과와 일치:", np.allclose(C, C_numpy))