import numpy as np
np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

# 1. 원본 데이터: 각 열은 x1과 x2
data = np.array([[1, 2], [2, 6], [3, 9], [5, 12]], dtype=float)

# 2. 데이터 개수와 각 열의 평균
n = data.shape[0]
mean = data.mean(axis=0)

# 3. 각 열에서 평균 빼기
X = data - mean

# 4. 표본 공분산 행렬 계산
C = (X.T @ X) / (n - 1)

# 5. NumPy 함수로 검산
# rowvar=False: 각 열을 변수로 취급
# ddof=1: n-1로 나눔
C_numpy = np.cov(data, rowvar=False, ddof=1)

print("각 열의 평균:", mean)

print("\n표본 공분산 행렬:")
print(C)

print(f"\n분산(x1) = {C[0, 0]:.4f}")
print(f"분산(x2) = {C[1, 1]:.4f}")
print(f"공분산(x1, x2) = {C[0, 1]:.4f}")

print("\nNumPy 결과와 일치:", np.allclose(C, C_numpy))