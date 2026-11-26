"""
붓꽃(iris) 데이터를 퍼셉트론으로 학습하는 예제.

워크북 '붓꽃 분류' 시트와 같은 50개 샘플을 사용한다.
Setosa 25개, Versicolor 25개로 구성한다.
표준 라이브러리만 쓰므로 별도 설치 없이 바로 실행된다.

    python iris_perceptron.py
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# 1. 데이터  (품종, 꽃받침길이 SL, 꽃받침너비 SW, 꽃잎길이 PL, 꽃잎너비 PW)
#    정답 t 는 setosa = -1, versicolor = +1
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["꽃받침 길이 SL", "꽃받침 너비 SW", "꽃잎 길이 PL", "꽃잎 너비 PW"]

RAW_DATA = [
    # 기존 Setosa 데이터 5개
    ("setosa",     5.1, 3.5, 1.4, 0.2),
    ("setosa",     4.9, 3.0, 1.4, 0.2),
    ("setosa",     4.7, 3.2, 1.3, 0.2),
    ("setosa",     5.0, 3.6, 1.4, 0.2),
    ("setosa",     5.4, 3.9, 1.7, 0.4),

    # 기존 Versicolor 데이터 5개
    ("versicolor", 7.0, 3.2, 4.7, 1.4),
    ("versicolor", 6.4, 3.2, 4.5, 1.5),
    ("versicolor", 6.9, 3.1, 4.9, 1.5),
    ("versicolor", 5.5, 2.3, 4.0, 1.3),
    ("versicolor", 6.5, 2.8, 4.6, 1.5),

    # 추가 Setosa 데이터 20개
    ("setosa",     4.6, 3.1, 1.5, 0.2),
    ("setosa",     4.6, 3.4, 1.4, 0.3),
    ("setosa",     5.0, 3.4, 1.5, 0.2),
    ("setosa",     4.4, 2.9, 1.4, 0.2),
    ("setosa",     4.9, 3.1, 1.5, 0.1),
    ("setosa",     5.4, 3.7, 1.5, 0.2),
    ("setosa",     4.8, 3.4, 1.6, 0.2),
    ("setosa",     4.8, 3.0, 1.4, 0.1),
    ("setosa",     4.3, 3.0, 1.1, 0.1),
    ("setosa",     5.8, 4.0, 1.2, 0.2),
    ("setosa",     5.7, 4.4, 1.5, 0.4),
    ("setosa",     5.4, 3.9, 1.3, 0.4),
    ("setosa",     5.1, 3.5, 1.4, 0.3),
    ("setosa",     5.7, 3.8, 1.7, 0.3),
    ("setosa",     5.1, 3.8, 1.5, 0.3),
    ("setosa",     5.4, 3.4, 1.7, 0.2),
    ("setosa",     5.1, 3.7, 1.5, 0.4),
    ("setosa",     4.6, 3.6, 1.0, 0.2),
    ("setosa",     5.1, 3.3, 1.7, 0.5),
    ("setosa",     4.8, 3.4, 1.9, 0.2),

    # 추가 Versicolor 데이터 20개
    ("versicolor", 5.7, 2.8, 4.5, 1.3),
    ("versicolor", 6.3, 3.3, 4.7, 1.6),
    ("versicolor", 4.9, 2.4, 3.3, 1.0),
    ("versicolor", 6.6, 2.9, 4.6, 1.3),
    ("versicolor", 5.2, 2.7, 3.9, 1.4),
    ("versicolor", 5.0, 2.0, 3.5, 1.0),
    ("versicolor", 5.9, 3.0, 4.2, 1.5),
    ("versicolor", 6.0, 2.2, 4.0, 1.0),
    ("versicolor", 6.1, 2.9, 4.7, 1.4),
    ("versicolor", 5.6, 2.9, 3.6, 1.3),
    ("versicolor", 6.7, 3.1, 4.4, 1.4),
    ("versicolor", 5.6, 3.0, 4.5, 1.5),
    ("versicolor", 5.8, 2.7, 4.1, 1.0),
    ("versicolor", 6.2, 2.2, 4.5, 1.5),
    ("versicolor", 5.6, 2.5, 3.9, 1.1),
    ("versicolor", 5.9, 3.2, 4.8, 1.8),
    ("versicolor", 6.1, 2.8, 4.0, 1.3),
    ("versicolor", 6.3, 2.5, 4.9, 1.5),
    ("versicolor", 6.1, 2.8, 4.7, 1.2),
    ("versicolor", 6.4, 2.9, 4.3, 1.3),
]
LABEL_OF = {"setosa": -1, "versicolor": +1}


def load_data(feature_idx=(0, 1, 2, 3)):
    """(X, t, 품종이름) 을 돌려준다. feature_idx 로 쓸 특징을 고를 수 있다."""
    X, t, names = [], [], []
    for row in RAW_DATA:
        species, values = row[0], row[1:]
        X.append([values[i] for i in feature_idx])
        t.append(LABEL_OF[species])
        names.append(species)
    return X, t, names


# ---------------------------------------------------------------------------
# 2. 퍼셉트론
# ---------------------------------------------------------------------------


class Perceptron:
    """s = w·x + b 를 계산하고 계단 함수로 -1 / +1 을 내놓는 퍼셉트론."""

    def __init__(self, n_features: int, lr: float = 0.01, seed: int | None = 0):
        rng = random.Random(seed)
        # 작은 난수로 시작한다. 전부 0으로 두어도 동작한다.
        self.w = [rng.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.b = 0.0
        self.lr = lr
        self.history: list[int] = []  # 에폭마다 틀린 개수

    # --- 앞 절반: 선형회귀와 같은 부분 -------------------------------------
    def net_input(self, x) -> float:
        """가중합 s = ω₁x₁ + ... + ωₙxₙ + b"""
        return sum(wi * xi for wi, xi in zip(self.w, x)) + self.b

    # --- 뒤 절반: 활성화 함수 ---------------------------------------------
    @staticmethod
    def step(s: float) -> int:
        """계단 함수. s >= 0 이면 +1, 아니면 -1"""
        return 1 if s >= 0 else -1

    def predict(self, x) -> int:
        return self.step(self.net_input(x))

    # --- 학습 --------------------------------------------------------------
    def fit(self, X, t, epochs: int = 50, verbose: bool = True):
        """퍼셉트론 학습 규칙으로 w, b 를 수정한다.

            틀렸을 때만:  w ← w + η(t - y)x ,  b ← b + η(t - y)
            맞혔으면 t - y = 0 이라 아무것도 바뀌지 않는다.
        """
        for epoch in range(1, epochs + 1):
            errors = 0
            for xi, ti in zip(X, t):
                yi = self.predict(xi)
                delta = self.lr * (ti - yi)
                if delta != 0.0:  # 틀린 경우에만 수정
                    self.w = [w + delta * x for w, x in zip(self.w, xi)]
                    self.b += delta
                    errors += 1
            self.history.append(errors)

            if verbose:
                print(f"  에폭 {epoch:3d}  |  틀린 개수 {errors}  |  "
                      f"w = [{', '.join(f'{v:+.3f}' for v in self.w)}]  b = {self.b:+.3f}")

            if errors == 0:  # 멈춤 조건: 전부 맞히면 더 고칠 것이 없다
                if verbose:
                    print(f"  → {epoch} 에폭 만에 수렴")
                return self
        if verbose:
            print("  → 최대 에폭까지 수렴하지 못했다 (직선으로 나뉘지 않는 데이터일 수 있다)")
        return self

    def accuracy(self, X, t) -> float:
        hit = sum(1 for xi, ti in zip(X, t) if self.predict(xi) == ti)
        return hit / len(t)


# ---------------------------------------------------------------------------
# 3. 출력 도우미
# ---------------------------------------------------------------------------


def print_table(model: Perceptron, X, t, names, feature_idx):
    head = ["품종"] + [FEATURE_NAMES[i].split()[-1] for i in feature_idx]
    head += ["s", "y(예측)", "정답", "일치"]
    widths = [12] + [6] * len(feature_idx) + [8, 8, 6, 5]

    print("  " + "".join(h.ljust(w) for h, w in zip(head, widths)))
    print("  " + "-" * sum(widths))
    for xi, ti, name in zip(X, t, names):
        s = model.net_input(xi)
        y = model.step(s)
        cells = [name] + [f"{v:.1f}" for v in xi]
        cells += [f"{s:+.3f}", f"{y:+d}", f"{ti:+d}", "O" if y == ti else "X"]
        print("  " + "".join(c.ljust(w) for c, w in zip(cells, widths)))


def print_equation(model: Perceptron, feature_idx):
    terms = " + ".join(
        f"({model.w[k]:+.3f})·{FEATURE_NAMES[i].split()[-1]}"
        for k, i in enumerate(feature_idx)
    )
    print(f"  s(x) = {terms} + ({model.b:+.3f})")
    print("  s(x) >= 0 이면 versicolor, s(x) < 0 이면 setosa")


# ---------------------------------------------------------------------------
# 4. 실행
# ---------------------------------------------------------------------------


def run(feature_idx, title, lr=0.01, epochs=50, seed=0):
    print("=" * 78)
    print(title)
    print("=" * 78)

    X, t, names = load_data(feature_idx)
    model = Perceptron(n_features=len(feature_idx), lr=lr, seed=seed)

    print(f"\n[학습]  학습률 η = {lr},  최대 에폭 = {epochs}")
    model.fit(X, t, epochs=epochs)

    print("\n[학습된 결정식]")
    print_equation(model, feature_idx)

    print("\n[각 샘플의 결과]")
    print_table(model, X, t, names, feature_idx)

    print(f"\n[정확도]  {model.accuracy(X, t) * 100:.1f}%  "
          f"({sum(1 for xi, ti in zip(X, t) if model.predict(xi) == ti)} / {len(t)})")
    print()
    return model


def main():
    # (1) 특징 4개를 모두 사용
    model = run(
        feature_idx=(0, 1, 2, 3),
        title="붓꽃 퍼셉트론 — 특징 4개 (SL, SW, PL, PW)",
        lr=0.01,
    )

    # (2) 꽃잎 정보 2개만 사용해도 충분히 나뉜다
    run(
        feature_idx=(2, 3),
        title="붓꽃 퍼셉트론 — 꽃잎 정보 2개만 (PL, PW)",
        lr=0.01,
    )

    # (3) 꽃받침 정보 2개만 쓰면 어려워진다
    run(
        feature_idx=(0, 1),
        title="붓꽃 퍼셉트론 — 꽃받침 정보 2개만 (SL, SW)",
        lr=0.01,
    )

    # (4) 학습이 끝난 모델로 새 꽃을 판정해 본다
    print("=" * 78)
    print("새로운 꽃 판정하기 (특징 4개 모델)")
    print("=" * 78)
    new_flowers = [
        ("A", [5.2, 3.4, 1.5, 0.2]),
        ("B", [6.3, 2.9, 4.4, 1.4]),
        ("C", [5.8, 3.0, 3.0, 0.9]),  # 경계 근처
    ]
    for label, x in new_flowers:
        s = model.net_input(x)
        y = model.step(s)
        species = "versicolor" if y > 0 else "setosa"
        print(f"  꽃 {label}  {x}  →  s = {s:+.3f}  →  {species}")
    print()


if __name__ == "__main__":
    main()
