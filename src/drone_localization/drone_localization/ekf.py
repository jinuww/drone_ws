#!/usr/bin/env python3
"""
6상태 등속도 추적 필터 — 선형 KF 갱신과 극좌표 EKF 갱신 두 경로.

ROS·sklearn·open3d에 의존하지 않는다(numpy만). 트래커 노드에서 분리해 둔
이유는 필터 수학만 따로 돌려볼 수 있게 하기 위해서다 — 노드를 띄우거나
LiDAR를 켜지 않고도 회귀를 잡을 수 있다(test/test_jacobian.py).

두 갱신 경로의 차이:

  update()        관측을 월드 직교좌표 위치로 보고 3x3 공분산 R로 반영.
                  H가 상수라 선형 KF. 두 LiDAR의 검출을 정보행렬로 미리
                  합쳐 한 번에 넣는, 기존 lidar_drone_tracker.py의 경로다.

  update_polar()  관측을 **그 센서의 극좌표**(거리·방위각·고도각)로 보고
                  반영. h()가 비선형이라 야코비안이 필요하다 — EKF.
                  센서마다 한 번씩 호출해 순차 갱신한다.

선형·가우시안 필터에서 관측 두 개를 차례로 넣는 것은 한꺼번에 넣는 것과
같으므로, 순차 갱신 자체는 융합 대비 손해가 없다. 갈리는 지점은 R이다.

극좌표를 쓰는 이유는 LiDAR 오차가 월드 XYZ에서 등방이 아니기 때문이다.

  시선 방향(range)   지배항은 거리 정밀도(OS1-128은 ~1cm)가 아니라, LiDAR가
                     드론의 **자기 쪽을 향한 면만** 맞힌다는 사실이다.
                     센트로이드가 실제 중심보다 기체 두께의 절반쯤 앞에
                     찍히는데, 그 두께는 참가팀 기체의 성질이라 알 수 없다.
                     점을 많이 모아도 줄지 않는 계통 성분이므로 √n으로
                     나누지 않는다.
                       σ_range ≈ 0.10 m

  시선 수직(angular) 빔 간격(0.35°) × 거리. 이쪽은 점 개수로 평균되어 준다.
                       σ_ang ≈ hypot(0.35°·d, σ_body) / (√n · d)

이걸 극좌표에서 대각행렬로 선언하고 야코비안이 월드로 회전시키게 두면,
센서마다 자기 시선 방향으로 길쭉한 오차 타원체가 생긴다. 남북(ns) 배치에서
두 센서는 드론을 크게 어긋난 방향에서 보므로 **한 센서의 약한 축이 다른
센서의 강한 축**이 된다. 근접면 편향이 추정 없이 기하로 상쇄된다.

실장비에서 시뮬과 달라지는 점: 센서 외부 파라미터(extrinsic)가 완벽하지
않다. 캘리브레이션 RMSE는 월드에서 등방인 오차라 극좌표 두 축 모두에
제곱합으로 얹는다 — sigma_ang()의 extrinsic_rmse 인자와, 호출부에서
sigma_range에 더하는 항이 그것이다. 이 항이 없으면 필터가 실제보다
확신하게 되어 NIS가 3을 크게 넘는다.

이 모듈은 lidar_drone_tracker.py(A)만 사용한다. lidar_drone_tracker_seq.py(B)는
비교 실험용이라 필터를 파일 안에 그대로 두는 정책이므로 건드리지 않았다 —
한 파일을 고쳤을 때 다른 방식이 조용히 따라 바뀌면 무엇을 재고 있는지
알 수 없게 되기 때문이다.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_pi(a: float) -> float:
    """각도를 (-π, π] 로 접는다.

    방위각 innovation에 반드시 필요하다. ±π 경계를 넘나들 때 접지 않으면
    innovation이 2π만큼 튀고, 필터는 그걸 거대한 관측 불일치로 받아들여
    한 프레임 만에 발산한다."""
    return (a + np.pi) % TWO_PI - np.pi


def cart_to_polar(d: np.ndarray) -> np.ndarray:
    """센서 좌표 (x, y, z) → (거리, 방위각, 고도각)."""
    rho = float(np.hypot(d[0], d[1]))
    return np.array([
        float(np.linalg.norm(d)),
        float(np.arctan2(d[1], d[0])),
        float(np.arctan2(d[2], rho)),
    ])


class KF6D:
    """Constant-velocity KF with state [x,y,z,vx,vy,vz]."""

    def __init__(self, accel_noise_std: float, default_meas_noise_std: float):
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 9.0
        self.accel_noise_std = float(accel_noise_std)
        self.default_R = np.eye(3, dtype=np.float64) * float(default_meas_noise_std) ** 2
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.initialized = False

    def init(self, pos: np.ndarray, pos_cov: np.ndarray | None = None):
        self.x[:3] = np.asarray(pos, dtype=np.float64)
        self.x[3:] = 0.0
        self.P = np.eye(6, dtype=np.float64) * 0.25
        if pos_cov is not None:
            self.P[:3, :3] = self._regularize_cov(pos_cov)
        self.initialized = True

    @staticmethod
    def _regularize_cov(cov: np.ndarray) -> np.ndarray:
        cov = np.asarray(cov, dtype=np.float64).reshape(3, 3)
        cov = 0.5 * (cov + cov.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-6, 100.0)
        return vecs @ np.diag(vals) @ vecs.T

    def predict(self, dt: float):
        if not self.initialized:
            return
        dt = max(1e-3, min(1.0, float(dt)))
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Physically consistent white-acceleration process-noise model.
        q = self.accel_noise_std ** 2
        q1 = np.array([
            [0.25 * dt**4, 0.5 * dt**3],
            [0.5 * dt**3, dt**2],
        ], dtype=np.float64) * q
        Q = np.zeros((6, 6), dtype=np.float64)
        for p, v in ((0, 3), (1, 4), (2, 5)):
            Q[p, p] = q1[0, 0]
            Q[p, v] = q1[0, 1]
            Q[v, p] = q1[1, 0]
            Q[v, v] = q1[1, 1]

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, R_meas: np.ndarray | None = None):
        if not self.initialized:
            self.init(z, R_meas)
            return
        Rm = self.default_R if R_meas is None else self._regularize_cov(R_meas)
        z = np.asarray(z, dtype=np.float64).reshape(3)
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + Rm
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(3))
        self.x = self.x + K @ innov
        IKH = np.eye(6) - K @ self.H
        # Joseph form for numerical robustness.
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    @property
    def cov_pos(self):
        return self.P[:3, :3].copy()


class EKF6D(KF6D):
    """상태·운동모델은 KF6D와 같고, 관측만 센서 극좌표에서 받는다.

    h()가 비선형이라 야코비안이 필요하다 — 그래서 EKF. 왜 극좌표가 맞는
    좌표계인지는 모듈 설명 참조."""

    # 수평거리 rho가 이보다 작으면 방위각이 정의되지 않는다(센서 바로 위/아래).
    # ns 배치는 센서가 필드 밖이라 실제로 걸리지 않지만, 배치를 바꿨을 때
    # 0으로 나누지 않도록 막아둔다.
    _RHO_MIN = 1e-3

    def update_polar(self, z_world: np.ndarray, R_s2w: np.ndarray,
                     t_s2w: np.ndarray, sigma_range: float,
                     sigma_ang: float):
        """센서 하나의 센트로이드를 그 센서의 극좌표에서 반영한다.

        z_world     그 센서 점들만으로 구한 센트로이드 (월드 좌표)
        R_s2w       센서→월드 회전 (3x3)
        t_s2w       센서 위치 (월드, 3)
        sigma_range 시선 방향 표준편차 [m]
        sigma_ang   방위·고도 각도 표준편차 [rad]

        반환: (NIS, 평균이득). 필터가 아직 초기화되지 않았거나 특이점에
        걸려 건너뛰면 None.

        NIS(정규화 innovation 제곱)는 3자유도 카이제곱을 따라야 하므로,
        평균이 3에서 크게 벗어나면 σ가 잘못 잡힌 것이다 — 튜닝 지표다.
        check_ekf_fusion.py가 이 값을 비행 뒤에 판정한다."""
        if not self.initialized:
            return None

        Rt = np.asarray(R_s2w, dtype=np.float64).T
        t_s2w = np.asarray(t_s2w, dtype=np.float64).reshape(3)
        z_world = np.asarray(z_world, dtype=np.float64).reshape(3)

        d = Rt @ (self.x[:3] - t_s2w)          # 예측 위치를 센서 좌표로
        r = float(np.linalg.norm(d))
        rho = float(np.hypot(d[0], d[1]))
        if rho < self._RHO_MIN or r < self._RHO_MIN:
            return None

        h = cart_to_polar(d)
        z = cart_to_polar(Rt @ (z_world - t_s2w))

        # ∂h/∂d (센서 좌표 기준). 체인룰로 ∂d/∂p = R_s2wᵀ 를 곱해 상태에 대한
        # 야코비안을 얻는다. 속도는 관측되지 않으므로 뒤 3열은 0.
        Jd = np.array([
            [d[0] / r, d[1] / r, d[2] / r],
            [-d[1] / rho**2, d[0] / rho**2, 0.0],
            [-d[0] * d[2] / (r * r * rho),
             -d[1] * d[2] / (r * r * rho),
             rho / (r * r)],
        ])
        H = np.zeros((3, 6))
        H[:, :3] = Jd @ Rt

        Rm = np.diag([sigma_range**2, sigma_ang**2, sigma_ang**2])

        y = z - h
        y[1] = wrap_pi(y[1])
        y[2] = wrap_pi(y[2])

        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        self.x = self.x + K @ y
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T

        return float(y @ np.linalg.solve(S, y)), float(np.trace(K[:3, :3]) / 3.0)

    @staticmethod
    def sigma_ang(beam_res_rad: float, dist: float, n_pts: int,
                  sigma_body: float, extrinsic_rmse: float = 0.0) -> float:
        """센트로이드의 시선-수직 불확실성을 각도로 환산한다.

        점 개수로 평균되어 √n으로 주는 성분이 둘:

          빔 양자화   beam_res × 거리 — 각분해능이 각도라서 **멀수록 커진다**
          기체 크기   sigma_body      — 자기 쪽 면 위 어디가 맞느냐는 거리와
                                        무관하다. **가까울수록 이쪽이 지배한다**

        빔 항만 넣으면 가까운 거리에서 σ가 실제보다 작게 나와 필터가 과신한다
        (시뮬 실측 NIS 3.6~5.3, 이상적 3.0). 기체 크기 항이 그 바닥을 잡아준다.
        참가팀 기체 치수는 알 수 없으므로 sigma_body는 예상 규모를 넣는
        파라미터다 — 미지량이라는 점에서 sigma_range와 같은 성격이다.

        extrinsic_rmse는 센서 자세 오차라 점을 모아도 줄지 않으므로 √n
        **밖에서** 더한다. 실장비에만 있는 항이다(시뮬 센서 pose는 정확)."""
        cross = np.hypot(beam_res_rad * dist, sigma_body) / max(n_pts, 1) ** 0.5
        cross = np.hypot(cross, max(0.0, extrinsic_rmse))
        return float(cross / max(dist, 1e-6))
