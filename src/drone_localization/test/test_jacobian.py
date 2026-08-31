"""극좌표 관측 야코비안을 수치미분과 대조한다.

EKF에서 야코비안이 틀리면 필터는 조용히 틀린다 — 발산하지 않고 그냥 조금씩
어긋난 위치로 수렴해서, 비행 로그만 봐서는 알아채기 어렵다. 그래서 h()를
직접 수치미분해 해석식과 맞춰 본다. ROS도 LiDAR도 필요 없다.

  python3 test/test_jacobian.py     (또는 pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation as R

from drone_localization.ekf import EKF6D, cart_to_polar, wrap_pi

# 실장비 공칭 배치 (params/lidar_localization_os1.yaml 기본값)
SENSORS = [
    (np.array([12.0, -1.0, 2.0]), R.from_euler('xyz', [0, 0, 1.5708]).as_matrix()),
    (np.array([12.0, 16.0, 2.0]), R.from_euler('xyz', [0, 0, -1.5708]).as_matrix()),
]


def h_of_p(p, R_s2w, t_s2w):
    return cart_to_polar(R_s2w.T @ (p - t_s2w))


def analytic_H(p, R_s2w, t_s2w):
    Rt = R_s2w.T
    d = Rt @ (p - t_s2w)
    r = float(np.linalg.norm(d))
    rho = float(np.hypot(d[0], d[1]))
    Jd = np.array([
        [d[0] / r, d[1] / r, d[2] / r],
        [-d[1] / rho**2, d[0] / rho**2, 0.0],
        [-d[0] * d[2] / (r * r * rho), -d[1] * d[2] / (r * r * rho), rho / (r * r)],
    ])
    return Jd @ Rt


def numeric_H(p, R_s2w, t_s2w, eps=1e-7):
    out = np.zeros((3, 3))
    for j in range(3):
        dp = np.zeros(3)
        dp[j] = eps
        hp = h_of_p(p + dp, R_s2w, t_s2w)
        hm = h_of_p(p - dp, R_s2w, t_s2w)
        d = hp - hm
        d[1] = wrap_pi(d[1])
        d[2] = wrap_pi(d[2])
        out[:, j] = d / (2 * eps)
    return out


def _probe_points():
    rng = np.random.default_rng(0)
    # 마커 4개(공칭) + 필드 전역 무작위
    pts = [np.array([5.0, 4.0, 2.0]), np.array([5.0, 11.0, 2.0]),
           np.array([19.0, 4.0, 2.0]), np.array([12.0, 11.0, 2.0])]
    pts += [np.array([rng.uniform(0, 24), rng.uniform(0, 15), rng.uniform(1, 3)])
            for _ in range(200)]
    return pts


def test_jacobian_matches_numeric():
    worst = 0.0
    for p in _probe_points():
        for t_s2w, R_s2w in SENSORS:
            Ha = analytic_H(p, R_s2w, t_s2w)
            Hn = numeric_H(p, R_s2w, t_s2w)
            # 상대오차 (행 크기로 정규화)
            scale = np.maximum(np.abs(Hn).max(axis=1, keepdims=True), 1e-9)
            worst = max(worst, float(np.abs(Ha - Hn).max() / scale.max()))
    print(f'해석 야코비안 vs 수치미분 — 최대 상대오차: {worst:.3e}')
    assert worst < 1e-6


def test_wrap_pi_folds_azimuth():
    # ±π 경계를 접지 않으면 innovation이 2π만큼 튀어 한 프레임에 발산한다.
    assert abs(wrap_pi(np.pi + 0.1) - (-np.pi + 0.1)) < 1e-12
    assert abs(wrap_pi(-np.pi - 0.1) - (np.pi - 0.1)) < 1e-12
    assert abs(wrap_pi(0.3) - 0.3) < 1e-12


def test_update_polar_pulls_state_toward_measurement():
    kf = EKF6D(accel_noise_std=2.0, default_meas_noise_std=0.15)
    truth = np.array([8.0, 6.0, 2.0])
    kf.init(truth + np.array([0.30, 0.0, 0.0]), np.eye(3) * 0.25)
    before = float(np.linalg.norm(kf.position - truth))
    for t_s2w, R_s2w in SENSORS:
        res = kf.update_polar(truth, R_s2w, t_s2w, 0.10, np.deg2rad(0.2))
        assert res is not None
        nis, gain = res
        assert np.isfinite(nis) and nis >= 0.0
        # 이득은 극좌표(rad) → 월드(m) 변환이 섞여 있어 [0,1]로 정규화되지
        # 않는다. 진단용 수치이므로 유한한지만 본다.
        assert np.isfinite(gain)
    after = float(np.linalg.norm(kf.position - truth))
    print(f'초기오차 {before:.3f} m → 두 센서 갱신 후 {after:.3f} m')
    assert after < before


def test_update_polar_before_init_is_ignored():
    kf = EKF6D(accel_noise_std=2.0, default_meas_noise_std=0.15)
    t_s2w, R_s2w = SENSORS[0]
    assert kf.update_polar(np.array([8.0, 6.0, 2.0]), R_s2w, t_s2w,
                           0.10, np.deg2rad(0.2)) is None


def test_sigma_ang_floor_and_growth():
    # 가까우면 기체 크기 항이, 멀면 빔 항이 지배한다.
    beam = np.deg2rad(0.35)
    near = EKF6D.sigma_ang(beam, 3.0, 30, 0.14) * 3.0
    far = EKF6D.sigma_ang(beam, 20.0, 30, 0.14) * 20.0
    assert near >= 0.14 / 30**0.5 * 0.99
    assert far > near
    # 외부 파라미터 오차는 √n 밖에서 더해지므로 항상 σ를 키운다.
    with_extr = EKF6D.sigma_ang(beam, 10.0, 30, 0.14, 0.05)
    without = EKF6D.sigma_ang(beam, 10.0, 30, 0.14, 0.0)
    assert with_extr > without


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS  {name}')
            except AssertionError as exc:
                fails += 1
                print(f'FAIL  {name}: {exc}')
    print('PASS' if fails == 0 else f'FAIL ({fails})')
    sys.exit(1 if fails else 0)
