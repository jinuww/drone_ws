#!/usr/bin/env python3
"""
EKF가 두 LiDAR를 하나의 위치로 융합했는지 판정한다.

`/drone/estimated_pose`에는 위치와 공분산만 실려서, 추정값 CSV만 봐서는
"두 LiDAR가 실제로 합쳐졌는가"를 알 수 없다. 한쪽 센서만 쓰고 있어도 겉보기
궤적은 멀쩡해 보이고, 경로점 오차가 이상하게 나온 뒤에야 알게 된다.

그래서 트래커가 프레임마다 남기는 ekf_diagnostics.csv 를 읽어 판정한다.

  판정 1  융합률      두 센서가 동시에 반영된 프레임 비율
  판정 2  대체 경로   극좌표 갱신을 못 해 융합 센트로이드 1회 갱신으로 빠진 비율
  판정 3  NIS         정규화 innovation 제곱. 3자유도 카이제곱이라 평균 3 근처가
                      정상. 크게 벗어나면 σ 설정이 틀린 것(과신/과보수)
  판정 4  센서 균형   한쪽 센서가 점을 독점하고 있지 않은지

사용:
  ros2 run drone_localization check_ekf_fusion
  ros2 run drone_localization check_ekf_fusion --output_dir ~/competition_results/teamA_run3

--output_dir 를 생략하면 ~/competition_results 아래에서 진단 로그가 있는
가장 최근 폴더를 자동으로 찾는다(GUI가 팀·런별로 폴더를 만든다).
"""

import argparse
import csv
import os

import numpy as np

DEFAULT_ROOT = os.path.expanduser('~/competition_results')
LOG_NAME = 'ekf_diagnostics.csv'


def _find_log(output_dir):
    """지정 폴더, 없으면 그 하위 폴더에서 가장 최근 진단 로그를 찾는다."""
    direct = os.path.join(output_dir, LOG_NAME)
    if os.path.isfile(direct):
        return direct
    if not os.path.isdir(output_dir):
        return None
    found = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name, LOG_NAME)
        if os.path.isfile(path):
            found.append(path)
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def _load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    # 센서 이름은 '<name>_nis' 열에서 뽑는다. '_pts'로 뽑으면 프레임 공통
    # 열인 sel_pts 까지 센서로 잡힌다.
    names = [k[:-4] for k in rows[0] if k.endswith('_nis')]
    return rows, names


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=DEFAULT_ROOT,
                    help='런 폴더 또는 그 상위 폴더 (기본 ~/competition_results)')
    ap.add_argument('--min_fusion_rate', type=float, default=0.80,
                    help='이 비율 이상 두 센서가 동시에 반영돼야 합격')
    args, _ = ap.parse_known_args()

    path = _find_log(os.path.expanduser(args.output_dir))
    if path is None:
        print(f'진단 로그({LOG_NAME})를 찾을 수 없습니다: {args.output_dir}\n'
              '트래커를 다시 빌드한 뒤, GUI에서 런을 설정하고(START) 비행하세요 '
              '— 런 폴더에 자동으로 기록됩니다.')
        return

    rows, names = _load(path)
    if not rows:
        print(f'진단 로그가 비어 있습니다: {path}')
        return

    n = len(rows)
    mode = np.array([r['mode'] for r in rows])
    nsens = np.array([int(r['n_sensors']) for r in rows])
    pts = {s: np.array([int(r[f'{s}_pts']) for r in rows]) for s in names}
    nis = np.concatenate([
        np.array([_f(r[f'{s}_nis']) for r in rows]) for s in names])
    nis = nis[np.isfinite(nis)]
    sig = np.array([_f(r['sigma_pos']) for r in rows])

    both = float((nsens >= 2).mean())
    one = float((nsens == 1).mean())
    fallback = float((mode == 'fused').mean())
    linear = float((mode == 'linear').mean())

    print('=' * 66)
    print(' EKF 센서 융합 판정')
    print('=' * 66)
    print(f' 로그 {path}')
    print(f' 검출 프레임 {n}개 · 센서 {len(names)}대 ({", ".join(names)})')
    print()

    if linear > 0.5:
        print(' ✗ use_polar_ekf 가 꺼져 있습니다 (선형 KF, 융합 센트로이드 1회 갱신).')
        print('   EKF 융합을 쓰려면 use_polar_ekf:=true 로 실행하세요.')
        return

    print('── 판정 1. 융합률 ──')
    print(f'  두 센서 동시 반영 : {both * 100:5.1f}%   ({int((nsens >= 2).sum())}프레임)')
    print(f'  한 센서만 반영    : {one * 100:5.1f}%   ({int((nsens == 1).sum())}프레임)')
    print(f'  평균 센서/프레임  : {nsens.mean():5.2f}')
    ok_fusion = both >= args.min_fusion_rate
    print(f'  → {"합격" if ok_fusion else "불합격"} '
          f'(기준 {args.min_fusion_rate * 100:.0f}% 이상)')
    if not ok_fusion:
        print()
        print('  [원인 가리기] 한쪽 센서가 빠진 프레임이 어느 단계에서 빠졌는가.')
        dist = {s: np.array([_f(r[f'{s}_dist']) for r in rows]) for s in names}
        cand = np.array([int(r['n_cand']) for r in rows])
        # 클라우드가 몇 개 들어왔는가 — 원인을 가르는 결정적 열.
        #   1 = 그 센서에서 프레임 자체가 안 왔다
        #   2 = 프레임은 왔는데 클러스터가 안 나왔거나, 시각차로 버려졌다
        has_clouds = 'n_clouds' in rows[0]
        clouds = (np.array([int(r['n_clouds']) for r in rows]) if has_clouds
                  else None)
        sync = np.array([_f(r.get('sync_dt_ms', '')) for r in rows])
        finite_sync = sync[np.isfinite(sync)]

        for s in names:
            miss = pts[s] == 0
            if not miss.any():
                continue
            d = dist[s][miss]
            d = d[np.isfinite(d)]
            line = f'    {s} 누락 {int(miss.sum())}프레임'
            if len(d):
                line += (f' · 그때 거리 중앙 {np.median(d):.1f}m '
                         f'(전체 중앙 {np.nanmedian(dist[s]):.1f}m)')
            line += f' · 후보 클러스터 중앙 {int(np.median(cand[miss]))}개'
            if clouds is not None:
                one = float((clouds[miss] < 2).mean())
                line += f' · 클라우드 1개뿐이던 비율 {one * 100:.0f}%'
            print(line)

        print()
        print('    (a) 클라우드가 1개뿐  → 그 센서에서 프레임이 안 온다.')
        print('        ros2 topic hz /lidar1/points /lidar2/points 로 확인.')
        print('        한 대만 연결한 벤치 시험이라면 정상이다.')
        print('    (b) 클라우드는 2개인데 누락  → 프레임은 왔는데 클러스터가 안 나왔다.')
        print('        ROI 밖(센서 배치·yaw 확인), 배경 제거가 드론까지 지움,')
        print('        또는 원거리 점 부족(cluster_min_points).')
        if len(finite_sync):
            print(f'    (c) 두 스캔 시각차 중앙 {np.median(finite_sync):.1f} ms '
                  f'· P95 {np.percentile(finite_sync, 95):.1f} ms')
            print('        → async_tolerance_sec 를 넘으면 늦은 쪽이 버려진다. '
                  '이 값을 올리거나 PTP phase lock 을 켤 것.')
        else:
            print('    (c) 두 센서가 한 프레임에 같이 들어온 적이 없어 시각차를 '
                  '잴 수 없었다 — (a)를 먼저 보라.')

    print()
    print('── 판정 2. 대체 경로 ──')
    print(f'  극좌표 갱신 실패로 융합 1회 갱신 : {fallback * 100:5.1f}%')
    ok_fb = fallback < 0.05
    print(f'  → {"합격" if ok_fb else "불합격"} (5% 미만이어야 함)')
    if not ok_fb:
        print('     타임스탬프 불일치(sync_tolerance_sec)나 센서 간 위치 불일치'
              '(sensor_agreement_dist)로 한쪽이 버려지고 있는지 확인하세요.')

    print()
    print('── 판정 3. NIS (σ 설정이 맞는가) ──')
    if len(nis) == 0:
        print('  표본 없음')
        ok_nis = False
    else:
        m = float(nis.mean())
        print(f'  평균 NIS {m:.2f}  (3자유도 카이제곱 → 이상적 3.0)')
        print(f'  중앙 {np.median(nis):.2f} · P95 {np.percentile(nis, 95):.2f}')
        ok_nis = 1.0 <= m <= 6.0
        print(f'  → {"합격" if ok_nis else "불합격"} (1.0~6.0 범위)')
        if m > 6.0:
            print('     필터가 과신 중입니다. ekf_sigma_body / ekf_sigma_range 를 키우거나,'
                  ' 외부 파라미터(lidarN_extrinsic_rmse)가 실제 캘리브레이션'
                  ' 잔차와 맞는지 확인하세요.')
        elif m < 1.0:
            print('     필터가 과보수적입니다. ekf_sigma_range 를 줄이면 '
                  '관측을 더 씁니다.')

    print()
    print('── 판정 4. 센서 균형 ──')
    for s in names:
        seen = pts[s] > 0
        print(f'  {s:8s} 기여 {seen.mean() * 100:5.1f}%  '
              f'점 중앙 {int(np.median(pts[s][seen])) if seen.any() else 0}개  '
              f'최대 {int(pts[s].max())}개')
    share = np.array([pts[s].sum() for s in names], dtype=float)
    share = share / max(share.sum(), 1)
    print(f'  점 점유율 {" : ".join(f"{v * 100:.0f}%" for v in share)}')
    ok_bal = share.min() > 0.15
    print(f'  → {"합격" if ok_bal else "불합격"} (한쪽이 15% 미만이면 사실상 단일센서)')

    print()
    print(f' 위치 불확실성 σ: 평균 {np.nanmean(sig) * 100:.1f}cm · '
          f'최대 {np.nanmax(sig) * 100:.1f}cm')
    print()
    print('=' * 66)
    allok = ok_fusion and ok_fb and ok_nis and ok_bal
    if allok:
        print(' ✓ 두 센서가 EKF로 하나의 위치 추정에 정상 융합되고 있습니다.')
    else:
        print(' ✗ 융합에 문제가 있습니다. 위 불합격 항목을 확인하세요.')
    print('=' * 66)


if __name__ == '__main__':
    main()
