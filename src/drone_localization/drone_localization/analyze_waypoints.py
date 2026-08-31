#!/usr/bin/env python3
"""
Offline waypoint(마커) 호버 구간 분석기.

flight_recorder가 기록한 estimated_path.csv(= /drone/estimated_pose 전체
궤적)를 비행 후에 읽어, 각 ArUco 마커 근처에서 저속으로 머문 구간을 찾아
median 위치로 재추정한다.

lidar_drone_tracker.py 안의 실시간 판정(_update_waypoint_tracking)과 달리
전체 궤적을 한 번에 놓고 판단하므로, 같은 마커 근처의 저속 구간이 속도추정
노이즈로 잠깐씩 끊겨도 시간 간격만 보고 유연하게 다시 이어붙일 수 있다 —
실시간 버전은 "지금 당장 끊을지 말지"를 프레임 단위로 결정해야 해서
유예시간을 아무리 늘려도 조각남에 근본적으로 더 취약하다.

사용법:
  ros2 run drone_localization flight_recorder   # 비행 시작 "전"에 켜서 끝까지 기록
  ... 비행 ...
  ros2 run drone_localization analyze_waypoints
  # 옵션: --gate_radius, --speed_threshold, --min_duration, --merge_gap, --output_dir
"""

import argparse
import csv
import os

import numpy as np


def _load_csv(path):
    if not os.path.isfile(path):
        return None
    t, x, y, z = [], [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 4:
                continue
            t.append(float(row[0]))
            x.append(float(row[1]))
            y.append(float(row[2]))
            z.append(float(row[3]))
    if not t:
        return None
    return np.array(t), np.array(x), np.array(y), np.array(z)


def _load_markers():
    """generate_competition_map.py의 marker_coords와 항상 동일하게 유지되는
    단일 소스(MARKERS)를 그대로 재사용한다 (좌표 중복 정의 방지)."""
    try:
        from drone_mission.generate_groundtruth_trajectory import MARKERS
        return MARKERS
    except Exception:
        return None


def _smoothed_speed(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                     half_window_s: float = 0.3) -> np.ndarray:
    """±half_window_s 구간의 중심차분으로 수평 속도를 추정.

    실시간 버전은 KF의 프레임간 속도(노이즈 큼)를 그대로 썼지만, 오프라인은
    전체 궤적이 있으므로 앞뒤로 넓게 걸친 차분을 써서 훨씬 매끄러운 속도
    추정이 가능하다 — 이게 실시간 버전보다 조각남이 덜한 이유 중 하나."""
    n = len(t)
    speed = np.zeros(n)
    for i in range(n):
        i0 = np.searchsorted(t, t[i] - half_window_s)
        i1 = np.searchsorted(t, t[i] + half_window_s) - 1
        i1 = max(i1, i0)
        if i1 <= i0:
            continue
        dt = t[i1] - t[i0]
        if dt > 0:
            speed[i] = float(np.hypot(x[i1] - x[i0], y[i1] - y[i0]) / dt)
    return speed


def find_visits(t, x, y, z, markers, gate_radius, speed_threshold,
                 min_duration_s, merge_gap_s):
    """전체 궤적에서 마커별 호버 구간을 찾아 median 위치 리스트를 반환."""
    n = len(t)
    speed = _smoothed_speed(t, x, y)

    marker_ids = list(markers.keys())
    marker_xy = np.array([markers[m] for m in marker_ids])
    pos_xy = np.stack([x, y], axis=1)
    dists = np.linalg.norm(pos_xy[:, None, :] - marker_xy[None, :, :], axis=2)
    nearest_i = np.argmin(dists, axis=1)
    nearest_d = dists[np.arange(n), nearest_i]
    nearest_marker = np.array([marker_ids[i] for i in nearest_i])

    near = nearest_d <= gate_radius
    slow = speed <= speed_threshold
    active = near & slow

    # 1) 연속으로 active인 구간(raw segment)을 먼저 찾는다.
    segments = []  # [marker_id, start_idx, end_idx]
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            vals, counts = np.unique(nearest_marker[i:j], return_counts=True)
            mid = vals[np.argmax(counts)]
            segments.append([mid, i, j - 1])
            i = j
        else:
            i += 1

    # 2) 같은 마커의 인접 구간을, 시간 간격이 merge_gap_s 이내면 하나로 합친다.
    merged = []
    for seg in segments:
        if merged and merged[-1][0] == seg[0] and \
                (t[seg[1]] - t[merged[-1][2]]) <= merge_gap_s:
            merged[-1][2] = seg[2]
        else:
            merged.append(list(seg))

    # 3) 병합된 구간마다 실제 active 표본만 median.
    dt_avg = (t[-1] - t[0]) / (n - 1) if n > 1 else 0.1
    results = []
    visit_counts = {}
    for mid, i0, i1 in merged:
        idxs = [k for k in range(i0, i1 + 1)
                if active[k] and nearest_marker[k] == mid]
        duration = len(idxs) * dt_avg
        if duration < min_duration_s:
            continue
        pts = np.stack([x[idxs], y[idxs], z[idxs]], axis=1)
        med = np.median(pts, axis=0)
        visit_counts[mid] = visit_counts.get(mid, 0) + 1
        results.append({
            'marker_id': mid,
            'visit': visit_counts[mid],
            'pos': med,
            'n_samples': len(idxs),
            'duration_s': duration,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                         default=os.path.expanduser('~/drone_project/flight_logs'))
    parser.add_argument('--gate_radius', type=float, default=4.5)
    parser.add_argument('--speed_threshold', type=float, default=0.25)
    parser.add_argument('--min_duration', type=float, default=0.4)
    parser.add_argument('--merge_gap', type=float, default=2.0)
    args, _ = parser.parse_known_args()

    markers = _load_markers()
    if markers is None:
        print('drone_mission.generate_groundtruth_trajectory의 MARKERS를 '
              '불러오지 못했습니다. drone_mission 패키지가 빌드/소싱됐는지 확인하세요.')
        return

    csv_path = os.path.join(args.output_dir, 'estimated_path.csv')
    data = _load_csv(csv_path)
    if data is None:
        print(f'기록된 CSV가 없습니다: {csv_path}\n'
              'flight_recorder를 비행 "전"에 켜서 끝까지 켜둔 채로 기록한 뒤 '
              '다시 실행하세요.')
        return
    t, x, y, z = data
    print(f'{len(t)}개 샘플, {t[-1] - t[0]:.1f}초 구간 로드 '
          f'(gate={args.gate_radius}m, speed<{args.speed_threshold}m/s, '
          f'merge_gap={args.merge_gap}s)')

    results = find_visits(t, x, y, z, markers, args.gate_radius,
                           args.speed_threshold, args.min_duration, args.merge_gap)

    if not results:
        print('호버 구간을 하나도 못 찾았습니다. '
              '--gate_radius / --speed_threshold를 조정해보세요.')
        return

    out_path = os.path.join(args.output_dir, 'waypoint_estimates_offline.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['marker_id', 'visit', 'x', 'y', 'z', 'n_samples', 'duration_s'])
        for r in results:
            p = r['pos']
            w.writerow([r['marker_id'], r['visit'], f'{p[0]:.4f}', f'{p[1]:.4f}',
                        f'{p[2]:.4f}', r['n_samples'], f"{r['duration_s']:.2f}"])
            print(f"id{r['marker_id']} 방문#{r['visit']}: "
                  f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  "
                  f"n={r['n_samples']}샘플 ({r['duration_s']:.1f}초)")

    print(f'\n결과 저장: {out_path}')


if __name__ == '__main__':
    main()
