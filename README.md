# Dual Ouster UAV Localization Workspace

ROS 2 기반의 듀얼 Ouster OS1-128 드론 외부 측위 및 대회 운영 워크스페이스입니다.
각 LiDAR의 점군을 독립적으로 처리한 뒤 센서별 드론 위치를 EKF로 결합하고,
추정 위치의 시각화·기록·채점 기능을 제공합니다.

```text
Ouster LiDAR 2대
  -> 센서별 좌표 변환 / ROI / 배경 제거 / DBSCAN
  -> 센서별 위치 추정
  -> Polar EKF
  -> /drone/estimated_pose
  -> RViz / 대회 GUI / 채점 / CSV 분석
```

> [!WARNING]
> 현재 `lidar_localization_os1.yaml`의 위치·자세·ROI 값은 실내 벤치 테스트용
> 임시 설정입니다. 실제 운용 전 두 LiDAR의 외부 파라미터를 다시 보정하고,
> 네트워크 주소와 배경 모델 경로를 현장 환경에 맞게 변경해야 합니다.

## 패키지 구성

| 패키지 | 역할 |
| --- | --- |
| `dual_ouster_driver` | Ouster 드라이버 두 대의 lifecycle 실행과 토픽 분리 |
| `lidar_calibration` | 대응점과 Kabsch/SVD를 이용한 LiDAR 외부 파라미터 보정 |
| `drone_localization` | 점군 필터링, DBSCAN 검출, EKF 추적, RViz 및 결과 분석 |
| `competition_score` | 임무 이벤트와 추정 궤적을 이용한 오프라인 채점 |
| `competition_gui` | 팀·비행 차수 관리와 측위/채점 프로세스 제어 |

## 개발 환경

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- Ouster ROS 드라이버(`ouster_ros`)

주요 Python 의존성은 NumPy, SciPy, scikit-learn, Matplotlib, PyYAML입니다.
`lidar_drone_tracker_seq`를 사용할 때는 Open3D가 추가로 필요합니다.

## 설치 및 빌드

ROS 2와 Ouster 드라이버를 먼저 설치한 뒤 워크스페이스 루트에서 실행합니다.

```bash
source /opt/ros/humble/setup.bash

rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install numpy scipy scikit-learn matplotlib PyYAML

colcon build --symlink-install
source install/setup.bash
```

Open3D 기반 비교용 트래커까지 사용할 경우 다음 의존성을 추가합니다.

```bash
python3 -m pip install open3d
```

## 설정

운용 전에 다음 파일을 환경에 맞게 확인합니다.

- `src/dual_ouster_driver/launch/dual_os1.launch.py`: 센서 주소와 UDP 목적지
- `src/dual_ouster_driver/config/lidar1.yaml`: LiDAR 1 설정
- `src/dual_ouster_driver/config/lidar2.yaml`: LiDAR 2 설정
- `src/drone_localization/params/lidar_localization_os1.yaml`: 외부 파라미터, ROI, 필터 및 EKF 설정
- `src/drone_localization/launch/lidar_localization.launch.py`: 배경 모델 기본 경로

### 외부 파라미터 보정

```bash
ros2 launch dual_ouster_driver dual_os1.launch.py
ros2 run lidar_calibration extrinsic_calibrator
```

보정 결과의 `x`, `y`, `z`, `roll`, `pitch`, `yaw`를
`lidar_localization_os1.yaml`에 반영합니다.

### 정적 배경 학습

외부 파라미터 보정이 끝난 뒤 비어 있는 비행장에서 한 번 실행합니다.

```bash
ros2 launch drone_localization lidar_localization.launch.py \
  learn_background:=true \
  use_background:=true \
  background_file:=$HOME/competition_results/calibration/dual_ouster_background.npz
```

## 실행

### LiDAR 측위

```bash
ros2 launch drone_localization lidar_localization.launch.py \
  learn_background:=false \
  use_background:=true \
  background_file:=$HOME/competition_results/calibration/dual_ouster_background.npz
```

두 번째 LiDAR 없이 드라이버만 확인하려면 다음처럼 실행할 수 있습니다.

```bash
ros2 launch dual_ouster_driver dual_os1.launch.py enable_lidar2:=false
```

### EKF 결과 시각화

```bash
ros2 launch drone_localization estimate_view.launch.py
```

RViz를 별도 장비에서 실행할 경우:

```bash
ros2 launch drone_localization estimate_view.launch.py rviz:=false
```

### 대회 운영 GUI

```bash
ros2 run competition_gui competition_gui
```

GUI는 측위 노드와 채점 노드를 실행하고 결과를 기본적으로
`~/competition_results/` 아래에 저장합니다.

### 결과 분석

```bash
ros2 run drone_localization plot_run --list
ros2 run drone_localization plot_run --save-only
```

## 주요 토픽

| 토픽 | 설명 |
| --- | --- |
| `/lidar1/points`, `/lidar2/points` | 센서별 원본 점군 입력 |
| `/drone/lidar1_pose`, `/drone/lidar2_pose` | 센서별 드론 위치 추정 |
| `/drone/estimated_pose` | EKF가 결합한 최종 위치 |
| `/competition/run_config` | 팀 및 실행 설정 |
| `/competition/mission_event` | 대회 시작·종료 등 임무 이벤트 |

## 테스트

```bash
colcon test
colcon test-result --verbose
```

측위 알고리즘 단위 테스트만 빠르게 실행하려면:

```bash
pytest -q src/drone_localization/test
```

테스트는 EKF Jacobian, 두 센서 갱신, 비동기 스캔 보정, 센서 불일치 처리와
합성 점군 기반 추적 파이프라인을 검증합니다.

## 저장소에 포함하지 않는 파일

`build/`, `install/`, `log/`, Python 캐시, 비행 기록과 배경 모델은
`.gitignore`에서 제외합니다. 필요한 결과 데이터는 별도 저장소나 릴리스
아티팩트로 관리하는 것을 권장합니다.
