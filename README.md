# Dual Ouster UAV Localization Workspace

ROS 2 기반의 듀얼 Ouster OS1-128 드론 외부 측위 및 대회 운영 워크스페이스입니다.
각 LiDAR의 점군을 독립적으로 처리한 뒤 센서별 드론 위치를 polar EKF로 결합하고,
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
> 현재 `lidar_localization_os1.yaml`의 외부 파라미터·ROI·마커 좌표에는
> 실내 벤치 테스트용 임시 값이 포함되어 있습니다. 실제 운용 전 반드시
> 현장의 네트워크, LiDAR 자세, 경기장 ROI, 마커 좌표를 다시 측정하십시오.

## 목차

1. [최초 설정](#1-최초-설정)
2. [LiDAR 외부 파라미터 캘리브레이션](#2-lidar-외부-파라미터-캘리브레이션)
3. [정적 배경 학습](#3-정적-배경-학습)
4. [경기 실행](#4-경기-실행)
5. [결과 분석](#5-결과-분석)
6. [문제 해결](#6-문제-해결)

## 패키지 구성

| 패키지 | 역할 | 주요 실행 항목 |
| --- | --- | --- |
| `dual_ouster_driver` | Ouster 드라이버 2대의 lifecycle 실행과 토픽 분리 | `dual_os1.launch.py` |
| `lidar_calibration` | 대응점과 Kabsch/SVD를 이용한 외부 파라미터 보정 | `extrinsic_calibrator` |
| `drone_localization` | 점군 필터링, DBSCAN 검출, EKF 추적, RViz와 분석 | `lidar_drone_tracker` 등 |
| `competition_score` | 임무 이벤트와 추정 궤적을 이용한 채점 | `compet_score` |
| `competition_gui` | 팀·비행 차수 관리와 측위/채점 프로세스 제어 | `competition_gui` |

## 전체 운영 순서

최초 설치 때는 1~3단계를 수행합니다. 외부 파라미터 또는 ROI를 변경했다면
2~3단계를 다시 수행하고, 평소 경기에서는 4~5단계만 반복합니다.

```text
최초 설정
  -> LiDAR 통신 확인
  -> 외부 파라미터 캘리브레이션
  -> 빈 경기장 배경 학습
  -> 사전 점검
  -> GUI에서 경기 실행
  -> FINISH 후 결과 분석
```

---

## 1. 최초 설정

### 1.1 준비물과 지원 환경

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- Ouster OS1-128 2대와 ROS 2용 `ouster_ros`
- 두 센서와 연결할 전용 유선 NIC 또는 스위치
- 캘리브레이션 타깃과 월드 좌표를 측정할 장비
- GUI 사용 시 `python3-tk`
- RViz 사용 시 `rviz2`, `tf2_ros`

선택 기능에는 다음 외부 패키지가 필요합니다.

- `px4_msgs`: `flight_recorder`에서 PX4 실제 경로를 함께 기록할 때 필요
- `drone_mission`: 계획 궤적 오버레이와 오프라인 경로점 분석에 필요
- 미션 제어/이벤트 발행 노드: 실제 비행과 채점 이벤트를 연결할 때 필요

위 두 구성요소는 이 저장소에 포함되어 있지 않습니다.
- Open3D: 비교용 `lidar_drone_tracker_seq`를 사용할 때만 필요

### 1.2 저장소 받기

이 저장소는 비공개이므로 GitHub 인증이 된 환경에서 clone합니다.

```bash
git clone https://github.com/jinuww/drone_ws.git
cd drone_ws
```

이미 이 폴더를 가지고 있다면 워크스페이스 루트로 이동하면 됩니다.

```bash
cd "/path/to/drone_ws"
```

### 1.3 ROS와 Python 의존성 설치

ROS 2 Humble을 설치한 뒤 다음을 실행합니다.

```bash
source /opt/ros/humble/setup.bash

sudo apt update
sudo apt install -y python3-pip python3-pytest python3-tk \
  ros-humble-rviz2 ros-humble-tf2-ros

rosdep update
rosdep install --from-paths src --ignore-src -r -y

/usr/bin/python3 -m pip install --user \
  numpy scipy scikit-learn matplotlib PyYAML
```

비교용 순차 트래커까지 사용할 경우에만 Open3D를 추가합니다.

```bash
/usr/bin/python3 -m pip install --user open3d
```

> [!IMPORTANT]
> ROS 2 Humble은 Ubuntu 22.04의 Python 3.10을 기준으로 합니다. Conda가
> 활성화되어 `python3`가 다른 버전을 가리키면 `rclpy` import 오류가 날 수
> 있습니다. 이 경우 `conda deactivate` 후 `/usr/bin/python3`를 사용하십시오.

Ouster 드라이버 설치 여부는 다음으로 확인합니다.

```bash
source /opt/ros/humble/setup.bash
ros2 pkg prefix ouster_ros
```

경로가 출력되지 않으면 현재 ROS 2 배포판과 센서 펌웨어에 맞는
`ouster_ros`를 먼저 설치하거나 소스 빌드해야 합니다.

### 1.4 빌드

```bash
cd "/path/to/drone_ws"
source /opt/ros/humble/setup.bash

colcon build --symlink-install
source install/setup.bash
```

새 터미널을 열 때마다 다음 세 줄을 먼저 실행합니다.

```bash
cd "/path/to/drone_ws"
source /opt/ros/humble/setup.bash
source install/setup.bash
```

실행 파일이 설치됐는지 확인합니다.

```bash
ros2 pkg executables drone_localization
ros2 pkg executables lidar_calibration
ros2 pkg executables competition_gui
ros2 pkg executables competition_score
```

### 1.5 LiDAR 네트워크 설정

현재 기본 네트워크 값은 다음과 같습니다.

| 장치 | 주소 | 출력 토픽 |
| --- | --- | --- |
| 호스트 유선 NIC | `192.168.6.100/24` | UDP 수신 |
| LiDAR 1 | `192.168.6.11` | `/lidar1/points` |
| LiDAR 2 | `192.168.6.12` | `/lidar2/points` |

전용 NIC의 IPv4 주소를 `192.168.6.100/24`로 설정합니다. NetworkManager를
사용한다면 먼저 연결 이름을 찾고, `<유선 연결 이름>`을 실제 값으로 바꿉니다.

```bash
nmcli -f NAME,DEVICE connection show
sudo nmcli connection modify "<유선 연결 이름>" \
  ipv4.method manual ipv4.addresses 192.168.6.100/24
sudo nmcli connection up "<유선 연결 이름>"
```

주소와 연결 상태를 확인합니다.

```bash
ip -br address
ping -c 3 192.168.6.11
ping -c 3 192.168.6.12
```

주소가 다르면 아래 파일을 수정합니다.

- `src/dual_ouster_driver/launch/dual_os1.launch.py`
  - `SENSORS`: LiDAR 주소
  - `UDP_DEST`: 호스트 NIC 주소
- `src/dual_ouster_driver/config/lidar1.yaml`
- `src/dual_ouster_driver/config/lidar2.yaml`

> [!NOTE]
> 실제 `dual_os1.launch.py`는 현재 센서 설정을 launch 파일 안에서 직접
> 전달합니다. `config/lidar1.yaml`, `config/lidar2.yaml`은 참고/대체 설정이며
> 기본 launch에서 자동으로 읽히지 않습니다. 기본 실행값을 바꾸려면 launch
> 파일의 `SENSORS`, `UDP_DEST`, `parameters`를 우선 수정하십시오.

설정 파일을 수정한 뒤 다시 빌드합니다.

```bash
colcon build --symlink-install --packages-select dual_ouster_driver
source install/setup.bash
```

### 1.6 최초 점군 통신 확인

터미널 A에서 드라이버만 실행합니다. launch 파일이 lifecycle 노드를 자동으로
`configure -> activate` 상태로 전환합니다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch dual_ouster_driver dual_os1.launch.py
```

터미널 B에서 두 토픽이 보이고 약 10 Hz로 들어오는지 확인합니다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 topic list | grep -E '^/lidar[12]/points$'
ros2 topic hz /lidar1/points
```

`Ctrl+C`로 첫 번째 측정을 멈춘 뒤 LiDAR 2도 확인합니다.

```bash
ros2 topic hz /lidar2/points
```

한 대만 연결한 벤치 시험에서는 다음처럼 실행합니다.

```bash
ros2 launch dual_ouster_driver dual_os1.launch.py enable_lidar2:=false
```

기본 launch는 한 센서가 통신 실패로 `finalized` 상태가 되면 전체 시스템을
종료하도록 되어 있습니다. 실제 듀얼 운용 전에는 두 센서가 모두 응답해야 합니다.

### 1.7 경기장 좌표와 파라미터 설정

주요 운용 파라미터는 다음 파일에 있습니다.

```text
src/drone_localization/params/lidar_localization_os1.yaml
```

최소한 다음 항목을 현장과 일치시킵니다.

- `lidar1_*`, `lidar2_*`: 센서 위치·자세와 캘리브레이션 RMSE
- `roi_x_*`, `roi_y_*`, `roi_z_*`: 실제 추적 영역
- `cluster_*`: 드론 점군의 크기와 최소 점 수
- `marker1_*`~`marker4_*`: 실제 마커 위치
- `home_*`: 필터 초기 위치
- `sensor_agreement_dist`: 두 센서 검출 일치 허용 거리
- `sync_tolerance_sec`, `async_tolerance_sec`: 스캔 시간차 허용값

채점기의 마커 좌표와 고도도 함께 일치해야 합니다.

```text
src/competition_score/competition_score/compet_score.py
```

현재 GUI는 채점기를 별도 YAML 없이 기본 파라미터로 실행하므로,
`marker_1`~`marker_4`, `home`, `target_altitude`를 바꿀 때는 트래커 YAML과
채점기 기본값을 함께 수정해야 합니다.

또한 GUI 실행은 배경 파일 인자를 별도로 전달하지 않습니다. 다음 파일의
`background_file` 기본값이 실제 사용자 경로를 가리키도록 최초 1회 수정합니다.

```text
src/drone_localization/launch/lidar_localization.launch.py
```

기본 소스 값 `/home/drone/competition_results/...`을 예를 들어 다음과 같이
현재 계정의 절대 경로로 바꿉니다.

```text
/home/<사용자명>/competition_results/calibration/dual_ouster_background.npz
```

파라미터 또는 launch 파일을 수정한 뒤 다시 빌드합니다.

```bash
colcon build --symlink-install --packages-select \
  drone_localization competition_score competition_gui
source install/setup.bash
```

### 1.8 기본 테스트

하드웨어 없이 측위 수학과 합성 점군 파이프라인을 확인할 수 있습니다.

```bash
/usr/bin/python3 -m pytest -q src/drone_localization/test
```

전체 ROS 패키지 테스트는 다음과 같습니다.

```bash
colcon test
colcon test-result --verbose
```

---

## 2. LiDAR 외부 파라미터 캘리브레이션

### 2.1 캘리브레이션 원리와 준비

캘리브레이터는 센서 좌표의 타깃 중심과 이미 알고 있는 월드 좌표 사이의
대응점을 모은 뒤 Kabsch/SVD로 다음 변환을 구합니다.

```text
p_world = R_sensor_to_world * p_sensor + t_sensor_to_world
```

준비 사항:

- 두 LiDAR가 움직이지 않도록 단단히 고정
- 월드 좌표계의 원점과 X/Y/Z 방향을 먼저 확정
- 중심 위치를 정확히 측정할 수 있는 타깃 준비
- 최소 3개, 권장 6개 이상의 서로 다른 대응점 준비
- 대응점을 한 직선이나 한곳에 몰지 말고 X/Y/Z 방향으로 넓게 분산
- 캡처에 사용할 ROI는 월드 좌표가 아니라 **각 LiDAR의 원시 센서 좌표**

### 2.2 드라이버와 캘리브레이터 실행

터미널 A:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch dual_ouster_driver dual_os1.launch.py
```

터미널 B:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run lidar_calibration extrinsic_calibrator
```

프롬프트가 `calibration>`으로 바뀌면 명령을 입력합니다.

### 2.3 대응점 캡처

명령 형식:

```text
capture <lidar> <world_x> <world_y> <world_z> \
  <xmin> <xmax> <ymin> <ymax> <zmin> <zmax>
```

예를 들어 월드 좌표 `(2.0, 2.0, 1.5)`에 놓인 타깃이 LiDAR 1 원시 좌표에서
`x=1.8~2.2`, `y=-0.3~0.3`, `z=-0.3~0.3`에 보인다면:

```text
calibration> capture 1 2.0 2.0 1.5 1.8 2.2 -0.3 0.3 -0.3 0.3
```

노드는 기본적으로 유효한 스캔 20프레임을 모아 타깃 중심을 저장합니다.
`Target ... captured` 메시지가 나온 뒤 타깃을 다음 위치로 옮깁니다. 같은 실제
타깃을 LiDAR 2에서도 캡처하되, LiDAR 2에서 보이는 원시 ROI를 별도로 지정합니다.

유용한 명령:

```text
calibration> list 1
calibration> list 2
calibration> delete 1
calibration> delete 2
calibration> save 1 ~/lidar1_calibration.csv
calibration> save 2 ~/lidar2_calibration.csv
calibration> quit
```

- `list N`: 저장된 센서/월드 대응점 확인
- `delete N`: 해당 센서의 마지막 대응점 삭제
- `save N FILE`: 대응점을 CSV로 보관

### 2.4 변환 계산과 적용

각 센서에 대응점이 충분히 모이면 계산합니다.

```text
calibration> solve 1
calibration> solve 2
```

출력되는 다음 값을 기록합니다.

- `lidarN_x`, `lidarN_y`, `lidarN_z`
- `lidarN_roll`, `lidarN_pitch`, `lidarN_yaw` — 단위는 radian
- `RMSE`, `Max error`

결과를 `lidar_localization_os1.yaml`의 동일한 키에 복사하고, 실제 RMSE를
`lidarN_extrinsic_rmse`에 반영합니다. 큰 오차가 한 타깃에 집중되어 있으면
`delete`로 제거한 뒤 타깃 중심과 원시 ROI를 다시 확인하고 재캡처합니다.

```bash
colcon build --symlink-install --packages-select drone_localization
source install/setup.bash
```

### 2.5 캘리브레이션 검증

배경 제거를 끄고 측위를 실행합니다.

```bash
ros2 launch drone_localization lidar_localization.launch.py \
  use_background:=false \
  learn_background:=false
```

확인 항목:

```bash
ros2 topic hz /lidar1/points_world
ros2 topic hz /lidar2/points_world
ros2 topic echo /drone/lidar1_pose --once
ros2 topic echo /drone/lidar2_pose --once
```

- 같은 물체가 두 센서의 월드 점군에서 같은 위치에 겹치는지 확인
- 드론을 움직였을 때 두 센서의 개별 pose가 같은 방향으로 움직이는지 확인
- 축이 뒤집히거나 거울처럼 움직이면 yaw/좌표축 정의부터 다시 확인
- 두 개별 pose 차이가 `sensor_agreement_dist`를 계속 넘으면 융합 전에
  캘리브레이션을 다시 수행

외부 파라미터 또는 ROI를 바꾸면 기존 배경 모델은 더 이상 유효하지 않으므로
다음 단계에서 반드시 다시 학습합니다.

---

## 3. 정적 배경 학습

### 3.1 학습 전 조건

- 외부 파라미터와 ROI 설정 완료
- 두 LiDAR 점군이 모두 정상 수신
- 경기장 안에 드론·사람·움직이는 물체가 없음
- 센서와 주변 구조물이 학습 중 움직이지 않음
- 저장 경로의 상위 디렉터리에 쓰기 권한이 있음

배경 파일 경로 예시:

```bash
mkdir -p "$HOME/competition_results/calibration"
```

### 3.2 학습 실행

다른 드라이버가 이미 실행 중이면 먼저 종료합니다. 아래 launch가 두 드라이버와
트래커를 함께 실행합니다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch drone_localization lidar_localization.launch.py \
  learn_background:=true \
  use_background:=true \
  background_file:="$HOME/competition_results/calibration/dual_ouster_background.npz"
```

기본값은 LiDAR별 200프레임입니다. 10 Hz 기준 약 20초가 필요하며 터미널에
다음과 같은 진행 메시지가 표시됩니다.

```text
Background learning [lidar1:.../200, lidar2:.../200]
Background calibration COMPLETE: .../dual_ouster_background.npz
```

`COMPLETE`가 나오기 전에는 종료하지 마십시오. 완료 후 `Ctrl+C`로 안전하게
종료하고 파일을 확인합니다.

```bash
ls -lh "$HOME/competition_results/calibration/dual_ouster_background.npz"
```

### 3.3 정상 추적 모드 확인

경기장에 드론을 놓고 정상 모드로 다시 실행합니다.

```bash
ros2 launch drone_localization lidar_localization.launch.py \
  learn_background:=false \
  use_background:=true \
  background_file:="$HOME/competition_results/calibration/dual_ouster_background.npz"
```

시작 로그에서 두 센서에 대해 다음 메시지를 확인합니다.

```text
Loaded ... background voxels for lidar1
Loaded ... background voxels for lidar2
```

추정 토픽이 나오는지도 확인합니다.

```bash
ros2 topic hz /drone/estimated_pose
ros2 topic echo /drone/estimated_pose --once
```

배경 사용 때문에 드론까지 사라지는지 비교하려면 잠시 다음처럼 실행합니다.

```bash
ros2 launch drone_localization lidar_localization.launch.py \
  learn_background:=false use_background:=false
```

외부 파라미터, ROI 또는 `background_voxel_size`가 바뀌면 배경 서명이 달라집니다.
노드는 서명이 맞지 않는 이전 파일을 자동으로 거부하므로 3.2를 다시 수행합니다.

---

## 4. 경기 실행

### 4.1 경기 전 체크리스트

경기마다 아래 항목을 먼저 확인합니다.

- [ ] 두 LiDAR에 ping 성공
- [ ] `/lidar1/points`, `/lidar2/points`가 각각 약 10 Hz
- [ ] 캘리브레이션 이후 센서가 움직이지 않음
- [ ] 현재 파라미터와 일치하는 배경 모델이 로드됨
- [ ] 경기장 ROI와 마커 좌표가 실제 배치와 일치
- [ ] `/drone/estimated_pose`가 안정적으로 출력됨
- [ ] 저장 장치에 충분한 공간과 쓰기 권한이 있음
- [ ] 미션 제어 노드가 아래 임무 이벤트 규약을 사용함

### 4.2 선택: RViz 모니터링

측위 또는 GUI를 실행한 뒤 별도 터미널에서 RViz를 붙일 수 있습니다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch drone_localization estimate_view.launch.py
```

옵션:

```bash
# RViz 없이 시각화 토픽과 TF만 실행
ros2 launch drone_localization estimate_view.launch.py rviz:=false

# 이미 map -> world TF가 있으면 중복 TF 비활성화
ros2 launch drone_localization estimate_view.launch.py static_tf:=false

# 궤적 길이 변경
ros2 launch drone_localization estimate_view.launch.py trail_max_points:=4000
```

### 4.3 선택: PX4 실제 경로 동시 기록

GUI 런 결과에 PX4 실제 경로와 LiDAR 추정을 함께 남기려면 GUI에서
`START RUN`을 눌러 상태가 `RUNNING`이 된 직후, 실제 이륙 전에
`flight_recorder`를 별도 터미널에서 실행합니다. `<Team>_<ID>`와 런 번호를
GUI에 입력한 값에 맞춥니다.

```bash
RUN_DIR="$HOME/competition_results/<Team>_<ID>/run_01"

ros2 run drone_localization flight_recorder --ros-args \
  -p output_dir:="$RUN_DIR"
```

`px4_msgs`가 없으면 `actual_path.csv`에는 데이터가 기록되지 않고 LiDAR의
`estimated_path.csv`만 기록됩니다. 경기 종료 후 이 터미널도 `Ctrl+C`로 닫습니다.

### 4.4 GUI로 경기 시작

측위 드라이버나 채점기를 따로 실행하지 않은 상태에서 GUI를 시작합니다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run competition_gui competition_gui
```

GUI 사용 순서:

1. `Team Name` 입력
2. `Team ID` 입력
3. 정수 `Run Number` 입력
4. `START RUN` 클릭
5. 상태가 `RUNNING`으로 바뀐 것을 확인한 뒤 비행 시작

`START RUN`은 다음 작업을 자동 수행합니다.

1. `~/competition_results/<Team>_<ID>/run_NN/` 생성
2. `lidar_localization.launch.py` 실행
3. `competition_score/compet_score` 실행
4. 두 노드가 ROS graph에 나타날 때까지 대기
5. `/competition/run_config` 발행
6. `/competition/mission_event`에 `START` 발행
7. 트래커와 채점기의 CSV 기록 시작

동일한 런 폴더가 이미 있으면 덮어쓸지 묻습니다. 이전 파일과 새 파일이
섞이지 않게 가능한 한 새 Run Number를 사용하는 것이 안전합니다.

### 4.5 미션 이벤트 규약

GUI는 `START`, `FINISH`, `ABORT`만 발행합니다. 실제 마커 검출·호버·구간
추적 이벤트는 비행 미션 제어 노드가 `/competition/mission_event`로 발행해야
정상 채점됩니다.

| 이벤트 문자열 | 의미 |
| --- | --- |
| `SEARCH_START` | 마커 탐색 시작 |
| `SEARCH_RESUME` | 호버 후 탐색 재개 |
| `MARKER_DETECTED:N` | 마커 N 검출, N은 1~4 |
| `HOVER_START:N` | 마커 N 호버 측정 시작 |
| `HOVER_END:N` | 마커 N 호버 측정 종료 |
| `TRACE_START:4:3` | 마커 4에서 3으로 가는 구간 채점 시작 |
| `TRACE_START:3:2` | 마커 3에서 2로 가는 구간 채점 시작 |
| `TRACE_START:2:1` | 마커 2에서 1로 가는 구간 채점 시작 |
| `RETURN_HOME` | 마커 1에서 HOME으로 복귀 구간 채점 |
| `FINISH` | 정상 종료와 결과 저장, GUI가 발행 |
| `ABORT` | 중단과 0점 결과 저장, GUI가 발행 |

미션 노드 없이 채점기 이벤트 수신만 점검할 때는 다음처럼 한 번씩 발행할 수
있습니다. 실제 경기에서는 GUI가 이미 `START`를 보내므로 다시 보내지 않습니다.

```bash
ros2 topic pub --once /competition/mission_event std_msgs/msg/String \
  "{data: 'SEARCH_START'}"

ros2 topic pub --once /competition/mission_event std_msgs/msg/String \
  "{data: 'MARKER_DETECTED:1'}"

ros2 topic pub --once /competition/mission_event std_msgs/msg/String \
  "{data: 'HOVER_START:1'}"

# 필요한 시간 동안 호버한 뒤
ros2 topic pub --once /competition/mission_event std_msgs/msg/String \
  "{data: 'HOVER_END:1'}"
```

채점기가 지원하는 직선 구간 키는 `4-3`, `3-2`, `2-1`, `1-0`입니다.
`1-0`은 `TRACE_START:1:0` 대신 `RETURN_HOME` 이벤트를 사용합니다.

### 4.6 실행 중 상태 확인

별도 터미널에서 다음을 확인합니다.

```bash
ros2 node list | grep -E 'lidar_drone_tracker|competition_score'
ros2 topic hz /drone/estimated_pose
ros2 topic echo /competition/run_config --once \
  --qos-durability transient_local
```

트래커는 주기적으로 다음 통계를 출력합니다.

- `Detection ...%`: 처리 프레임 중 드론 검출 비율
- `lidarN:...pts/...frames`: 센서별 기여 점 수와 프레임 수
- `Polar EKF ... sensors/frame`: 한 프레임에 실제 반영된 센서 수
- `NIS mean`: 이상값은 약 3.0, 높으면 과신, 낮으면 과보수 가능성
- `sync median/p95`: 두 LiDAR 스캔 시간차

### 4.7 정상 종료와 중단

정상 비행을 마치면 GUI에서 `FINISH RUN`을 누릅니다.

1. GUI가 `FINISH` 이벤트 발행
2. 트래커가 CSV를 flush하고 닫음
3. 채점기가 `scoring_trajectory.csv`와 `result.json` 저장
4. GUI가 `result.json` 생성을 확인한 뒤 하위 프로세스 종료
5. 상태가 `RUN FINISHED`로 변경

비행을 무효 처리해야 하면 `ABORT RUN`을 누릅니다. `ABORT` 결과의
`final_score`는 0점이며, 0.5초 후 측위와 채점 프로세스를 종료합니다.

GUI 창을 닫을 때 실행 중인 런이 있으면 확인 창이 나타납니다. 터미널을 강제로
종료하기보다 GUI의 `FINISH RUN` 또는 `ABORT RUN`을 사용해야 CSV 손상을 줄일 수
있습니다.

### 4.8 GUI 없이 측위만 실행

채점이나 팀별 기록이 필요 없는 센서 점검에서는 다음처럼 직접 실행합니다.

```bash
ros2 launch drone_localization lidar_localization.launch.py \
  params_file:=lidar_localization_os1.yaml \
  learn_background:=false \
  use_background:=true \
  background_file:="$HOME/competition_results/calibration/dual_ouster_background.npz"
```

이 모드에서는 `/competition/run_config`와 `START`가 없으므로 GUI 런용 CSV는
생성되지 않습니다. 토픽과 RViz 점검용으로 사용합니다.

### 4.9 런 결과 파일

기본 저장 경로:

```text
~/competition_results/<정리된_Team>_<정리된_ID>/run_NN/
```

| 파일 | 생성 주체 | 내용 |
| --- | --- | --- |
| `raw_detections.csv` | 트래커 | 센서별 원시 검출과 융합 정보 |
| `drone_trajectory.csv` | 트래커 | EKF 위치·속도·공분산 |
| `waypoint_estimates.csv` | 트래커 | 실시간 호버 구간의 경로점 median |
| `ekf_diagnostics.csv` | 트래커 | 센서별 기여, NIS, 동기 상태 |
| `scoring_trajectory.csv` | 채점기 | 시각별 상태와 항목별 오차 |
| `result.json` | 채점기 | 항목 점수, 감점, 최종 점수 |
| `actual_path.csv` | 선택적 recorder | PX4 실제 경로 |
| `estimated_path.csv` | 선택적 recorder | 비교용 LiDAR 추정 경로 |

---

## 5. 결과 분석

아래 예시의 `RUN_DIR`을 분석할 런 폴더로 바꿉니다.

```bash
RUN_DIR="$HOME/competition_results/<Team>_<ID>/run_01"
```

### 5.1 최종 점수 확인

```bash
/usr/bin/python3 -m json.tool "$RUN_DIR/result.json"
```

주요 필드:

- `status`: `FINISHED` 또는 `ABORTED`
- `completion_time_s`: 총 수행 시간
- `detected_markers`, `valid_hover_markers`: 검출 및 유효 호버 마커
- `grid_rmse_m`, `segment_rmse_m`, `altitude_rmse_m`: 항목별 RMSE
- `hover_rmse_m`: 마커별 호버 RMSE
- `scores`: grid/hover/segments/altitude/time 세부 점수
- `penalty`: 미검출·무효 호버 감점
- `final_score`: 0~100 최종 점수

### 5.2 GUI 런 궤적 그리기

런 목록 확인:

```bash
ros2 run drone_localization plot_run --list
```

특정 런 분석:

```bash
ros2 run drone_localization plot_run --run "$RUN_DIR"
```

화면 없이 PNG만 생성:

```bash
ros2 run drone_localization plot_run --run "$RUN_DIR" --save-only
```

같은 런 폴더에 `trajectory.png`가 저장됩니다. 그림의 네 패널은 다음을 뜻합니다.

- 평면 궤적: ROI 이탈과 경로점 위치
- 고도: `roi_z_min`~`roi_z_max` 범위 이탈
- 공분산: 관측이 끊겨 코스팅한 구간
- 샘플 간격: 목표 100 ms 대비 처리 지연

### 5.3 두 LiDAR EKF 융합 검사

```bash
ros2 run drone_localization check_ekf_fusion --output_dir "$RUN_DIR"
```

기본 판정 항목:

- 두 센서가 동시에 반영된 프레임 비율 80% 이상
- fused fallback 비율 5% 미만
- 평균 NIS 1.0~6.0
- 각 센서 점 점유율 15% 초과

검사 결과가 불합격이면 출력되는 `n_clouds`, 센서별 점 수, 스캔 시각차 진단을
따라 문제 해결 절의 센서 누락·ROI·동기화 항목을 확인합니다.

### 5.4 PX4 실제 경로와 비교

`flight_recorder`를 함께 실행해 `actual_path.csv`와 `estimated_path.csv`가 있을 때
사용합니다.

```bash
ros2 run drone_localization plot_flight \
  --output_dir "$RUN_DIR" --save_only

ros2 run drone_localization plot_flight_3d \
  --output_dir "$RUN_DIR" --color_by_error --save_only
```

생성 파일:

- `flight_plot.png`: XY, 고도, 3D 및 PX4 대비 오차 요약
- `flight_plot_3d.png`: 필드·LiDAR·ROI를 포함한 3D 경로

### 5.5 오프라인 경로점 재분석

이 기능은 외부 `drone_mission` 패키지의 `MARKERS` 정의가 필요합니다.

```bash
ros2 run drone_localization analyze_waypoints \
  --output_dir "$RUN_DIR"
```

기본 판정값:

- 마커 gate 반경: 4.5 m
- 저속 기준: 0.25 m/s 이하
- 최소 지속 시간: 0.4 s
- 같은 마커 구간 병합 간격: 2.0 s

필요하면 조정합니다.

```bash
ros2 run drone_localization analyze_waypoints \
  --output_dir "$RUN_DIR" \
  --gate_radius 2.0 \
  --speed_threshold 0.20 \
  --min_duration 1.0 \
  --merge_gap 1.0
```

결과는 `waypoint_estimates_offline.csv`에 저장됩니다.

### 5.6 결과 보관

`competition_results/`, CSV, 배경 `.npz`, ROS bag은 `.gitignore`에서 제외됩니다.
실험 결과는 날짜·장소·파라미터 버전과 함께 별도 스토리지에 백업하십시오.
재현성을 위해 다음도 같이 보관하는 것이 좋습니다.

- 사용한 `lidar_localization_os1.yaml`
- 캘리브레이션 대응점 CSV
- 배경 모델 생성 시각과 센서 배치 사진
- Git 커밋 해시

---

## 6. 문제 해결

### 6.1 드라이버가 `Couldn't communicate with lidarN`으로 종료됨

확인:

```bash
ip -br address
ping -c 3 192.168.6.11
ping -c 3 192.168.6.12
```

조치:

- 호스트 NIC가 `192.168.6.100/24`인지 확인
- `dual_os1.launch.py`의 `SENSORS`, `UDP_DEST` 확인
- 센서 전원·케이블·스위치와 중복 IP 확인
- 여러 NIC가 있으면 UDP 목적지가 실제 LiDAR NIC를 가리키는지 확인

### 6.2 노드는 보이지만 `/lidarN/points`가 나오지 않음

`os_driver`는 lifecycle 노드입니다. 직접 `ros2 run ouster_ros os_driver`만 실행하면
`unconfigured` 상태에 멈출 수 있으므로 이 저장소의 launch를 사용합니다.

```bash
ros2 lifecycle get /lidar1/os_driver_lidar1
ros2 lifecycle get /lidar2/os_driver_lidar2
ros2 topic info /lidar1/points --verbose
```

두 드라이버의 상태가 `active`인지 확인합니다.

### 6.3 `No NEW LiDAR frame pairs/frames processed` 경고

```bash
ros2 topic hz /lidar1/points
ros2 topic hz /lidar2/points
ros2 topic echo /lidar1/points --once --field header
ros2 topic echo /lidar2/points --once --field header
```

- 한 센서의 프레임이 멈췄는지 확인
- 두 센서와 ROS PC의 시간 기준 확인
- 동일 타임스탬프만 반복되는지 확인
- 서로 다른 PC를 쓴다면 `ROS_DOMAIN_ID`와 DDS 네트워크 설정을 일치

### 6.4 한 센서만 EKF에 반영되거나 융합률이 낮음

```bash
ros2 run drone_localization check_ekf_fusion --output_dir "$RUN_DIR"
```

- `n_clouds=1`: 한 센서에서 프레임 자체가 오지 않음
- `n_clouds=2`, 센서 점 수 0: ROI·배경 제거·DBSCAN 설정 문제
- 스캔 P95가 `async_tolerance_sec`보다 큼: PTP/phase lock을 구성하거나
  실제 시각차를 근거로 허용값 조정
- 두 개별 pose가 멀리 떨어짐: 외부 파라미터 재보정
- 한 센서가 원거리에서 점이 부족함: `cluster_min_points`와 ROI 확인

허용값만 크게 늘려 문제를 숨기기보다 먼저 센서 시간과 캘리브레이션 원인을
해결하십시오.

### 6.5 점군은 나오지만 드론이 검출되지 않음

확인할 토픽:

```bash
ros2 topic hz /lidar1/points_world
ros2 topic hz /lidar1/filtered_points
ros2 topic hz /drone/lidar1_pose
ros2 topic hz /drone/estimated_pose
```

순서대로 확인합니다.

1. 변환된 드론 점이 `roi_*` 안에 있는지
2. `lidarN_x/y/z/roll/pitch/yaw`가 올바른지
3. `use_background:=false`일 때는 검출되는지
4. `background_distance`가 드론까지 제거하지 않는지
5. 실제 드론 점 수가 `cluster_min_points`보다 많은지
6. 클러스터 크기가 `cluster_min_extent`~`cluster_max_extent` 안인지

### 6.6 배경 파일을 찾지 못함

로그 예:

```text
Background file not found: ... Continuing without learned background.
```

확인:

```bash
ls -lh "$HOME/competition_results/calibration/dual_ouster_background.npz"
```

직접 launch할 때는 `background_file:=...`을 명시합니다. GUI는 이 인자를 전달하지
않으므로 `lidar_localization.launch.py`의 기본 경로가 현재 사용자와 일치해야 합니다.

### 6.7 배경 서명이 맞지 않음

로그 예:

```text
Background model calibration signature does not match current extrinsics/ROI.
```

외부 파라미터, ROI 또는 배경 voxel 설정이 바뀐 것입니다. 이전 모델은 사용하지
말고 빈 경기장에서 [3. 정적 배경 학습](#3-정적-배경-학습)을 다시 수행합니다.

### 6.8 RViz에 `Fixed Frame [world] does not exist`가 표시됨

기본 시각화 launch는 `map -> world` 정적 TF를 함께 실행합니다.

```bash
ros2 launch drone_localization estimate_view.launch.py static_tf:=true
```

제공된 RViz 설정의 Fixed Frame은 `world`입니다. 기본 launch가 `map -> world`
정적 TF를 만들어 `world` 프레임을 TF 트리에 등록합니다. 다른 시스템이 동일
TF를 이미 발행한다면 충돌을 막기 위해 `static_tf:=false`를 사용합니다.

### 6.9 GUI가 `STARTING SYSTEM`에서 계속 멈춤

```bash
ros2 node list
```

다음 노드가 모두 보여야 GUI가 `START`를 발행합니다.

```text
/lidar_drone_tracker
/competition_score
```

GUI를 실행한 터미널에서 하위 launch의 오류를 먼저 확인합니다. 한 LiDAR가
응답하지 않으면 기본 드라이버 정책상 측위 launch 전체가 종료되어 START에
실패합니다.

### 6.10 `FINISHING RUN`에서 끝나지 않음

GUI는 다음 파일이 생길 때까지 기다립니다.

```text
~/competition_results/<Team>_<ID>/run_NN/result.json
```

확인:

```bash
ros2 node list | grep competition_score
ls -la "$RUN_DIR"
```

- 채점기 프로세스가 살아 있는지 확인
- 출력 폴더 쓰기 권한 확인
- GUI 터미널의 `Failed to save result.json` 오류 확인
- `/competition/run_config`가 START 전에 전달됐는지 확인

복구가 불가능하면 `ABORT RUN`으로 종료하고 해당 런 폴더의 CSV가 정상적으로
flush됐는지 확인합니다.

### 6.11 점수가 0이거나 대부분의 채점 항목이 비어 있음

- `ABORT`로 끝난 런은 의도적으로 최종 점수가 0
- 미션 제어 노드가 `MARKER_DETECTED`, `HOVER_START/END`, `TRACE_START`,
  `RETURN_HOME` 이벤트를 보냈는지 확인
- 트래커 YAML과 채점기의 마커 좌표가 같은지 확인
- `scoring_trajectory.csv`의 `state`, `marker` 열 확인
- `HOVER_START`와 `HOVER_END` 사이가 기본 4초 이상인지 확인

이벤트 모니터링:

```bash
ros2 topic echo /competition/mission_event
```

### 6.12 `rclpy` 또는 ROS 메시지 import 오류

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source install/setup.bash
which python3
python3 --version
```

Python 3.10이 아닌 Conda 환경이면 시스템 Python으로 다시 빌드합니다.

```bash
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

> [!CAUTION]
> 위 삭제 명령은 워크스페이스 루트에서 실행할 때만 사용하십시오. 소스가 아닌
> `build/`, `install/`, `log/` 생성물만 삭제합니다.

### 6.13 PX4 실제 경로가 기록되지 않음

`flight_recorder` 시작 로그에 `px4_msgs not found`가 보이면 현재 환경에
`px4_msgs`가 없습니다. 패키지를 설치·빌드하고 다음 토픽을 확인합니다.

```bash
ros2 topic hz /fmu/out/vehicle_local_position_v1
```

토픽 이름이 다르면 recorder 실행 시 변경합니다.

```bash
ros2 run drone_localization flight_recorder --ros-args \
  -p actual_topic:=/실제/PX4/토픽 \
  -p output_dir:="$RUN_DIR"
```

### 6.14 `analyze_waypoints`가 `drone_mission`을 찾지 못함

오프라인 분석기는 마커 좌표의 단일 소스로 외부
`drone_mission.generate_groundtruth_trajectory.MARKERS`를 사용합니다.
해당 패키지를 같은 overlay에서 빌드하고 source하거나, 이 도구 대신 GUI 런의
`waypoint_estimates.csv`를 사용하십시오.

### 6.15 Matplotlib 또는 `mpl_toolkits` 충돌

3D 그림이 열리지 않으면 동일 Python 환경에 Matplotlib을 다시 설치합니다.

```bash
/usr/bin/python3 -m pip install --user --force-reinstall matplotlib
```

화면이 없는 서버에서는 `--save-only` 옵션으로 PNG만 생성합니다.

---

## 주요 토픽

| 토픽 | 형식 | 설명 |
| --- | --- | --- |
| `/lidar1/points`, `/lidar2/points` | `sensor_msgs/PointCloud2` | 센서별 원본 점군 |
| `/lidar1/points_world`, `/lidar2/points_world` | `sensor_msgs/PointCloud2` | 월드 좌표 변환 점군 |
| `/lidar1/filtered_points`, `/lidar2/filtered_points` | `sensor_msgs/PointCloud2` | ROI·배경·필터 적용 점군 |
| `/drone/lidar1_pose`, `/drone/lidar2_pose` | `geometry_msgs/PoseWithCovarianceStamped` | 센서별 위치 |
| `/drone/estimated_pose` | `geometry_msgs/PoseWithCovarianceStamped` | EKF 최종 위치 |
| `/lidar/cluster_markers` | `visualization_msgs/MarkerArray` | 클러스터 시각화 |
| `/drone/waypoint_estimates` | `visualization_msgs/MarkerArray` | 경로점 추정 시각화 |
| `/competition/run_config` | `std_msgs/String` JSON | 팀·런·출력 폴더 설정 |
| `/competition/mission_event` | `std_msgs/String` | 경기 상태 이벤트 |

## 저장소에 포함하지 않는 파일

`build/`, `install/`, `log/`, Python 캐시, 비행 기록과 배경 모델은
`.gitignore`에서 제외됩니다. 소스와 파라미터만 Git으로 관리하고, 실제 경기
데이터는 별도 백업 정책을 사용하십시오.
