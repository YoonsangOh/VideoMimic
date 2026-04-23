# VideoMimic Scene/Motion 좌표계와 시뮬레이터별 Ground 설정 정리

## 한 줄 요약

VideoMimic의 stage 2 학습에서는 `z=0` 기본 바닥 plane을 따로 쓰지 않고, 각 clip의 `background_mesh.obj`를 그대로 Isaac Gym의 static triangle mesh terrain으로 넣는다. 따라서 `scene`과 `motion`이 함께 음수 `z` 영역에 있어도, 둘의 좌표계가 일치하기만 하면 학습 자체는 정상적으로 진행될 수 있다.

이 문서는 다음 질문에 답하기 위해 작성했다.

- 왜 어떤 clip은 MuJoCo 등에서 열어보면 scene과 motion이 지하로 들어간 것처럼 보이는가?
- VideoMimic의 Isaac Gym 학습에서는 왜 그 문제가 바로 드러나지 않는가?
- Isaac Gym과 Isaac Sim은 이 장면을 어떻게 다르게 다뤄야 하는가?
- 학습 시 기본 세팅에서 무엇을 건드리면 이 좌표계/ground 문제가 다시 생길 수 있는가?

## 문제의 핵심

Real2Sim 결과물은 clip마다 다음 두 파일로 저장된다.

- `retarget_poses_g1.h5`: G1 reference motion
- `background_mesh.obj`: 해당 clip의 scene mesh

중요한 점은 이 둘이 반드시 `z=0` 위에 있어야 하는 것은 아니라는 점이다. 실제로는 scene 전체가 음수 `z` 영역에 있을 수 있고, reference motion도 같은 좌표계에서 함께 음수 `z`를 가질 수 있다.

대표 예시로 `anthony_apr21_IMG_5585` clip을 직접 확인하면:

- `root_pos[:, 2]` 범위: `-0.4867 ~ 0.0629`
- `link_pos[..., 2]` 범위: `-1.2569 ~ 0.3620`
- `background_mesh.obj` vertex `z` 범위: `-1.3692 ~ -0.6641`

즉 이 clip은 scene 바닥 자체가 전반적으로 `z < 0`에 있다. 하지만 motion도 같은 좌표계에 있으므로, `scene`과 `motion`만 함께 보면 정합성은 유지된다.

문제가 생기는 경우는 주로 외부 툴이 암묵적으로 다음 가정을 할 때다.

- `z=0`에 기본 ground plane이 존재한다.
- imported mesh는 그 ground 위에 있어야 한다.
- motion root도 world ground 기준으로 위에 있어야 한다.

이 경우 원본 데이터가 가진 절대 좌표계를 보존하지 못하고, 장면이 "지하로 내려간 것처럼" 보인다.

## 이 프로젝트에서 `npz`는 무엇인가

이 프로젝트에서 `simulation/data/videomimic_captures_npz/*.npz`는 새로운 의미의 데이터가 아니라, 원래 폴더형 데이터의 단일 파일 번들이다.

- 원본 폴더형 데이터: `retarget_poses_g1.h5` + `background_mesh.obj`
- 번들형 데이터: `<clip>.npz`

변환 스크립트는 `simulation/convert_videomimic_captures_to_npz.py` 이다.

이 스크립트는:

- `h5`에서 `root_pos`, `root_quat`, `joints`, `link_pos`, `link_quat`, `contacts_*` 등을 읽고
- `obj`에서 `mesh_vertices`, `mesh_faces`를 읽어서
- 하나의 `npz`에 저장한다.

즉 좌표계 문제는 `npz` 고유의 문제가 아니라, 원래 `h5 + obj`가 공유하는 절대 좌표계의 문제다.

## VideoMimic Isaac Gym 학습에서는 왜 덜 문제처럼 보이는가

핵심 이유는 간단하다.

- 이 프로젝트의 human-video stage 2 학습은 기본 plane을 쓰지 않는다.
- 대신 `background_mesh.obj`를 simulator의 유일한 terrain collision으로 넣는다.

### 1. 기본 plane 대신 triangle mesh를 쓴다

`create_sim()`은 terrain 설정을 보고 분기한다.

- `mesh_type == 'trimesh'` 이면 `_create_trimesh()`
- 아니면 `_create_ground_plane()`

코드:

- `simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py`

human-video deepmimic terrain 설정은 기본적으로 다음과 같다.

- `mesh_type = 'trimesh'`
- `cast_mesh_to_heightfield = False`

코드:

- `simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py`

따라서 stage 2 human-video 학습에서는 명시적으로 ground plane을 추가하지 않는다.

### 2. scene mesh를 Isaac Gym static collision으로 넣는다

`DeepMimicTerrain`은 각 clip의 `background_mesh.obj`를 `trimesh.load()`로 읽고, 여러 clip을 한 큰 mesh로 concat한다.

그 뒤 Isaac Gym에는 `gym.add_triangle_mesh(...)` 로 넣는다.

즉 이 mesh는:

- 단순 시각화용이 아니라
- 실제 rigid body가 접촉하는 static collision geometry다.

따라서 scene이 음수 `z`에 있더라도, 그 mesh 자체가 곧 terrain이기 때문에 문제 없다.

### 3. motion도 같은 좌표계로 reset/target에 사용한다

`ReplayDataLoader`는 `retarget_poses_g1.h5`를 읽고:

- `root_pos`
- `root_quat`
- `joints`
- `link_pos`
- `link_quat`
- `contacts`

를 메모리에 올린다.

reset 시에는 현재 clip의 시작 frame을 사용해:

- root state를 `root_pos`, `root_quat`로 초기화하고
- joint state를 `dofs`로 초기화한다.

또한 step 중에는 같은 clip의 현재/미래 frame을 reference target으로 계속 사용한다.

즉 scene과 motion이 애초에 같은 절대 좌표계에 묶여 있기 때문에:

- 둘 다 `z=0` 아래에 있어도
- 서로의 상대 관계만 맞으면
- 학습은 정상적으로 진행된다.

## Isaac Gym에서 실제로 쓰는 scene 좌표계 처리

VideoMimic은 clip별로 다른 scene mesh를 한 world에 합쳐 넣는다. 이때 clip이 섞이지 않도록 `env_offsets`를 사용한다.

순서는 다음과 같다.

1. 각 clip의 `background_mesh.obj`를 읽는다.
2. 여러 mesh를 x/y 방향 grid로 배치하며 하나의 큰 triangle mesh로 합친다.
3. 각 env는 자신이 샘플링한 clip index에 따라 해당 scene tile의 offset을 받는다.
4. reset 시 reference root 위치에 이 offset을 더해 world 좌표로 변환한다.

즉 world 좌표계에서는 여러 terrain tile이 옆으로 펼쳐져 있고, 각 env의 로봇은 자기 clip의 tile 위에서만 움직인다.

중요한 점은:

- 이 offset은 주로 x/y 정렬용이다.
- z를 0으로 끌어올리는 정규화가 아니다.

따라서 clip 원본 mesh가 아래쪽에 있으면, 그 tile 전체가 아래쪽에 있는 상태로 유지된다.

## Stage 2 학습의 기본 세팅과 ground 관련 포인트

이 프로젝트에서 자주 쓴 stage 2 스크립트들을 보면 공통적으로 다음 override가 들어간다.

- `--task=g1_deepmimic_proj_heightfield`
- `--env.deepmimic.use_human_videos=True`
- `--env.deepmimic.human_motion_source=...`
- `--env.deepmimic.respawn_z_offset=0.1`
- `--env.terrain.cast_mesh_to_heightfield=False`
- `--env.deepmimic.upsample_data=True`

예:

- `simulation/videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh`
- `simulation/videomimic_gym/legged_gym/scripts/train_stage_2_single_clip_single_gpu_0331.sh`
- `simulation/videomimic_gym/legged_gym/scripts/train_holosoma_stairs_heightfield_single_gpu.sh`

여기서 ground/좌표계와 관련해 중요한 항목은 네 가지다.

### `mesh_type='trimesh'`

기본 plane 대신 scene mesh를 simulator terrain으로 사용한다.

### `cast_mesh_to_heightfield=False`

OBJ를 heightfield로 근사하지 않고, 원래 triangle mesh를 그대로 collision으로 쓴다.

### `respawn_z_offset=0.1`

reset 시 reference root 높이 위에 소량의 추가 z 오프셋을 준다.
이는 feet penetration이나 초기 충돌을 줄이기 위한 완충값이지, world 전체를 `z=0`에 맞추기 위한 정렬이 아니다.

### `randomize_start_offset=True`

clip의 첫 frame이 아니라 clip 내부 임의 frame에서 reset될 수 있다.
즉 “항상 첫 frame의 지면 정합”만 가정하고 설계된 파이프라인은 아니다.

## MuJoCo에서 왜 더 이상하게 보일 수 있는가

MuJoCo 쪽에서 문제가 더 도드라지는 이유는 두 층으로 나눠 볼 수 있다.

### 1. ground 기준의 차이

외부 MuJoCo viewer나 로더가:

- `z=0` ground를 자동으로 깔거나
- mesh와 motion을 world ground 위에 있어야 한다고 가정하면

원래 VideoMimic clip의 절대 좌표계와 충돌한다.

이 경우에는 scene과 motion이 함께 음수 `z`에 있어도, viewer가 "지하"처럼 보이게 만든다.

### 2. non-convex mesh collision의 차이

MuJoCo는 mesh를 렌더링할 수는 있지만, collision은 기본적으로 convex geoms 중심이다.
공식 문서 기준으로, mesh는 collision 시 convex hull로 대체된다.

즉 stairs, sofa, 복잡한 실내 장면 같은 non-convex mesh를 하나의 OBJ로 넣으면:

- visual mesh는 원본처럼 보일 수 있어도
- collision은 원본과 달라진다.

그래서 VideoMimic처럼 복잡한 scene mesh를 그대로 static contact terrain으로 쓰는 용도에는 MuJoCo가 불리하다.

## Isaac Gym과 Isaac Sim의 차이

둘 다 PhysX 계열이지만, 이 문제를 다루는 방식은 다르다.

### Isaac Gym

이 프로젝트가 실제로 쓰는 방식이다.

- Python에서 바로 `vertices`, `triangles`를 만들어
- `gym.add_triangle_mesh(...)` 로 static terrain collision으로 넣는다.
- 구현이 단순하고, VideoMimic 파이프라인과 직접 맞물려 있다.

장점:

- `obj -> trimesh -> vertices/faces -> static terrain` 흐름이 짧다.
- 학습 코드가 이미 이 방식을 전제로 작성되어 있다.
- reference motion과 scene mesh를 같은 좌표계로 유지하기 쉽다.

주의점:

- scene mesh가 world `z=0`보다 아래에 있어도 그대로 들어간다.
- 따라서 외부 툴에서 임의로 z축 기준을 재정렬하면 안 된다.

### Isaac Sim

Isaac Sim은 USD 기반이다.
같은 PhysX 계열이지만, scene을 USD prim + Collision API 방식으로 구성해야 한다.

static scene에 대해서는:

- rigid body 없이 collider만 두는 것이 맞다.
- mesh collider approximation을 적절히 설정해야 한다.

공식 schema상 `UsdPhysicsMeshCollisionAPI`는 approximation으로 다음을 지원한다.

- `none`
- `convexDecomposition`
- `convexHull`
- `meshSimplification`
- 기타 bounding primitive

즉 static mesh scene이라면 원본 mesh collider를 직접 쓰는 방향도 가능하다.
하지만 Isaac Sim 쪽으로 포팅할 때는 다음 실수가 자주 문제를 만든다.

- stage에 기본 ground plane을 추가해 둠
- static mesh 대신 rigid body + 부적절한 collider approximation을 사용함
- import 과정에서 mesh transform이나 up-axis를 바꿔 버림

즉 Isaac Sim에서도 원칙은 같다.

- 기본 plane을 중복으로 넣지 말 것
- scene mesh와 motion의 절대 좌표계를 같이 보존할 것
- static collider 설정을 명시적으로 확인할 것

## 왜 VideoMimic은 이 문제를 "숨기고" 있었는가

엄밀히 말하면 숨긴 것은 아니다.
그냥 파이프라인 자체가 다음 가정을 쓰고 있었기 때문에 문제로 드러나지 않았던 것이다.

- world ground의 기준은 `z=0` plane이 아니라, clip별 scene mesh다.
- reference motion은 그 scene mesh와 같은 절대 좌표계에 있다.
- reset도 reward도 observation도 그 좌표계를 그대로 사용한다.

따라서 외부 viewer가 `z=0` 기준을 강하게 가정하지 않는 한, 학습 자체는 정상이다.

## 실전 체크리스트

다른 시뮬레이터나 viewer로 이 데이터를 열 때는 아래를 먼저 점검하는 것이 안전하다.

1. 기본 ground plane이 자동으로 추가되는가?
2. scene mesh를 static collider로 쓰는가, 아니면 단순 visual mesh인가?
3. motion과 scene에 같은 world transform이 적용되는가?
4. mesh import 시 up-axis, scale, origin이 바뀌지 않았는가?
5. non-convex mesh collision이 원형 유지인지, convex hull 근사인지?

특히 MuJoCo나 자체 viewer에서 결과가 이상하면, 가장 먼저 확인할 것은 다음 두 가지다.

- `z=0` 기본 plane을 넣었는지
- mesh와 motion을 같은 transform으로 가져왔는지

## 결론

- VideoMimic의 human-video stage 2는 `z=0` ground plane 위에서 학습하는 구조가 아니다.
- clip의 `background_mesh.obj` 자체가 terrain이며, Isaac Gym에 static triangle mesh collision으로 들어간다.
- reference motion도 같은 절대 좌표계의 `h5`를 그대로 사용하므로, scene과 motion이 함께 음수 `z`에 있어도 학습은 가능하다.
- 외부 시각화 툴에서 이상해 보이는 것은 대개:
  - 기본 plane 추가
  - 좌표계 재정렬
  - non-convex mesh collision 근사
  중 하나 때문이다.
- Isaac Sim으로 옮길 때도 같은 원칙이 적용되지만, USD collider 설정과 기본 stage ground를 더 조심해야 한다.

## 코드 경로

- scene/motion 경로 수집:
  - `simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic.py`
- motion 로딩:
  - `simulation/videomimic_gym/legged_gym/tensor_utils/replay_data.py`
- terrain mesh 로딩/concat:
  - `simulation/videomimic_gym/legged_gym/utils/deepmimic_terrain.py`
- Isaac Gym triangle mesh 추가:
  - `simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py`
- reset 시 reference root/dof 세팅:
  - `simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py`
- stage 2 학습 스크립트:
  - `simulation/videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh`

## 외부 참고 자료

- Isaac Sim / Omniverse mesh collision:
  - https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/_apidocs/classusdrt_1_1UsdPhysicsMeshCollisionAPI.html
  - https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/latest/capabilities/physics_bodies/physics_rigid_bodies/capability-physics_rigid_bodies.html
  - https://docs.omniverse.nvidia.com/kit/docs/omni_physics/106.5/dev_guide/rigid_bodies_articulations/collision.html
- MuJoCo mesh collision:
  - https://mujoco.readthedocs.io/en/latest/computation/
  - https://mujoco.readthedocs.io/en/3.2.6/XMLreference.html
