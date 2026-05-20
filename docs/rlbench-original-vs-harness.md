# RLBench: Original vs vla-evaluation-harness — Differences for VLA Evaluation

> **Status:** comparison + how-to reference for using **original RLBench** to
> evaluate VLAs while borrowing harness know-how. Companion to
> [`rlbench-enablement-plan.md`](./rlbench-enablement-plan.md) (which proposes
> fixes inside the harness). This document treats the harness integration as a
> *reference implementation* — what it gets right, what it removes, what to
> mirror, and what to fix if you go native.
> **Sources compared:**
> - Original: `/home/theo_lab/RLBench/` (master HEAD `02720bba`, RLBench `1.2.0`,
>   stepjam/RLBench, single-arm, 106 task `.py` files).
> - Harness: `/home/theo_lab/vla-evaluation-harness/` (the `RLBenchBenchmark`
>   class + Docker image, current `main`).

---

## 0. 한국어 요약 (TL;DR)

원본 RLBench는 **거의 그대로 VLA 평가에 쓸 수 있다.** harness가 추가하는 가장
유용한 것은 (1) Docker 패키징 + license gate, (2) headless Xvfb + OpenGL3
segfault 우회, (3) 통합 runner 추상화 — 단 그뿐이고, **그 외의 거의 모든 축에서
원본보다 기능을 *제거*했다**:

- camera 5개 중 3개 disable, depth/pointcloud/mask 모두 off, proprio는 sim에
  켜둔 채 `make_obs()`에서 안 forward
- 21 PerAct 18-task 표준 대신 5개 무작위 default + 1-task stub eval.yaml
- 변이 `sample_variation()` (random, 비결정성) → PerAct 표준의
  `set_variation(ep_idx % N)` 부재
- action mode를 `JointVelocity()`로 하드코딩 → 현존 VLA 7D EE 출력과 비호환
- 언어 설명은 variation당 3~6 phrasings 다양성이 있는데도 항상 `[0]`만 사용
- demo/dataset 파이프라인 (`dataset_generator.py`) 완전 미사용 — PerAct
  *학습*에 100 demos/task 필요 (이미 사전학습된 VLA를 *평가*만 할 거면 무관)

**그리고 잠재 버그도 하나 발견** — harness의 `benchmark.py:74-80`이
`CameraConfig(render_resolution=...)` 키워드를 쓰는데 원본 `CameraConfig`는
`image_size`만 받음 (`observation_config.py:13`). 현재 GitHub HEAD 기준으로는
`_ensure_env()`에서 `TypeError`. (issue #44가 "Likely works"로만 표기한 이유와
정합.)

**결론:** 원본을 native로 쓰면서 harness는 "쇼핑 리스트"로 참고. 아래 §5 레시피
하나로 PerAct-호환 평가 셋업 가능.

---

## 1. What this document is for

If you want to evaluate a VLA on RLBench and the harness Docker setup is
inconvenient (different GPU/distro, custom training pipeline, you already have
demos collected, you want depth/pointclouds, etc.), original `stepjam/RLBench`
gives you everything you need natively. This document tells you, layer by layer:

- which harness choice you should **keep** when going native (good defaults)
- which harness choice you should **drop or invert** (overconstrained or wrong)
- which native features the harness silently disables (recover them)
- one minimal copy-paste recipe (§5) to run a VLA on original RLBench

For the orthogonal direction — fixing the harness itself — see
[`rlbench-enablement-plan.md`](./rlbench-enablement-plan.md).

---

## 2. Same vs different at a glance

| Axis | Original RLBench (`02720bba`) | Harness `RLBenchBenchmark` | What VLA users want |
|---|---|---|---|
| Arm action mode | All 6: JointVelocity / JointPosition / JointTorque / EndEffectorPoseViaIK / EndEffectorPoseViaPlanning / ERJointViaIK | Hardcoded `JointVelocity()` | `EndEffectorPoseViaIK(absolute_mode=False, frame=RelativeFrame.EE)` |
| Gripper mode | `Discrete` (1=open/0=closed) or `GripperJointPosition` (continuous) | `Discrete()` | `Discrete()` (matches `binary_close_*` VLAs) |
| Cameras enabled | 5: left/right_shoulder, wrist, overhead, front | front + wrist only | depends; π₀-style = front+wrist; PerAct/RVT = all 4 shoulder/front |
| Sensor modalities/camera | RGB+depth+point_cloud+mask all on, 128² | RGB only, 256² (and uses a **broken kwarg name**, §4.1) | RGB always; depth/PCD for 3D-aware models |
| Proprio | joint_positions/velocities/forces, gripper_open/pose/matrix/joint, task_low_dim_state — all available | sim-enabled but `make_obs()` discards them | depends; most state-conditioned VLAs need `joint_positions` + `gripper_open` + `gripper_pose` |
| Task count | 106 runnable `.py` | 5 default, 1 in `eval.yaml`, 18 listed in leaderboard | PerAct-18 standard |
| Variation selection | `set_variation(i)` (deterministic) or `sample_variation()` (random) | only `sample_variation()` → non-reproducible | `set_variation(ep_idx % variation_count())` + seeded `np.random` |
| Language descriptions | `List[str]`, 1–6 phrasings/variation | only `descriptions[0]` | sample uniformly (PerAct) or fix deterministically |
| Reward | `float(success)` sparse, or shaped via `Task.reward()` if `shaped_rewards=True` | sparse only; `success = reward > 0.99` | sparse is fine |
| Episode timeout | none built-in | adds `max_steps` (default 200) | needed; 200–500 typical |
| Robot embodiments | 5: `panda`/`jaco`/`mico`/`sawyer`/`ur5` | `panda` only (default) | `panda` (every published RLBench VLA uses Panda) |
| Headless docs | "real X server" (`X :99` + nvidia-xconfig) | Xvfb + `DISPLAY=:99` + OpenGL3 disabled | Xvfb path generally easier; **OpenGL3 workaround required** |
| Demos / dataset | `dataset_generator.py`, `get_demos()`, `Demo`, `reset_to_demo` | none of it | yes, PerAct uses 100 demos/task |
| Extras | `gym.py` wrapper, `sim2real` DR, `tools/{task_builder,task_validator,cinematic_recorder}` | none wired | Gym + DR are genuinely useful |
| Bimanual / RLBench2 | not in this clone | not supported | requires a separate fork |
| Install | unpinned (no `python_requires`), README assumes CoppeliaSim 4.1.0 | Python 3.8 conda, CoppeliaSim 4.1.0 Ubuntu20.04, PyRep+RLBench `--depth 1` unpinned | pin everything (commit-pin PyRep & RLBench) |
| PerAct protocol | natively supported (all primitives exist) | every primitive missing or hardcoded wrong | follow §5 |

The pattern is consistent: **original is full-featured but unopinionated; harness is
opinionated to a small smoke-test slice**. For VLA work, you want the
full-featured original with a *small* set of opinionated choices (the ones in §5).

---

## 3. Layer-by-layer comparison

### 3.1 Action modes

**Original** (`/home/theo_lab/RLBench/rlbench/action_modes/`):

| Class | shape | semantics | absolute/delta |
|---|---|---|---|
| `JointVelocity` | `(7,)` | sets joint target velocities for one sim tick, then zeroes them | velocity (inherently delta-like) |
| `JointPosition(absolute_mode=True)` | `(7,)` | sets joint target positions; delta mode adds to current | both |
| `JointTorque` | `(7,)` | sets max velocity ±9999, constrains forces; raw torque | — |
| `EndEffectorPoseViaIK(absolute_mode=True, frame=WORLD, collision_checking=False)` | `(7,)` `[x,y,z,qx,qy,qz,qw]` (unit quat) | Jacobian IK, target must be near current | both, frame-aware |
| `EndEffectorPoseViaPlanning(...)` | `(7,)` same | plans full trajectory (linear IK → RRTConnect fallback) | both, frame-aware |
| `ERJointViaIK(commanded_joints=[0])` | `(7+k,)` | IK with `k` constrained joints | both |

Gripper modes:

| Class | shape | semantics |
|---|---|---|
| `Discrete(attach_grasped_objects=True, detach_before_open=True)` | `(1,)` | thresholded at `0.5` → `1.0=open` or `0.0=closed`; hysteresis on open; grasp/release physics |
| `GripperJointPosition(absolute_mode=True)` | `(1,)` | continuous in `[0, 0.04]` m, repeated to both fingers |

`MoveArmThenGripper(arm, gripper)` concatenates: total `action_shape =
arm.action_shape + gripper.action_shape`.

**Harness:** hardcodes `MoveArmThenGripper(JointVelocity(), Discrete())` →
**8D = 7 joint-velocity + 1 discrete gripper**. Asserts shape 8 in `step()`.

**Critical subtlety for the EE mode** (already in the enablement plan):
`EndEffectorPoseViaIK(absolute_mode=False)` with the **default `frame=WORLD`**
calls `calculate_delta_pose(robot, action)` which composes the rotation as
`Quaternion(delta) * Quaternion(current_world)` — a **world-frame left
multiply**. VLA per-step rotation deltas (OpenVLA / π₀ / OFT robosuite OSC
convention) are **EE-frame increments**. Use **`frame=RelativeFrame.EE`** — it
skips `calculate_delta_pose` and solves IK relative to the tip, which is
geometrically correct for EE-frame deltas. With WORLD frame you get
systematically wrong orientation from step 1. (`reach_target` is
position-only and cannot detect this; validate on a grasp/orientation task —
`turn_tap`, `pick_up_cup` — instead.)

### 3.2 Observations & sensors

Original `ObservationConfig` (`observation_config.py:35-108`) defaults are
**maximalist**: all 5 cameras enabled with `CameraConfig()` defaults
(RGB+depth+point_cloud+mask, **128×128**, `RenderMode.OPENGL3`), and every
proprio field (`joint_velocities/positions/forces=True`,
`gripper_open/pose=True`, `gripper_matrix/joint_positions/touch_forces=False`,
`task_low_dim_state=False`).

The full per-step `Observation` exposes **20 image attributes** (5 cameras ×
{rgb, depth, mask, point_cloud}) + **9 low-dim attrs** + `misc` (camera
intrinsics/extrinsics + variation_index). `Observation.get_low_dim_data()`
concatenates all enabled low-dim into a flat vector.

**Harness** (`benchmark.py:74-92`):

```python
cam = CameraConfig(
    rgb=True, depth=False, point_cloud=False, mask=False,
    render_resolution=(256, 256),    # ← BROKEN KWARG NAME, see §4.1
)
cam_off = CameraConfig(); cam_off.set_all(False)
obs_cfg = ObservationConfig(
    front_camera=cam, wrist_camera=cam,
    left_shoulder_camera=cam_off, right_shoulder_camera=cam_off,
    overhead_camera=cam_off,
    joint_positions=True, gripper_open=True,
)
```

And `make_obs()` (`:137-148`) returns only `{"images": {"front":..., "wrist":...},
"task_description": ...}` — **proprio is enabled in sim but never forwarded**.

**Recovery checklist for native use** (which modalities each VLA family needs):

| VLA family | Cameras | Modalities | Proprio |
|---|---|---|---|
| OpenVLA / RT-2 | front (224²) | RGB | none |
| π₀ / Octo / GR-1 | front + wrist (256²) | RGB | `joint_positions`, `gripper_open`, `gripper_pose` |
| PerAct / RVT (multi-view) | all 4 (front + 2 shoulder + wrist) (128²) | RGB | `joint_positions`, `gripper_open` |
| 3D-aware (Act3D, SPA) | front + shoulder + wrist (256²) | RGB + depth + point_cloud | `joint_positions`, `gripper_pose` |

Concrete original-API snippet for an OpenVLA setup (single front 224² RGB):

```python
cam = CameraConfig(rgb=True, depth=False, point_cloud=False, mask=False,
                   image_size=(224, 224))   # note: image_size, NOT render_resolution
cam_off = CameraConfig(); cam_off.set_all(False)
obs_cfg = ObservationConfig(
    front_camera=cam,
    left_shoulder_camera=cam_off, right_shoulder_camera=cam_off,
    overhead_camera=cam_off, wrist_camera=cam_off,
    joint_velocities=False, joint_forces=False,
    gripper_open=True, gripper_pose=False,
)
```

### 3.3 Task catalog

Original task directory contains **106 runnable `.py` files** (each
auto-discovered via `rlbench.utils.name_to_task_class`).

Harness `DEFAULT_TASKS` (5): `reach_target`, `pick_up_cup`, `push_button`,
`close_drawer`, `open_door`. **Zero overlap with the PerAct-18 subset.** And
`eval.yaml` overrides further to just `reach_target × 1 episode`.

PerAct-18 (the de-facto VLA standard) — all 18 `.py` files **confirmed present**
in the local clone. Leaderboard short name → actual file:

| Leaderboard | `.py` | variations |
|---|---|---|
| `close_jar` | `close_jar.py` | 20 |
| `drag_stick` | `reach_and_drag.py` | 20 |
| `insert_peg` | `insert_onto_square_peg.py` | 20 |
| `meat_off_grill` | `meat_off_grill.py` | 2 |
| `open_drawer` | `open_drawer.py` | 3 |
| `place_cups` | `place_cups.py` | 3 |
| `place_wine` | `stack_wine.py` | 1 |
| `push_buttons` | `push_buttons.py` | 50 |
| `put_in_cupboard` | `put_groceries_in_cupboard.py` | 9 |
| `put_in_drawer` | `put_item_in_drawer.py` | 3 |
| `put_in_safe` | `put_money_in_safe.py` | 3 |
| `screw_bulb` | `light_bulb_in.py` | 20 |
| `slide_block` | `slide_block_to_target.py` | 1 |
| `sort_shape` | `place_shape_in_shape_sorter.py` | 5 |
| `stack_blocks` | `stack_blocks.py` | 60 |
| `stack_cups` | `stack_cups.py` | 20 |
| `sweep_to_dustpan` | `sweep_to_dustpan.py` | 1 |
| `turn_tap` | `turn_tap.py` | 2 |
| | **total** | **243** |

The leaderboard description says "249 total variations" — likely a paper-version
discrepancy (per-task caps differ slightly between RLBench tags); 243 is the
HEAD truth.

83 tasks are unused by both harness and PerAct subset (open/close articulated
objects, kitchen/household, long-horizon `set_the_table`/`solve_puzzle`/
`empty_dishwasher`, precision-tool `screw_nail`/`pour_from_cup_to_cup`, etc.).

**243 (HEAD) vs 249 (leaderboard/PerAct paper):** the leaderboard summary cites
"249 total variations" but HEAD sums to 243. The 6-variation gap is concentrated
in a few tasks (`meat_off_grill`, `place_cups`, `light_bulb_in` are the usual
suspects — exact distribution depends on the RLBench fork the paper was
evaluated against; PerAct-era papers used `MohitShridhar/RLBench@peract`, which
has slightly higher per-task caps than `stepjam/RLBench` HEAD). For
*paper-faithful* reproduction, install from the PerAct fork; for evaluating
*your own VLA* without paper-matching, HEAD is fine but expect ≲2–3 pp of
variation-mix drift versus published numbers.

### 3.4 Variations, seeding, determinism

Original `TaskEnvironment` API (`task_environment.py`):

```python
def sample_variation(self) -> int:                   # RANDOM
    self._variation_number = np.random.randint(0, self._task.variation_count())
def set_variation(self, v: int) -> None:             # DETERMINISTIC
    if v >= self.variation_count(): raise TaskEnvironmentError(...)
    self._variation_number = v
def variation_count(self) -> int:
    return self._task.variation_count()
```

`Environment` has **no global seed** parameter anywhere. Object placement inside
`Scene.init_episode()` calls `SpawnBoundary.sample()`, which uses NumPy's
**global RNG** — so to make placement reproducible you must `np.random.seed(...)`
externally *before* `reset()`.

Harness uses **only** `sample_variation()` (random) and has no `seed`/`episode_idx`
parameter on `RLBenchBenchmark.__init__`. Two harness runs with the same model
will see different variations and different object placements.

PerAct convention (and what every leaderboard `notes` entry implies):

```python
task_env.set_variation(episode_idx % task_env.variation_count())
np.random.seed(episode_idx)                          # MUST come before reset()
descs, obs = task_env.reset()
```

### 3.5 Episode lifecycle — `step()`, `reward`, `terminate`, max_steps

Original `TaskEnvironment.step(action) -> (Observation, float, bool)`:

```python
self._action_mode.action(self._scene, action)
success, terminate = self._task.success()
reward = float(success)                              # sparse 1.0 or 0.0
if self._shaped_rewards:
    reward = self._task.reward()                     # task-specific, may be None → RuntimeError
return self._scene.get_observation(), reward, terminate
```

`Task.success() -> (bool, bool)`: returns `(success, terminate)`. `terminate=True`
fires immediately when all `_success_conditions` are met, OR all `_fail_conditions`
are met. Most tasks only override `init_task()` (to register conditions); only
`reach_target` overrides `reward()` for dense reward.

Exceptions raised mid-step (NOT caught inside `TaskEnvironment.step()`):
`InvalidActionError`, `IKError`, `BoundaryError`, `WaypointError`. Action-mode
specific wrappers re-raise PyRep IK failures as `InvalidActionError`.

**There is no built-in episode timeout.** `Environment.__init__` has no
`episode_length`. `TaskEnvironment` has no step counter. You must enforce
max-steps in your runner.

Harness adds `max_steps` (default 200, orchestrator falls back to 300) and uses
`success = reward > 0.99` (a float-noise guard equivalent to `== 1.0`). This is
correct for sparse rewards and matches PerAct convention.

### 3.6 Robot & scene

Original `Environment.__init__` accepts 14 kwargs; key ones for VLAs:

| kwarg | default | meaning |
|---|---|---|
| `headless` | `False` | suppresses CoppeliaSim GUI (still needs `DISPLAY`) |
| `robot_setup` | `'panda'` | one of `'panda' / 'jaco' / 'mico' / 'sawyer' / 'ur5'` |
| `static_positions` | `False` | freeze object init poses |
| `attach_grasped_objects` | `True` | kinematic-attach on grasp (recommended) |
| `shaped_rewards` | `False` | switch to `Task.reward()` |
| `arm_max_velocity / arm_max_acceleration` | `1.0 / 4.0` | motion limits |
| `randomize_every / visual_randomization_config / dynamics_randomization_config` | `None` | sim2real DR (see §3.10) |

`Supported robots` (`const.py:38-44`): Panda(7-DOF/PandaGripper),
Sawyer(7/Baxter), Jaco(6), Mico(6), UR5(6/Robotiq85). Their CoppeliaSim models
live in `rlbench/robot_ttms/{panda,jaco,mico,sawyer,ur5}.ttm`.

Every published RLBench VLA (PerAct, RVT, RVT-2, Act3D, GR-1, 3D Diffuser
Actor, π₀ on RLBench…) uses **Panda exclusively**. The harness's Panda-only
default is correct for all current VLA work.

Base scene = `task_design.ttt`; per-task assets = `task_ttms/*.ttm` (108 files).

### 3.7 Headless rendering

Original `README.md:64-137` "Running Headless" prescribes a **real X server
backed by a GPU**:

```bash
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
sudo nohup X :99 & disown
export DISPLAY=:99
```

Then `Environment(..., headless=True)` (which only hides the CoppeliaSim GUI,
does not switch the rendering backend). No mention of EGL, Xvfb, or
`QT_QPA_PLATFORM=offscreen` anywhere in the original codebase.

Harness uses a different (and frankly more portable) approach in
`Dockerfile.rlbench` + `rlbench_entrypoint.sh`:

1. `xvfb` + `tini` installed.
2. Entrypoint: `Xvfb :99 -screen 0 1280x1024x24 +extension GLX +render -noreset &;
   export DISPLAY=:99`.
3. Dockerfile sets `QT_QPA_PLATFORM=offscreen` as a base ENV, but
   `DISPLAY=:99` at entrypoint causes Qt to pick the `xcb` plugin (the
   `benchmark.py:18-19` comment documents this: "Do NOT set
   `QT_QPA_PLATFORM=offscreen` — CoppeliaSim needs xcb+Xvfb").
4. **Disables `libsimExtOpenGL3Renderer.so`** via `mv → .disabled` — the
   bundled Qt under Xvfb segfaults in that plugin; falling back to CoppeliaSim's
   built-in renderer works. This is **not in the original README**, but it
   matters for anyone running CoppeliaSim under Xvfb on bare metal.

`tini` PID-1 is required to reap CoppeliaSim child processes.

### 3.8 Language descriptions

Every `Task.init_episode(index) -> List[str]` returns **multiple natural-language
phrasings** of the same variation. Examples:

```python
# reach_target.py — 20 colors × 3 phrasings
return ['reach the %s target' % c,
        'touch the %s ball with the panda gripper' % c,
        'reach the %s sphere' % c]

# stack_blocks.py — 60 variations × 6 phrasings
return ['stack %d %s blocks' % (n, c),
        'place %d of the %s cubes on top of each other' % (n, c),
        # ... 4 more
       ]
```

Harness stores all of these from `reset()` but **always uses `_descriptions[0]`**
when building the observation (`benchmark.py:144`). For `stack_blocks` this
discards 5 of 6 phrasings; for color-tasks 2 of 3.

PerAct's convention is to **sample uniformly at random** from `descriptions` per
episode (mirrors the language diversity each VLA saw during training). For
reproducibility, seed the choice on `episode_idx` or fix it deterministically.

### 3.9 Demonstrations & datasets

Original has a complete demo/dataset pipeline that the harness ignores entirely:

- `rlbench/dataset_generator.py` — multiprocessing CLI. `--episodes_per_task`,
  `--variations`, `--image_size`, `--processes`. Writes per-episode
  `low_dim_obs.pkl` (a `Demo`) + per-step `{cam}_{rgb,depth,mask}/<i>.png`
  per camera + `variation_descriptions.pkl` (the language list).
- `TaskEnvironment.get_demos(amount, live_demos, image_paths, ...)` — collect
  live (in-sim scripted waypoint replay) or load from `dataset_root`.
- `Demo` (`rlbench/demo.py`) — list of `Observation`; carries `random_seed` for
  reproducible replay; `restore_state()` puts NumPy RNG back to that state.
- `TaskEnvironment.reset_to_demo(demo)` — restore exact scene state from a saved
  demo.

Language is attached **at the variation level**, not per timestep — `init_episode`
returns the description list once, and `dataset_generator` pickles it next to the
episodes. PerAct then samples from that list at training time.

For VLA training/fine-tuning on RLBench the typical pipeline is:

```bash
python -m rlbench.dataset_generator \
    --tasks close_jar drag_stick ... (the 18) \
    --episodes_per_task 100 --variations -1 \
    --save_path /data/rlbench/peract \
    --processes 8
```

Harness contributes nothing here; this is purely an original-RLBench workflow.

### 3.10 Extras (Gym, sim2real, tools)

| Asset | Path | What |
|---|---|---|
| Gym wrapper | `rlbench/gym.py` | `RLBenchEnv(gym.Env)`; auto-registers `rlbench/<task>-{state\|vision}-v0`; supports `gym.make_vec(..., vectorization_mode="async")` for parallel envs |
| sim2real DR | `rlbench/sim2real/{domain_randomization, domain_randomization_scene}.py` | `RandomizeEvery`, `VisualRandomizationConfig` (random textures from PNG pool), `DynamicsRandomizationConfig` (stub) |
| Task builder | `tools/task_builder.py` | Interactive REPL for editing tasks in a live CoppeliaSim scene |
| Task validator | `tools/task_validator.py` | Smoke-runs N demos / task, checks success ≥ 50%; used in CI |
| Cinematic recorder | `tools/cinematic_recorder.py` | MP4 demo recordings with free-flying cam |
| Few-shot splits | `examples/few_shot_rl.py` | `FS10_V1['train'/'test']` task split |
| Multi-task splits | `examples/multi_task_rl.py` | `MT30_V1['train']` |
| Robot swap | `examples/swap_arm.py` | `Environment(robot_setup='sawyer')` |
| RLBench2 / bimanual | — | **NOT in this clone** (no `dual_*` tasks, no bimanual action modes; would need a separate fork) |

For VLA users: the **Gym wrapper** (esp. `make_vec`) and **`dataset_generator`**
are the biggest wins; both are zero-config to use natively, and both are absent
from the harness path.

### 3.11 Install / runtime stack

Original `setup.py` `install_requires`:

```python
"pyrep @ git+https://github.com/stepjam/PyRep.git",   # unpinned VCS
"numpy", "Pillow", "pyquaternion", "scipy", "natsort"
```

No `python_requires`. `extras_require = {"gym": ["gymnasium==1.0.0a2"], "dev": ["pytest"]}`.

Original README install (verbatim, except whitespace):

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
    -C $COPPELIASIM_ROOT --strip-components 1
pip install git+https://github.com/stepjam/RLBench.git
```

Three divergences in the harness Dockerfile that you should adopt going native:

1. `QT_QPA_PLATFORM_PLUGIN_PATH` → `$COPPELIASIM_ROOT/platforms` (subdir, not
   root). The README's value is technically the wrong directory; Qt's plugin
   loader expects `/platforms/libqxcb.so` etc.
2. `mv $COPPELIASIM_ROOT/libsimExtOpenGL3Renderer.so → .disabled` (segfault
   workaround; needed under Xvfb).
3. Python **3.8** explicitly (the harness pins this; original `setup.py` doesn't
   pin a floor but PyRep's `ctypes` glue + `gymnasium==1.0.0a2` extra make 3.8
   the only verified version).

PyRep and RLBench are both **unpinned VCS clones** in both original and harness.
For reproducibility, capture commits at install:

```bash
git clone https://github.com/stepjam/PyRep && cd PyRep && git rev-parse HEAD
git clone https://github.com/stepjam/RLBench && cd RLBench && git rev-parse HEAD
```

### 3.12 PerAct protocol support

The PerAct protocol (the de-facto VLA benchmark on RLBench) is:

- **18 tasks** (§3.3) × **25 eval episodes** = 450 episodes total
- **100 training demos / task** (multi-task learning, one policy for all 18)
- variations cycled deterministically: `set_variation(ep_idx % variation_count())`
- score = arithmetic mean of per-task success rate

Original RLBench supports **every primitive** natively (see §3.4, §3.9).

> **Training vs evaluation — disambiguation.** "100 demos/task" is for
> *training/fine-tuning* a model from scratch on RLBench data (the PerAct
> training recipe). If you already have a pretrained VLA checkpoint and just
> want to *evaluate* it, you do NOT need to run `dataset_generator.py` — jump
> straight to §5. Generate demos only when (a) fine-tuning your model on
> RLBench, or (b) replaying scripted trajectories for debugging.

Harness has all four PerAct primitives broken:

1. wrong action mode (JointVelocity, §3.1)
2. random `sample_variation()` (§3.4)
3. no demo generation (§3.9)
4. `eval.yaml` is 1-task × 1-episode stub (§3.3)

Zero of the 136 leaderboard `rlbench` entries are harness-measured; all are
`reported_paper` references curated via the leaderboard pipeline. Issue #44
flags RLBench as "Likely works" — never end-to-end confirmed.

---

## 4. Differences that bite (callouts)

### 4.1 Latent kwarg bug in harness benchmark.py

`/home/theo_lab/vla-evaluation-harness/src/vla_eval/benchmarks/rlbench/benchmark.py:74-80`
passes `render_resolution=(...)` to `CameraConfig(...)`. But original
`CameraConfig.__init__` (`observation_config.py:6-16`, verified at current
master HEAD `02720bba` of `stepjam/RLBench`) accepts **`image_size=(128,128)`,
not `render_resolution`**. There is no `render_resolution` kwarg.

Consequence: `_ensure_env()` raises `TypeError: __init__() got an unexpected
keyword argument 'render_resolution'` the moment the first episode is reset.

This is consistent with:
- Issue #44 listing RLBench as "Likely works" (never confirmed end-to-end).
- Zero harness-measured RLBench entries in the leaderboard.
- The harness rename in commit `f0e7994` (PR #25): the `RLBenchBenchmark.__init__`
  parameter was renamed `image_size → render_resolution`, but the kwarg passed
  through to the library `CameraConfig` was renamed too — and that downstream
  rename was wrong (the library still uses `image_size`).

Fix: either pass `image_size=(self._render_resolution, self._render_resolution)`
inside `_ensure_env`, or rename the param back. Same one-character fix either way.

If you go native, just use `image_size=` as `CameraConfig` defines it.

### 4.2 JointVelocity action mode is incompatible with delta-EE VLAs

Every modern VLA model server in the harness (`openvla`, `pi0`, `oft`, `groot`,
`xvla`, `vlanext`, `cogact`) emits a flat 7D delta-EE action under `{"actions":
...}`. The harness's `JointVelocity` mode + `assert act.shape[-1] == 8` will
crash on shape mismatch, or (worse) interpret the 7 numbers as joint velocities
and produce physically meaningless motion. See `rlbench-enablement-plan.md`
§1.4–1.5 + §3 for the full fix.

For native use: pick `EndEffectorPoseViaIK(absolute_mode=False,
frame=RelativeFrame.EE)` + `Discrete()`. See §3.1 above and the minimum recipe
in §5.

### 4.3 `sample_variation()` makes scores non-reproducible

Two harness runs ≠ same variation, ≠ same object placement, ≠ comparable
numbers. For a published-score reproduction, you must use
`set_variation(ep_idx % var_count)` + `np.random.seed(ep_idx)` before
`reset()`. (§3.4)

### 4.4 Language diversity collapsed to index 0

Original tasks return 1–6 phrasings per variation. Harness's
`self._descriptions[0]` always picks the first. For VLAs trained on the natural
diversity, this can systematically bias scores. Sample at random (PerAct) or
fix deterministically per `(variation_index, episode_idx)` — but **do not**
always pick `[0]`. (§3.8)

### 4.5 OpenGL3 plugin segfault under Xvfb (not in original docs)

The original README's headless recipe assumes a real GPU-backed X server. If
you're on a cloud VM / cluster with Xvfb instead, you **will** hit a silent
segfault inside `libsimExtOpenGL3Renderer.so` at `pyrep.launch()`. The harness's
`mv → .disabled` workaround is mandatory for that environment. (§3.7)

### 4.6 No episode timeout in original

`Environment.__init__` and `TaskEnvironment.step()` have no step counter — your
runner must enforce it. The harness's default 200 is a reasonable starting
point; PerAct papers typically use higher (300–500 for long-horizon tasks like
`stack_blocks`, `put_in_drawer`). (§3.5)

### 4.7 Proprio enabled but discarded

`obs_cfg.joint_positions=True` and `gripper_open=True` are set but
`make_obs()` never reads them. If your VLA is state-conditioned (π₀, OFT,
GR-1, X-VLA), you need to forward these explicitly. (§3.2)

---

## 5. Minimum recipe: evaluate a VLA on original RLBench

Self-contained — does **not** depend on the harness. Run on Ubuntu 20.04, Python
3.8, GPU optional (Xvfb path is CPU-renderable).

```bash
# 1. System deps
sudo apt-get install -y xvfb libfontconfig1
conda create -y -n rlbench python=3.8 pip && conda activate rlbench

# 2. CoppeliaSim 4.1.0 Edu — exact version
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
mkdir -p ${COPPELIASIM_ROOT}
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C ${COPPELIASIM_ROOT} --strip-components=1
rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# 3. OpenGL3 segfault workaround — CONDITIONAL.
#    Apply ONLY when running under Xvfb without a real GPU-backed X server
#    (cloud VM / headless cluster). On a workstation with a real $DISPLAY backed
#    by a GPU, KEEP OpenGL3 enabled — it's the higher-quality renderer and the
#    "fallback" renderer is CPU-software (slower, worse depth quality).
#    Symptom of needing the workaround: SIGSEGV inside libQt5Gui or
#    libsimExtOpenGL3Renderer at pyrep.launch().
mv ${COPPELIASIM_ROOT}/libsimExtOpenGL3Renderer.so \
   ${COPPELIASIM_ROOT}/libsimExtOpenGL3Renderer.so.disabled
# If you keep OpenGL3 enabled, also set render_mode=RenderMode.OPENGL explicitly
# in CameraConfig (see Python recipe) — defensive against stale plugin dlopen.

# 4. Env vars (persist in ~/.bashrc or conda activate.d)
export LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}
export QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}/platforms

# 5. PyRep + RLBench (capture commits for reproducibility)
git clone https://github.com/stepjam/PyRep && cd PyRep
PYREP_COMMIT=$(git rev-parse HEAD) && pip install -e . && cd ..
git clone https://github.com/stepjam/RLBench && cd RLBench
RLBENCH_COMMIT=$(git rev-parse HEAD) && pip install -e . && cd ..
echo "PyRep=$PYREP_COMMIT  RLBench=$RLBENCH_COMMIT"  # record these
pip install scipy gymnasium  # gymnasium only if you want the Gym wrapper

# 6. Headless display
Xvfb :99 -screen 0 1280x1024x24 +extension GLX +render -noreset &
export DISPLAY=:99
# Do NOT set QT_QPA_PLATFORM=offscreen here — Xvfb wants xcb
```

### Per-model configuration (read this BEFORE running the recipe)

The recipe below is **OpenVLA-tuned by default**. For other VLAs you must change
at least the `image_size`, `states` composition, `POS_SCALE`, and gripper
mapping. The wire conventions differ per model — silent score degradation if
you don't adjust:

| Model family | Cameras | `image_size` | `states` content | `POS_SCALE` start | Gripper at wire | Action chunk |
|---|---|---|---|---|---|---|
| **OpenVLA / OpenVLA-OFT** | front | `(224, 224)` | (model doesn't consume state — omit) | `1.0` | `binary_close_positive` (`+1=close`) → **invert** for RLBench Discrete | 1 (per-step) |
| **OFT (standalone)** | front + wrist | `(256, 256)` | `[joint_positions(7), gripper_open(1)]` — **NOT `gripper_pose`** | `1.0` | `binary_close_positive` → invert | 8 |
| **π₀ / π₀-FAST** | front + wrist | `(224, 224)` | `[eef_pos(3), eef_quat_xyzw(4), gripper(1)]` *normalized* — requires the training-time normalizer to be reapplied | `~0.05–0.1` (deltas are normalized to `[-1,1]`, not meters) | continuous `[0,1]` or `[-1,1]`, threshold at `0.5` — **NOT sign-based** | 4–10 |
| **GR-1 / GR00T-N1** | front + wrist | `(224, 224)` | `[eef_pos(3), eef_axisangle(3), gripper(2)]` | `1.0` (verify) | continuous `[0,1]` per finger, take `min`, threshold `0.5` | varies |
| **RVT / RVT-2** *(keyframe)* | front + 2 shoulder + wrist | `(128,128)` (RVT) or `(220,220)` (RVT-2) | (consumed internally from voxel grid; omit) | n/a (absolute keyframe pose) | `binary_open` `{0,1}`, pass through | n/a (one keyframe → multi-step planning) |
| **PerAct / Act3D** *(keyframe + 3D)* | front + 2 shoulder + wrist (+ depth + point_cloud) | `(128, 128)` | (consumed internally; omit) | n/a (absolute keyframe) | `binary_open`, pass through | n/a (keyframe) |

**Keyframe models (RVT, PerAct, Act3D) need a different recipe.** They emit one
absolute target pose every 5–20 sim steps, executed by motion planning between
keyframes. Use `EndEffectorPoseViaPlanning(absolute_mode=True, frame=WORLD)`
*instead of* `EndEffectorPoseViaIK(absolute_mode=False, frame=EE)`, pass the
absolute pose directly, and skip the delta decode. For the simplest path to a
PerAct paper number, use PerAct's own eval scripts
(`MohitShridhar/peract`), not §5 — the recipe below targets continuous delta-EE
VLAs only.

**Action chunking (continuous VLAs only).** π₀ / OFT return an `(N, 7)` chunk
per call; consume the first `K` (typically `K=4–8` for π₀, `K=8` for OFT),
then re-query. Wrap with a queue:

```python
action_queue = []
for _ in range(MAX_STEPS):
    if not action_queue:
        chunk = model_predict(obs_dict)
        chunk = np.atleast_2d(chunk)
        action_queue = list(chunk[:K])   # K from the table above
    raw_action = action_queue.pop(0)
    ...
```

OpenVLA base is per-step (`K=1`), so the queue degenerates to the simple loop
shown below.

**Image resolution discipline.** Set `image_size` to your VLA's training
resolution at `CameraConfig` time. Do NOT collect at one resolution and resize
client-side — depth/point_cloud are tied to the camera resolution and resizing
RGB after the fact desynchronizes them.

### Recipe (continuous delta-EE VLAs)

Then in Python:

```python
import gc
import numpy as np
from scipy.spatial.transform import Rotation
from pyrep.const import RenderMode
from rlbench import utils as rlb_utils
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, RelativeFrame
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.backend.exceptions import (
    InvalidActionError, BoundaryError, WaypointError,
)
from pyrep.errors import IKError                    # underlying PyRep IK failure

# === 1) ObservationConfig — set per the per-model table above ===========
# Default below: OpenVLA template (front only, 224², no proprio).
# For other models, swap in their row from the table (e.g. OFT: front+wrist,
# 256², joint_positions+gripper_open; π₀: front+wrist, 224², full EE state).
#
# render_mode: pass OPENGL explicitly if you DISABLED the OpenGL3 plugin in
# bash step 3; keep OPENGL3 (default) if you have a real GPU-backed X server.
RENDER_MODE = RenderMode.OPENGL                     # change to OPENGL3 if not disabled
cam_front = CameraConfig(rgb=True, depth=False, point_cloud=False, mask=False,
                         image_size=(224, 224), render_mode=RENDER_MODE)
cam_off = CameraConfig(); cam_off.set_all(False)
obs_cfg = ObservationConfig(
    front_camera=cam_front,
    wrist_camera=cam_off,                           # enable for π₀/OFT/GR-1
    left_shoulder_camera=cam_off,
    right_shoulder_camera=cam_off,
    overhead_camera=cam_off,
    # Proprio — sim still computes these; set to False for OpenVLA (unused) to
    # save sim cost, set to True for state-conditioned VLAs (π₀/OFT/GR-1/X-VLA).
    joint_positions=False,
    joint_velocities=False,
    joint_forces=False,
    gripper_open=False,
    gripper_pose=False,
)

# === 2) Action mode — delta-EE in EE frame (continuous VLAs) ============
# For keyframe models (RVT/PerAct/Act3D), replace with:
#   EndEffectorPoseViaPlanning(absolute_mode=True, frame=RelativeFrame.WORLD)
# and pass absolute target poses (not deltas) — see the per-model table.
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaIK(
        absolute_mode=False,
        frame=RelativeFrame.EE,                     # NOT WORLD — §3.1 / §4.2
        collision_checking=False,
    ),
    gripper_action_mode=Discrete(),
)

env = Environment(
    action_mode=action_mode,
    obs_config=obs_cfg,
    headless=True,                                  # hides GUI, still needs DISPLAY
    robot_setup='panda',                            # only Panda has published VLA scores
    shaped_rewards=False,                           # sparse reward = float(success)
    attach_grasped_objects=True,
)
env.launch()

# === 3) PerAct-18 task list =============================================
PERACT_18 = [
    "close_jar", "reach_and_drag", "insert_onto_square_peg", "meat_off_grill",
    "open_drawer", "place_cups", "stack_wine", "push_buttons",
    "put_groceries_in_cupboard", "put_item_in_drawer", "put_money_in_safe",
    "light_bulb_in", "slide_block_to_target", "place_shape_in_shape_sorter",
    "stack_blocks", "stack_cups", "sweep_to_dustpan", "turn_tap",
]

EPISODES_PER_TASK = 25
MAX_STEPS = 300                                     # original has no built-in timeout
POS_SCALE = 1.0                                     # OpenVLA/OFT=1.0; π₀≈0.05; see table
CHUNK_K  = 1                                        # OpenVLA=1; π₀=4-10; OFT=8

# Replace with your VLA inference. Return (N,7) chunk or (7,) per call.
# See harness reference implementations:
#   src/vla_eval/model_servers/openvla.py   (per-step)
#   src/vla_eval/model_servers/pi0.py       (chunked)
#   src/vla_eval/model_servers/oft.py       (chunked)
def model_predict(obs_dict):
    raise NotImplementedError("plug in your VLA forward pass")

def build_obs_dict(raw, lang):
    """Build the observation dict for your model. Per-model `states` content
    differs — see the per-model table above. Default below = OpenVLA (no state)."""
    obs = {
        "images": {"front": np.asarray(raw.front_rgb, dtype=np.uint8)},
        "task_description": lang,
    }
    # For OFT, append:
    #   obs["states"] = np.concatenate([raw.joint_positions, [raw.gripper_open]])
    # For π₀, append (after applying training-time normalizer):
    #   obs["states"] = np.concatenate([raw.gripper_pose, [raw.gripper_open]])  # 8D
    # For GR-1, axis-angle state:
    #   aa = Rotation.from_quat(raw.gripper_pose[3:]).as_rotvec()
    #   obs["states"] = np.concatenate([raw.gripper_pose[:3], aa, raw.gripper_joint_positions])
    return obs

def discretize_gripper(grip_raw):
    """Convert wire gripper to RLBench Discrete (1.0=open, 0.0=closed).

    OpenVLA / OFT — wire is `binary_close_positive` (+1=close, -1=open).
        return 0.0 if grip_raw >= 0 else 1.0   (← default below)
    π₀ / π₀-FAST — continuous in [0,1] (or [-1,1]), threshold 0.5, 1=close.
        return 0.0 if grip_raw >= 0.5 else 1.0
    GR-1 / GR00T — two-finger continuous [0,1], take min, 0=close.
        return 1.0 if min(grip_raw) >= 0.5 else 0.0
    Always verify against your model server's `get_action_spec()`.
    """
    return 0.0 if grip_raw >= 0 else 1.0            # OpenVLA/OFT default

results = {}
ik_fail_streak_threshold = 10                       # give up episode after N consecutive IK fails

for task_name in PERACT_18:
    task_cls = rlb_utils.name_to_task_class(task_name)
    task_env = env.get_task(task_cls)
    var_count = task_env.variation_count()
    succ = 0
    try:
        for ep in range(EPISODES_PER_TASK):
            # --- 4) Deterministic variation + seeded placement ---------
            task_env.set_variation(ep % var_count)
            np.random.seed(ep)                      # MUST come before reset()
            try:
                descs, raw = task_env.reset()
            except (BoundaryError, WaypointError) as e:
                # Scene-config error (not a model error); skip this episode.
                print(f"[{task_name} ep{ep}] reset failed: {e}")
                continue
            # --- 5) Language: PerAct samples uniformly per episode -----
            lang = descs[np.random.randint(len(descs))]
            action_queue = []
            ik_fail_streak = 0
            for _step in range(MAX_STEPS):
                if not action_queue:
                    chunk = model_predict(build_obs_dict(raw, lang))
                    chunk = np.atleast_2d(np.asarray(chunk, dtype=np.float64))
                    action_queue = list(chunk[:CHUNK_K])
                raw_action = action_queue.pop(0)
                assert raw_action.shape[0] == 7, "model must emit 7D"
                # --- 6) Decode model 7D → RLBench 8D --------------------
                pos = raw_action[0:3] * POS_SCALE
                quat = Rotation.from_rotvec(raw_action[3:6]).as_quat()   # [x,y,z,w]
                grip = discretize_gripper(float(raw_action[6]))
                act = np.concatenate([pos, quat, [grip]])                # 8D
                # --- 7) Step with typed exception handling --------------
                try:
                    raw, reward, terminate = task_env.step(act)
                    ik_fail_streak = 0
                except (IKError, InvalidActionError):
                    # Bad action: oversized delta, unreachable target, or
                    # near-singular Jacobian. Drop the chunk and re-query.
                    action_queue.clear()
                    ik_fail_streak += 1
                    if ik_fail_streak > ik_fail_streak_threshold:
                        # Likely wrong frame / scale / gripper. Abort episode.
                        break
                    continue
                if terminate:
                    if reward > 0.99:
                        succ += 1
                    break
    finally:
        # Per-task cleanup: RLBench has no per-task shutdown but releasing the
        # reference + GC mitigates CoppeliaSim handle accumulation across tasks.
        del task_env
        gc.collect()

    results[task_name] = succ / EPISODES_PER_TASK
    print(f"{task_name}: {results[task_name]:.2%}")

print(f"\nOverall (mean): {np.mean(list(results.values())):.2%}")
env.shutdown()
```

Smoke order before running the full PerAct-18:

1. **`reach_target` × 5** — sanity: IK works, env loop wires together. Position-only;
   will pass with wrong gripper or wrong rotation frame.
2. **`pick_up_cup` × 5 (or `turn_tap`)** — sanity: gripper sign + orientation
   frame + `POS_SCALE` are correct. These are the silent-failure axes; if the
   gripper never closes on the cup, your gripper convention is inverted; if the
   arm orbits the cup without grasping, your rotation frame is wrong; if it
   overshoots/jitters, `POS_SCALE` is too large.
3. **PerAct-18 × 25** once smoke 2 passes.

If IK errors storm from step 1: `POS_SCALE` is too large or rotation frame is
wrong. If gripper visibly inverts (opens when it should close): flip the
`discretize_gripper` branch.

**First reproduction targets, ranked by ease with the §5 recipe alone:**

| Difficulty | Model | Published RLBench score | Why |
|---|---|---|---|
| ★ Easiest | **OpenVLA-7B / OpenVLA-OFT** | ~30–50% (LIBERO-trained → RLBench transfer) | Per-step, single front cam, no state. Public checkpoint at `openvla/openvla-7b`. Recipe's defaults are tuned to it. |
| ★★ | **OFT (standalone)** | varies | Chunk size 8, front+wrist@256, `joint_positions` state. Same gripper sign as OpenVLA. |
| ★★★ | **π₀-FAST** | ~55% (HybridVLA / similar) | Chunk handling required; deltas are normalized → need the training-time unnormalizer; continuous gripper. No single canonical RLBench checkpoint — use a LIBERO-finetuned variant. |
| ★★★★ | **GR-1 / GR00T-N1** | varies | Different state spec, two-finger gripper, model-specific unnormalizer. |
| ✖ out of §5 scope | **RVT / RVT-2** | 81.4% (RVT-2) | Keyframe model: needs `EndEffectorPoseViaPlanning`, 4 cameras at 220², voxel-grid 3D. Use RVT's own eval script. |
| ✖ out of §5 scope | **PerAct / BridgeVLA / TVVE** | 49.4 / 88.2 / 86.6% | Keyframe + 3D (depth + PCD) + multi-view. Use their respective repos' eval scripts. The PerAct checkpoint is public at `MohitShridhar/peract`. |

Full published numbers (paper-reported) are in
`leaderboard/data/leaderboard.json` — all 136 `rlbench` entries are
`reported_paper` references; zero are harness-measured, so any score you
produce with the §5 recipe is the first end-to-end measurement on this pipeline.

---

## 6. When to use harness vs original

| Situation | Use |
|---|---|
| You want everything dockerized, license-gated, reproducible across machines, with a shared model-server protocol | **Harness** — after fixing the bugs in `rlbench-enablement-plan.md` |
| You have a custom training pipeline already, want depth/point-cloud/multi-camera, need `dataset_generator` for IL data, want the Gym wrapper | **Original** — recipe in §5 above |
| You want to reproduce a single PerAct paper score with your model end-to-end and get there fast | **Original** — fewer abstractions to fight; §5 is enough |
| You want to compare 5+ models head-to-head with one harness across many benchmarks (LIBERO + CALVIN + RLBench + …) | **Harness** — after Phase 1–3 in `rlbench-enablement-plan.md` |
| You want bimanual / RLBench2 | Neither — get a separate fork |

---

## 7. References (exact paths)

Original (clone at `02720bba` of `stepjam/RLBench` master):

- Action modes: `rlbench/action_modes/{arm,gripper,}_action_mode{s,}.py`
- Observation: `rlbench/observation_config.py`, `rlbench/backend/observation.py`
- Tasks: `rlbench/tasks/*.py` (106 files), task assets `rlbench/task_ttms/*.ttm`
- Lifecycle: `rlbench/task_environment.py`, `rlbench/backend/task.py`
- Env: `rlbench/environment.py`, robot models `rlbench/robot_ttms/`
- Gym: `rlbench/gym.py`
- sim2real: `rlbench/sim2real/`
- Tools: `tools/{task_builder,task_validator,cinematic_recorder}.py`
- Demos: `rlbench/{dataset_generator,demo}.py`
- README install: `README.md:62-137`

Harness (current `main`):

- Benchmark wrapper: `src/vla_eval/benchmarks/rlbench/benchmark.py`
- Eval config: `configs/benchmarks/rlbench/{eval,README}.yaml/md`
- Docker: `docker/Dockerfile.rlbench`, `docker/rlbench_entrypoint.sh`,
  `docker/build.sh`, `docker/push.sh` (NO_REDIST)
- Protocol notes: `leaderboard/benchmarks/rlbench.md`
- Paper baselines: `leaderboard/data/leaderboard.json` (136 entries,
  all `reported_paper`)
- Fix plan: `docs/rlbench-enablement-plan.md`

---

## 8. Verification record

This doc was assembled from 12 parallel Sonnet investigations (action modes,
observations, tasks, variations, lifecycle, robot/scene, headless, language,
demos, extras, install, PerAct protocol — each comparing original vs harness
with file:line citations), then independently cross-checked by two Opus
reviewers on 2026-05-20.

**Factual review (load-bearing claims vs ground truth):** all 12 load-bearing
claims CONFIRMED, no WRONG items. The headline `image_size` vs
`render_resolution` kwarg bug claim was independently re-verified against
`observation_config.py:6-16` and the Dockerfile's clone URL — TypeError is
mechanically certain on first `_ensure_env()`.

**Usability review (adversarial, VLA-practitioner perspective):** initial draft
was flagged as "would not ship as-is" because §5 silently assumed OpenVLA
conventions, making it wrong for π₀ (deltas normalized, not metres), OFT
(joint-state, not EE-state), GR-1 (two-finger), and keyframe models (need
planning mode, not IK). Corrections applied in this version:

- Added the **per-model configuration table** at the top of §5 (cameras,
  `image_size`, `states` content, `POS_SCALE` start, gripper convention, action
  chunk size) — single highest-impact addition.
- Added **action chunking** guidance for π₀/OFT (consume `K` actions before
  re-querying); recipe now wraps `model_predict` with an action queue.
- Made the **OpenGL3 segfault workaround conditional** in §5 step 3 (apply
  under Xvfb only; keep enabled on real GPU-backed X servers).
- Added **`render_mode=RenderMode.OPENGL`** explicit fallback in the recipe.
- Replaced blanket `except Exception: pass` with **typed exception handling**
  (`IKError` / `InvalidActionError` / `BoundaryError` / `WaypointError`) and a
  consecutive-IK-fail streak break.
- Added **per-task cleanup** (`del task_env; gc.collect()`) to mitigate
  CoppeliaSim handle accumulation.
- Factored gripper inversion into a `discretize_gripper()` helper with branches
  for OpenVLA/OFT, π₀, GR-1.
- Clarified **`states` composition is per-model**, not RLBench-fixed (OFT needs
  `joint_positions`, not `gripper_pose`).
- Added **train-vs-eval disambiguation** to §3.12 and §0 TL;DR (100 demos/task
  is a *training* requirement; eval-only users skip `dataset_generator`).
- Specified the **243 (HEAD) vs 249 (PerAct-fork) variation gap** in §3.3 with
  fork-pinning guidance.
- Replaced the "pick a published number" hand-wave with a **ranked
  reproduction-difficulty table** in §5 (OpenVLA easiest; RVT/PerAct/keyframe
  models out of §5 scope).

**Items the reviewers explicitly confirmed survive scrutiny:**

- The `image_size`-vs-`render_resolution` kwarg bug claim.
- `EndEffectorPoseViaIK(frame=RelativeFrame.EE)` correctness argument
  (WORLD-frame left-multiply is geometrically wrong for EE-frame VLA deltas).
- `Discrete` gripper semantics (1.0=open, 0.0=closed, threshold 0.5).
- Original README's headless recipe assumes a real GPU-backed X server (no EGL,
  no Xvfb, no `QT_QPA_PLATFORM=offscreen` mentions).
- 136 leaderboard rlbench entries are all `reported_paper`; zero
  harness-measured.
- `sample_variation()` random / `set_variation(i)` deterministic API.
- `Observation.misc` carries per-camera intrinsics/extrinsics.
- `scipy.Rotation.from_rotvec().as_quat()` returns `[x,y,z,w]`.
- 18-task PerAct file mapping in §3.3.

**Reviewer's residual reservation:** keyframe models (RVT/PerAct/Act3D) are not
covered by the §5 recipe and require a different action-mode plus camera setup.
This doc explicitly defers them to their own repos' eval scripts rather than
inventing a brittle keyframe path here.
