# RLBench Enablement Plan — Running Arbitrary VLAs on RLBench

> **Status:** design / implementation plan (not yet implemented)
> **Author:** prepared 2026-05-19
> **Audience:** whoever implements this in an environment with GPU + Docker + the
> `rlbench` image. This document is self-contained — it does not assume access
> to the conversation that produced it.

---

## 한국어 요약 (TL;DR)

RLBench는 **막혀 있던 게 아니다.** "8D joint-velocity vs EE 모델" 불일치라는 초기 진단은
아키텍처 오독이었다. 이 harness의 설계 의도는 **"각 benchmark의 `step()`이 모델이 보내는
flat 7D `action["actions"]` 배열을 그 시뮬레이터 컨트롤러 형식으로 직접 decode한다"** 이고,
LIBERO·CALVIN·ManiSkill2·SimplerEnv가 전부 이 패턴을 따른다. RLBench만 이 decode를
생략하고 `JointVelocity` action mode를 하드코딩한 **유일한 outlier**다.

따라서 할 일은 "새 인프라/어댑터 작성"이 아니라 **이미 코드베이스에 4번 구현된 패턴을
RLBench `benchmark.py` 한 파일에 이식**하는 것이다. 모델 코드는 한 줄도 안 바뀐다.
예상 작업량 ≈ 2일 (Docker 빌드 포함, 모델 1개 reproduction 기준).

---

## 1. The core misconception, corrected

### 1.1 What was wrongly concluded first

> "All model servers output `{position, rotation, gripper}` (delta-EE 7D); RLBench's
> `step()` reads `action["actions"]` and asserts 8D joint velocities; there is no
> adapter; therefore RLBench is blocked until someone writes an EE→joint converter
> or a dedicated joint-velocity model server."

This conflated two *different* things: the **spec dict** and the **runtime wire payload**.

### 1.2 How the action interface actually works

- `get_action_spec()` returns a dict like `{"position": POSITION_DELTA, "rotation":
  ROTATION_AA, "gripper": GRIPPER_CLOSE_POS}`. This is **only** sent during the
  HELLO handshake and fed to `check_specs()` for **diagnostic logging**. See
  `src/vla_eval/specs.py:1-7` (module docstring) and `:110-136` (`check_specs`):
  it returns a list of warning strings; the orchestrator merely `logger.warning`s
  them. **It never converts or blocks anything.**
- At **runtime**, every EE model server emits a single flat array under the
  `"actions"` key: `{"actions": np.ndarray([dx, dy, dz, aa_x, aa_y, aa_z, grip])}`
  (7D for delta-EE models). The `position/rotation/gripper` keys are documentation
  of what the 7 numbers *mean*, not the wire format.
- Each benchmark's `step()` is responsible for **decoding that flat 7D array into
  whatever its simulator's controller expects.** This is the intended extension
  point. The connection/runner pass the action dict verbatim — there is no
  transform hook between server and `benchmark.step()`
  (`src/vla_eval/connection.py` `act()` is a pure send/recv; the sync runner calls
  `benchmark.apply_action(action)` directly).

### 1.3 Evidence: LIBERO does exactly this (the reference pattern)

`src/vla_eval/benchmarks/libero/benchmark.py:211-226`:

```python
def step(self, action: Action) -> StepResult:
    raw_action = action.get("actions", action.get("action"))
    if isinstance(raw_action, np.ndarray):
        raw_action = raw_action.tolist()
    assert len(raw_action) == 7, f"...got {len(raw_action)}, expected 7"

    # Discretize gripper
    if raw_action[-1] < 0:
        gripper = -1.0
    else:
        gripper = 1.0
    processed_action = raw_action[:-1] + [gripper]

    assert self._env is not None
    obs, reward, done, info = self._env.step(processed_action)   # OSC delta-EE controller
    return StepResult(obs=obs, reward=reward, done=done, info=info)
```

LIBERO takes the flat 7D `actions`, binarizes the gripper, and feeds it to its
robosuite OSC **delta-EE** controller. CALVIN / ManiSkill2 / SimplerEnv follow the
same shape — read `action["actions"]`, decode into the sim's native controller
input (CALVIN additionally converts axis-angle → euler in `_process_absolute_action`).

### 1.4 Evidence: RLBench is the outlier

`src/vla_eval/benchmarks/rlbench/benchmark.py:94-97` hardcodes the wrong action mode:

```python
action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),     # ← 7 joint velocities, NOT what EE VLAs emit
    gripper_action_mode=Discrete(),
)
```

and `:123-135` skips the decode entirely:

```python
def step(self, action: Action) -> StepResult:
    raw_action = action.get("actions", action.get("action"))
    act = np.asarray(raw_action, dtype=np.float64)
    assert act.shape[-1] == 8, f"...got {act.shape[-1]}, expected 8"
    if act.shape[0] < 8:
        act = np.pad(act, (0, 8 - act.shape[0]))
    act = act[:8]
    obs, reward, terminate = self._task_env.step(act)            # interprets as joint velocities
    return StepResult(obs=obs, reward=reward, done=terminate, info={})
```

A delta-EE model sends a 7D vector; this asserts 8D and (worse) would feed the
numbers in as joint velocities — physically meaningless. **The bug is the missing
decode + wrong action mode, not a missing infrastructure layer.**

### 1.5 RLBench natively supports the action mode we need

Verified in the local RLBench source
`/home/theo_lab/RLBench/rlbench/action_modes/arm_action_modes.py`:

- `class EndEffectorPoseViaIK(ArmActionMode)` and
  `class EndEffectorPoseViaPlanning(ArmActionMode)` both:
  - `action_shape(scene) -> (7,)`
  - `action()` does `assert_action_shape(action, (7,))` then
    `assert_unit_quaternion(action[3:])` — i.e. input is
    `[x, y, z, qx, qy, qz, qw]` with a **unit quaternion** rotation.
  - `__init__(absolute_mode: bool = True, frame: RelativeFrame = WORLD,
    collision_checking: bool = False)`.
  - When `absolute_mode=False` and `frame != EE`, internally calls
    `calculate_delta_pose(scene.robot, action)` — i.e. it natively supports
    **delta** end-effector control, which is exactly what OpenVLA/π₀/OFT/X-VLA emit.
- `MoveArmThenGripper(EndEffectorPoseViaIK(...), Discrete())` therefore expects a
  total of **7 (pose) + 1 (discrete gripper) = 8D** array. The existing
  `assert act.shape[-1] == 8` stays correct *dimensionally*; only the **content
  and the producing decode** change.

`scipy.spatial.transform.Rotation.as_quat()` returns `[x, y, z, w]`, which matches
both RLBench's `assert_unit_quaternion(action[3:])` ordering and the harness's
`specs.ROTATION_QUAT = DimSpec("rotation", 4, "quaternion_xyzw", (-1, 1))`.

---

## 2. Search confirmation (why nothing pre-existing helps, and nothing is missing)

Exhaustive search was performed across all refs, history, and GitHub:

- **Branches/stashes/tags:** the only branches are `main`, `origin/docs/readme-news`,
  `origin/fix/smoke-test-failures`, `origin/update-leaderboard-data`, `origin/gh-pages`.
  None contain any RLBench model server, adapter, or reproduction work beyond `main`.
  No stashes. Tags `v0.0.1`–`v0.2.0` only carry older/equal versions of the same
  benchmark integration.
- **History archaeology:** no deleted/reverted RLBench model or adapter ever existed.
  Commit `ae23527`'s "stale embedded rlbench results" were paper-scraped numbers an
  earlier LLM run misplaced — never harness-measured. The 5 RLBench Docker commits
  (`5ea6e51`, `db8f8d3`, `173a777`, `73f5452`, `29724e1`) are infra-only.
- **GitHub:** PR #55 (license gate) is the only RLBench-specific PR. No model-server
  PR, no issue requesting one, no maintainer note. RLBench was the lowest-priority
  ("May") proposal item. Issue #44 lists RLBench as "Likely works" on ROCm — never
  confirmed end-to-end on any hardware.
- **Leaderboard:** all 136 `rlbench` entries in `leaderboard/data/leaderboard.json`
  are `reported_paper` references. **Zero** are harness-measured. So the first
  harness reproduction does not exist yet — we will be creating it.

**Conclusion:** nothing reusable exists, but nothing new (infra/adapter) needs to
be invented either. The work is a one-file transplant of the existing pattern.

---

## 3. The implementation

All code changes are confined to **one file**:
`src/vla_eval/benchmarks/rlbench/benchmark.py`. Model servers are **not** modified.

### 3.1 Switch the action mode (`_ensure_env`)

Replace the `JointVelocity` import and the `action_mode` construction
(`benchmark.py:69, 94-97`):

```python
# was:
from rlbench.action_modes.arm_action_modes import JointVelocity
...
action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),
    gripper_action_mode=Discrete(),
)

# becomes (recommended default — continuous delta-EE):
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, RelativeFrame
from rlbench.action_modes.action_mode import MoveArmThenGripper
...
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaIK(
        absolute_mode=False,
        frame=RelativeFrame.EE,        # NOT the WORLD default — see correctness note below
    ),
    gripper_action_mode=Discrete(),
)
```

> **`frame=RelativeFrame.EE` is a correctness decision, not a tuning knob.**
> Verified in `arm_action_modes.py`: when `absolute_mode=False` **and**
> `frame != EE`, RLBench runs `calculate_delta_pose(robot, action)` which composes
> the rotation as `Quaternion(delta) * Quaternion(current_world_pose)` — a
> *world-frame left-multiply*. But VLA per-step rotation deltas (OpenVLA/π₀/OFT,
> the robosuite OSC convention LIBERO feeds) are **end-effector-frame
> increments**. Feeding an EE-frame increment as a WORLD-frame left product is
> geometrically wrong and produces systematically incorrect orientation **from
> step 1** — not gentle drift. With `frame=RelativeFrame.EE`, RLBench skips
> `calculate_delta_pose` and solves IK relative to the current tip, which is the
> correct interpretation for EE-frame deltas. If a specific model is later found
> to emit world-frame rotation deltas, that model — not the default — switches to
> `frame=WORLD`.

Make the mode selectable via a constructor arg so keyframe-style models can opt
into planning later:

```python
def __init__(self, tasks=None, render_resolution=256, max_steps=200,
             action_mode: str = "ee_ik_delta", send_state: bool = False,
             seed: int = 0) -> None:
    ...
    self._action_mode_name = action_mode   # "ee_ik_delta" | "ee_planning_delta" | "joint_velocity"
    self._send_state = send_state
    self._seed = seed
```

`_ensure_env` then branches:

| `action_mode` value | RLBench arm action mode | Use for |
|---|---|---|
| `ee_ik_delta` (default) | `EndEffectorPoseViaIK(absolute_mode=False, frame=RelativeFrame.EE)` | continuous delta-EE VLAs: OpenVLA, π₀, OFT, X-VLA, GR00T |
| `ee_planning_delta` | `EndEffectorPoseViaPlanning(absolute_mode=False, frame=RelativeFrame.EE)` | keyframe / waypoint models (PerAct-style) |
| `joint_velocity` | `JointVelocity()` (legacy) | a future joint-space model server |

### 3.2 Add the decode (`step`)

Mirror LIBERO's structure, but assemble the RLBench EE input
`[dx, dy, dz, qx, qy, qz, qw, gripper_discrete]` (8D). Three things are NOT
trivial and were wrong in an earlier draft of this plan — they are corrected and
called out below: (a) gripper sign for the recommended first models, (b) delta
position scaling, (c) IK failure must not crash the episode.

```python
from scipy.spatial.transform import Rotation

# RLBench consumes raw metres/radians; LIBERO-style deltas are OSC-gain-scaled.
# Start at 1.0, tune on a grasp task (see §6). Make it a constructor arg.
self._pos_scale = pos_scale          # e.g. 1.0; per-model

def step(self, action: Action) -> StepResult:
    raw = action.get("actions", action.get("action"))
    raw = np.asarray(raw, dtype=np.float64).flatten()

    if self._action_mode_name == "joint_velocity":
        # legacy path: pass 8D through unchanged
        act = np.pad(raw, (0, max(0, 8 - raw.shape[0])))[:8]
    else:
        # EE path: model emits [dx,dy,dz, aa_x,aa_y,aa_z, grip] (7D)
        assert raw.shape[0] == 7, f"expected 7D delta-EE action, got {raw.shape[0]}"
        pos = raw[0:3] * self._pos_scale
        aa = raw[3:6]
        # axis-angle == rotation vector (angle*axis). Confirmed against the
        # harness's own convention in src/vla_eval/rotation.py and CALVIN's
        # axisangle_to_matrix(raw[3:6]) — from_rotvec is the correct inverse.
        quat = Rotation.from_rotvec(aa).as_quat()    # [x,y,z,w], unit-norm
        # GRIPPER: RLBench Discrete → 1.0=OPEN, 0.0=CLOSED (verified in
        # rlbench/action_modes/gripper_action_modes.py: action = float(a>0.5),
        # action==0.0 → grasp). OpenVLA/OFT emit binary_close_positive at the
        # wire (openvla.py:144 `-np.sign(2x-1)` → +1=CLOSE, -1=OPEN; matches
        # specs.GRIPPER_CLOSE_POS and LIBERO libero/benchmark.py:218). So for the
        # recommended first models the mapping is INVERTED vs RLBench:
        grip = 0.0 if raw[6] >= 0 else 1.0           # close_positive → Discrete
        act = np.concatenate([pos, quat, [grip]])    # 8D

    assert self._task_env is not None
    try:
        obs, reward, terminate = self._task_env.step(act)
    except Exception as e:
        # EndEffectorPoseViaIK raises InvalidActionError/IKError when the target
        # is out of reach or the Jacobian solve diverges (large/unscaled delta).
        # One bad action must NOT abort the episode — treat as a no-op-ish
        # terminal failure for this step and let episode-level isolation handle it.
        from vla_eval.benchmarks.base import StepResult
        return StepResult(obs=self._last_obs, reward=0.0, done=True,
                          info={"ik_error": repr(e)})
    self._last_obs = obs
    return StepResult(obs=obs, reward=reward, done=terminate, info={})
```

> **Gripper convention — this is a per-model CORRECTNESS fork, decided up front,
> not "tune later".** RLBench `Discrete` is `1.0`=OPEN / `0.0`=CLOSED. The wire
> convention differs by model and must be read from the model server, not guessed:
> OpenVLA (`openvla.py:144`) and OFT emit **`binary_close_positive`** (positive =
> CLOSE), so the correct mapping is `grip = 0.0 if raw[6] >= 0 else 1.0` (used
> above). A `binary_close_negative` / `GRIPPER_01` model needs the opposite or a
> threshold. **`reach_target` (the smoke task) does not grasp, so it cannot
> validate this** — verify on a grasping task (§6).

> **Delta frame — already resolved in §3.1 (`frame=RelativeFrame.EE`), not here.**
> The position part is a pure additive EE-frame translation; the rotation part is
> composed correctly only in EE frame. Do **not** revert to `frame=WORLD` as a
> "drift" workaround — that reintroduces the world-vs-EE rotation bug. The only
> open scalar here is `pos_scale` (magnitude/units), tuned on a grasp task.

### 3.3 Update specs & metadata

```python
def get_action_spec(self) -> dict[str, DimSpec]:
    if self._action_mode_name == "joint_velocity":
        return {"joints": DimSpec("joints", 7, "joint_velocity"), "gripper": GRIPPER_RAW}
    return {                                  # EE path: matches what VLAs declare
        "position": POSITION_DELTA,
        "rotation": ROTATION_AA,
        "gripper": GRIPPER_CLOSE_POS,         # set to the verified convention
    }

def get_metadata(self) -> dict[str, Any]:
    return {"max_steps": self._max_steps, "action_dim": 8}   # action_dim added
```

(`action_dim` is currently absent; adding it keeps the async runner correct if it
is ever used. Import `POSITION_DELTA, ROTATION_AA, GRIPPER_CLOSE_POS` from
`vla_eval.specs`.)

### 3.4 Required `reset()` change + optional proprio

**REQUIRED — deterministic variation (not optional).** The current `reset()` calls
`self._task_env.sample_variation()` — a *random* variation every episode. The
stated goal is "reproduce a published PerAct score within ~2pp"; published PerAct
numbers use a fixed variation/episode protocol. Random sampling makes scores
non-reproducible and not apples-to-apples, defeating the goal. Select the
variation deterministically from `task.get("episode_idx", 0)` and `self._seed`
(mirroring how LIBERO indexes `initial_states[episode_idx]`). Also set
`self._last_obs = obs` at the end of `reset()` so the §3.2 IK-error path has a
valid observation to return on the first step.

**Optional — proprio forwarding.** `ObservationConfig` already enables
`joint_positions=True, gripper_open=True`, but `make_obs()` never forwards them.
If a target model needs state, add an opt-in `states` key in `make_obs()` (gated
by `self._send_state`), mirroring LIBERO's `send_state` block.

---

## 4. Configs

### 4.1 Benchmark side

Current `configs/benchmarks/rlbench/eval.yaml` is a 1-task stub
(`reach_target`, 1 episode) despite the README claiming "18 tasks × 25 episodes".

Create two files:

`configs/benchmarks/rlbench/smoke_test.yaml` (fast sanity loop):

```yaml
server:
  url: "ws://localhost:8000"
docker:
  image: ghcr.io/allenai/vla-evaluation-harness/rlbench:latest
  runtime: nvidia
output_dir: "./results"
benchmarks:
  - benchmark: "vla_eval.benchmarks.rlbench.benchmark:RLBenchBenchmark"
    episodes_per_task: 2
    params:
      tasks: [reach_target]
      action_mode: ee_ik_delta
      render_resolution: 256
      max_steps: 200
    action_dim: 8
```

`configs/benchmarks/rlbench/eval.yaml` (real protocol — 18-task PerAct subset from
`leaderboard/benchmarks/rlbench.md`): `close_jar, drag_stick, insert_peg,
meat_off_grill, open_drawer, place_cups, place_wine, push_buttons,
put_in_cupboard, put_in_drawer, put_in_safe, screw_bulb, slide_block, sort_shape,
stack_blocks, stack_cups, sweep_to_dustpan, turn_tap`; `episodes_per_task: 25`.
Fix the README table + comment to match reality once finalized.

### 4.2 Model side — zero model-code changes

Add a per-model config, e.g. `configs/model_servers/openvla/rlbench.yaml`,
copying an existing benchmark config for that model (e.g. its `libero.yaml`) and
adjusting only image keys / resize to match RLBench's `front` + `wrist` 256² RGB
observation (`make_obs()` returns `images={"front":..., "wrist":...}`,
`task_description=...`). No adapter, no new server class.

---

## 5. Docker / license (unchanged, but required)

The `rlbench` image is **not** on ghcr.io (CoppeliaSim Edu + RLBench carry
non-redistribution / academic-only licenses; `docker/push.sh` has `rlbench` in
`NO_REDIST`). It must be built locally with an explicit license opt-in:

```bash
docker/build.sh rlbench --accept-license rlbench
```

Internals (from `docker/Dockerfile.rlbench`): Python 3.8 conda env, **CoppeliaSim
4.1.0 exactly** (newer segfaults), PyRep + RLBench from GitHub HEAD, Xvfb `:99`
headless via `docker/rlbench_entrypoint.sh`, `libsimExtOpenGL3Renderer.so`
disabled, `tini` PID 1.

**Reproducibility hole to fix while here:** PyRep and RLBench are
`git clone --depth 1` with **no commit pin**. Add `ARG PYREP_COMMIT` /
`ARG RLBENCH_COMMIT` and check those out, so the image (and therefore any
reported score) is reproducible. This aligns with the repo's stated
Reproducibility design principle.

---

## 6. End-to-end run (two terminals)

```bash
# Terminal 1 — model server (host, GPU). No model code changes; just a new config.
uv run vla-eval serve -c configs/model_servers/openvla/rlbench.yaml \
  --address 0.0.0.0:8000 -v

# Terminal 2 — benchmark (Docker). Smoke first.
vla-eval run -c configs/benchmarks/rlbench/smoke_test.yaml \
  --server-url ws://localhost:8000 --accept-license rlbench --yes
```

**Two-stage smoke — `reach_target` alone is insufficient.** `reach_target` is
position-only: it validates the IK/translation path and the env wiring, but it
**cannot** catch the gripper-convention bug or any rotation-frame error. After
`reach_target` looks sane, run a **grasp + orientation** task (e.g. `turn_tap` or
`pick_up_cup`) and confirm the gripper actually closes on the object and
orientation tracks — this is where gripper sign and `pos_scale` are tuned. Only
then scale out:

```bash
./scripts/run_sharded.sh -c configs/benchmarks/rlbench/eval.yaml -n <shards>
vla-eval merge -c configs/benchmarks/rlbench/eval.yaml -o results/rlbench.json
```

---

## 7. Validation checklist

1. `make check` (ruff + ty, line length 119) and `make test` pass.
2. `vla-eval test -c configs/benchmarks/rlbench/smoke_test.yaml` (config validation).
3. Docker image builds with the license flag; container starts (Xvfb socket
   appears, CoppeliaSim launches without segfault under `EndEffectorPoseViaIK`).
4. Spec-check: for an `openvla`/`oft` server (declares `position/rotation/gripper`)
   the benchmark's new EE spec produces **no key-mismatch warning**. Note: a
   `RAW`-spec server (e.g. **pi0 declares `{actions: RAW}`**) will still log
   "no overlapping keys" — that warning is expected and harmless (warn-only), not
   a regression. Do not chase it.
5. `reach_target` smoke: sane motion, success clearly above random (validates
   IK/translation + wiring only).
6. **Grasp/orientation smoke** (`turn_tap` or `pick_up_cup`): gripper visibly
   closes on the object, orientation tracks. This is the gate for the gripper-sign
   and `pos_scale` correctness items — must pass before the full run.
7. Robustness: feed an oversized delta and confirm the §3.2 `try/except` turns it
   into an episode-level failure (not a process crash).
8. Determinism: same `--shard`/seed → identical variation/episode selection across
   two runs.
9. Full 18-task PerAct run completes; result JSON merged.
10. Compare to a published paper score for the *same* model+protocol
    (`leaderboard/data/leaderboard.json` has the references) — repo convention is
    "reproduced within ~2pp".

---

## 8. Phased plan & estimate

| Phase | Work | Files | Est. |
|---|---|---|---|
| 0 | Docker build + commit-pin PyRep/RLBench | `docker/Dockerfile.rlbench` | 0.5 d |
| 1 | Action-mode switch + `step()` decode + spec/metadata (§3) | `src/vla_eval/benchmarks/rlbench/benchmark.py` | 0.5–1 d |
| 2 | `smoke_test.yaml` + real `eval.yaml` + README fix (§4.1) | `configs/benchmarks/rlbench/*` | 0.5 d |
| 3 | Per-model config + two-stage smoke→full eval; **resolve correctness items**: gripper sign per model, `pos_scale`, IK-error rate, orientation sanity (§4.2, §6, §7) | `configs/model_servers/<model>/rlbench.yaml` | **1–3 d (long pole)** |
| 4 | `docs/reproductions/rlbench.md`, update reproductions README, badge `◇`→`✓`, leaderboard | docs / leaderboard | 0.5 d |
| | **Total (first model)** | | **~3–5 d** |

**Phase 3 is the long pole and is a correctness investigation, not tuning.** The
one-file code change (Phase 1) is genuinely small, but getting a *trustworthy*
score requires empirically nailing gripper sign, `pos_scale`, frame, and IK
stability on grasp/orientation tasks — `reach_target` cannot validate any of
these. Budget conservatively; the earlier "~2 d, all tuning-level" framing was
optimistic and is corrected here. Additional models afterward are mostly
config-only (Phase 3 repeated) **but each still needs its own gripper-sign and
`pos_scale` check** — these are per-model, not one-time.

---

## 9. Risk register

**The architectural thesis is sound (independently verified, §11). The residual
risks are correctness items that the naive smoke task hides — not benign tuning.**

| Risk | Severity | Mitigation |
|---|---|---|
| Rotation-frame: EE-frame deltas composed as world-frame product | **Correctness** | `frame=RelativeFrame.EE` (§3.1) — decided up front, NOT a drift workaround. Validate on an orientation task, not `reach_target` |
| Gripper sign inverted vs RLBench `Discrete` | **Correctness** | Per-model mapping read from server convention; default for OpenVLA/OFT is `binary_close_positive` → inverted (§3.2). Validate on a grasp task |
| Unscaled delta magnitude → IK divergence / wrong reach | **Correctness** | `pos_scale` constructor arg; tune on grasp task; `try/except` so a bad action fails the episode, not the process (§3.2) |
| `reach_target` smoke is blind to the three items above | **Process** | Mandatory two-stage smoke incl. grasp/orientation task before full run (§6, §7) |
| `RAW`-spec server (pi0) logs a spurious "no overlapping keys" warning | Cosmetic | Expected & harmless (warn-only); documented in §7 so it isn't chased |
| `EndEffectorPoseViaIK` instability under Xvfb | Tuning | Validate Phase 0; `ee_planning_delta` fallback (slower, robust) |
| Continuous high-freq deltas vs IK solve cost | Tuning | Acceptable for sync runner; planning mode for keyframe models |
| No apples-to-apples published baseline for some models | Process | First model = one with a clear PerAct-protocol paper number in `leaderboard.json` |
| Non-reproducible Docker (unpinned PyRep/RLBench; local HEAD already drifted — `ERJointViaIK` present) | Reproducibility | Pin commits in Phase 0 (§5) |

---

## 10. References (exact locations)

- Misread vs reality: `src/vla_eval/specs.py:1-7, 110-136` (warn-only); spec dict
  is HELLO-handshake docs, not wire format.
- Reference decode pattern: `src/vla_eval/benchmarks/libero/benchmark.py:211-226`.
- RLBench outlier: `src/vla_eval/benchmarks/rlbench/benchmark.py:69, 94-97`
  (wrong mode), `:123-135` (missing decode).
- RLBench native EE modes: `/home/theo_lab/RLBench/rlbench/action_modes/arm_action_modes.py`
  — `EndEffectorPoseViaIK` / `EndEffectorPoseViaPlanning`: `action_shape=(7,)`,
  `assert_unit_quaternion(action[3:])`, `absolute_mode` param,
  `calculate_delta_pose` for delta mode.
- Quaternion order match: `scipy.spatial.transform.Rotation.as_quat()` = `[x,y,z,w]`
  == `specs.ROTATION_QUAT` (`quaternion_xyzw`) == RLBench `assert_unit_quaternion`.
  Verified empirically on scipy 1.11.4; the scalar-last (xyzw) default is stable
  across scipy versions (≥1.14 adds an opt-in `scalar_first` kwarg; default
  unchanged). **Note:** the RLBench container ships its own scipy inside the
  Python 3.8 conda env — the xyzw guarantee holds regardless, but the decode runs
  there, not in the host env.
- Axis-angle == rotation vector convention corroborated by
  `src/vla_eval/rotation.py` (`quat_to_axisangle` → `angle*axis`) and CALVIN's
  `axisangle_to_matrix(raw[3:6])` use of the same model outputs.
- Gripper semantics: RLBench `Discrete` in
  `/home/theo_lab/RLBench/rlbench/action_modes/gripper_action_modes.py`
  (`1.0`=open, `0.0`=closed); OpenVLA wire convention `src/vla_eval/model_servers/openvla.py:144`.
- 18-task PerAct subset: `leaderboard/benchmarks/rlbench.md`.
- Docker/license: `docker/Dockerfile.rlbench`, `docker/build.sh`,
  `docker/push.sh` (`NO_REDIST`), `docker/rlbench_entrypoint.sh`.
- Paper baselines: `leaderboard/data/leaderboard.json` (all `rlbench` entries are
  `reported_paper`, none harness-measured).

---

## 11. Independent verification record

This plan's central thesis and every load-bearing claim were independently
cross-checked by a separate Opus reviewer against the actual source (harness,
RLBench at `/home/theo_lab/RLBench/`, configs, git history) on 2026-05-19.

**Confirmed (verbatim against code):**

- Spec dict is HELLO-handshake diagnostic only; `check_specs` returns warnings,
  `orchestrator.py:184-188` only `logger.warning`s them — never converts/blocks.
  Runtime payload is a flat `{"actions": array}` (openvla.py:145, pi0.py:141,
  oft.py:191).
- LIBERO/CALVIN/ManiSkill2/SimplerEnv all self-decode `action["actions"]` in
  `step()`; RLBench is the lone outlier hardcoding `JointVelocity`.
- RLBench EE modes: `action_shape=(7,)`, `assert_unit_quaternion(action[3:])`,
  `absolute_mode`/`frame`/`calculate_delta_pose` logic; `MoveArmThenGripper(...,
  Discrete())` → 8D. `as_quat()` xyzw verified.
- Docker/license/NO_REDIST, 18-task list, "136 entries all reported_paper",
  leaderboard mismatch — all exact.

**Defects the review found in the first draft — now FIXED in this document:**

1. Rotation frame: original recommended `EndEffectorPoseViaIK(absolute_mode=False)`
   (default `frame=WORLD`) composes EE-frame rotation deltas as a world-frame
   product → systematically wrong orientation. **Fixed:** §3.1 now mandates
   `frame=RelativeFrame.EE` as a correctness decision.
2. Gripper default `raw[6]>=0→open` was **inverted** for the recommended first
   models (OpenVLA/OFT are `binary_close_positive`). **Fixed:** §3.2 now uses
   `grip = 0.0 if raw[6] >= 0 else 1.0` with full derivation.
3. `reach_target` smoke cannot catch (1) or (2). **Fixed:** §6/§7 mandate a
   two-stage smoke incl. a grasp/orientation task.
4. No delta scaling, no IK-error handling in `step()`. **Fixed:** §3.2 adds
   `pos_scale` and `try/except` → episode-level failure, not process crash.
5. Deterministic variation was mislabeled "optional". **Fixed:** §3.4 marks it
   REQUIRED for the reproduction goal.
6. Validation checklist over-promised a clean spec check (false for pi0 `RAW`).
   **Fixed:** §7 documents the expected harmless warning.

**Reviewer's overall judgment:** architecture sound; the plan is safe to hand off
**with the above fixes applied** (they are). The top items to stress-test first
remain the three §9 *Correctness*-severity rows — they cannot be validated on
`reach_target` and are the reason Phase 3 is the long pole.
