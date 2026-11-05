"""
Taichi kernels for TI field computation — works on both GPU (CUDA) and CPU.

This file MUST NOT use `from __future__ import annotations`,
because Taichi's AST transformer needs to inspect type annotations at compile time.
"""
import os
import multiprocessing as _mp
import tempfile as _tempfile

import taichi as ti

# Taichi 离线缓存目录: 优先放在项目根目录下 (可打包部署, 跨机器复用),
# 开发模式下回退到 %TEMP% (与旧行为兼容)。
try:
    from core.paths import PROJECT_ROOT
    _TI_CACHE_DIR = os.path.join(str(PROJECT_ROOT), "ti_cache")
except Exception:
    _TI_CACHE_DIR = os.path.join(_tempfile.gettempdir(), "ti_stim_ticache")
os.makedirs(_TI_CACHE_DIR, exist_ok=True)

# --- cpu_max_num_threads 安全值 ---
# Taichi 1.7.4 在某些环境（cgroups / 虚拟机 / 打包后）cpu_count 返回异常值,
# 传 0 会导致 ThreadPool 内部断言 `this->desired_num_threads > 0` 失败直接 abort.
# 这里手动算一个安全值, 至少 1 线程. 也允许用户用 TISTIM_CPU_THREADS 显式覆盖.
_raw_cpu_count = _mp.cpu_count() or 4
_env_threads = os.environ.get("TISTIM_CPU_THREADS", "").strip()
if _env_threads.isdigit() and int(_env_threads) >= 1:
    _SAFE_THREADS = int(_env_threads)
else:
    _SAFE_THREADS = max(1, min(_raw_cpu_count, 64))   # 下限 1, 上限 64
print(f"[Taichi] CPU 逻辑核心数={_raw_cpu_count}, 使用线程数={_SAFE_THREADS}"
      f" (可用环境变量 TISTIM_CPU_THREADS 覆盖)", flush=True)

# --- 初始化策略 ---
# 硬性规则：taichi 全局只能 init 一次，第二次 ti.init 会静默毁状态或抛异常。
# 所以这里先检测"有没有人已 init 过"——有就接受它的后端，不重新 init。
#
# 典型"先 init 的"场景：
#   - Linux + SimNIBS 某些版本的依赖链（如 numba/cupy/PyCUDA）间接触发 taichi
#   - multiprocessing fork 继承了父进程的 taichi 运行时
#   - 同一进程其他模块提前 import 并 init 了 taichi
#
# Windows 桌面版 SimNIBS 不带这些依赖，所以顶层 init 就是第一次，CUDA 正常。
# 这就是 Windows 跑 GPU 正常而 Linux 总是 fallback CPU 的根因。
TAICHI_ARCH = "cpu"  # 默认值，后续按实际初始化结果更新
TAICHI_AVAILABLE = False   # 只有 init 成功才置 True


def _is_taichi_initialized():
    """检测 taichi 运行时是否已被初始化（不管谁 init 的、用的什么后端）。

    Taichi 1.7.4 内部有三层 API：ti.core._taichi_core.is_initialized()（最底层 C++ 句柄）、
    ti.core.is_initialized()（Python 封装）、ti.cfg（已 init 才有 arch 属性）。
    这里三层都试一遍，任一层成功就视为已初始化。
    """
    # 1. 最底层 C++ runtime 句柄（最可靠，不受上层 Python 封装影响）
    try:
        from taichi import _lib as _tl
        if hasattr(_tl, 'taichi_python'):
            return bool(_tl.taichi_python.is_initialized())
    except Exception:
        pass
    # 2. ti.core.is_initialized()（1.7.x Python 层暴露）
    try:
        if hasattr(ti.core, 'is_initialized'):
            return bool(ti.core.is_initialized())
    except Exception:
        pass
    # 3. ti.cfg.arch 可读取（已 init 才有有效值，未 init 会抛 AttributeError）
    try:
        _arch = ti.cfg.arch
        # ti.core.Arch 值里含 'cuda'/'cpu' 之类字符串 → 能取到就算已 init
        return True
    except Exception:
        pass
    return False


def _get_current_arch():
    """读取当前已初始化的后端名 ('cuda' / 'cpu' / ...)。"""
    try:
        arch = ti.cfg.arch
        s = str(arch).lower()
        if 'cuda' in s or 'gpu' in s:
            return 'cuda'
        if 'cpu' in s:
            return 'cpu'
        # 其他后端(opengl/vulkan/metal) 也视为 CPU 等价 —— 我们的 kernel 只依赖
        # f32 ndarray 支持，这几种后端都能跑，只是性能不同。
        return 'cpu'
    except Exception:
        return 'cpu'


# ---- 执行 init ----
_force_cpu = os.environ.get("TISTIM_FORCE_CPU", "").strip().lower() in {
    "1", "true", "yes", "on", "cpu"}

_already_init = _is_taichi_initialized()

if _already_init and not _force_cpu and _get_current_arch() != 'cuda':
    # ★ 关键兜底：已 init 过但后端不是 CUDA —— 尝试 reset 后强制 CUDA
    # （SimNIBS/Linux 场景：SimNIBS 先创建了 CUDA context，导致 taichi 自己的
    #  CUDA init 抛异常 → fallback CPU → 现在 reset 清掉旧运行时，再抢 CUDA）
    try:
        ti.reset()
    except Exception:
        try:
            ti.shutdown()
        except Exception:
            pass
    _already_init = False  # reset 后干净了，走下面的第一次 init 流程

if _already_init:
    # ★ 已被别人 init 过且不用强制切换 —— 接受它的后端，绝不第二次 ti.init
    TAICHI_ARCH = _get_current_arch()
    TAICHI_AVAILABLE = True
    print(f"[Taichi] 运行时已初始化（后端={TAICHI_ARCH}），沿用该后端（跳过二次 ti.init）",
          flush=True)
else:
    # 第一次 init：正常流程（CUDA 优先，无 GPU fallback CPU）
    try:
        if _force_cpu:
            ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False,
                    cpu_max_num_threads=_SAFE_THREADS,
                    offline_cache_file_path=_TI_CACHE_DIR)
            TAICHI_ARCH = "cpu"
            TAICHI_AVAILABLE = True
        else:
            try:
                ti.init(arch=ti.cuda, default_fp=ti.f32, debug=False,
                        offline_cache_file_path=_TI_CACHE_DIR)
                # 验证 CUDA 是否真正可用（init 不报错不代表有 GPU）
                _test = ti.ndarray(ti.f32, shape=(1,))
                _test.fill(1.0)
                del _test
                TAICHI_ARCH = "cuda"
                TAICHI_AVAILABLE = True
            except Exception:
                # 无 NVIDIA GPU 或 CUDA 驱动不可用 → 回退到 CPU 后端
                ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False,
                        cpu_max_num_threads=_SAFE_THREADS,
                        offline_cache_file_path=_TI_CACHE_DIR)
                TAICHI_ARCH = "cpu"
                TAICHI_AVAILABLE = True
    except Exception as _e:
        # init 完全失败 (极少见, 如打包环境缺 libomp 等) → 标记不可用, 上层 NumPy 兜底
        TAICHI_AVAILABLE = False
        TAICHI_ARCH = "cpu"
        import sys as _sys
        print(f"[Taichi] init failed ({_e}), falling back to NumPy.threads={_SAFE_THREADS}",
              file=_sys.stderr)


# ---------------------------------------------------------------------------
# Basic TI envelope kernel — operates on pre-split 1D component arrays
# ---------------------------------------------------------------------------
@ti.kernel
def ti_compute_ti_envelope(
    e1x: ti.types.ndarray(dtype=ti.f32, ndim=1), e1y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    e1z: ti.types.ndarray(dtype=ti.f32, ndim=1),
    e2x: ti.types.ndarray(dtype=ti.f32, ndim=1), e2y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    e2z: ti.types.ndarray(dtype=ti.f32, ndim=1),
    out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    total: ti.i32,
):
    """Fused TI max envelope kernel — entire formula in one GPU kernel."""
    for idx in range(total):
        a1x = e1x[idx]; a1y = e1y[idx]; a1z = e1z[idx]
        b1x = e2x[idx]; b1y = e2y[idx]; b1z = e2z[idx]

        n1 = ti.sqrt(a1x * a1x + a1y * a1y + a1z * a1z)
        n2 = ti.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)

        if n2 > n1:
            a1x, b1x = b1x, a1x
            a1y, b1y = b1y, a1y
            a1z, b1z = b1z, a1z
            n1, n2 = n2, n1

        d = a1x * b1x + a1y * b1y + a1z * b1z
        if d < 0.0:
            b1x = -b1x; b1y = -b1y; b1z = -b1z

        n2 = ti.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)
        d2 = a1x * b1x + a1y * b1y + a1z * b1z
        ca = d2 / (n1 * n2 + 1e-8)

        dx = a1x - b1x; dy = a1y - b1y; dz = a1z - b1z
        cx = b1y * dz - b1z * dy
        cy = b1z * dx - b1x * dz
        cz = b1x * dy - b1y * dx

        cn = ti.sqrt(cx * cx + cy * cy + cz * cz)
        dn = ti.sqrt(dx * dx + dy * dy + dz * dz)

        val = 2.0 * cn / (dn + 1e-8)
        if n2 <= n1 * ca:
            val = 2.0 * n2

        out[idx] = val


# ---------------------------------------------------------------------------
# Phase 1 pure-Taichi kernel — fused: diff lookup + current mul + TI + A_ROI
#
# Each GPU thread handles one combo:
#   - Iterates over N_cu current values
#   - For each current, iterates over N_roi elements:
#     * Looks up diff vectors from flattened tables
#     * Multiplies by current
#     * Computes TI max envelope
#     * Accumulates A_ROI (weighted sum or max)
#   - Finds best current for this combo
#
# No intermediate (N_cu × M × N_roi × 3) arrays needed!
# ---------------------------------------------------------------------------
@ti.kernel
def ti_phase1_fused(
    # Diff table for pair1: (n_elec, n_elec, N_roi) — flattened to 1D
    dt1_x: ti.types.ndarray(dtype=ti.f32, ndim=1),
    dt1_y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    dt1_z: ti.types.ndarray(dtype=ti.f32, ndim=1),
    # Diff table for pair2: (n_elec, n_elec, N_roi) — flattened to 1D
    dt2_x: ti.types.ndarray(dtype=ti.f32, ndim=1),
    dt2_y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    dt2_z: ti.types.ndarray(dtype=ti.f32, ndim=1),
    # Combo electrode indices: flattened (M*4)
    combos: ti.types.ndarray(dtype=ti.i32, ndim=1),
    # Current values: (N_cu,)
    cu_vals: ti.types.ndarray(dtype=ti.f32, ndim=1),
    cu2_vals: ti.types.ndarray(dtype=ti.f32, ndim=1),
    # Volume weights for ROI: (N_roi,)
    vol_roi: ti.types.ndarray(dtype=ti.f32, ndim=1),
    # Output: best A_ROI and best current index per combo
    out_A_ROI: ti.types.ndarray(dtype=ti.f32, ndim=1),
    out_curr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    # Dimensions
    M: ti.i32,
    N_cu: ti.i32,
    N_roi: ti.i32,
    n_elec: ti.i32,
    # Parameters
    alpha: ti.f32,
    inv_sum_vol: ti.f32,
    thres: ti.f32,
    intersection_only: ti.i32,
):
    for combo_idx in range(M):
        e1 = combos[combo_idx * 4]
        e2 = combos[combo_idx * 4 + 1]
        e3 = combos[combo_idx * 4 + 2]
        e4 = combos[combo_idx * 4 + 3]

        # Base offset into flattened diff tables
        base1 = (e1 * n_elec + e2) * N_roi
        base2 = (e3 * n_elec + e4) * N_roi

        best_A = -1.0
        best_cu = 0

        for cu_idx in range(N_cu):
            I1 = cu_vals[cu_idx]
            I2 = cu2_vals[cu_idx]

            # Accumulate A_ROI over ROI elements
            acc = 0.0
            max_val = 0.0

            for roi_idx in range(N_roi):
                idx1 = base1 + roi_idx
                idx2 = base2 + roi_idx

                # E1 = I1 * diff_table_pair1[e1, e2, roi]
                a1x = I1 * dt1_x[idx1]
                a1y = I1 * dt1_y[idx1]
                a1z = I1 * dt1_z[idx1]
                # E2 = I2 * diff_table_pair2[e3, e4, roi]
                b1x = I2 * dt2_x[idx2]
                b1y = I2 * dt2_y[idx2]
                b1z = I2 * dt2_z[idx2]

                # ---- TI max envelope (inline) ----
                n1 = ti.sqrt(a1x * a1x + a1y * a1y + a1z * a1z)
                n2 = ti.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)

                if n2 > n1:
                    a1x, b1x = b1x, a1x
                    a1y, b1y = b1y, a1y
                    a1z, b1z = b1z, a1z
                    n1, n2 = n2, n1

                d = a1x * b1x + a1y * b1y + a1z * b1z
                if d < 0.0:
                    b1x = -b1x; b1y = -b1y; b1z = -b1z

                n2 = ti.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)
                d2 = a1x * b1x + a1y * b1y + a1z * b1z
                ca = d2 / (n1 * n2 + 1e-8)

                dx = a1x - b1x; dy = a1y - b1y; dz = a1z - b1z
                cx = b1y * dz - b1z * dy
                cy = b1z * dx - b1x * dz
                cz = b1x * dy - b1y * dx

                cn = ti.sqrt(cx * cx + cy * cy + cz * cz)
                dn = ti.sqrt(dx * dx + dy * dy + dz * dz)

                val = 2.0 * cn / (dn + 1e-8)
                if n2 <= n1 * ca:
                    # single-pair-dominant (假包络): intersection_only 时置零
                    if intersection_only > 0:
                        val = 0.0
                    else:
                        val = 2.0 * n2

                # ---- Accumulate A_ROI ----
                if alpha > 0.5:
                    acc += (val ** alpha) * vol_roi[roi_idx]
                elif alpha > -0.5:
                    # alpha == 0: threshold-weighted
                    if val > thres:
                        acc += vol_roi[roi_idx]
                else:
                    # alpha == -1: max
                    if val > max_val:
                        max_val = val

            # Finalize A_ROI for this (combo, current)
            A_roi = 0.0
            if alpha > 0.5:
                A_roi = (acc * inv_sum_vol) ** (1.0 / alpha)
            elif alpha > -0.5:
                A_roi = acc * inv_sum_vol
            else:
                A_roi = max_val

            if A_roi > best_A:
                best_A = A_roi
                best_cu = cu_idx

        out_A_ROI[combo_idx] = best_A
        out_curr[combo_idx] = best_cu


# ---------------------------------------------------------------------------
# Phase 2 pure-Taichi kernel — fused: fd lookup + diff + current mul + TI + A_Other
#
# Each GPU thread handles one combo:
#   - Reads known current value (best from Phase 1)
#   - For each spatial element in sub-chunk:
#     * Looks up field data for 4 electrodes
#     * Computes diff vectors → multiply by current → TI envelope
#     * Accumulates weighted sum or tracks max
#   No intermediate (B, sub_N, 3) arrays needed!
# ---------------------------------------------------------------------------
@ti.kernel
def ti_phase2_fused(
    # Field data sub-chunk: (n_elec, sub_N, 3) — flattened per component
    fd_x: ti.types.ndarray(dtype=ti.f32, ndim=1),
    fd_y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    fd_z: ti.types.ndarray(dtype=ti.f32, ndim=1),
    # Combo electrode indices: (B*4) flattened
    combos: ti.types.ndarray(dtype=ti.i32, ndim=1),
    # Current values: (B,)
    cu_vals: ti.types.ndarray(dtype=ti.f32, ndim=1),
    cu2_vals: ti.types.ndarray(dtype=ti.f32, ndim=1),
    # Volume weights: (sub_N,)
    vol: ti.types.ndarray(dtype=ti.f32, ndim=1),
    # Output: (B,) — partial weighted sum or partial max
    out_A: ti.types.ndarray(dtype=ti.f32, ndim=1),
    # Dimensions
    B: ti.i32,
    sub_N: ti.i32,
    # Parameters
    alpha: ti.f32,
    thres: ti.f32,
    intersection_only: ti.i32,
):
    for combo_idx in range(B):
        ie1 = combos[combo_idx * 4]
        ie2 = combos[combo_idx * 4 + 1]
        ie3 = combos[combo_idx * 4 + 2]
        ie4 = combos[combo_idx * 4 + 3]
        I1 = cu_vals[combo_idx]
        I2 = cu2_vals[combo_idx]

        acc = 0.0
        max_val = 0.0

        for si in range(sub_N):
            # fd[ie1, si] - fd[ie2, si]
            idx1 = ie1 * sub_N + si
            idx2 = ie2 * sub_N + si
            a1x = I1 * (fd_x[idx1] - fd_x[idx2])
            a1y = I1 * (fd_y[idx1] - fd_y[idx2])
            a1z = I1 * (fd_z[idx1] - fd_z[idx2])

            # fd[ie3, si] - fd[ie4, si]
            idx3 = ie3 * sub_N + si
            idx4 = ie4 * sub_N + si
            b1x = I2 * (fd_x[idx3] - fd_x[idx4])
            b1y = I2 * (fd_y[idx3] - fd_y[idx4])
            b1z = I2 * (fd_z[idx3] - fd_z[idx4])

            # ---- TI max envelope (inline) ----
            n1 = ti.sqrt(a1x * a1x + a1y * a1y + a1z * a1z)
            n2 = ti.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)

            if n2 > n1:
                a1x, b1x = b1x, a1x
                a1y, b1y = b1y, a1y
                a1z, b1z = b1z, a1z
                n1, n2 = n2, n1

            d = a1x * b1x + a1y * b1y + a1z * b1z
            if d < 0.0:
                b1x = -b1x; b1y = -b1y; b1z = -b1z

            n2 = ti.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)
            d2 = a1x * b1x + a1y * b1y + a1z * b1z
            ca = d2 / (n1 * n2 + 1e-8)

            dx = a1x - b1x; dy = a1y - b1y; dz = a1z - b1z
            cx = b1y * dz - b1z * dy
            cy = b1z * dx - b1x * dz
            cz = b1x * dy - b1y * dx

            cn = ti.sqrt(cx * cx + cy * cy + cz * cz)
            dn = ti.sqrt(dx * dx + dy * dy + dz * dz)

            val = 2.0 * cn / (dn + 1e-8)
            if n2 <= n1 * ca:
                # single-pair-dominant (假包络): intersection_only 时置零
                if intersection_only > 0:
                    val = 0.0
                else:
                    val = 2.0 * n2

            # ---- Accumulate ----
            if alpha > 0.5:
                acc += (val ** alpha) * vol[si]
            elif alpha > -0.5:
                # alpha == 0: threshold-weighted
                if val > thres:
                    acc += vol[si]
            else:
                # alpha == -1: max
                if val > max_val:
                    max_val = val

        # Output partial result (Python side does final reduction)
        if alpha > 0.5:
            out_A[combo_idx] = acc
        elif alpha > -0.5:
            out_A[combo_idx] = acc
        else:
            out_A[combo_idx] = max_val


# ---------------------------------------------------------------------------
# CPU-OPTIMIZED Phase 2 kernel — loop order transposed for CPU memory hierarchy
#
# GPU kernel (above) parallelizes over combos: each combo re-reads 4 electrode
# field vectors over ~1.5M elements → ~23 GB RAM traffic per call on a typical
# mesh. Fine on GPU (300+ GB/s), catastrophic on tablet CPUs (~25 GB/s,
# memory-stalled cores → ~20% utilization, days of runtime).
#
# This kernel instead:
#   - parallel over n_blocks spatial blocks (one block per CPU thread)
#   - serial inner loops over elements then combos
#   - each block writes ONLY its own row of out_blocks (n_blocks × B) →
#     no cross-thread sharing, no contended atomics
# At a fixed element, the 75 electrode vectors (~900 B) stay in L1 while the
# combo loop runs, cutting RAM traffic by ~100x.
#
# fd:       1-D float32 view of field data (n_elec, N_total, 3) C-contiguous
# idx:      global element indices of the Other region (sub_N,) int32
# combos:   (B*4,) int32
# cu/cu2:   (B,) float32 best currents
# vol:      (sub_N,) float32 volumes aligned with idx
# out_blocks: (n_blocks*B,) float32, pre-zeroed (or -inf for alpha == -1)
# ---------------------------------------------------------------------------
@ti.kernel
def ti_phase2_cpu_blocks(
    fd: ti.types.ndarray(dtype=ti.f32, ndim=1),
    idx: ti.types.ndarray(dtype=ti.i32, ndim=1),
    combos: ti.types.ndarray(dtype=ti.i32, ndim=1),
    cu_vals: ti.types.ndarray(dtype=ti.f32, ndim=1),
    cu2_vals: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vol: ti.types.ndarray(dtype=ti.f32, ndim=1),
    out_blocks: ti.types.ndarray(dtype=ti.f32, ndim=1),
    B: ti.i32,
    sub_N: ti.i32,
    N_total: ti.i32,
    n_blocks: ti.i32,
    block_size: ti.i32,
    alpha: ti.f32,
    thres: ti.f32,
    intersection_only: ti.i32,
):
    stride_elec = N_total * 3
    for blk in range(n_blocks):
        si_lo = blk * block_size
        si_hi = si_lo + block_size
        if si_hi > sub_N:
            si_hi = sub_N
        for si in range(si_lo, si_hi):
            gi = idx[si]
            v = vol[si]
            base = gi * 3
            for c in range(B):
                e1 = combos[c * 4]
                e2 = combos[c * 4 + 1]
                e3 = combos[c * 4 + 2]
                e4 = combos[c * 4 + 3]
                I1 = cu_vals[c]
                I2 = cu2_vals[c]

                o1 = e1 * stride_elec + base
                o2 = e2 * stride_elec + base
                o3 = e3 * stride_elec + base
                o4 = e4 * stride_elec + base

                a1x = I1 * (fd[o1] - fd[o2])
                a1y = I1 * (fd[o1 + 1] - fd[o2 + 1])
                a1z = I1 * (fd[o1 + 2] - fd[o2 + 2])
                b1x = I2 * (fd[o3] - fd[o4])
                b1y = I2 * (fd[o3 + 1] - fd[o4 + 1])
                b1z = I2 * (fd[o3 + 2] - fd[o4 + 2])

                n1 = ti.sqrt(a1x * a1x + a1y * a1y + a1z * a1z)
                n2 = ti.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)

                if n2 > n1:
                    a1x, b1x = b1x, a1x
                    a1y, b1y = b1y, a1y
                    a1z, b1z = b1z, a1z
                    n1, n2 = n2, n1

                d = a1x * b1x + a1y * b1y + a1z * b1z
                if d < 0.0:
                    b1x = -b1x; b1y = -b1y; b1z = -b1z

                n2 = ti.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)
                d2 = a1x * b1x + a1y * b1y + a1z * b1z
                ca = d2 / (n1 * n2 + 1e-8)

                dx = a1x - b1x; dy = a1y - b1y; dz = a1z - b1z
                cx = b1y * dz - b1z * dy
                cy = b1z * dx - b1x * dz
                cz = b1x * dy - b1y * dx

                cn = ti.sqrt(cx * cx + cy * cy + cz * cz)
                dn = ti.sqrt(dx * dx + dy * dy + dz * dz)

                val = 2.0 * cn / (dn + 1e-8)
                if n2 <= n1 * ca:
                    if intersection_only > 0:
                        val = 0.0
                    else:
                        val = 2.0 * n2

                # 本 block 私有行, 无跨线程竞争
                key = blk * B + c
                if alpha > 0.5:
                    out_blocks[key] += (val ** alpha) * v
                elif alpha > -0.5:
                    if val > thres:
                        out_blocks[key] += v
                else:
                    cur = out_blocks[key]
                    if val > cur:
                        out_blocks[key] = val
