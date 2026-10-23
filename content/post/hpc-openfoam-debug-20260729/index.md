+++
date = '2026-07-29T00:00:00+08:00'
lastmod = '2026-10-23T00:00:00+08:00'
title = 'HPC 环境下 OpenFOAM 与 ParaView 脚本调试记录'
slug = 'hpc-openfoam-debug-20260729'
categories = ['高性能计算']
tags = ['HPC', 'OpenFOAM', 'Slurm', 'ParaView']
description = '从作业秒退到并行求解、结果重构和 VTS→NPZ 转换，回溯 CFFF 平台上的环境、MPI、ParaView 兼容与批处理稳定性排查。'
+++

## 工作目标与记录来源

这次工作的目标，是在 CFFF 高性能计算平台上完成一个 OpenFOAM 算例的全流程运行：并行求解、重构最新结果、利用 ParaView 重采样到统一网格，再转换成供后续分析使用的 NumPy 数据。

保留下来的目录名为 `7.29debug`。其中一个重要线索是 `test.py`：它并不是 CFD 测试程序，而是一份生成《CFFF 平台 OpenFOAM 作业调试全过程与最终结果汇报》的 Word 文档脚本，内部保留了排查过程和结果摘要。结合其他 Shell、MPI 与 ParaView 脚本，可以恢复这次工作的主要技术脉络。

> 本文按工作归档日期记为 2026 年 7 月 29 日，于 2026 年 10 月回溯整理。原始作业日志、计算结果及汇报提及的最终脚本未完整保留；运行结果依据已有汇报记录，本文没有重新提交 HPC 作业。示例中的项目绝对路径已改为通用路径。

调试的核心变化，是把一次完整 CFD 作业拆成可单独检查的阶段：先确认程序和资源，再处理求解、重构、后处理兼容及数据转换。这样能够区分“程序没有启动”“求解失败”和“后处理失败”，避免每次修改都从头运行全部计算。

## 保留下来的文件及其作用

文件夹共有 **15 个文件**，其中有一份空文件，以及三个内容完全相同的运行脚本。按文件内容去重后，共有 **12 份非空材料**。

| 文件 | 可以恢复的信息 |
| --- | --- |
| `test.py` | Word 汇报生成脚本，保留 18 段调试经过、结果摘要和改进清单，是恢复过程的主要文字来源 |
| `7.29debug.txt` | 内容实际为 Bash 运行脚本，包含逐项加载环境的打点；不是终端运行日志 |
| `run.sh` | 对每项环境加载增加 BEFORE / AFTER 标记，用于定位提前退出的位置 |
| `test_mpi.sh` | 检查 Slurm 资源、OpenFOAM、Python 和 MPI，并执行 MPI hostname 测试 |
| `newcase_cfff.sh` | 将分区、求解、重构、VTS 生成和 NPZ 转换串联起来的版本 |
| `newcase_cfff - 副本 (2).sh`、`newcase_cfff - 副本 (3).sh` | 与 `newcase_cfff.sh` 字节内容相同，不能作为两份独立修订记录 |
| `Newcase_cfff .sh` | 文件名含空格且首字母大写；对 FPE 误报与正常结束判断作了更详细的说明 |
| `LSF_3.sh` | 增加浮点异常提示过滤、后处理日志和 NPZ 文件检查 |
| `LSF_4.sh` | 明确拆分 OpenFOAM、ParaView、Python 三阶段，并检查 VTS / NPZ 非空、输出失败日志尾部 |
| `uniform_grid_data_generate_paraviewscript.py` | ParaView 5.12 兼容后处理脚本，包含最新时刻选择、统一网格和重采样流程 |
| `新建 文本文档 (2).txt`、`新建 文本文档 (3).txt` | 粗糙度 NPZ 数据检查、经纬度瓦片及 9×9 分块定位笔记 |
| `新建 文本文档 (4).txt` | 本地 Docker 镜像拉取与桌面软件环境笔记，和本次 HPC 主流程没有明确关联 |
| `新建 文本文档 (5).txt` | 空文件，没有可恢复内容 |

`LSF_3.sh`、`LSF_4.sh` 的文件名虽然带有 LSF，但脚本内部使用的是 `SLURM_NTASKS`、`SLURM_JOB_ID` 等 Slurm 环境变量。本文按实际代码与汇报内容讨论 Slurm，不根据文件名判断调度器。

这些文件可以展示不同方案及修改点，但没有足够的版本元数据将每一份文件精确排成时间线。下文的排查顺序主要依据 `test.py` 中的汇报叙述。

## 初始问题：排队很久，运行却很快结束

汇报记载，最初作业在资源准备阶段等待较长，但真正进入运行后约 1～2 秒便退出，平台显示“异常”“中断”或类似状态，错误输出又不足以定位问题。

这类现象首先需要确认脚本执行到了哪里。环境模块加载、求解器启动、MPI、ParaView 和 Python 都可能失败，仅看平台状态无法区分。

### 用日志打点定位停止位置

`run.sh` 为环境加载命令分别加上前后标记，例如：

```bash
echo "[DEBUG-01 BEFORE] module load openmpi"
module load CentOS/7.9/openmpi/4.1.0
echo "[DEBUG-01 AFTER ] module load openmpi SUCCESS"
```

如果只出现 BEFORE 而没有 AFTER，可以优先检查该命令及其依赖。脚本还记录作业编号、节点、MPI 进程数、开始时间及当前算例目录，并在 `logs/case_status.txt` 中写入阶段状态。

这里的 `SUCCESS` 是脚本准备打印的文字。仅在源文件里看见这句话，不代表对应命令实际成功执行过；确认运行结果仍需要日志或作业记录。

### 模块加载成功之后，还要检查程序

早期方案同时加载多个模块，并手工 `source` OpenFOAM 的 `bashrc`。汇报中记载过 `WM_PROJECT_DIR` 与安装位置不符、找不到 `simpleFoam` 等问题。后续方案改用集群提供的 OpenFOAM 模块，并检查实际命令路径。

以下为整理后的预检示例：

```bash
module load CentOS/7.9/openfoam/V2312

module list
command -v simpleFoam
command -v mpirun
printf 'WM_PROJECT=%s\n' "${WM_PROJECT:-NA}"
printf 'WM_PROJECT_VERSION=%s\n' "${WM_PROJECT_VERSION:-NA}"
printf 'WM_PROJECT_DIR=%s\n' "${WM_PROJECT_DIR:-NA}"

simpleFoam -help
mpirun --version
```

模块能够加载、程序能够被定位、程序实际能够启动，是三个不同的检查点。这里保留的是当时环境的模块名，不表示其他集群也使用相同名称。

## Slurm 与 MPI：先做最小化测试

### 分清任务数、进程数与 CPU 配额

`test_mpi.sh` 单独检查资源分配和 MPI 启动，避免将最基础的环境问题带入完整 CFD 作业。其核心命令是：

```bash
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-NA}"
printf 'SLURM_NTASKS=%s\n' "${SLURM_NTASKS:-NA}"
printf 'SLURM_CPUS_ON_NODE=%s\n' "${SLURM_CPUS_ON_NODE:-NA}"
printf 'SLURM_JOB_CPUS_PER_NODE=%s\n' "${SLURM_JOB_CPUS_PER_NODE:-NA}"

mpirun -np "$SLURM_NTASKS" hostname
```

汇报记载，当时申请 1 个节点、36 个任务，并通过 hostname 测试确认 36 个 MPI 进程能够启动。这里应表述为“36 个 MPI 进程”，不能在缺少 CPU 绑定及资源配置记录时，将任务数直接等同于物理核心数。Slurm 中 `--ntasks` 和 `--cpus-per-task` 的含义不同。[Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)

### 分区数必须与实际并行方案对应

对于这次标准的单区域 OpenFOAM 域分解运行，需要核对：

```text
Slurm 分配的任务数
        ↕
mpirun 使用的进程数
        ↕
decomposeParDict 中的 numberOfSubdomains
```

原脚本采用 `NP=${SLURM_NTASKS:-36}`，在没有 Slurm 变量时会回退到 36。调试时便于临时运行，但正式提交前仍需要核对真实资源，不能让默认值代替资源检查。OpenFOAM 的 `numberOfSubdomains` 定义整个计算的分区数。[OpenFOAM v2312 并行说明](https://doc.openfoam.com/2312/tools/parallel/)

`LSF_4.sh` 只打印了 `numberOfSubdomains` 和 `NP`，未实现二者不一致时退出。汇报称最终增加了自动一致性检查，但对应最终脚本没有保留。因此，下面是**依据汇报补写的检查示例**，不是恢复出的原始代码：

```bash
NP="${SLURM_NTASKS:?SLURM_NTASKS is unset}"
SUBDOMAINS="$(foamDictionary system/decomposeParDict \
    -entry numberOfSubdomains -value)"

if [ "$SUBDOMAINS" != "$NP" ]; then
    printf 'Partition mismatch: subdomains=%s, MPI=%s\n' \
        "$SUBDOMAINS" "$NP" >&2
    exit 1
fi
```

## 环境拆分：OpenFOAM、ParaView 与 Python 各自检查

`LSF_4.sh` 已经明确将运行过程拆成三段。这是可以直接从保存的脚本确认的改进。

| 阶段 | 主要环境 | 任务 | 需要留下的证据 |
| --- | --- | --- | --- |
| Stage A | OpenFOAM V2312 及其 MPI 依赖 | 分区、并行求解、最新结果重构 | 程序路径、任务数、求解和重构日志 |
| Stage B | ParaView 5.12 | 读取重构结果、统一网格、输出 VTS | `pvpython` 版本、后处理日志、新生成的 VTS |
| Stage C | Anaconda 与 `py310_env` | VTS→NPZ、检查结果 | Python 路径和版本、转换日志、数组结构 |

每段开始时先 `module purge`，再加载本段所需模块，减少多个工具的路径和动态库相互干扰。`module purge` 的主要作用是卸载已加载模块，并不等于将整个 Shell 还原为全新环境；手工修改和环境激活造成的状态仍需核对。[Environment Modules 文档](https://modules.readthedocs.io/en/stable/module.html)

### Python 环境也要验证解释器

汇报记录了 Anaconda 加载后命令不可用、指定环境没有成功激活等问题。后续检查不再停留于激活命令本身，而是查看实际 Python：

```bash
module load CentOS/7.9/Anaconda3/24.5.0
source activate /path/to/py310_env

command -v python
python --version
printf 'CONDA_PREFIX=%s\n' "${CONDA_PREFIX:-NA}"
```

汇报中记录的最终解释器版本为 Python 3.10.18。该版本来自原工作记录，不是本次整理时在集群上重新检测的结果。

### 注意模块名本身存在差异

原文件保存了几种不同写法：

| 位置 | 实际写法 |
| --- | --- |
| 早期脚本中的 GLU 模块 | `CentOS/7.9/libglu-mesa` 或带版本的类似写法 |
| 较完整运行脚本中的 GLU 模块 | `CentOS/7.9/libglu1-mesa/9.0.0` |
| `LSF_4.sh` 的 ParaView 模块 | `CentOS7.9/paraview/5.12` |
| 汇报中的 ParaView 模块写法 | `CentOS/7.9/paraview/5.12` |

其中 `CentOS7.9` 与 `CentOS/7.9` 并不相同。保留文件不足以确认最终作业究竟使用哪一种名称，复用时应以集群当时的 `module avail` 和实际加载结果为准，不能仅为了格式一致而改写模块名。

## ParaView：动态库、Python 与 Trace API 分开排查

### libGLU 问题只是第一层

汇报记载，早期 `pvpython` 曾因 `libGLU.so.1` 无法加载而失败。检查模块名称与动态库路径后，才进入后处理脚本的执行阶段。

因此，“ParaView 不能运行”需要继续区分：可执行程序是否存在、动态库能否加载、Python 代码能否解析、需要的 filter 是否存在。修好动态库之后又遇到语法错误，并不代表之前的定位无效，而是执行进入了下一层。

### 原 Trace 与集群版本不一致

汇报中记录的兼容关系是：原 Trace 来自 ParaView 6.1.0，初期使用的集群环境为 ParaView 5.4.1，后续改用 ParaView 5.12.1。先遇到 Python 语法问题，再遇到类似下面的属性错误：

```text
vtkSMSourceProxy object has no attribute 'Set'
```

解决方向是核对目标版本实际暴露的属性，保留业务流程，改写调用方式。保存下来的 Python 文件顶部也明确写着：

```python
# ParaView 5.12 compatible version
# Converted from a trace originally generated using ParaView 6.1.0
```

这是脚本的兼容目标说明，不代表已经对所有 ParaView 5.12 安装方式完成验证。ParaView 提供自己的 Python 接口及跨版本 API 说明，批处理脚本应使用匹配版本的 `pvpython`。[ParaView 5.12 Python 文档](https://www.paraview.org/paraview-docs/v5.12.0/python/index.html)

### 通过最小探针检查能力

以下为可用于复查目标环境的示例：

```python
from paraview.simple import OpenFOAMReader, FastUniformGrid

grid = FastUniformGrid()
print("FastUniformGrid properties:")
print(grid.ListProperties())

reader = OpenFOAMReader(FileName="/path/to/case/case.foam")
print("OpenFOAMReader properties:")
print(reader.ListProperties())
```

这个探针分别检查 filter 是否存在及属性名称，而不是直接执行完整重采样。输入的 `.foam` 文件和算例路径需要对应真实数据。

## 重建后处理链：从 OpenFOAM 到统一网格

### 保存脚本中的处理流程

```text
OpenFOAMReader ──→ CellDatatoPointData ──────┐
                                           ├─→ ResampleWithDataset ─→ SaveData
FastUniformGrid ──→ Transform ───────────────┘
```

这里有两条输入分支：一条提供 OpenFOAM 计算得到的场，另一条构造目标网格。重采样将前者映射到后者，输出结构与目标网格相对应。[ParaView ResampleWithDataset 文档](https://www.paraview.org/paraview-docs/v5.12.0/python/paraview.simple.ResampleWithDataset.html)

原脚本最初列出多个 Cell Arrays，随后又将有效读取列表设为 `['U', 'k']`。`U` 为速度矢量场，后续转换为 `u`、`v`、`w` 三个分量；`k` 为湍动能场。重采样结果还保留有效点标记，用于区分有效与无效采样点。

### 根据脚本位置确定算例路径

保存的 ParaView 脚本通过自身路径寻找上一级算例目录：

```python
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "uniform_grid_data"
OUT_DIR.mkdir(exist_ok=True)
```

这一写法要求脚本位于算例的 `scripts/` 下。原目录只是这些文件的归档集合，不能直接把 `7.29debug` 当作完整 OpenFOAM 算例目录运行。

脚本还会查找 `.foam` 文件：没有时创建入口文件；存在多个时列出候选并选取第一个。批量运行时更适合固定唯一入口或显式指定文件，避免将“选中了一个文件”误认为已经选中了预期算例。

### 从固定时间改为最新可用结果

汇报称早期脚本曾写死 `UpdatePipeline(time=1000.0)`，而该算例最终输出到 `1500/`。保存的兼容脚本已经采用：

```python
timesteps = list(reader.TimestepValues)
if not timesteps:
    raise RuntimeError("No reconstructed timestep was found")

target_time = max(timesteps)
UpdatePipeline(time=target_time, proxy=reader)
```

这里的 `reader` 是为了简化示例使用的名称，原脚本变量名为 `a20260524foam`。后续 filter 更新也使用同一个 `target_time`。

“最新”指读取器实际发现的可用结果，而不是自动证明算例已经达到预期终点。还需要核对重构目录和求解日志。`simpleFoam` 是稳态求解器，因此不能把记录中的 `Time=1500` 直接表述为“完成 1500 秒物理时间仿真”。[OpenFOAM v2312 simpleFoam 说明](https://doc.openfoam.com/2312/tools/processing/solvers/rtm/incompressible/simpleFoam/)

### 统一网格参数与数组尺寸

| 参数 | 保存脚本中的值 |
| --- | --- |
| `WholeExtent` | `[0, 240, 0, 240, 0, 70]` |
| `Transform.Translate` | `[2800.0, -2970.0, 0.0]` |
| `Transform.Scale` | `[10.0, 10.0, 10.0]` |
| 输出目录 | `scripts/uniform_grid_data/` |
| VTS 文件 | `uniform_grid_uvwk.vts` |

按包含端点的索引范围，对应网格点数为 `241 × 241 × 71`。汇报中也记录了同样的数组形状。但这些参数并不替代 VTS 中的实际坐标检查；转换时还需要确认轴顺序、坐标范围、单位及有效区域。

## 批处理中的 ResampleWithDataset 崩溃

### 交互环境成功，不能直接外推到 Batch

汇报称，兼容脚本在交互 / VNC 环境中已经能够生成 VTS 和 NPZ，但放入正式 Slurm 作业后，仍在 `ResampleWithDataset` 阶段崩溃，错误包括：

```text
double free or corruption (!prev)
```

汇报列出的调用栈涉及 `vtkAbstractCellLocator`、`vtkCellLocator`、`vtkPolyhedron`、`vtkFiltersCore` 和 `libtbb.so.12`。据此，当时将排查重点转向 VTK/TBB 相关的并行执行路径。

这些是已有汇报保存的现象与判断，原始 backtrace 不在目录中。调用栈出现 TBB 并不能单独证明根因一定是某个具体线程竞争缺陷。

### 当时采用的单线程规避组合

汇报记录，在启动 ParaView 后处理前设置：

```bash
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export TBB_NUM_THREADS=1
export VTK_SMP_MAX_THREADS=1
```

随后交互测试得到 `PVPYTHON_EXIT=0`，并将该组合用于正式批处理。需要区分：线程限制值为 `1`，进程成功退出码为 `0`。

这一组合是原工作记录中的经验性规避措施。不同 VTK 构建可能使用不同 SMP 后端，各变量是否被实际采用，需要结合构建和运行时确认，不能认为设置四个变量就证明所有代码都只使用一个线程。[VTK SMP 框架说明](https://docs.vtk.org/en/v9.4.2/design_documents/smptools.html)

将限制放在 Stage A 的 OpenFOAM 求解结束后、Stage B 的 `pvpython` 启动前，可以保留“OpenFOAM 使用 36 个 MPI 进程，ParaView 后处理限制线程”的阶段设计。**保存的 `LSF_4.sh` 没有这四行配置**，它们仅能从汇报中恢复，不能将该文件直接称为包含全部修复的最终版本。

## 让脚本正确判断成功与失败

### 保留管道中的失败状态

求解命令需要同时显示和保存日志：

```bash
set -eo pipefail

if ! mpirun -np "$NP" simpleFoam -parallel \
    2>&1 | tee "$LOG_DIR/log.simpleFoam"; then
    echo "FAILED_SIMPLEFOAM" > "$LOG_DIR/case_status.txt"
    exit 1
fi
```

若没有 `pipefail`，管道状态通常取最后一个命令；`tee` 成功可能掩盖上游失败。启用后可以发现管道中的非零状态，但仅凭这个状态还不能区分是 MPI、求解器还是日志写入失败。[GNU Bash 管道说明](https://www.gnu.org/software/bash/manual/html_node/Pipelines)

另外，`set -e` 存在条件语句和管道等例外，不应当作所有错误都能自动处理的保证。[GNU Bash 手册](https://www.gnu.org/s/bash/manual/bash.html)

### 浮点异常提示不能直接按关键词判错

早期脚本将所有 `Floating point exception` 都视为失败，可能把“启用了浮点异常捕获”的正常提示也匹配进去。`LSF_3.sh`、`Newcase_cfff .sh` 和 `LSF_4.sh` 中可以看到逐步增加的过滤逻辑。

`LSF_4.sh` 采用的核心检查是：

```bash
if grep -Ei \
    'FOAM FATAL ERROR|MPI_ABORT|Segmentation fault|core dumped' \
    "$LOG_DIR/log.simpleFoam" > /dev/null; then
    echo "FAILED_SIMPLEFOAM_FATAL_LOG" > "$LOG_DIR/case_status.txt"
    exit 1
fi

if grep -Ei 'Floating point exception' "$LOG_DIR/log.simpleFoam" \
    | grep -Eiv 'trapping enabled|exception trapping' > /dev/null; then
    echo "FAILED_SIMPLEFOAM_FLOATING_POINT" > "$LOG_DIR/case_status.txt"
    exit 1
fi

if ! grep -Eq '^[[:space:]]*End[[:space:]]*$' \
    "$LOG_DIR/log.simpleFoam"; then
    echo "FAILED_SIMPLEFOAM_NO_END" > "$LOG_DIR/case_status.txt"
    exit 1
fi
```

这些规则用于该工作中的运行状态检查，并不是覆盖全部 OpenFOAM 错误的通用解析器。出现 `End` 说明运行正常结束到相应位置，不等于已经满足残差、守恒或物理可信度要求；这些仍需要单独检查。

### 提交前检查语法与诊断命令

汇报记载过 CRLF 换行、反斜杠续行、管道位置和 `if` 结构造成的语法错误。因此，先统一 LF 换行，再执行：

```bash
bash -n job.sh
```

`bash -n` 只检查语法，不会验证模块是否存在、命令能否启动、路径是否正确或计算能否完成。

保留的预检脚本还使用了 `simpleFoam -help | head -5` 等写法。在 `pipefail` 下，提前关闭管道可能使上游因 SIGPIPE 返回非零。整理后的示例改为先保存帮助输出，再截取文件内容，避免诊断命令本身干扰退出判断：

```bash
simpleFoam -help > "$LOG_DIR/simpleFoam-help.txt" 2>&1
sed -n '1,5p' "$LOG_DIR/simpleFoam-help.txt"
```

## 产物检查：避免把旧结果当成新结果

### 从文件存在升级到内容检查

早期脚本仅使用 `-f` 检查文件存在，`LSF_4.sh` 改用 `-s` 检查 VTS 和 NPZ 非空，并在失败时输出相关日志尾部。

但文件非空仍可能对应旧结果。汇报进一步记载：生成新 VTS 前清理旧 VTS，转换前清理旧 NPZ，转换成功后才移除中间 VTS。**这两处生成前清理没有出现在保存的 `LSF_4.sh` 中**，需要与已落实到该文件中的检查区分。

复用时也可以为每次作业使用独立输出目录，避免覆盖旧结果。无论采用哪一种方式，成功判断都需要关联本次作业，而不是只发现一个同名文件。

### NPZ 的结构验证

原始 VTS→NPZ 转换脚本没有保留，因此无法恢复字段拆分、坐标排序和 reshape 的具体实现。现有汇报记录的结果是：

| 项目 | 汇报中的结果 |
| --- | --- |
| VTS 大小 | 约 162 MB |
| NPZ 大小 | 约 53 MB |
| `u`、`v`、`w`、`k`、`mask` | 形状均为 `(241, 241, 71)` |
| 坐标轴长度 | `x`、`y` 为 241，`z` 为 71 |
| 有效网格比例 | 约 `0.96081`，即 96.08% |
| 其他字段 | `nx`、`ny`、`nz` |

下面是依据这些约定补写的**结构检查示例**，不替代缺失的转换器，也没有对原始 NPZ 实际执行：

```python
from pathlib import Path
import numpy as np

path = Path("scripts/uniform_grid_data/uniform_grid_uvwk.npz")
expected_shape = (241, 241, 71)

with np.load(path, allow_pickle=False) as data:
    required = {"x", "y", "z", "u", "v", "w", "k", "mask", "nx", "ny", "nz"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")

    for key in ("u", "v", "w", "k", "mask"):
        if data[key].shape != expected_shape:
            raise ValueError(f"Unexpected shape: {key}={data[key].shape}")

    for axis, size in zip(("x", "y", "z"), expected_shape):
        if data[axis].shape != (size,):
            raise ValueError(f"Unexpected coordinate shape: {axis}")

    valid = data["mask"].astype(bool)
    if not valid.any():
        raise ValueError("No valid sampling points")

    for key in ("u", "v", "w", "k"):
        if not np.isfinite(data[key][valid]).all():
            raise ValueError(f"Non-finite values in valid region: {key}")

    print("Shape:", expected_shape)
    print("Valid fraction:", valid.mean())
```

结构检查通过后，还需要核对坐标与数值场的对应、轴顺序、单位及采样区域。这些不能只凭文件大小或数组尺寸确认。

## 汇报记录的最终运行结果

`test.py` 中记载的成功作业编号为 **24586061**，算例为 `case0003_WD184_WS13`。整理后的结果链为：

| 阶段 | 原汇报记录 |
| --- | --- |
| 资源与 MPI | 1 个节点，36 个任务；MPI 最小测试通过 |
| OpenFOAM | V2312，36 个 MPI 进程；运行到 `Time=1500`，出现 `End` 和 `Finalising parallel run` |
| 重构 | `reconstructPar -latestTime` 成功 |
| ParaView | 5.12.1，在所记录的线程限制配置下完成重采样与保存 |
| VTS | 生成约 162 MB 的统一网格文件 |
| Python | `py310_env`，Python 3.10.18 |
| NPZ | 生成约 53 MB 文件，并记录数组尺寸和有效点比例 |
| 结束状态 | 平台显示“已完成”，脚本记录 `SUCCESS` 并以 0 退出 |

这些数值和成功状态都来自当时汇报的文字，不是从本次目录中的原始终端日志重新统计得到的。现有材料能够恢复“当时记录了怎样的结果”，无法替代对原作业输出的独立复核。

## 可以复用的流程与尚缺材料

### 运行流程

| 顺序 | 执行内容 | 继续条件 |
| --- | --- | --- |
| 预检 | Bash 语法、模块、命令路径、Slurm 任务数和分区数 | 语法和资源配置满足当前作业要求 |
| Stage A | `decomposePar` → `mpirun ... simpleFoam -parallel` → 日志判断 → `reconstructPar -latestTime` | 求解退出状态、日志及重构结果均可确认 |
| Stage B | 切换 ParaView 环境 → 按需限制线程 → 选定可用结果 → 重采样 → 保存 VTS | 本次 VTS 生成成功，且可读取 |
| Stage C | 切换 Python 环境 → VTS→NPZ → 检查字段、形状和有效值 | 本次 NPZ 内容满足约定 |
| 收尾 | 保存日志和状态；确认结果后按需清理中间文件 | 失败时仍有足够材料定位或恢复 |

原脚本在重构后就清理 `processor` 目录，在 NPZ 非空后清理 VTS。后续复用时，应结合重构字段的完整性、恢复需求和存储空间决定清理时机，不能仅因某条命令返回 0 就无条件丢弃全部排错材料。

### 文字记录与保存脚本的对应情况

| 修改 | 当前能够找到的证据 |
| --- | --- |
| 环境加载前后打点 | `run.sh` 和 `7.29debug.txt` |
| 最小化 MPI 测试 | `test_mpi.sh` |
| OpenFOAM / ParaView / Python 三阶段 | `LSF_4.sh` |
| FPE 提示过滤及 `End` 检查 | `LSF_3.sh`、`Newcase_cfff .sh`、`LSF_4.sh` |
| 最新可用时间步选择 | ParaView Python 脚本 |
| 非空产物检查、失败日志尾部 | `LSF_4.sh` |
| 四个线程限制变量 | 汇报文字与其中的流程示例；保存的 `LSF_4.sh` 未包含 |
| 分区数自动比较、旧 VTS / NPZ 生成前清理 | 汇报文字；保存的 `LSF_4.sh` 未落实这些检查 |
| 最终成功作业和结果规模 | 汇报文字；原始运行日志和产物未保留 |

汇报中提到的 `LSF_5.sh`、原 Trace 备份 `.bak_61`，以及运行脚本调用的以下文件，都不在当前文件夹中：

```text
scripts/uniform_grid_data_generate_run_cfff.sh
scripts/uniform_grid_data_convert_vts_to_npz.py
```

同时缺少算例的 `0/`、`constant/`、`system/`、网格、完整 Slurm 提交配置和输出数据。因此，这份归档适合作为调试复盘，尚不能作为完整的可运行复现包。

## 附带的数据检查笔记

两份短文本还保存了粗糙度数据的辅助检查，与 HPC 调试主线分开记录。

第一份用于查看 NPZ 的字段、形状、数据类型与数值范围。第二份根据经纬度确定 1° 瓦片，再将其划为 9×9 子块；列从西向东，行号由南向编号转换为从北向南编号。

对笔记中的测试点 `lat=31.85`、`lon=118.45`，按该函数可得：

```text
tile_id = N31E118
row_from_north = 1
col = 4
```

这是可以由保存代码直接计算的索引结果；对应粗糙度数据文件并未保留，无法重新计算其数值范围。原切片使用 `H // 9` 和 `W // 9`，如果数组尺寸不能被 9 整除，末端会留下未覆盖的行或列，复用前需要核对瓦片尺寸和边界处理。

Docker 笔记涉及本地镜像下载和桌面软件环境，没有证据表明它属于这次 CFFF 作业的修复步骤；其中历史镜像地址与系统升级建议不作为本文的 HPC 操作方案。

## 调试中留下的经验

这次工作最有价值的部分，是建立了能够逐层定位问题的检查链：先确定实际执行环境与资源，再核对程序、求解日志、后处理接口及数据产物。交互环境成功之后，还需要在正式计算节点上验证。

对于后续同类作业，应同时保存提交脚本、模块清单、解释器和可执行程序路径、各阶段日志、关键配置及输出结构检查结果。这样才能让一个 `SUCCESS` 对应到明确的运行证据，也能在脚本遗失或环境变化后恢复排查过程。
