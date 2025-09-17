- [算法简介](#算法简介)
- [功能特性](#功能特性)
- [安装方式](#安装方式)
  - [自动安装 (推荐)](#自动安装-推荐)
  - [手动安装](#手动安装)
  - [离线安装](#离线安装)
- [使用教程](#使用教程)
  - [作为python库使用](#作为python库使用)
  - [作为命令行工具使用](#作为命令行工具使用)
- [论文引用](#论文引用)
- [联系方式](#联系方式)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

[README-EN](https://github.com/ZPGuiGroupWhu/scalefc-pkg?tab=readme-ov-file) | [中文简体](https://github.com/ZPGuiGroupWhu/scalefc-pkg/blob/main/README-CN.md)

# 算法简介

**论文题目**：`ScaleFC: A scale-aware geographical flow clustering algorithm for heterogeneous origin-destination data`


本研究提出一种尺度感知的地理流聚类方法（ScaleFC），以解决长度不均、密度异质和弱连接等异质地理流特征引发的流聚类问题。该方法引入尺度因子调节不同长度流的邻居搜索范围，以兼顾长短距离流簇识别。同时，受边界检测聚类的启发，引入分割流（流簇间的边界流）识别密度异质流簇，并分离弱连接流簇。如下图所示，方法包含`4`个步骤：

- `S1`：流组识别。基于尺度因子计算流邻域搜索范围，统计邻居数量。根据$MinFlows$阈值区分特征流和噪声流，通过连接操作生成流连接簇。

- `S2`：强、弱连接簇识别。计算流簇紧凑性指标，若小于质心流邻域范围则为强连接簇并直接保留，否则为弱连接簇需进一步处理。
- `S3`：弱连接簇处理。识别局部密度变化最显著的分割流，用分割流将流簇分割为两个子簇，随后递归处理每一个子簇。
- `S4`：分割流标签分配和聚类结果输出。将分割流预分配至最邻近流簇，检验紧凑性指标约束是否满足。满足约束则保留至该簇，否则将其视为噪声。处理完成后输出最终聚类结果。

下图中使用一个示例数据展示了算法的详细处理流程。

![](https://raw.githubusercontent.com/ZPGuiGroupWhu/scalefc-pkg/refs/heads/main/data/Fig3.png)

# 功能特性

本仓库为算法的官方实现，在[ZPGuiGroupWhu/ScaleFC](https://github.com/ZPGuiGroupWhu/ScaleFC)版本的实现上改进了算法效率，降低了算法的内存消耗。需要注意的是，该算法为了加速处理，在分割流的分配步骤上做了近似计算，与原版算法得到的聚类结果会略微有一些差异。总的来说，当前版本的算法实现有以下特性：

- **高效率**：本算法支持多进程加速处理。处理约`20,000`条`OD`流只需要`25s`左右。
- **低内存**：内存使用峰值与`OD`流数量无关，内存消耗峰值约为`2GB`，确保算法可以在大部分电脑上运行。
- **多方式**：提供了命令行方式和`python`库方式使用。
- **可扩展**：算法参数提供了灵活的接口，以适应不同的应用场景。
- **易集成**：算法主要依赖科学计算库`sklearn`，支持与其他算法集成。

算法在不同规模人造数据集上的运行效率和内存消耗如下图所示：

![](https://raw.githubusercontent.com/ZPGuiGroupWhu/scalefc-pkg/refs/heads/main/data/scalefc_performance_analysis.png)

# 安装方式

> [!IMPORTANT]
> 支持的`python`版本为`3.10`及以上版本

## 自动安装 (推荐)

本项目已上传[pypi](https://pypi.org/project/scalefc/)上，支持直接从`pypi`上下载并安装。

使用`pip`的安装命令为：
```shell
pip install scalefc
```

也支持[conda](https://anaconda.org/anaconda/conda)、[uv](https://github.com/astral-sh/uv)、[pipenv](https://pipenv.pypa.io/en/latest/)、[poetry](https://python-poetry.org/)等`python`环境管理工具进行安装。

## 手动安装

这种安装方式的优点是可与`github`上算法仓库保持一致，方便更新。

安装命令为：
```shell
# 克隆仓库
git clone https://github.com/ZPGuiGroupWhu/scalefc-pkg.git
cd scalefc-pkg
# 注意，需要使用pip以editable模式安装
pip install -e .
```

之后使用`git pull`拉取最新的代码，即可获取最新的算法包。

## 离线安装

在[pypi]()上下载编译好的`wheel`包，然后使用`pip`安装即可。例如安装包为`scalefc-0.1.0-py3-none-any.whl`，则安装命令为：

```shell
pip install scalefc-0.1.0-py3-none-any.whl
```

# 使用教程

`scalefc`算法包支持两种使用方式，作为`python`库使用，和命令行方式使用。以下详细介绍这两种使用方式。

## 作为python库使用

`scalefc`算法包提供了`flow_cluster_scalefc`函数，用于对地理流数据聚类。函数中每个参数的含义如下：

```python
def flow_cluster_scalefc(
    OD: ODArray,  # OD 流矩阵，形状为 (N,4)的numpy数组，每一行为 [origin_x, origin_y, destination_x, destination_y] 的平面直角坐标信息
    scale_factor: float | None = 0.1,  # 尺度因子，取值范围 (0, 1]，用于动态计算每个流的搜索邻域范围（epsilon）
    min_flows: int = 5,  # 最小流数，形成有效簇时所需的最小流数量，小于该值的组将视为噪声
    scale_factor_func: Union[
        Literal["linear", "square", "sqrt", "tanh"],
        Callable[[np.ndarray, float], np.ndarray],
    ] = "linear",  # 控制尺度因子模型的类型，默认为linear
    fixed_eps: float | None = None,  # 固定的邻域半径 epsilon，若提供则会覆盖 scale_factor 的自动计算方式，所有的流使用固定的邻域范围
    n_jobs: int | None = None,  # 并行计算时的线进程数，None 顺序执行，-1 用所有CPU，正整数指定核数
    debug: bool | Literal["simple", "full"] = False,  # 调试输出，是否打印详细的调试信息，包括中间结果和耗时
    show_time_usage: bool = False,  # 是否显示每一步的时间消耗
    **kwargs,  # 其他高级自定义参数（见下说明）
) -> Label:
    """执行 ScaleFC 算法。

        本函数实现了论文 ScaleFC: A scale-aware geographical flow clustering algorithm for heterogeneous origin-destination data 中所提出的地理流聚类算法。
        算法主要分为以下步骤：

            1. 基于空间连接机制识别流组
            2. 判断流组是强连接流组还是弱连接流组，强连接流阻直接保留为流簇，弱连接流组进行后续处理
            3. 处理弱连接流组，基于局部密度寻找流组的分割流，然后将流簇分割为2个子组，随后递归处理这2个子组
            4. 处理所有的分割流，尝试将其分配到最邻近的流簇内，并输出聚类结果


        ScaleFC 算法对于长度不均、密度异质以及弱连接的流聚类效果较好。算法细节见论文原文。

        参数:
            OD (ODArray): OD 流矩阵，numpy 数组，形状为 (N, 4)，每一行为 [origin_x, origin_y, destination_x, destination_y] 坐标
            scale_factor (float | None, optional): 尺度因子，取值范围 (0, 1]，用于计算每个流的搜索邻域范围（epsilon）。默认 0.1
            min_flows (int, optional): 形成一个流簇所需的最小流数，少于该阈值的组被视为噪声，必须为正整数。默认 5
            scale_factor_func (Union[Literal["linear", "square", "sqrt", "tanh"], Callable], optional): 控制如何用 scale_factor 计算 epsilon 的方法，可以是指定字符串或自定义函数。自定义函数需接受 (flow_data, scale_factor) 并返回 epsilon。默认 "linear"
            fixed_eps (float | None, optional): 固定邻域半径 epsilon。如果提供，则不再通过 scale_factor 计算。默认 None
            n_jobs (int | None, optional): 指定并行计算进程数，None 为顺序串行，-1 用所有CPU，正整数指定具体核数。默认 None
            debug (bool | Literal["simple", "full"], optional): 是否在运行中输出详细的调试信息，包括聚类中间结果和耗时信息。默认 False
            show_time_usage (bool, optional): 是否显示每一步的时间消耗信息。默认 False
            **kwargs: 其他高级自定义参数：
                - spatially_connected_flow_groups_label (np.ndarray, optional): 空间连接流组的预计算标签数组，须与 OD 数组长度一致
                - is_strongly_connected_flow_group_func (Callable, optional): 判断流组是否强连接组的自定义函数，参数为 (OD_subset, **params)，返回布尔值
                - can_discard_flow_group_func (Callable, optional): 判断是否丢弃某一流组的自定义函数，参数为 (OD_subset, **params)，返回布尔值

        返回值:
            Label: 聚类标签，numpy 整型数组，形状为 (N,)，N 为输入流数量。
                - 非负整数: 聚类编号 (0, 1, 2, ...)
                - -1: 噪声流（未归属于任何聚类的流）# 聚类标签数组
        异常:
            AssertionError: 若输入校验不通过（包括：OD 不是 4 列的二维数组、min_flows 非正整数、fixed_eps 非正数、n_jobs 小于 -1 等）
            ValueError: 若传入了无效的关键字参数，或 spatially_connected_flow_groups_label 长度不符。
    """
```

安装好`scalefc`库后，调用该函数的方式如下：

```python
from scalefc import flow_cluster_scalefc
import numpy as np
OD = np.random.randint(0, 100, size=(1000, 4))
label = flow_cluster_scalefc(OD, scale_factor=0.3, min_flows=5)
print(label)
```

## 作为命令行工具使用
除了使用python库的方式调用，`scalefc`还提供了命令行工具。使用命令行工具的方式如下：

```shell
# 查看命令行帮助
Usage: python -m scalefc [OPTIONS]

  ScaleFC: A scale-aware geographical flow clustering algorithm for
  heterogeneous origin-destination data

  Paper link: https://doi.org/10.1016/j.compenvurbsys.2025.102338

  This tool performs flow clustering on Origin-Destination (OD) flow data
  using the ScaleFC algorithm. The input can be: - Local files:
  /path/to/file.csv or C:\path\to\file.csv - HTTP/HTTPS URLs:
  https://example.com/data.csv - FTP URLs: ftp://server.com/data.csv - Cloud
  storage: s3://bucket/data.csv, gs://bucket/data.csv

  The input file should contain flow coordinates in the format [ox, oy, dx,
  dy] or [ox, oy, dx, dy, label].

Options:
  -f, --file, --od-file TEXT      Input OD flow file (txt or csv) or URL.
                                  Supports: 1) Local files, 2) HTTP/HTTPS
                                  URLs, 3) FTP URLs, 4) Cloud storage URLs
                                  (s3://, gs://, etc.). Must be Nx4 or Nx5
                                  matrix with columns [ox,oy,dx,dy] or
                                  [ox,oy,dx,dy,label].  [required]
  -s, --scale-factor FLOAT RANGE  Scale factor for calculating neighborhood
                                  size (0 < scale_factor <= 1).  [0.0<x<=1.0;
                                  required]
  -m, --min-flows INTEGER RANGE   Minimum number of flows required to form a
                                  cluster.  [x>=1; required]
  -sf, --scale-factor-func [linear|square|sqrt|tanh]
                                  Function to calculate epsilon from scale
                                  factor. Default: linear.
  -e, --eps, --fixed-eps FLOAT    Fixed epsilon value for neighborhood
                                  queries. If provided, overrides
                                  scale_factor.
  -n, --n-jobs INTEGER            Number of parallel jobs. None for
                                  sequential, -1 for all CPUs.
  -d, --debug                     Enable debug mode to print intermediate
                                  algorithm results.
  -su, --show-time-usage          Show time usage of each step.
  -o, --output PATH               Output file path for cluster labels. If not
                                  specified, results will be printed to
                                  stdout.
  --output-mode [append|default]  Output mode for file saving. APPEND: save
                                  ox,oy,dx,dy,label; DEFAULT: save only label.
  -o, --output PATH               Output file path for cluster labels. If not
                                  specified, results will be printed to
                                  stdout.
  --output-mode [append|default]  Output mode for file saving. APPEND: save
  -o, --output PATH               Output file path for cluster labels. If not
                                  specified, results will be printed to
  -o, --output PATH               Output file path for cluster labels. If not
                                  specified, results will be printed to
                                  stdout.
  --output-mode [append|default]  Output mode for file saving. APPEND: save
                                  ox,oy,dx,dy,label; DEFAULT: save only label.
                                  Default: DEFAULT.
  --stdout-format [list|json|default]
                                  Format for stdout output. LIST: Python list
                                  string, JSON: JSON object with 'label' key,
                                  DEFAULT: human-readable format.
  -v, --verbose                   Enable verbose mode to show detailed
                                  processing information.
  -h, --help                      Show this message and exit.
```

命令行参数与`flow_cluster_scalefc`函数参数含义是完全一致的。注意，使用`-f`指定`OD`文件路径，该路径可以是本地路径，可以是网络路径。并且，文件的格式必须要满足以下两个条件之一才可以被执行：  
- 文件头为`ox,oy,dx,dy`
- 文件头为`ox,oy,dx,dy,label`（这里的`label`表示数据的真实标签）

必须将待处理`OD`流数据的文件格式统一为上述格式后，才能使用命令行调用处理。

调用的示例如下：

```shell
$ python -m scalefc --file https://raw.githubusercontent.com/ZPGuiGroupWhu/scalefc-pkg/refs/heads/main/data/DataA.txt --scale-factor 0.2 --min-flows 5 --n-jobs 4 --debug --stdout-format list

2025-08-21 20:36:35 - DEBUG - Start ScaleFC algorithm on 300 flows, scale factor: 0.2, min flows: 5.
2025-08-21 20:36:35 - DEBUG - Initially, there are 9 spatially-connected flow groups.
2025-08-21 20:36:35 - DEBUG - Process flow groups in parallel.
2025-08-21 20:36:40 - DEBUG - There are no partitioning flows.
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
```

示例中的`DataA.txt`文件可以点击[链接](https://raw.githubusercontent.com/ZPGuiGroupWhu/scalefc-pkg/refs/heads/main/data/DataA.txt)下载。

# 论文引用

论文引用格式如下：

> Chen, H., Gui, Z., Peng, D., Liu, Y., Ma, Y., & Wu, H. (2025). ScaleFC: A scale-aware geographical flow clustering algorithm for heterogeneous origin-destination data. Computers, Environment and Urban Systems, 122, 102338. https://doi.org/10.1016/j.compenvurbsys.2025.102338

# 联系方式

对算法、论文以及该仓库有任何疑问，可以发邮件至`chen_huan@whu.edu.cn`咨询

# 贡献指南

请参考[Contributing to a project - GitHub Docs](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)对本项目进行贡献。如有疑问，可通过`issue`或者邮件的方式联系

# 许可证

该仓库遵循`MIT`许可证
