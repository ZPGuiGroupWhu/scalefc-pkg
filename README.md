- [Algorithm Introduction](#algorithm-introduction)
- [Features](#features)
- [Installation](#installation)
  - [Automatic Installation (Recommended)](#automatic-installation-recommended)
  - [Manual Installation](#manual-installation)
  - [Offline Installation](#offline-installation)
- [Usage Tutorial](#usage-tutorial)
  - [Use as Python Library](#use-as-python-library)
  - [Use as Command Line Tool](#use-as-command-line-tool)
- [Paper Citation](#paper-citation)
- [Contact](#contact)
- [Contributing](#contributing)
- [License](#license)



[README-EN](https://github.com/ZPGuiGroupWhu/scalefc-pkg?tab=readme-ov-file) | [中文简体](https://github.com/ZPGuiGroupWhu/scalefc-pkg/blob/main/README-CN.md)

# Algorithm Introduction

**Paper Title**: `ScaleFC: A scale-aware geographical flow clustering algorithm for heterogeneous origin-destination data`



This study proposes a scale-aware geographical flow clustering method (ScaleFC) to address flow clustering problems caused by heterogeneous geographical flow characteristics such as uneven length, heterogeneous density, and weak connectivity. The method introduces a scale factor to adjust the neighborhood search range for flows of different lengths, balancing the identification of both short- and long-distance flow clusters. Meanwhile, inspired by boundary-seeking clustering, ScaleFC introduces partitioning flows (boundary flows between adjacent flow clusters) to identify density-heterogeneous flow clusters and separate weakly-connected flow clusters. As shown in the figure below, the method consists of `4` steps:

- `S1`: Identifying flow groups via spatial connectivity. Calculate flow neighborhood search range based on the scale factor and count the number of neighbors. Distinguish feature flows from noise flows according to $MinFlows$ threshold, and generate flow groups through connection operations.

- `S2`: Recognizing strongly-connected flow groups. Calculate the spatial compactness indicator of each flow group. If smaller than the neighborhood range of the centroid flow of the group, it is a strongly-connected group and directly retained as a flow cluster; otherwise, it is a weakly-connected group that needs further processing.
- `S3`: Handling weakly-connected flow groups. Identify partitioning flows with the most significant local density changes, use partitioning flows to split a flow group into two sub-groups, then recursively process each sub-group.
- `S4`: Reallocating partitioning flows and outputting cluster results. Pre-assign partitioning flows to the nearest flow cluster and verify whether compactness indicator constraints are satisfied. If constraints are satisfied, retain in that cluster; otherwise, treat as noise. Output final clustering results after processing is complete.

The figure below shows the detailed processing flow of the algorithm using example data.

![](https://raw.githubusercontent.com/ZPGuiGroupWhu/scalefc-pkg/refs/heads/main/data/Fig3.png)

# Features

This repository is the official implementation of the algorithm, which improves algorithm efficiency and reduces memory consumption compared to the [ZPGuiGroupWhu/ScaleFC](https://github.com/ZPGuiGroupWhu/ScaleFC) version implementation. Note that this algorithm uses approximate calculations in the partitioning flow assignment step for acceleration, so the clustering results may have slight differences from the original algorithm. Overall, the current version of the algorithm implementation has the following features:

- **High Efficiency**: This algorithm supports multi-process acceleration. Processing approximately `20,000` `OD` flows takes only about `25s`.
- **Low Memory**: Peak memory usage is independent of the number of `OD` flows, with peak memory consumption of approximately `2GB`, ensuring the algorithm can run on most computers.
- **Multiple usage**: Provides both command-line and `python` library usage methods.
- **Extensible**: Algorithm parameters provide flexible interfaces to adapt to different application scenarios.
- **Easy Integration**: The algorithm mainly depends on the scientific computing library `sklearn`, supporting integration with other algorithms.

The algorithm's runtime efficiency and memory consumption on synthetic datasets of different scales are shown in the figure below:

![](https://raw.githubusercontent.com/ZPGuiGroupWhu/scalefc-pkg/refs/heads/main/data/scalefc_performance_analysis.png)

# Installation

> [!IMPORTANT]
> Supported `python` versions are `3.10` and above

## Automatic Installation (Recommended)

This project has been uploaded to [pypi](https://pypi.org/project/scalefc/), supporting direct download and installation from `pypi`.

The installation command using `pip` is:
```shell
pip install scalefc
```

It also supports installation using `python` environment management tools such as [conda](https://anaconda.org/anaconda/conda), [uv](https://github.com/astral-sh/uv), [pipenv](https://pipenv.pypa.io/en/latest/), [poetry](https://python-poetry.org/), etc.

## Manual Installation

The advantage of this installation method is that it can stay consistent with the algorithm repository on `github`, making updates convenient.

The installation command is:
```shell
# Clone repository
git clone https://github.com/ZPGuiGroupWhu/scalefc-pkg.git
cd scalefc-pkg
# Note: need to install using pip in editable mode
pip install -e .
```

After that, use `git pull` to pull the latest code to get the latest algorithm package.

## Offline Installation

Download the compiled `wheel` package from [pypi](https://pypi.org/project/scalefc/#files) and then install using `pip`. For example, if the installation package is `scalefc-0.1.0-py3-none-any.whl`, the installation command is:

```shell
pip install scalefc-0.1.0-py3-none-any.whl
```

# Usage Tutorial

The `scalefc` package supports two usage methods: as a `python` library and as a command-line tool. The following provides detailed introductions to these two usage methods.

## Use as Python Library

The `scalefc` algorithm package provides the `flow_cluster_scalefc` function for clustering geographical flow data. The meaning of each parameter in the function is as follows:

```python
def flow_cluster_scalefc(
    OD: ODArray,  # OD flow matrix, numpy array with shape (N,4), each row contains [origin_x, origin_y, destination_x, destination_y] planar rectangular coordinate information
    scale_factor: float | None = 0.1,  # Scale factor, value range (0, 1], used to dynamically calculate the search neighborhood range (epsilon) for each flow
    min_flows: int = 5,  # Minimum number of flows, minimum number of flows required to form a valid cluster, groups with fewer than this value are considered noise
    scale_factor_func: Union[
        Literal["linear", "square", "sqrt", "tanh"],
        Callable[[np.ndarray, float], np.ndarray],
    ] = "linear",  # Controls the type of scale factor model, default is linear
    fixed_eps: float | None = None,  # Fixed neighborhood radius epsilon, if provided, it will override the automatic calculation method of scale_factor, all flows use a fixed neighborhood range
    n_jobs: int | None = None,  # Number of parallel processes, None for sequential execution, -1 for all CPUs, positive integer specifies number of cores
    debug: bool | Literal["simple", "full"] = False,  # Debug output, whether to print detailed debug information, including intermediate results and timing
    show_time_usage: bool = False,  # Whether to display time consumption for each step
    **kwargs,  # Other advanced custom parameters (see explanation below)
) -> Label:
    """Execute ScaleFC algorithm.

        This function implements the geographical flow clustering algorithm proposed in the paper ScaleFC: A scale-aware geographical flow clustering algorithm for heterogeneous origin-destination data.
        The algorithm mainly consists of the following steps:

            1. Identify flow groups based on spatial connection mechanism
            2. Determine whether flow groups are strongly connected or weakly connected flow groups, strongly connected flow groups are directly retained as flow clusters, weakly connected flow groups undergo subsequent processing
            3. Process weakly connected flow groups, find partitioning flows of flow groups based on local density, then split flow clusters into 2 sub-groups, and recursively process these 2 sub-groups
            4. Process all partitioning flows, attempt to assign them to the nearest flow cluster, and output clustering results


        ScaleFC algorithm has good clustering effects for flows with uneven lengths, density heterogeneity, and weak connections. See the original paper for algorithm details.

        Parameters:
            OD (ODArray): OD flow matrix, numpy array with shape (N, 4), each row contains [origin_x, origin_y, destination_x, destination_y] coordinates
            scale_factor (float | None, optional): Scale factor, value range (0, 1], used to calculate the search neighborhood range (epsilon) for each flow. Default 0.1
            min_flows (int, optional): Minimum number of flows required to form a flow cluster, groups with fewer than this threshold are considered noise, must be a positive integer. Default 5
            scale_factor_func (Union[Literal["linear", "square", "sqrt", "tanh"], Callable], optional): Controls how to use scale_factor to calculate epsilon, can be a specified string or custom function. Custom function needs to accept (flow_data, scale_factor) and return epsilon. Default "linear"
            fixed_eps (float | None, optional): Fixed neighborhood radius epsilon. If provided, no longer calculated through scale_factor. Default None
            n_jobs (int | None, optional): Specifies number of parallel computing processes, None for sequential serial, -1 for all CPUs, positive integer specifies specific number of cores. Default None
            debug (bool | Literal["simple", "full"], optional): Whether to output detailed debug information during runtime, including clustering intermediate results and timing information. Default False
            show_time_usage (bool, optional): Whether to display time consumption information for each step. Default False
            **kwargs: Other advanced custom parameters:
                - spatially_connected_flow_groups_label (np.ndarray, optional): Pre-computed label array for spatially connected flow groups, must be consistent with OD array length
                - is_strongly_connected_flow_group_func (Callable, optional): Custom function to determine whether a flow group is a strongly connected group, parameters are (OD_subset, **params), returns boolean value
                - can_discard_flow_group_func (Callable, optional): Custom function to determine whether to discard a certain flow group, parameters are (OD_subset, **params), returns boolean value

        Returns:
            Label: Clustering labels, numpy integer array with shape (N,), where N is the number of input flows.
                - Non-negative integers: Cluster numbers (0, 1, 2, ...)
                - -1: Noise flows (flows not belonging to any cluster) # Clustering label array
        Exceptions:
            AssertionError: If input validation fails (including: OD is not a 4-column 2D array, min_flows is not a positive integer, fixed_eps is not a positive number, n_jobs is less than -1, etc.)
            ValueError: If invalid keyword parameters are passed, or spatially_connected_flow_groups_label length does not match.
    """
```

After installing the `scalefc` library, the way to call this function is as follows:

```python
from scalefc import flow_cluster_scalefc
import numpy as np
OD = np.random.randint(0, 100, size=(1000, 4))
label = flow_cluster_scalefc(OD, scale_factor=0.3, min_flows=5)
print(label)
```

## Use as Command Line Tool
In addition to using the python library method, `scalefc` also provides a command-line tool. The way to use the command-line tool is as follows:

```shell
# View command line help
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

The command line parameters have exactly the same meaning as the `flow_cluster_scalefc` function parameters. Note that use `-f` to specify the `OD` file path, which can be a local path or a network path. Moreover, the file format must satisfy one of the following two conditions to be executed:
- File header is `ox,oy,dx,dy`
- File header is `ox,oy,dx,dy,label` (where `label` represents the true labels of the data)

The `OD` flow data file format must be unified to the above format before it can be processed using command line calls.

An example call is as follows:

```shell
$ python -m scalefc --file https://raw.githubusercontent.com/ZPGuiGroupWhu/scalefc-pkg/refs/heads/main/data/DataA.txt --scale-factor 0.2 --min-flows 5 --n-jobs 4 --debug --stdout-format list

2025-08-21 20:36:35 - DEBUG - Start ScaleFC algorithm on 300 flows, scale factor: 0.2, min flows: 5.
2025-08-21 20:36:35 - DEBUG - Initially, there are 9 spatially-connected flow groups.
2025-08-21 20:36:35 - DEBUG - Process flow groups in parallel.
2025-08-21 20:36:40 - DEBUG - There are no partitioning flows.
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
```

The `DataA.txt` file in the example can be downloaded by clicking this [link](https://raw.githubusercontent.com/ZPGuiGroupWhu/scalefc-pkg/refs/heads/main/data/DataA.txt).

# Paper Citation

The paper citation format is as follows:

> Chen, H., Gui, Z., Peng, D., Liu, Y., Ma, Y., & Wu, H. (2025). ScaleFC: A scale-aware geographical flow clustering algorithm for heterogeneous origin-destination data. Computers, Environment and Urban Systems, 122, 102338. https://doi.org/10.1016/j.compenvurbsys.2025.102338

# Contact

For any questions about the algorithm, paper, or this repository, you can send an email to `chen_huan@whu.edu.cn` for consultation.

# Contributing

Please refer to [Contributing to a project - GitHub Docs](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) to contribute to this project. If you have any questions, you can contact us through `issue` or email.

# License

This repository follows the `MIT` license