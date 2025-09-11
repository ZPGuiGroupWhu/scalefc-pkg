import sys
import numpy as np
import pandas as pd
import click
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import requests
from .scalefc import flow_cluster_scalefc

__all__ = ["flow_cluster_scalefc"]

def validate_input_path(ctx, param, value):
    """Custom validator for input file path that supports both local files and URLs.
    
    Args:
        ctx: Click context
        param: Click parameter
        value: Input value to validate
        
    Returns:
        str or Path: Validated path or URL
        
    Raises:
        click.BadParameter: If the path/URL is invalid
    """
    if not value:
        return value
        
    # Check if it's a URL
    parsed = urlparse(str(value))
    if parsed.scheme in ('http', 'https'):
        # It's a HTTP/HTTPS URL, validate accessibility
        try:
            response = requests.head(str(value), timeout=10, allow_redirects=True)
            if response.status_code >= 400:
                raise click.BadParameter(f"URL '{value}' returned status code {response.status_code}")
            return str(value)
        except requests.RequestException as e:
            raise click.BadParameter(f"Cannot access URL '{value}': {e}")
    elif parsed.scheme in ('ftp', 'ftps'):
        # FTP URLs - let pandas handle them
        return str(value)
    elif parsed.scheme and parsed.scheme not in ('file', ''):
        # Other schemes like s3://, gs:// etc. - return as string
        return str(value)
    else:
        # It's a local path, validate existence
        try:
            path = Path(value)
            if not path.exists():
                raise click.BadParameter(f"File '{value}' does not exist.")
            return path
        except Exception as e:
            raise click.BadParameter(f"Invalid path '{value}': {e}")

def log_message(message: str, level: str = "INFO", verbose: bool = True, force: bool = False):
    """Print formatted log message with timestamp and color.
    
    Args:
        message: The message to print
        level: Log level (DEBUG, INFO, ERROR)
        verbose: Whether verbose mode is enabled
        force: Force print even if verbose is False (for errors)
    """
    if not verbose and not force:
        return
        
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if level == "DEBUG":
        prefix = click.style(f"[{timestamp}] [DEBUG]", fg="green")
    elif level == "ERROR":
        prefix = click.style(f"[{timestamp}] [ERROR]", fg="red")
    else:  # INFO
        prefix = f"[{timestamp}] [INFO]"
    
    click.echo(f"{prefix} {message}")

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-f", "--file", "--od-file", "input_file",
    required=True,
    callback=validate_input_path,
    help="Input OD flow file (txt or csv) or URL. Supports: 1) Local files, 2) HTTP/HTTPS URLs, 3) FTP URLs, 4) Cloud storage URLs (s3://, gs://, etc.). Must be Nx4 or Nx5 matrix with columns [ox,oy,dx,dy] or [ox,oy,dx,dy,label]."
)
@click.option(
    "-s", "--scale-factor",
    required=True,
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=False),
    help="Scale factor for calculating neighborhood size (0 < scale_factor <= 1)."
)
@click.option(
    "-m", "--min-flows",
    required=True,
    type=click.IntRange(min=1),
    help="Minimum number of flows required to form a cluster."
)
@click.option(
    "-sf", "--scale-factor-func",
    type=click.Choice(["linear", "square", "sqrt", "tanh"], case_sensitive=False),
    default="linear",
    help="Function to calculate epsilon from scale factor. Default: linear."
)
@click.option(
    "-e", "--eps", "--fixed-eps", "fixed_eps",
    type=float,
    help="Fixed epsilon value for neighborhood queries. If provided, overrides scale_factor."
)
@click.option(
    "-n", "--n-jobs",
    type=int,
    help="Number of parallel jobs. None for sequential, -1 for all CPUs."
)
@click.option(
    "-d", "--debug",
    is_flag=True,
    help="Enable debug mode to print intermediate algorithm results."
)
@click.option(
    "-su", "--show-time-usage",
    is_flag=True,
    help="Show time usage of each step."
)
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=True),
    help="Output file path for cluster labels. If not specified, results will be printed to stdout."
)
@click.option(
    "--output-mode",
    type=click.Choice(["APPEND", "DEFAULT"], case_sensitive=False),
    default="DEFAULT",
    help="Output mode for file saving. APPEND: save ox,oy,dx,dy,label; DEFAULT: save only label. Default: DEFAULT."
)
@click.option(
    "--stdout-format",
    type=click.Choice(["LIST", "JSON", "DEFAULT"], case_sensitive=False),
    default="DEFAULT",
    help="Format for stdout output. LIST: Python list string, JSON: JSON object with 'label' key, DEFAULT: human-readable format."
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose mode to show detailed processing information."
)
def cli(
    input_file,
    scale_factor: float,
    min_flows: int,
    scale_factor_func: str,
    fixed_eps: float,
    n_jobs: int,
    debug: bool,
    show_time_usage: bool,
    output: Path,
    stdout_format: str,
    verbose: bool,
    output_mode: str
):
    """ScaleFC: A scale-aware geographical flow clustering algorithm for heterogeneous origin-destination data
    
    Paper link: https://doi.org/10.1016/j.compenvurbsys.2025.102338
    
    This tool performs flow clustering on Origin-Destination (OD) flow data using
    the ScaleFC algorithm. The input can be:
    - Local files: /path/to/file.csv or C:\\path\\to\\file.csv
    - HTTP/HTTPS URLs: https://example.com/data.csv
    - FTP URLs: ftp://server.com/data.csv
    - Cloud storage: s3://bucket/data.csv, gs://bucket/data.csv
    
    The input file should contain flow coordinates in the format [ox, oy, dx, dy] or [ox, oy, dx, dy, label].
    """
    try:
        # Load input data
        try:
            # Determine if input is URL or local path for logging
            input_type = "URL" if isinstance(input_file, str) and urlparse(input_file).scheme else "file"
            log_message(f"Reading data from {input_type}: {input_file}", "DEBUG", verbose)
            
            # Try to read with header first
            df = pd.read_csv(input_file)
            if set(df.columns[:4]) == {'ox', 'oy', 'dx', 'dy'}:
                # Has proper header
                od_data = df[['ox', 'oy', 'dx', 'dy']].values
                if len(df.columns) == 5 and 'label' in df.columns:
                    log_message(f"Input {input_type} has {len(df)} flows with existing labels.", "DEBUG", verbose)
                elif len(df.columns) == 4:
                    log_message(f"Input {input_type} has {len(df)} flows without labels.", "DEBUG", verbose)
                else:
                    raise ValueError("Invalid number of columns. Expected 4 or 5 columns.")
            else:
                # No proper header, read as numeric data
                df = pd.read_csv(input_file, header=None)
                if df.shape[1] not in [4, 5]:
                    raise ValueError(f"Invalid number of columns: {df.shape[1]}. Expected 4 or 5.")
                od_data = df.iloc[:, :4].values
                log_message(f"Input {input_type} has {len(df)} flows (no header detected).", "DEBUG", verbose)
        except Exception as e:
            log_message(f"Error reading data from {input_file}: {e}", "ERROR", verbose, force=True)
            sys.exit(1)
        
        # Validate data
        if np.any(np.isnan(od_data)) or np.any(np.isinf(od_data)):
            log_message("Input data contains NaN or infinite values.", "ERROR", verbose, force=True)
            sys.exit(1)
        
        # Convert to float32 for consistency with ODArray type
        od_data = od_data.astype(np.float32)
        
        # Validate fixed_eps if provided
        if fixed_eps is not None and fixed_eps <= 0:
            log_message("fixed_eps must be positive.", "ERROR", verbose, force=True)
            sys.exit(1)
        
        # Perform clustering
        log_message("Starting ScaleFC clustering...", "DEBUG", verbose)
        
        labels = flow_cluster_scalefc(
            OD=od_data,
            scale_factor=scale_factor,
            min_flows=min_flows,
            scale_factor_func=scale_factor_func.lower(),
            fixed_eps=fixed_eps,
            n_jobs=n_jobs,
            debug=debug,
            show_time_usage=show_time_usage
        )
        
        # Report results
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(labels == -1)
        
        log_message("Clustering completed!", "DEBUG", verbose)
        log_message(f"Number of flow clusters: {n_clusters}", "DEBUG", verbose)
        log_message(f"Number of noise flows: {n_noise}", "DEBUG", verbose)
        log_message(f"Number of feature flows: {len(labels) - n_noise}", "DEBUG", verbose)
        
        # Output results
        if output:
            try:
                # Determine output file path
                if output.is_dir():
                    # If output is a directory, generate filename based on input file
                    if isinstance(input_file, str) and urlparse(input_file).scheme:
                        # Input is URL, extract filename from URL path
                        parsed_url = urlparse(input_file)
                        url_path = Path(parsed_url.path)
                        input_stem = url_path.stem if url_path.stem else "url_data"
                    else:
                        # Input is local file
                        input_stem = input_file.stem
                    output_file = output / f"{input_stem}-clustering-result.csv"
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    # If output is a file path, use it directly
                    output_file = output
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                

                if output_mode.upper() == "APPEND":
                    result_df = pd.DataFrame({
                        'ox': od_data[:, 0],
                        'oy': od_data[:, 1], 
                        'dx': od_data[:, 2],
                        'dy': od_data[:, 3],
                        'label': labels
                    })
                else:  # DEFAULT mode
                    result_df = pd.DataFrame({
                        'label': labels
                    })
                result_df.to_csv(output_file, index=False)
                log_message(f"Results saved to: {output_file}", "DEBUG", verbose)
            except Exception as e:
                log_message(f"Error saving results: {e}", "ERROR", verbose, force=True)
                sys.exit(1)
        else:
            # Print to stdout based on format
            if stdout_format.upper() == "LIST":
                click.echo(str(labels.tolist()))
            elif stdout_format.upper() == "JSON":
                import json
                result = {"label": labels.tolist()}
                click.echo(json.dumps(result))
            else:  # DEFAULT format
                click.echo("\nCluster labels:")
                for i, label in enumerate(labels):
                    click.echo(f"Flow {i}: Cluster {label}")
    
    except Exception as e:
        log_message(f"Error during clustering: {e}", "ERROR", verbose, force=True)
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()