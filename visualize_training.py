import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import sys
import os

def parse_training_log(log_file_path):
    """Parse the training log file and extract relevant data."""
    
    data = []
    task_names = set()
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # First, find all unique task names from the Task Averages sections
    task_avg_pattern = r'Task Averages:\s+\{([^}]+)\}'
    task_matches = re.findall(task_avg_pattern, content)
    
    for task_match in task_matches:
        # Extract individual task names from the dictionary string
        task_pattern = r"'([^']+)'"
        tasks_in_match = re.findall(task_pattern, task_match)
        task_names.update(tasks_in_match)
    
    task_names = sorted(list(task_names))  # Sort for consistent ordering
    print(f"Detected tasks: {task_names}")
    
    # Use a more flexible approach: extract step info and task averages separately
    step_base_pattern = r'Step\s+(\d+)\s+\|\s+LR:\s+([\d.e+-]+)\s+\|\s+Epoch:\s+(\d+)\s+\|\s+Progress:\s+([\d.]+)%.*?Task Averages:\s+\{([^}]+)\}'
    
    matches = re.findall(step_base_pattern, content, re.DOTALL)
    
    for match in matches:
        step, lr, epoch, progress, task_dict_str = match
        
        entry = {
            'step': int(step),
            'learning_rate': float(lr),
            'epoch': int(epoch),
            'progress': float(progress),
        }
        
        # Parse the task averages dictionary string
        task_value_pattern = r"'([^']+)':\s+([\d.]+)"
        task_matches = re.findall(task_value_pattern, task_dict_str)
        
        # Add task averages
        for task_name, task_value in task_matches:
            entry[f'{task_name}_avg'] = float(task_value)
        
        # Ensure all known tasks have values (fill with NaN if missing)
        for task_name in task_names:
            if f'{task_name}_avg' not in entry:
                entry[f'{task_name}_avg'] = np.nan
        
        data.append(entry)
    
    df = pd.DataFrame(data)
    # Store task names as a proper attribute using pandas metadata
    df.attrs['task_names'] = task_names
    return df

def create_task_averages_plot(df, save_path=None):
    """Create a comprehensive plot of task averages over training steps."""
    
    task_names = df.attrs.get('task_names', [])
    # Generate distinct colors for each task
    cmap = plt.colormaps.get_cmap('tab10')  # Good for up to 10 distinct colors
    colors = [cmap(i / max(1, len(task_names) - 1)) for i in range(len(task_names))]
    
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot 1: Task averages over steps
    for i, task_name in enumerate(task_names):
        col_name = f'{task_name}_avg'
        if col_name in df.columns:
            smoothed = df[col_name].rolling(window=10, min_periods=1).mean()
            ax1.plot(df['step'], smoothed, 
                    label=f'{task_name.title()} Loss (Smoothed)', 
                    color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Task Average Loss')
    ax1.set_title('Task Averages Over Training Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig

def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python visualize_training.py <log_filename>")
        print("Note: File will be looked for in the 'tmp' directory")
        sys.exit(1)
    
    # Get filename from command line and construct full path
    log_filename = sys.argv[1]
    log_file = os.path.join("tmp", log_filename)
    
    # Check if file exists
    if not os.path.exists(log_file):
        print(f"Error: File '{log_file}' not found!")
        sys.exit(1)
    
    print(f"Parsing training log: {log_file}")
    df = parse_training_log(log_file)
    
    if df.empty:
        print("No training data found in the log file!")
        return
    
    print(f"Found {len(df)} training steps")
    
    # Generate output filename based on input filename
    base_filename = os.path.splitext(log_filename)[0]
    output_path = os.path.join("tmp", f"training_analysis_{base_filename}.png")
    
    # Create visualizations
    create_task_averages_plot(df, save_path=output_path)

if __name__ == "__main__":
    main()