#!/usr/bin/env python3

import os
import json
import glob
from pathlib import Path

def get_greedy_experiment_results():
    """
    Extract accuracy results from all greedy experiments in the results directory.
    """
    results_dir = Path("results")
    
    # Find all greedy experiment directories
    greedy_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("greedy-")]
    
    results = []
    
    for exp_dir in sorted(greedy_dirs):
        # Each greedy experiment has a subdirectory containing the actual results
        subdirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            print(f"Warning: No subdirectories found in {exp_dir}")
            continue
        
        # Take the first (and typically only) subdirectory
        result_subdir = subdirs[0]
        summary_path = result_subdir / "summary.json"
        
        if not summary_path.exists():
            print(f"Warning: No summary.json found in {result_subdir}")
            continue
        
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # Extract key information
            accuracy = summary.get("accuracy", "N/A")
            dataset = summary.get("configuration", {}).get("task", {}).get("name", "unknown")
            
            # Extract dtype from experiment directory name (more reliable)
            exp_name = exp_dir.name
            # Expected format: greedy-{dataset}-{dtype}
            parts = exp_name.split("-")
            if len(parts) >= 3:
                dtype = "-".join(parts[2:])  # Handle multi-part dtypes like "naive_fp8"
            else:
                dtype = "unknown"
            
            results.append({
                "experiment": exp_dir.name,
                "dataset": dataset,
                "dtype": dtype,
                "accuracy": accuracy
            })
            
        except Exception as e:
            print(f"Error reading {summary_path}: {e}")
    
    return results

def print_results(results):
    """
    Print the results in a formatted table.
    """
    if not results:
        print("No results found!")
        return
    
    print("="*80)
    print("GREEDY EXPERIMENT ACCURACY RESULTS")
    print("="*80)
    
    # Group by dataset
    datasets = {}
    for result in results:
        dataset = result["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(result)
    
    for dataset, dataset_results in sorted(datasets.items()):
        print(f"\n{dataset.upper()} Dataset:")
        print("-" * 50)
        
        # Sort by dtype for consistent ordering
        dtype_order = ["float32", "float16", "bfloat16", "naive_fp8", "provided_fp8"]
        dataset_results.sort(key=lambda x: dtype_order.index(x["dtype"]) if x["dtype"] in dtype_order else len(dtype_order))
        
        for result in dataset_results:
            accuracy_str = f"{result['accuracy']:.4f}" if isinstance(result['accuracy'], float) else str(result['accuracy'])
            print(f"  {result['dtype']:>15} | Accuracy: {accuracy_str}")
    
    print("\n" + "="*80)
    
    # Summary table
    print("\nSUMMARY TABLE:")
    print("-" * 90)
    print(f"{'Dataset':<15} | {'Dtype':<15} | {'Accuracy':<10}")
    print("-" * 90)
    
    for result in results:
        accuracy_str = f"{result['accuracy']:.4f}" if isinstance(result['accuracy'], float) else str(result['accuracy'])
        print(f"{result['dataset']:<15} | {result['dtype']:<15} | {accuracy_str:<10}")

if __name__ == "__main__":
    results = get_greedy_experiment_results()
    print_results(results)
