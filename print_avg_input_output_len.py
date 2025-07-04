#!/usr/bin/env python3
"""
Script to analyze average input and output lengths for greedy results 
on AIME24 and AIME25 datasets across different precisions using the Qwen3-8B tokenizer.
"""

import json
import os
import glob
from pathlib import Path
from collections import defaultdict
import statistics
from transformers import AutoTokenizer

def load_tokenizer():
    """Load the Qwen3-8B tokenizer from ~/models directory."""
    tokenizer_path = os.path.expanduser("~/models/Qwen3-8B")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        print(f"✓ Loaded tokenizer from: {tokenizer_path}")
        return tokenizer
    except Exception as e:
        print(f"✗ Error loading tokenizer from {tokenizer_path}: {e}")
        return None

def extract_precision_from_path(path):
    """Extract precision information from directory path."""
    path_str = str(path)
    if "float32" in path_str:
        return "float32"
    elif "float16" in path_str:
        return "float16"
    elif "bfloat16" in path_str:
        return "bfloat16"
    elif "naive_fp8" in path_str:
        return "naive_fp8"
    elif "provided_fp8" in path_str:
        return "provided_fp8"
    return "unknown"

def get_text_from_conversation(input_conversation):
    """Extract text from input conversation."""
    if not input_conversation or len(input_conversation) == 0:
        return ""
    
    # Get the user message (first message should be from user)
    user_message = input_conversation[0].get("content", "")
    return user_message

def get_text_from_responses(responses):
    """Extract text from responses."""
    if not responses or len(responses) == 0:
        return ""
    
    # Get the first response content
    response_content = responses[0].get("content", "")
    return response_content

def tokenize_text(tokenizer, text):
    """Tokenize text and return token count."""
    if not text or not tokenizer:
        return 0
    
    try:
        # Tokenize without adding special tokens for pure content length
        tokens = tokenizer(text, add_special_tokens=False)
        return len(tokens['input_ids'])
    except Exception as e:
        print(f"Warning: Error tokenizing text: {e}")
        return 0

def analyze_results_file(file_path, tokenizer):
    """Analyze a single results.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        input_lengths = []
        output_lengths = []
        
        for key, entry in data.items():
            # Extract input text
            input_conversation = entry.get("input_conversation", [])
            input_text = get_text_from_conversation(input_conversation)
            input_length = tokenize_text(tokenizer, input_text)
            
            # Extract output text
            responses = entry.get("responses", [])
            output_text = get_text_from_responses(responses)
            output_length = tokenize_text(tokenizer, output_text)
            
            if input_length > 0:  # Only count valid entries
                input_lengths.append(input_length)
            if output_length > 0:
                output_lengths.append(output_length)
        
        return input_lengths, output_lengths
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []

def find_results_files(results_dir, dataset_pattern):
    """Find all results.json files for a given dataset pattern."""
    results_files = []
    pattern = os.path.join(results_dir, dataset_pattern)
    
    for dir_path in glob.glob(pattern):
        if os.path.isdir(dir_path):
            # Look for subdirectories containing results.json
            for subdir in os.listdir(dir_path):
                subdir_path = os.path.join(dir_path, subdir)
                if os.path.isdir(subdir_path):
                    results_file = os.path.join(subdir_path, "results.json")
                    if os.path.exists(results_file):
                        precision = extract_precision_from_path(dir_path)
                        results_files.append((results_file, precision))
    
    return results_files

def compute_stats(lengths):
    """Compute statistics for a list of lengths."""
    if not lengths:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0}
    
    return {
        "count": len(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths)
    }

def print_results_table(dataset_name, results):
    """Print results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} RESULTS")
    print(f"{'='*60}")
    
    # Header
    print(f"{'Precision':<15} {'Input Avg':<12} {'Output Avg':<12} {'Input Med':<12} {'Output Med':<12}")
    print(f"{'-'*60}")
    
    # Sort by precision name for consistent output
    for precision in sorted(results.keys()):
        input_stats = results[precision]['input']
        output_stats = results[precision]['output']
        
        print(f"{precision:<15} "
              f"{input_stats['mean']:<12.1f} "
              f"{output_stats['mean']:<12.1f} "
              f"{input_stats['median']:<12.1f} "
              f"{output_stats['median']:<12.1f}")

def main():
    """Main function to analyze results."""
    # Load tokenizer
    tokenizer = load_tokenizer()
    if not tokenizer:
        print("Failed to load tokenizer. Exiting.")
        return
    
    results_dir = "results"
    
    # Analyze AIME24 results
    print("Analyzing AIME24 results...")
    aime24_files = find_results_files(results_dir, "greedy-aime24-*")
    aime24_results = defaultdict(lambda: {"input": [], "output": []})
    
    for file_path, precision in aime24_files:
        print(f"Processing {precision}: {file_path}")
        input_lengths, output_lengths = analyze_results_file(file_path, tokenizer)
        aime24_results[precision]["input"].extend(input_lengths)
        aime24_results[precision]["output"].extend(output_lengths)
    
    # Analyze AIME25 results
    print("\nAnalyzing AIME25 results...")
    aime25_files = find_results_files(results_dir, "greedy-aime25-*")
    aime25_results = defaultdict(lambda: {"input": [], "output": []})
    
    for file_path, precision in aime25_files:
        print(f"Processing {precision}: {file_path}")
        input_lengths, output_lengths = analyze_results_file(file_path, tokenizer)
        aime25_results[precision]["input"].extend(input_lengths)
        aime25_results[precision]["output"].extend(output_lengths)
    
    # Analyze MATH500 results
    print("\nAnalyzing MATH500 results...")
    math500_files = find_results_files(results_dir, "greedy-math500-*")
    math500_results = defaultdict(lambda: {"input": [], "output": []})
    
    for file_path, precision in math500_files:
        print(f"Processing {precision}: {file_path}")
        input_lengths, output_lengths = analyze_results_file(file_path, tokenizer)
        math500_results[precision]["input"].extend(input_lengths)
        math500_results[precision]["output"].extend(output_lengths)
    
    # Compute and display statistics
    aime24_stats = {}
    for precision in aime24_results:
        aime24_stats[precision] = {
            "input": compute_stats(aime24_results[precision]["input"]),
            "output": compute_stats(aime24_results[precision]["output"])
        }
    
    aime25_stats = {}
    for precision in aime25_results:
        aime25_stats[precision] = {
            "input": compute_stats(aime25_results[precision]["input"]),
            "output": compute_stats(aime25_results[precision]["output"])
        }
    
    math500_stats = {}
    for precision in math500_results:
        math500_stats[precision] = {
            "input": compute_stats(math500_results[precision]["input"]),
            "output": compute_stats(math500_results[precision]["output"])
        }
    
    # Print results
    print_results_table("AIME24", aime24_stats)
    print_results_table("AIME25", aime25_stats)
    print_results_table("MATH500", math500_stats)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_precisions = set(aime24_stats.keys()) | set(aime25_stats.keys()) | set(math500_stats.keys())
    
    print(f"\n{'Dataset':<10} {'Precision':<15} {'Avg Input':<12} {'Avg Output':<12}")
    print(f"{'-'*50}")
    
    for precision in sorted(all_precisions):
        if precision in aime24_stats:
            stats = aime24_stats[precision]
            print(f"{'AIME24':<10} {precision:<15} "
                  f"{stats['input']['mean']:<12.1f} "
                  f"{stats['output']['mean']:<12.1f}")
        
        if precision in aime25_stats:
            stats = aime25_stats[precision]
            print(f"{'AIME25':<10} {precision:<15} "
                  f"{stats['input']['mean']:<12.1f} "
                  f"{stats['output']['mean']:<12.1f}")
        
        if precision in math500_stats:
            stats = math500_stats[precision]
            print(f"{'MATH500':<10} {precision:<15} "
                  f"{stats['input']['mean']:<12.1f} "
                  f"{stats['output']['mean']:<12.1f}")
    
    print(f"\nTokenizer: Qwen3-8B")
    print(f"Total AIME24 precisions analyzed: {len(aime24_stats)}")
    print(f"Total AIME25 precisions analyzed: {len(aime25_stats)}")
    print(f"Total MATH500 precisions analyzed: {len(math500_stats)}")

if __name__ == "__main__":
    main()
