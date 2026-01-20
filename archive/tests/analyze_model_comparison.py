"""
Analyze model comparison results and create comprehensive visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

sns.set_style('whitegrid')
sns.set_palette('husl')


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def extract_metrics(results):
    """Extract metrics from all models."""
    metrics = []

    for model_name, model_data in results['models'].items():
        if 'error' in model_data:
            print(f"⚠ Skipping {model_name} (error during testing)")
            continue

        stats = model_data['stats']
        samples = model_data['samples']

        # Collect all IoUs
        ious = []

        # Original prediction
        if samples['original']:
            ious.append(samples['original']['iou'])

        # Perturbed predictions
        for pert in samples['perturbations']:
            ious.append(pert['iou'])

        metrics.append({
            'model': model_name,
            'parse_success_rate': stats['parse_success_rate'] * 100,
            'avg_iou': np.mean(ious),
            'std_iou': np.std(ious),
            'median_iou': np.median(ious),
            'min_iou': np.min(ious),
            'max_iou': np.max(ious),
            'avg_time_per_pred': stats['avg_time_per_call'],
            'total_time_min': stats['total_time'] / 60,
            'total_calls': stats['total_calls'],
            'parse_failures': stats['parse_failures']
        })

    return pd.DataFrame(metrics)


def create_visualizations(results, metrics_df, output_dir='./'):
    """Create comprehensive visualizations."""
    output_dir = Path(output_dir)

    fig = plt.figure(figsize=(20, 12))

    # 1. IoU Performance Comparison (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(metrics_df))
    width = 0.35

    ax1.bar(x - width/2, metrics_df['avg_iou'], width, label='Mean IoU', alpha=0.8)
    ax1.bar(x + width/2, metrics_df['median_iou'], width, label='Median IoU', alpha=0.8)

    ax1.set_xlabel('Model')
    ax1.set_ylabel('IoU Score')
    ax1.set_title('IoU Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['model'], rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)

    # 2. IoU Distribution (Top Middle)
    ax2 = plt.subplot(2, 3, 2)

    # Collect IoU distributions per model
    for model_name, model_data in results['models'].items():
        if 'error' in model_data:
            continue

        samples = model_data['samples']
        ious = []

        if samples['original']:
            ious.append(samples['original']['iou'])
        for pert in samples['perturbations']:
            ious.append(pert['iou'])

        ax2.hist(ious, alpha=0.5, label=model_name, bins=10)

    ax2.set_xlabel('IoU Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('IoU Distribution by Model')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Parse Success Rate (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    colors = plt.cm.RdYlGn(metrics_df['parse_success_rate'] / 100)
    bars = ax3.barh(metrics_df['model'], metrics_df['parse_success_rate'], color=colors)

    ax3.set_xlabel('Success Rate (%)')
    ax3.set_title('Bbox Parsing Success Rate')
    ax3.set_xlim(0, 100)
    ax3.axvline(90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='90% threshold')
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, (model, rate) in enumerate(zip(metrics_df['model'], metrics_df['parse_success_rate'])):
        ax3.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=8)

    # 4. Inference Time (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    bars = ax4.bar(metrics_df['model'], metrics_df['avg_time_per_pred'])

    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Average Inference Time per Prediction')
    ax4.set_xticklabels(metrics_df['model'], rotation=45, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    # Color bars by speed
    max_time = metrics_df['avg_time_per_pred'].max()
    for bar, time in zip(bars, metrics_df['avg_time_per_pred']):
        bar.set_color(plt.cm.RdYlGn(1 - time/max_time))

    # Add time labels
    for i, (model, time) in enumerate(zip(metrics_df['model'], metrics_df['avg_time_per_pred'])):
        ax4.text(i, time + max_time*0.02, f'{time:.1f}s', ha='center', fontsize=8)

    # 5. Performance vs Speed Trade-off (Bottom Middle)
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(metrics_df['avg_time_per_pred'], metrics_df['avg_iou'],
                         s=200, alpha=0.6, c=range(len(metrics_df)), cmap='viridis')

    for idx, row in metrics_df.iterrows():
        ax5.annotate(row['model'],
                    (row['avg_time_per_pred'], row['avg_iou']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax5.set_xlabel('Avg Time per Prediction (s)')
    ax5.set_ylabel('Avg IoU')
    ax5.set_title('Performance vs Speed Trade-off')
    ax5.grid(alpha=0.3)

    # 6. Summary Table (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Create summary table
    table_data = []
    for idx, row in metrics_df.iterrows():
        table_data.append([
            row['model'].split(':')[0],  # Shorten name
            f"{row['avg_iou']:.3f}",
            f"{row['parse_success_rate']:.0f}%",
            f"{row['avg_time_per_pred']:.1f}s"
        ])

    table = ax6.table(cellText=table_data,
                     colLabels=['Model', 'Avg IoU', 'Success', 'Time/pred'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code by performance
    for i in range(len(table_data)):
        iou = metrics_df.iloc[i]['avg_iou']
        color = plt.cm.RdYlGn(iou)
        table[(i+1, 1)].set_facecolor(color)

    ax6.set_title('Summary Statistics', fontsize=12, weight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / 'model_comparison_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_file}")

    plt.show()


def print_detailed_report(metrics_df):
    """Print detailed text report."""
    print("\n" + "="*80)
    print("DETAILED MODEL COMPARISON REPORT")
    print("="*80)

    # Sort by average IoU
    metrics_sorted = metrics_df.sort_values('avg_iou', ascending=False)

    print(f"\n{'Rank':<6}{'Model':<20}{'Avg IoU':<10}{'Median':<10}{'Success%':<10}{'Time/pred':<12}")
    print("-"*80)

    for rank, (idx, row) in enumerate(metrics_sorted.iterrows(), 1):
        print(f"{rank:<6}{row['model']:<20}{row['avg_iou']:<10.3f}{row['median_iou']:<10.3f}"
              f"{row['parse_success_rate']:<10.1f}{row['avg_time_per_pred']:<12.2f}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Best overall
    best_iou = metrics_sorted.iloc[0]
    print(f"\n🏆 Best IoU Performance: {best_iou['model']}")
    print(f"   Avg IoU: {best_iou['avg_iou']:.3f}")
    print(f"   Median IoU: {best_iou['median_iou']:.3f}")
    print(f"   Success Rate: {best_iou['parse_success_rate']:.1f}%")

    # Fastest
    fastest = metrics_df.loc[metrics_df['avg_time_per_pred'].idxmin()]
    print(f"\n⚡ Fastest Model: {fastest['model']}")
    print(f"   Time/pred: {fastest['avg_time_per_pred']:.2f}s")
    print(f"   Avg IoU: {fastest['avg_iou']:.3f}")

    # Best trade-off (IoU per second)
    metrics_df['efficiency'] = metrics_df['avg_iou'] / metrics_df['avg_time_per_pred']
    best_tradeoff = metrics_df.loc[metrics_df['efficiency'].idxmax()]
    print(f"\n⚖️  Best Performance/Speed Trade-off: {best_tradeoff['model']}")
    print(f"   Efficiency (IoU/sec): {best_tradeoff['efficiency']:.4f}")
    print(f"   Avg IoU: {best_tradeoff['avg_iou']:.3f}")
    print(f"   Time/pred: {best_tradeoff['avg_time_per_pred']:.2f}s")

    # Ministral vs Qwen3-VL comparison
    print("\n" + "="*80)
    print("MINISTRAL vs QWEN3-VL COMPARISON")
    print("="*80)

    ministral_models = metrics_df[metrics_df['model'].str.contains('ministral')]
    qwen_models = metrics_df[metrics_df['model'].str.contains('qwen')]

    print(f"\nMinistral Models (n={len(ministral_models)}):")
    print(f"  Avg IoU: {ministral_models['avg_iou'].mean():.3f} ± {ministral_models['avg_iou'].std():.3f}")
    print(f"  Avg Success Rate: {ministral_models['parse_success_rate'].mean():.1f}%")
    print(f"  Avg Time/pred: {ministral_models['avg_time_per_pred'].mean():.2f}s")

    print(f"\nQwen3-VL Models (n={len(qwen_models)}):")
    print(f"  Avg IoU: {qwen_models['avg_iou'].mean():.3f} ± {qwen_models['avg_iou'].std():.3f}")
    print(f"  Avg Success Rate: {qwen_models['parse_success_rate'].mean():.1f}%")
    print(f"  Avg Time/pred: {qwen_models['avg_time_per_pred'].mean():.2f}s")

    # Statistical comparison
    iou_diff = qwen_models['avg_iou'].mean() - ministral_models['avg_iou'].mean()
    time_diff = qwen_models['avg_time_per_pred'].mean() - ministral_models['avg_time_per_pred'].mean()

    print(f"\nDifferences (Qwen3-VL - Ministral):")
    print(f"  IoU: {iou_diff:+.3f} ({iou_diff/ministral_models['avg_iou'].mean()*100:+.1f}%)")
    print(f"  Time: {time_diff:+.2f}s ({time_diff/ministral_models['avg_time_per_pred'].mean()*100:+.1f}%)")

    print()


def main():
    """Run complete analysis."""
    # Find the most recent results file
    results_files = list(Path('').glob('model_comparison_*.json'))
    if not results_files:
        print("✗ No model comparison results found!")
        return

    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")

    # Load and analyze
    results = load_results(latest_file)

    print(f"\nTest Configuration:")
    print(f"  Models tested: {len(results['models'])}")
    print(f"  Samples: {results['config']['n_samples']}")
    print(f"  Perturbations: {results['config']['n_perturbations']}")
    print(f"  Total predictions per model: {results['config']['n_samples'] * (1 + results['config']['n_perturbations'])}")

    # Extract metrics
    metrics_df = extract_metrics(results)

    # Print detailed report
    print_detailed_report(metrics_df)

    # Create visualizations
    create_visualizations(results, metrics_df)

    # Save metrics to CSV
    output_file = 'model_comparison_metrics.csv'
    metrics_df.to_csv(output_file, index=False)
    print(f"✓ Saved metrics to: {output_file}")


if __name__ == '__main__':
    main()
