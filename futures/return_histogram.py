import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def histogram(mean_return, std_return, returns_pct, ax_hist):

    # Define the boundaries for bucketing (3 standard deviations)
    upper_bound = mean_return + 3 * std_return
    lower_bound = mean_return - 3 * std_return
    
    # Add a small buffer (10% of range) on each side for better visualization
    range_width = upper_bound - lower_bound
    buffer = range_width * 0.1
    plot_min = lower_bound - buffer
    plot_max = upper_bound + buffer

    # Create a copy of the returns data for modification
    modified_returns = returns_pct.copy()

    # Replace values beyond 3 std dev with the boundary values (bucketing)
    modified_returns = np.clip(modified_returns, lower_bound, upper_bound)

    # Create histogram with the modified dataset
    n, bins, patches = ax_hist.hist(modified_returns, bins=50, color='gray', alpha=0.75, 
                                range=(lower_bound, upper_bound))
    ax_hist.set_title('Distribution of Daily Returns (Outliers Bucketed)', fontsize=12)
    ax_hist.set_xlabel('Daily Returns (%)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True, alpha=0.3)

    # Set x-axis limits to the 3 std dev boundaries
    ax_hist.set_xlim(lower_bound, upper_bound)

    # Set x-axis limits to the calculated bounds
    ax_hist.set_xlim(plot_min, plot_max)

    # Create x values for the normal distribution curve (within the 3 std dev range)
    x = np.linspace(lower_bound, upper_bound, 1000)
    # Create the normal distribution based on mean and std of returns
    y = stats.norm.pdf(x, mean_return, std_return)

    # Scale the normal distribution to match histogram height
    bin_width = (upper_bound - lower_bound) / 50  # assuming 50 bins
    hist_area = len(returns_pct)  # total count of data points
    scale_factor = hist_area * bin_width

    # Plot the scaled curve
    ax_hist.plot(x, y * scale_factor, 'r-', linewidth=2, label='Normal Distribution')

    # Count the outliers that were bucketed
    outliers_beyond_upper = (returns_pct > upper_bound).sum()
    outliers_beyond_lower = (returns_pct < lower_bound).sum()

    # Add text to show how many values were bucketed on each side
    if outliers_beyond_upper > 0:
        ax_hist.text(upper_bound * 0.98, ax_hist.get_ylim()[1] * 0.9, 
                    f"{outliers_beyond_upper} values\n> +3σ bucketed", 
                    ha='right', va='top', fontsize=8, style='italic',
                    bbox=dict(facecolor='white', alpha=0.7))

    if outliers_beyond_lower > 0:
        ax_hist.text(lower_bound * 1.02, ax_hist.get_ylim()[1] * 0.9, 
                    f"{outliers_beyond_lower} values\n< -3σ bucketed", 
                    ha='left', va='top', fontsize=8, style='italic',
                    bbox=dict(facecolor='white', alpha=0.7))

    # Add reference lines for mean and standard deviation
    ax_hist.axvline(mean_return, color='green', linestyle='--', alpha=0.8, 
                    label=f'Mean: {mean_return:.2f}%')
    ax_hist.axvline(mean_return + std_return, color='red', linestyle=':', alpha=0.8,
                    label=f'+1 Std: {(mean_return + std_return):.2f}%')
    ax_hist.axvline(mean_return - std_return, color='red', linestyle=':', alpha=0.8,
                    label=f'-1 Std: {(mean_return - std_return):.2f}%')

    # Add lines for +3/-3 standard deviations (which now correspond to the limits)
    ax_hist.axvline(upper_bound, color='purple', linestyle=':', alpha=0.8,
                    label=f'+3 Std: {upper_bound:.2f}%')
    ax_hist.axvline(lower_bound, color='purple', linestyle=':', alpha=0.8,
                    label=f'-3 Std: {lower_bound:.2f}%')

    # Add legend and adjust layout
    ax_hist.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])