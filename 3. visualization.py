#3. visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def boxplot_cloud_sun(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Flow Regime", y="CLOUD_COVER", color="gray")
    plt.title("Cloud Cover by Flood Season")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Flow Regime", y="SUN_ELEVATION", color="red")
    plt.title("Sun Elevation by Flood Season")
    plt.show()


def plot_mean_reflectance(merged, bands):
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    axes = axes.flatten()

    for i, band in enumerate(bands):
        ax = axes[i]
        ax.plot(merged["Year"], merged[f"{band}_WF"], marker="o")
        ax.plot(merged["Year"], merged[f"{band}_BF"], marker="o")
    plt.tight_layout()
    plt.show()


def correlation_heatmap(df, bands):
    corr = df[bands].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr, mask=mask, annot=True,
        cmap="ocean", vmin=-1, vmax=1, linewidths=0.5
    )
    plt.tight_layout()
    plt.show()
