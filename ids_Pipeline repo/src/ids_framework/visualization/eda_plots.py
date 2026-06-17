"""
EDA visualization utilities for IDS Framework
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import os

from ..utils.logger import Logger


class EDAVisualizer:
    """Visualization utilities for Exploratory Data Analysis"""
    
    def __init__(self, config, logger: Logger):
        self.config = config
        self.logger = logger
        self.plot_dpi = config.get("evaluation.plot_dpi", 300)
    
    def create_pca_tsne_comparison(self, X_pca: np.ndarray, X_tsne: np.ndarray, 
                                  y_train: np.ndarray, explained_variance: np.ndarray,
                                  output_dir: str) -> None:
        """Create PCA vs t-SNE comparison plot"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # PCA plot
        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Label": y_train
        })
        
        for label, color, name in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Attack")]:
            subset = pca_df[pca_df["Label"] == label]
            axes[0].scatter(
                subset["PC1"], subset["PC2"],
                c=color, label=name, alpha=0.4, s=8, edgecolors="none"
            )
        
        axes[0].set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}% variance)", fontsize=11)
        axes[0].set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}% variance)", fontsize=11)
        axes[0].set_title("PCA — 2D Projection", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=11, markerscale=3)
        
        # t-SNE plot
        for label, color, name in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Attack")]:
            mask = y_train == label
            axes[1].scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                c=color, label=name, alpha=0.45, s=10, edgecolors="none"
            )
        
        axes[1].set_xlabel("t-SNE Dimension 1", fontsize=11)
        axes[1].set_ylabel("t-SNE Dimension 2", fontsize=11)
        axes[1].set_title(f"t-SNE — 2D Projection", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=11, markerscale=3)
        
        plt.tight_layout()
        
        path = os.path.join(output_dir, "pca_tsne_comparison.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"PCA vs t-SNE comparison saved: {path}")
    
    def create_feature_correlation_network(self, corr_matrix: pd.DataFrame, output_dir: str) -> None:
        """Create feature correlation network visualization"""
        try:
            import networkx as nx
        except ImportError:
            self.logger.warning("NetworkX not available, skipping correlation network plot")
            return
        
        # Create network from high correlations
        G = nx.Graph()
        threshold = 0.7
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=abs(corr_val))
        
        if len(G.edges()) == 0:
            self.logger.info("No high correlations found for network visualization")
            return
        
        plt.figure(figsize=(15, 12))
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue", alpha=0.8)
        
        # Draw edges with varying width based on correlation
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.6, edge_color="gray")
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
        
        plt.title(f"Feature Correlation Network (|r| ≥ {threshold})", fontsize=16, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        
        path = os.path.join(output_dir, "correlation_network.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Correlation network saved: {path}")
    
    def create_feature_analysis_dashboard(self, train_df: pd.DataFrame, output_dir: str) -> None:
        """Create comprehensive feature analysis dashboard"""
        numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["label", "difficulty"]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Select top features by variance
        feature_variances = train_df[numerical_cols].var().sort_values(ascending=False)
        top_features = feature_variances.head(16).index.tolist()
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            ax = axes[i]
            
            # Plot distribution by class
            normal_data = train_df[train_df["label"] == 0][feature]
            attack_data = train_df[train_df["label"] == 1][feature]
            
            ax.hist(normal_data, bins=30, alpha=0.6, color="#2ecc71", label="Normal", density=True)
            ax.hist(attack_data, bins=30, alpha=0.6, color="#e74c3c", label="Attack", density=True)
            
            ax.set_title(feature.replace("_", " ").title(), fontsize=10, fontweight="bold")
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
        
        plt.suptitle("Feature Analysis Dashboard - Top 16 Features by Variance", 
                    fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        path = os.path.join(output_dir, "feature_analysis_dashboard.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Feature analysis dashboard saved: {path}")
    
    def create_data_quality_report(self, train_df: pd.DataFrame, output_dir: str) -> None:
        """Create data quality visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Missing values
        missing_data = train_df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            axes[0, 0].bar(range(len(missing_data)), missing_data.values)
            axes[0, 0].set_xticks(range(len(missing_data)))
            axes[0, 0].set_xticklabels(missing_data.index, rotation=45, ha="right")
            axes[0, 0].set_title("Missing Values by Feature")
            axes[0, 0].set_ylabel("Count")
        else:
            axes[0, 0].text(0.5, 0.5, "No Missing Values", ha="center", va="center", 
                           transform=axes[0, 0].transAxes, fontsize=14)
            axes[0, 0].set_title("Missing Values Check")
        
        # Data types
        dtype_counts = train_df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct="%1.1f%%")
        axes[0, 1].set_title("Data Types Distribution")
        
        # Class balance
        class_counts = train_df["label"].value_counts()
        class_labels = ["Normal", "Attack"]
        colors = ["#2ecc71", "#e74c3c"]
        axes[0, 2].pie(class_counts.values, labels=class_labels, colors=colors, autopct="%1.1f%%")
        axes[0, 2].set_title("Class Distribution")
        
        # Feature distribution (numerical)
        numerical_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [f for f in numerical_features if f not in ["label", "difficulty"]]
        
        if len(numerical_features) > 0:
            sample_features = numerical_features[:6]
            for i, feature in enumerate(sample_features[:3]):
                ax = axes[1, i]
                ax.hist(train_df[feature], bins=30, alpha=0.7, color="steelblue", edgecolor="black")
                ax.set_title(feature.replace("_", " ").title())
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
        else:
            axes[1, 0].text(0.5, 0.5, "No Numerical Features", ha="center", va="center",
                           transform=axes[1, 0].transAxes, fontsize=14)
        
        # Remove empty subplots
        for i in range(3, 6):
            if len(numerical_features) <= 3:
                fig.delaxes(axes[1, i])
        
        plt.suptitle("Data Quality Report", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        path = os.path.join(output_dir, "data_quality_report.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Data quality report saved: {path}")
    
    def create_statistical_summary_plot(self, train_df: pd.DataFrame, output_dir: str) -> None:
        """Create statistical summary visualization"""
        numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["label", "difficulty"]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Calculate statistics
        stats_df = train_df[numerical_cols].describe().T
        stats_df["skewness"] = train_df[numerical_cols].skew()
        stats_df["kurtosis"] = train_df[numerical_cols].kurtosis()
        
        # Select interesting features
        interesting_stats = stats_df.head(12)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mean vs Std
        axes[0, 0].scatter(interesting_stats["mean"], interesting_stats["std"], alpha=0.7)
        axes[0, 0].set_xlabel("Mean")
        axes[0, 0].set_ylabel("Standard Deviation")
        axes[0, 0].set_title("Mean vs Standard Deviation")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add feature labels for outliers
        for idx, row in interesting_stats.iterrows():
            if row["mean"] > interesting_stats["mean"].quantile(0.9) or row["std"] > interesting_stats["std"].quantile(0.9):
                axes[0, 0].annotate(idx.replace("_", "\n"), (row["mean"], row["std"]), 
                                   fontsize=8, ha="center")
        
        # Skewness vs Kurtosis
        axes[0, 1].scatter(interesting_stats["skewness"], interesting_stats["kurtosis"], alpha=0.7)
        axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.5)
        axes[0, 1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        axes[0, 1].set_xlabel("Skewness")
        axes[0, 1].set_ylabel("Kurtosis")
        axes[0, 1].set_title("Skewness vs Kurtosis")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Range analysis
        ranges = interesting_stats["max"] - interesting_stats["min"]
        axes[1, 0].bar(range(len(ranges)), ranges)
        axes[1, 0].set_xticks(range(len(ranges)))
        axes[1, 0].set_xticklabels([f.replace("_", "\n") for f in ranges.index], 
                                   rotation=45, ha="right", fontsize=8)
        axes[1, 0].set_ylabel("Range (Max - Min)")
        axes[1, 0].set_title("Feature Ranges")
        
        # Distribution of means
        axes[1, 1].hist(interesting_stats["mean"], bins=15, alpha=0.7, color="steelblue", edgecolor="black")
        axes[1, 1].set_xlabel("Mean Value")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Feature Means")
        
        plt.suptitle("Statistical Summary Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        path = os.path.join(output_dir, "statistical_summary.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Statistical summary saved: {path}")
    
    def create_pair_plot_subset(self, train_df: pd.DataFrame, output_dir: str) -> None:
        """Create pair plot for subset of features"""
        # Select most important features
        numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["label", "difficulty"]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Select features with highest variance
        feature_variances = train_df[numerical_cols].var().sort_values(ascending=False)
        selected_features = feature_variances.head(5).index.tolist()
        selected_features.append("label")
        
        # Create pair plot
        pair_data = train_df[selected_features].sample(n=min(1000, len(train_df)), random_state=42)
        
        g = sns.pairplot(pair_data, hue="label", palette=["#2ecc71", "#e74c3c"], 
                         plot_kws={"alpha": 0.6, "s": 30}, diag_kws={"alpha": 0.7})
        g.fig.suptitle("Pair Plot - Top 5 Features by Variance", y=1.02)
        
        path = os.path.join(output_dir, "pair_plot.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Pair plot saved: {path}")
    
    def create_comprehensive_eda_report(self, train_df: pd.DataFrame, output_dir: str) -> None:
        """Create comprehensive EDA report combining all visualizations"""
        self.logger.section("CREATING COMPREHENSIVE EDA REPORT")
        
        # Create all EDA visualizations
        self.create_data_quality_report(train_df, output_dir)
        self.create_statistical_summary_plot(train_df, output_dir)
        self.create_feature_analysis_dashboard(train_df, output_dir)
        
        # Create summary report
        self._generate_eda_summary(train_df, output_dir)
        
        self.logger.info("Comprehensive EDA report completed")
    
    def _generate_eda_summary(self, train_df: pd.DataFrame, output_dir: str) -> None:
        """Generate EDA summary report"""
        numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["label", "difficulty"]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        summary_stats = {
            "total_samples": len(train_df),
            "total_features": len(train_df.columns) - 2,  # Exclude label and difficulty
            "numerical_features": len(numerical_cols),
            "categorical_features": len(train_df.select_dtypes(include=["object"]).columns),
            "missing_values": int(train_df.isnull().sum().sum()),
            "class_balance": {
                "normal": int((train_df["label"] == 0).sum()),
                "attack": int((train_df["label"] == 1).sum())
            },
            "attack_types": int(train_df["attack"].nunique()) if "attack" in train_df.columns else 0
        }
        
        # Save summary
        import json
        summary_path = os.path.join(output_dir, "eda_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=4)
        
        self.logger.info(f"EDA summary saved: {summary_path}")
