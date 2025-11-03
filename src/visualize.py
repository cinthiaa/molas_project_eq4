"""
Esqueleto OOP para pipeline de Bike Sharing (sólo estructura, sin implementación de métodos).
Clases incluidas: DataLoader, Preprocessor, Model, Evaluator, Visualizer, Orchestrator.
"""
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
import os


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """Generates visualizations from model results and metrics."""

    def __init__(self, output_dir):
        """Initialize Visualizer with configurable output directory.
        
        Args:
            output_dir: Directory to save generated plots
        """

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set consistent style for all plots
        plt.style.use('default')
        sns.set_palette("husl")

    def plot_metrics(self, df_results: pd.DataFrame) -> Path:
        """Generate comprehensive model comparison visualizations.
        
        Args:
            df_results: DataFrame with columns: Model, Test RMSE, Test MAE, 
                       Test R2, Test MAPE, CV RMSE, Train Time (s)
        
        Returns:
            Path to saved comparison plot
            
        Raises:
            ValueError: If required columns are missing from DataFrame
        """
        self._validate_metrics_dataframe(df_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        self._plot_rmse_comparison(axes[0, 0], df_results)
        self._plot_r2_comparison(axes[0, 1], df_results)
        self._plot_mae_mape_comparison(axes[1, 0], df_results)
        self._plot_time_vs_performance(axes[1, 1], df_results)
        
        plt.tight_layout()
        output_path = self.output_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def plot_predictions(
        self,
        y_true,
        y_pred,
        title,
    ) -> Path:
        """Generate actual vs predicted values scatter plot.
        
        Args:
            y_true: Actual target values
            y_pred: Predicted target values
            title: Optional plot title
            
        Returns:
            Path to saved predictions plot
            
        Raises:
            ValueError: If input arrays have different lengths or are empty
        """
        y_true_arr, y_pred_arr = self._validate_prediction_arrays(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        ax.scatter(y_true_arr, y_pred_arr, alpha=0.6, s=20, edgecolors='k', linewidth=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true_arr.min(), y_pred_arr.min())
        max_val = max(y_true_arr.max(), y_pred_arr.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate and display metrics
        r2 = r2_score(y_true_arr, y_pred_arr)
        rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
        
        # Add metrics text box
        metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        plot_title = title or 'Actual vs Predicted Values'
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ensure equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        output_path = self.output_dir / 'predictions_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def _validate_metrics_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame contains required columns for metrics plotting."""
        required_columns = ['Model', 'Test RMSE', 'Test MAE', 'Test R2', 'CV RMSE', 'Train Time (s)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

    def _validate_prediction_arrays(
        self, 
        y_true, 
        y_pred
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and convert prediction arrays to numpy arrays."""
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        
        if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(f"Array length mismatch: y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}")
        
        return y_true_arr, y_pred_arr

    def _plot_rmse_comparison(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot RMSE comparison between Test and CV."""
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['Test RMSE'], width, label='Test RMSE', alpha=0.8)
        ax.bar(x + width/2, df['CV RMSE'], width, label='CV RMSE', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    def _plot_r2_comparison(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot R2 score comparison."""
        ax.barh(df['Model'], df['Test R2'], alpha=0.8, color='green')
        ax.set_xlabel('R² Score', fontsize=12)
        ax.set_title('R² Score Comparison (Higher is Better)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    def _plot_mae_mape_comparison(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot MAE and MAPE comparison with dual y-axis."""
        x = np.arange(len(df))
        width = 0.35
        
        ax2 = ax.twinx()
        ax.bar(x - width/2, df['Test MAE'], width, label='MAE', alpha=0.8, color='orange')
        
        if 'Test MAPE' in df.columns:
            ax2.bar(x + width/2, df['Test MAPE'], width, label='MAPE (%)', alpha=0.8, color='red')
            ax2.set_ylabel('MAPE (%)', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12, color='orange')
        ax.set_title('MAE and MAPE Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=15, ha='right')
        ax.tick_params(axis='y', labelcolor='orange')
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    def _plot_time_vs_performance(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot training time vs performance scatter plot."""
        scatter = ax.scatter(df['Train Time (s)'], df['Test R2'], 
                           s=200, alpha=0.6, c=df.index, cmap='viridis')
        
        for idx, row in df.iterrows():
            ax.annotate(row['Model'], (row['Train Time (s)'], row['Test R2']),
                       fontsize=9, ha='center', va='bottom')
        
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('Test R² Score', fontsize=12)
        ax.set_title('Training Time vs Performance', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)  
