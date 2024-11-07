#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:29:21 2024

@author: pavithira seenivasagan
"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
file_path = 'Employee.csv'
data = pd.read_csv(file_path)

# Set visual styles for plots
sns.set(style="whitegrid", palette="muted")


def clean_data(df):
    """Cleans the dataset by removing missing values and encoding categorical variables."""
    df_cleaned = df.dropna()
    df_cleaned['EverBenched'] = df_cleaned['EverBenched'].map(
        {'Yes': 1, 'No': 0})
    df_cleaned['LeaveOrNot'] = df_cleaned['LeaveOrNot'].astype('category')
    return df_cleaned


def statistical_summary(df):
    """Displays descriptive statistics, skewness, and kurtosis of numerical columns."""
    desc_stats = df.describe(include='number')
    skewness = df.select_dtypes(include='number').skew()
    kurtosis = df.select_dtypes(include='number').kurtosis()

    print("Descriptive Statistics:\n", desc_stats)
    print("\nSkewness:\n", skewness)
    print("\nKurtosis:\n", kurtosis)

    return desc_stats, skewness, kurtosis


def plot_relational_graph(df):
    """Scatter plot showing relationship between Age and Experience in Current Domain by Gender."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='ExperienceInCurrentDomain', hue='Gender',
                    data=df,
                    palette='coolwarm', edgecolor='black', s=100, alpha=0.85)
    plt.title('Age vs Experience in Current Domain',
              fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Age', fontsize=12, fontweight='bold')
    plt.ylabel('Experience in Current Domain (Years)',
               fontsize=12, fontweight='bold')
    plt.legend(title='Gender', fontsize=10,
               title_fontsize=12, loc='best', frameon=True)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_categorical_graph(df):
    """Horizontal bar graph comparing employee counts across different Education levels by Gender."""
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Education', hue='Gender', data=df,
                  palette='viridis', edgecolor='black')
    plt.title('Count of Employees by Education Level and Gender',
              fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Count of Employees', fontsize=12, fontweight='bold')
    plt.ylabel('Education Level', fontsize=12, fontweight='bold')
    plt.legend(title='Gender', fontsize=10, title_fontsize=12,
               loc='center right', frameon=True)
    plt.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_statistical_graph(df):
    """Line plot of average Payment Tier by Joining Year and Gender with a custom color palette."""
    plt.figure(figsize=(10, 6))
    avg_payment = df.groupby(['JoiningYear', 'Gender'])[
        'PaymentTier'].mean().reset_index()

    sns.lineplot(data=avg_payment, x='JoiningYear', y='PaymentTier',
                 hue='Gender', marker='o',
                 palette='magma', linewidth=2)

    plt.title('Average Payment Tier by Joining Year and Gender',
              fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Joining Year', fontsize=12, fontweight='bold')
    plt.ylabel('Average Payment Tier', fontsize=12, fontweight='bold')
    plt.legend(title='Gender', fontsize=10,
               title_fontsize=12, loc='best', frameon=True)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """Correlation heatmap for numerical columns in the dataset with a light color map."""
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Education'] = df['Education'].map({'Bachelors': 1, 'Masters': 2})
    numeric_data = df.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(10, 6))

    heatmap = sns.heatmap(correlation_matrix, annot=True,
                          cmap='Blues', fmt=".2f", linewidths=0.5, square=True)

    plt.title("Correlation Heatmap of Employee Features",
              fontsize=16, fontweight='bold', color='black')
    plt.xlabel("Features", fontsize=12, fontweight='bold')
    plt.ylabel("Features", fontsize=12, fontweight='bold')
    plt.show()


def plot_age_histogram_with_density(df):
    """Adds a density curve to the Age histogram plot and calculates skewness and kurtosis."""

    skewness = df['Age'].skew()
    kurtosis = df['Age'].kurtosis()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=15, kde=True, color='blue')
    plt.title("Histogram and Density Curve of Employee Age",
              fontsize=16, fontweight='bold')
    plt.xlabel("Age", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency / Density", fontsize=12, fontweight='bold')

    plt.figtext(0.15, 0.8, f'Skewness: {skewness:.2f}', fontsize=10)
    plt.figtext(0.15, 0.75, f'Kurtosis: {kurtosis:.2f}', fontsize=10)

    plt.tight_layout()
    plt.show()


# Clean the dataset
cleaned_data = clean_data(data)

# Generate statistical summary
desc_stats, skewness, kurtosis = statistical_summary(cleaned_data)

# Create visualizations
plot_relational_graph(cleaned_data)
plot_categorical_graph(cleaned_data)
plot_statistical_graph(cleaned_data)
plot_correlation_heatmap(cleaned_data)
plot_age_histogram_with_density(cleaned_data)
