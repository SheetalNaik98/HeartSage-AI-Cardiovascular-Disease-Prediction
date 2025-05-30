import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def exploratory_data_analysis(filepath, output_dir="results/figures"):
    df = pd.read_csv(filepath)

    # Distribution of target variable
    plt.figure(figsize=(6,4))
    df['HeartDisease'].value_counts().plot(kind='bar', color='maroon', edgecolor='black')
    plt.title("Heart Disease Cases")
    plt.ylabel("Count")
    plt.xticks(ticks=[0,1], labels=["No Disease","Disease"], rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heart_disease_distribution.png")
    plt.close()

    # Correlation heatmap
    df_numeric = pd.get_dummies(df, drop_first=True)
    corr = df_numeric.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
