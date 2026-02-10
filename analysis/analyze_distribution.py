
import os
import matplotlib.pyplot as plt
import pandas as pd

def analyze_dataset(root_dir):
    data = []
    splits = ['train', 'valid', 'test']
    classes = sorted([d for d in os.listdir(os.path.join(root_dir, 'train')) if os.path.isdir(os.path.join(root_dir, 'train', d))])
    
    for split in splits:
        counts = {}
        for cls in classes:
            path = os.path.join(root_dir, split, cls)
            if os.path.exists(path):
                counts[cls] = len(os.listdir(path))
            else:
                counts[cls] = 0
        data.append(counts)
    
    df = pd.DataFrame(data, index=splits).T
    df.index.name = 'Stage'
    return df

def plot_distribution(df):
    ax = df.plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title('CVM Stage Distribution (Clean ROI Dataset)', fontsize=15)
    plt.xlabel('CVM Stage', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('distribution_analysis.png')
    print("Distribution plot saved as 'distribution_analysis.png'")

if __name__ == '__main__':
    DATA_DIR = 'Aariz_CVM_Clean'
    if os.path.exists(DATA_DIR):
        distribution_df = analyze_dataset(DATA_DIR)
        print("\n--- Dataset Distribution Table ---")
        print(distribution_df)
        print("-" * 35)
        plot_distribution(distribution_df)
    else:
        print(f"Error: Directory '{DATA_DIR}' not found. Please run prepare_classification_data.py first.")
