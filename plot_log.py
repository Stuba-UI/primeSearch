# plot_log.py
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_log(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["combined_fitness"], label="Combined Fitness")
    plt.plot(df["strict_hits"], label="Strict Hits")
    plt.xlabel("Generation")
    plt.legend()
    plt.title("Evolution Progress")
    plt.show()

if __name__ == "__main__":
    plot_log(sys.argv[1])
