# plot_log.py
import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_log.py evolution_log.csv")
    sys.exit(1)

log_file = sys.argv[1]
df = pd.read_csv(log_file)

metrics = {
    "combined_fitness": {"color": "blue", "linestyle": "-"},
    "novelty": {"color": "red", "linestyle": "--"},
    "complexity": {"color": "purple", "linestyle": "-."},
    "strict_hits": {"color": "green", "linestyle": "-"},
    "closeness": {"color": "orange", "linestyle": "--"},
    "variance": {"color": "gray", "linestyle": ":"},
}

plt.figure(figsize=(14, 7))
for metric, style in metrics.items():
    if metric in df.columns:
        plt.plot(df['generation'], df[metric], label=metric.replace("_", " ").capitalize(),
                 color=style["color"], linestyle=style["linestyle"])
    else:
        print(f"Warning: '{metric}' column not found in CSV. Skipping.")

plt.title("Evolution Progress")
plt.xlabel("Generation")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
