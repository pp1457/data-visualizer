import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def draw_radar(metrics, scores, chunking_method, output_path):
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1] 

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    ax.set_theta_zero_location('N')
    # ax.yaxis.set_visible(False)  # Hide radial grid lines
    # ax.xaxis.set_visible(False)  # Hide angular grid lines
    # ax.spines['polar'].set_visible(False)  # Hide the outer frame


    ax.set_ylim(0, 1)
    break_point = [round(i*0.2, 1) for i in range(6)]
    ax.set_yticks(break_point)
    ax.set_yticklabels(break_point)  # You can choose to label the y-axis

    ax.set_xticks(angles[:-1])  # Set the x-ticks to the angles, excluding the last one to avoid duplication
    ax.set_xticklabels(metrics)  # Set the x-tick labels

    line_color = '#1f77b4'  # A strong blue for the line
    fill_color = '#aec7e8'  # A lighter blue for the fill
    text_color = '#ff7f0e'  # A contrasting orange for text
    scatter_color = '#2ca02c'  # A vibrant green for the scatter points

    # Draw hexagon
    ax.plot(angles, scores, linewidth=2, linestyle="solid", color=line_color, label="Metrics")
    ax.fill(angles, scores, color=fill_color, alpha=0.25)

    for ti, di in zip(angles, scores):
        ax.text(ti, di-0.15, round(di, 4), color=text_color, ha='center', va='center')

    ax.scatter(angles, scores, color=scatter_color, s=50)

    # Add a title
    ax.set_title(f"{chunking_method} Radar Graph")


    plt.savefig(output_path)
    # plt.show()

def draw_box(chunking_methods, scores, metric, output_path):

    data = []

    for index, method in enumerate(chunking_methods):
        for question_id, score in enumerate(scores[index]):
            data.append([method, question_id, score])

    df = pd.DataFrame(data, columns=["chunking_method", "question_id", "score"])

    # Create the box plot
    plt.figure(figsize=(12, 10))
    sns.boxplot(x="chunking_method", y="score", data=df)
    plt.title(f"{metric} of Different Chunking Methods")
    plt.xlabel("Chunking Method")
    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.savefig(output_path)

def draw_bar(methods, scores, metric, output_path):
    plt.figure(figsize=(10, 15))
    plt.bar(methods, scores, color='skyblue', edgecolor='black')

    plt.title(metric)
    plt.xlabel("Chunking Method")
    plt.ylabel("Score")

    # 顯示每個條形的數值
    for i, value in enumerate(scores):
        plt.text(i, value + 0.02, str(round(value, 4)), ha='center', va='bottom')

    plt.tight_layout()
    plt.xticks(rotation=90, fontsize=12)

    ax = plt.gca()
    ax.set_aspect(aspect=7, adjustable='box')

    # 保存圖表到文件
    plt.savefig(output_path)

def main():
    # metrics = ["Hit Rate", "MAP", "MRR", "NDCG", "TNR"]
    # scores = [0.3, 0.5, 0.99999, 0.1, 0.2]
    # draw(metrics, scores)

    methods = ["by_markdown", "by_page", "semantic"]
    scores = []
    scores.append([0.1, 0.3, 0.2])
    scores.append([0.4, 0.3, 0.1])
    scores.append([0.2, 0.2, 0.5])

    draw_box(methods, scores, "hit_rate")


if __name__ == "__main__":
    main()

