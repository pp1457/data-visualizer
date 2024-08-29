import json
import os
import pathlib
from data_visualizer.draw import draw_radar, draw_box, draw_bar

def read_json_to_dict(file_path):
    """Read JSON data from a file and return it as a dictionary."""

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        return None

def split(s):
    return "\n".join(s.split("@"))

def visualize(json_files, result_dir):
    metrics = ["hit_rate", "map", "mrr", "ndcg", "tnr"]
    chunking_methods = []
    
    method_metric_score = {}

    for json_path in json_files:

        data = read_json_to_dict(json_path)

        scores = [ data[metric]/100 for metric in metrics]

        filename = os.path.splitext(os.path.basename(data["search_filename"]))[0]
        method = str(data["search_chunking_method"]).replace("|", "@")
        embedding_model = str(data["search_embedding_model"])

        if not method in chunking_methods:
            chunking_methods.append(method)

        method_metric_score[method] = {}
        k_value = data["k_value"]
        threshold = data["threshold"]

        for metric in metrics:
            method_metric_score[method][metric] = data[f"{metric}_list"]

        output_dir = pathlib.Path(f"{result_dir}/{filename}/{embedding_model}/k={k_value}&threshold={threshold}/method")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / method
        draw_radar(metrics, scores, split(method), output_path)

    for metric in metrics:
        scores = []
        score_averages = []

        for method in chunking_methods:
            tmp = method_metric_score[method][metric]
            scores.append(tmp)
            score_averages.append(sum(tmp) / len(tmp))

        output_dir = pathlib.Path(f"{result_dir}/{filename}/{embedding_model}/k={k_value}&threshold={threshold}/metric_box")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / metric
        split_methods = [split(method) for method in chunking_methods]
        draw_box(split_methods, scores, metric, output_path)
        output_dir = pathlib.Path(f"{result_dir}/{filename}/{embedding_model}/k={k_value}&threshold={threshold}/metric_bar")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{metric}_bar"
        draw_bar(split_methods, score_averages, metric, output_path)


def main():
    visualize(
        [
        "/Users/liaoyunyang/Downloads/fubon_k=3,th=0.5.json/k=3@threshold=0.5@methods=by_markdown_number_of_hash=4.json",
        "/Users/liaoyunyang/Downloads/fubon_k=3,th=0.5.json/k=3@threshold=0.5@methods=by_page.json",
        "/Users/liaoyunyang/Downloads/fubon_k=3,th=0.5.json/k=3@threshold=0.5@methods=recursive_char_chunk_size=1000_overlap_size=100.json",
        "/Users/liaoyunyang/Downloads/fubon_k=3,th=0.5.json/k=3@threshold=0.5@methods=semantic_model=mxbai-embed-large_buffer_size=3_chunk_number=60.json",
        "/Users/liaoyunyang/Downloads/fubon_k=3,th=0.5.json/k=3@threshold=0.5@methods=semantic_model=text-embedding-3-small_buffer_size=3_chunk_number=60.json",
        "/Users/liaoyunyang/Downloads/fubon_k=3,th=0.5.json/k=3@threshold=0.5@methods=semantic_model=text-embedding-ada-002_buffer_size=3_chunk_number=60.json"
        ],
        "data/result"
    )

if __name__ == "__main__":
    main()
        

