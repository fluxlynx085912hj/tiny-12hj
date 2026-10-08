"""Tiny embedding similarity search utility. Computes cosine similarity between query vector and stored embeddings."""
import argparse, math

embeddings = [
    ("red",     [1.0, 0.0, 0.0]),
    ("green",   [0.0, 1.0, 0.0]),
    ("blue",    [0.0, 0.0, 1.0]),
    ("yellow",  [0.9, 0.8, 0.0]),
    ("cyan",    [0.0, 0.9, 0.8]),
]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot/(norm_a*norm_b) if norm_a and norm_b else 0.0

def main():
    p = argparse.ArgumentParser(description="Find most similar embeddings")
    p.add_argument("--query", required=True, help="comma‑separated query vector")
    p.add_argument("--top", type=int, default=3, help="number of results")
    args = p.parse_args()
    query = [float(x) for x in args.query.split(",")]
    sims = [(name, cosine(query, vec)) for name, vec in embeddings]
    sims.sort(key=lambda x: x[1], reverse=True)
    for name, s in sims[:args.top]:
        print(f"{name:>7} : {s:.4f}")

if __name__ == "__main__":
    main()