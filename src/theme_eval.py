# src/theme_eval.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

smooth = SmoothingFunction().method1

def cluster_themes(image_embeds, text_embeds, captions, n_clusters=3, random_state=42):
    """
    Cluster using concatenated embeddings (image + text per item).
    Returns:
      - labels: array of length N (cluster assignments)
      - cluster_info: list of dicts with keys { 'center', 'top_captions' }
    """
    # Ensure numpy arrays
    img = np.asarray(image_embeds)
    txt = np.asarray(text_embeds)

    # Flatten per-sample if embeddings shaped (N, D, 1) etc
    if img.ndim > 2:
        img = img.reshape(img.shape[0], -1)
    if txt.ndim > 2:
        txt = txt.reshape(txt.shape[0], -1)

    X = np.concatenate([img, txt], axis=1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    cluster_info = []
    for c in range(n_clusters):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            top_caps = []
        else:
            # compute distance to center -> pick top 3 closest captions as keywords
            dists = np.linalg.norm(X[idxs] - centers[c], axis=1)
            topk = np.argsort(dists)[:3]
            top_caps = [captions[int(idxs[i])] for i in topk]
        cluster_info.append({
            "center": centers[c],
            "top_captions": top_caps,
            "indices": idxs.tolist()
        })
    return labels, cluster_info

# ---------- Evaluation ----------
_scoring = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_story_metrics(story, captions, caption_embeds, sentence_model=None):
    """
    Compute BLEU, ROUGE-L, and embedding-based cosine similarity.
    """
    # BLEU
    references = [" ".join(captions)]
    ref_tokens = [r.split() for r in references]
    hyp_tokens = story.split()
    bleu = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth)

    # ROUGE-L
    rouge_scores = _scoring.score(" ".join(references), story)
    rouge_l_f1 = rouge_scores["rougeL"].fmeasure

    # Embedding similarity using SAME model for story + captions
    if sentence_model is None:
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    story_emb = sentence_model.encode([story])[0]
    caption_emb = sentence_model.encode([" ".join(captions)])[0]

    cos = cosine_similarity([story_emb], [caption_emb])[0][0]

    return {
        "bleu": float(bleu),
        "rouge_l_f1": float(rouge_l_f1),
        "embedding_cosine": float(cos)
    }
