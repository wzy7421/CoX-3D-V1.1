import torch
from sklearn.cluster import KMeans

try:
    from bertopic import BERTopic
    _HAS_BERTOPIC = True
except Exception:
    _HAS_BERTOPIC = False


class ELLM(torch.nn.Module):
    """
    ELLM (BERTopic–LLM) semantic modeling module.

    - If bertopic installed: use BERTopic with precomputed embeddings.
    - Else: fallback to KMeans (still reproducible).

    llm_filter:
    - 'heuristic': rule-based relevance scoring (no API)
    - 'openai': placeholder hook for GPT-4o topic-level filtering
    """

    def __init__(
        self,
        text_encoder,
        topic_k=12,
        use_bertopic=True,
        llm_filter="heuristic",
        lambda_kl=0.5,
        device="cuda",
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.topic_k = int(topic_k)
        self.use_bertopic = bool(use_bertopic) and _HAS_BERTOPIC
        self.llm_filter = llm_filter
        self.lambda_kl = float(lambda_kl)
        self.device = device if torch.cuda.is_available() else "cpu"

        if self.use_bertopic:
            self.topic_model = BERTopic()
        else:
            self.kmeans = KMeans(n_clusters=self.topic_k, random_state=0)

        self.design_terms = [
            "造型","材质","颜色","纹理","工艺","比例","曲线","极简","复古","未来感",
            "金属","透明","灯带","圆角","手感","质感","结构","细节","风格"
        ]

    def _kl(self, p: torch.Tensor, q: torch.Tensor):
        p = torch.clamp(p, 1e-8, 1.0)
        q = torch.clamp(q, 1e-8, 1.0)
        return torch.sum(p * torch.log(p / q), dim=-1).mean()

    def _heuristic_semantic_judgement(self, texts, topic_ids):
        K = self.topic_k
        score = torch.zeros(K, device=self.device)
        for t, k in zip(texts, topic_ids):
            s = 0.0
            for term in self.design_terms:
                if term in t:
                    s += 1.0
            score[int(k)] += s

        p = score + 1e-3
        p = p / p.sum()
        q = torch.ones_like(p) / K
        return p, q

    def _topics_to_prompt(self, texts, topic_ids):
        prompts = []
        for t, k in zip(texts, topic_ids):
            prompts.append(f"product design concept: {t}")
        return prompts

    @torch.no_grad()
    def forward(self, texts, timestamps):
        h = self.text_encoder(texts)  # [N,768]

        if self.use_bertopic:
            docs = [""] * len(texts)
            topic_ids, _ = self.topic_model.fit_transform(docs, embeddings=h.cpu().numpy())
            topic_ids = [int(x) if x != -1 else 0 for x in topic_ids]
        else:
            topic_ids = self.kmeans.fit_predict(h.cpu().numpy()).tolist()

        if self.llm_filter in ("heuristic", "openai"):
            p, q = self._heuristic_semantic_judgement(texts, topic_ids)
        else:
            p, q = self._heuristic_semantic_judgement(texts, topic_ids)

        L_kl = self._kl(p, q) * self.lambda_kl
        prompts = self._topics_to_prompt(texts, topic_ids)

        return {
            "embeddings": h,
            "topic_ids": topic_ids,
            "p": p,
            "q": q,
            "L_kl": L_kl,
            "prompts": prompts,
        }
