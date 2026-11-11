from ragkit.retrieval.sparse import BM25Encoder, token_id
from ragkit.retrieval.text import tokenize


def dot(q: dict, d: dict) -> float:
    dv = dict(zip(d["indices"], d["values"], strict=True))
    return sum(v * dv.get(i, 0.0) for i, v in zip(q["indices"], q["values"], strict=True))


def test_tokenize_keeps_identifiers_and_drops_stopwords():
    toks = tokenize("The IAM role uses s3:GetObject on bucket_policy.json and REL-02")
    assert {"iam", "role", "bucket_policy.json", "rel-02"} <= set(toks)
    assert "the" not in toks and "on" not in toks


def test_tokenize_is_language_aware_and_strips_accents():
    toks = tokenize("Autenticação do usuário com senha", language="pt")
    assert toks == ["autenticacao", "usuario", "senha"]


def test_token_id_is_stable_uint32():
    assert 0 <= token_id("rel-02") < 2**32
    assert token_id("rel-02") == token_id("rel-02")


def test_bm25_ranks_rare_term_match_first():
    corpus = [
        "enable MFA for the root user",
        "rotate the access keys of the root user",
        "use S3 lifecycle rules to reduce storage cost",
    ]
    enc = BM25Encoder().fit(corpus)
    q = enc.encode_query("how do I enable MFA?")
    scores = [dot(q, enc.encode_document(t)) for t in corpus]
    assert scores[0] == max(scores) > 0
    assert scores[1] == 0


def test_unknown_terms_ignored_and_roundtrip(tmp_path):
    enc = BM25Encoder(language="pt").fit(["login com senha"])
    assert enc.encode_query("blockchain quantum")["indices"] == []
    enc.save(tmp_path / "bm25.json")
    loaded = BM25Encoder.load(tmp_path / "bm25.json")
    assert (loaded.df, loaded.avgdl, loaded.language) == (enc.df, enc.avgdl, "pt")
