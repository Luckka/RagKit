"""Language-aware tokenization for the lexical (BM25) side of hybrid search."""

import re
import unicodedata

# Keeps identifiers whole: SEC-01, create_user, bucket_policy.json, v1.2, ec2.describe
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

_EN = """a an the of and or but to in on at by for from with without into onto over under
is are was were be been being am do does did has have had can could should would will may
might must this that these those it its as if then than so not no yes what which who whom
whose when where why how all any each both few more most other some such only own same very
your you we our they their them he she his her i me my also about after before between"""

_PT = """a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas
para pra com sem sob sobre entre ate e ou mas que se nao sim ao aos ja mais menos muito como
quando onde qual quais quem cujo seu sua seus suas este esta estes estas esse essa esses essas
isso isto aquele aquela ser sao foi sera tem ter ha pode deve"""

STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(_EN.split()),
    "pt": frozenset(_PT.split()),
}


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(text: str, language: str = "en") -> list[str]:
    """Lowercase, strip accents, split on non-identifier chars, drop stopwords and 1-char tokens."""
    stop = STOPWORDS.get(language, frozenset())
    return [
        t for t in TOKEN_RE.findall(strip_accents(text.lower())) if t not in stop and len(t) > 1
    ]
