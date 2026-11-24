"""Small text helpers shared by indexing and retrieval."""
import re

STOPWORDS = set("""a an the of to in on for and or by with as at from is are be been was were it its this that these
those any all such shall may must will which who whom whose than then there their under over into upon not no""".split())


def tokenize(text):
    """Lower-case word tokens for BM25; keeps numbers like 5.2.1 and 10,000 intact."""
    tokens = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def cite(chunk):
    """Human-readable citation, e.g. [RBI | KYC_MD_2016 | 5.2]."""
    return f"[{chunk['issuer']} | {chunk['doc_id']} | {chunk['clause']}]"


def doc_prefix(chunk):
    # nomic-embed-text expects task prefixes for documents and queries
    return f"search_document: {chunk['title']} {chunk['clause']}: {chunk['text']}"


def query_prefix(q):
    return f"search_query: {q}"
