import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time 
from app.config import settings

BATCH_SIZE = 50  # Number of documents to process in each batch
_GEMINI__DIM = 3072
_FALLBACK_DIM = 768 #all-mpnet-base-v2

_active_model =None 
_model_type:str |None = None

def _probe_gemini():
    """Check and verify whether Gemini API is available and working."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            api_key=settings.GEMINI_API_KEY,
        )
        # Test embedding a simple text
        test_embedding = embeddings.embed_query("test")
        if len(test_embedding) == _GEMINI__DIM:
            logfire.info("✅ Gemini API is available and working.")
            return True
        else:
            logfire.warning("⚠️ Gemini API returned unexpected embedding size.")
            return False
    except Exception as e:
        logfire.warning(f"❌ Gemini API probe failed: {e}")
        return False


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    global _active_model, _model_type
    if _probe_gemini():
        _active_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            api_key=settings.GEMINI_API_KEY,
        )
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


def get_embedding_dim()_>int:
    """
    Returns the dimension of the active embedding model.
    """
    if _active_model is None:
        _init()
    if _model_type == "gemini":
        return _GEMINI__DIM
    elif _model_type == "fallback":
        return _FALLBACK_DIM
    else:
        raise ValueError("Unknown embedding model type.")
    


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """
    Embed a batch of documents using the active embedding model.

    Args:
        batch (list[str]): A list of document texts to embed.

    Returns:
        list[list[float]]: A list of embeddings corresponding to the input documents.
    """
    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings