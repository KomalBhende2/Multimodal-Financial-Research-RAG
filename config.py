from pathlib import Path
import os
import ssl
import sys
import warnings

# Bypass SSL certificate verification globally
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Disable insecure request warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Monkeypatch requests and httpx globally to force verify=False (ignoring system SSL certificates)
try:
    import requests
    original_requests_init = requests.Session.__init__
    def patched_requests_init(self, *args, **kwargs):
        original_requests_init(self, *args, **kwargs)
        self.verify = False
    requests.Session.__init__ = patched_requests_init
except Exception:
    pass

try:
    import httpx
    original_httpx_client_init = httpx.Client.__init__
    def patched_httpx_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_httpx_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = patched_httpx_client_init

    original_httpx_async_client_init = httpx.AsyncClient.__init__
    def patched_httpx_async_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_httpx_async_client_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = patched_httpx_async_client_init
except Exception:
    pass

# Monkeypatch sentence-transformers Pooling to prevent TypeError on older model configurations (like BAAI/bge models)
try:
    from sentence_transformers.sentence_transformer.modules.pooling import Pooling
    original_pooling_init = Pooling.__init__
    def patched_pooling_init(self, *args, **kwargs):
        if 'word_embedding_dimension' in kwargs and 'embedding_dimension' not in kwargs:
            kwargs['embedding_dimension'] = kwargs.pop('word_embedding_dimension')
        original_pooling_init(self, *args, **kwargs)
    Pooling.__init__ = patched_pooling_init
except Exception:
    pass

# Reconfigure stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError when printing special characters (e.g. currency symbols)
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ===============================
# Project Paths
# ===============================

BASE_DIR = Path(__file__).parent

DOCUMENT_FOLDER = BASE_DIR / "data" / "documents"
METADATA_FOLDER = BASE_DIR / "data" / "metadata"

# ===============================
# Chunking
# ===============================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# ===============================
# Embedding
# ===============================

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# ===============================
# Retrieval
# ===============================

COLLECTION_NAME = "financial_documents"
QDRANT_PATH = str(BASE_DIR / "data" / "qdrant_db")

TOP_K = 10

SIMILARITY_THRESHOLD = 0.80

MMR_FETCH_K = 40

RERANK_TOP_K = 10
EMBEDDING_DIMENSION = 1024
QDRANT_BATCH_SIZE = 100

# ===============================
# Company
# ===============================

DEFAULT_COMPANY = {

    "company_id": "ATHER",

    "company_name": "Ather Energy Limited"
}

# ===============================
# Hybrid Search Configuration
# ===============================

VECTOR_WEIGHT = 0.70
LEXICAL_WEIGHT = 0.30

VECTOR_SEARCH_LIMIT = 150
KEYWORD_SEARCH_LIMIT = 80

# Metadata Boosting Configs
BOOST_COMPANY = 0.05
BOOST_DOCUMENT = 0.03
BOOST_TOPIC = 0.02
BOOST_QUARTER = 0.02
BOOST_SECTION = 0.02

# Adaptive Threshold Limits
THRESHOLD_SHORT_QUERY = 0.50
THRESHOLD_MEDIUM_QUERY = 0.40
THRESHOLD_LONG_QUERY = 0.30

SHORT_QUERY_MAX_WORDS = 5
MEDIUM_QUERY_MAX_WORDS = 15

# Reranking Pool Limits
RERANK_LIMIT = 40
RERANK_LIMIT_SECTION = 40

# Adjacent Chunk Expansion
ENABLE_ADJACENT_EXPANSION = True