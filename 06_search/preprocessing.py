from pathlib import Path
from typing import Generator, Tuple, Dict, Any, List

import nltk
from collections import Counter
import numpy as np

from scipy.sparse import coo_array, coo_matrix, csc_matrix
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

from config import DATA_DIR, SAVE_DIR
from files import save_pickle, save_svd, get_file_text
import re

stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.porter.PorterStemmer()


def process_text(text: str) -> dict:
    """For a given text returns bag of words (a dict)"""

    text = text.lower()
    text = re.sub(r'[\\\/=.,]', ' ', text)

    words = nltk.tokenize.word_tokenize(text)
    words = [w for w in words if w not in stopwords]
    words = [w for w in words if all([c not in w for c in "-_/+.,:;|\\"])]
    words = [w for w in words if not (w.isnumeric() and len(w) > 3)]
    words = [stemmer.stem(w) for w in words]
    words = [w for w in words if len(w) > 2 and len(w) < 15]

    return dict(Counter(words))


def processed_docs(n=None) -> Generator:
    """Generates a list of processed documents (bow dicts along with id)."""
    filenames = Path(DATA_DIR).glob('*')
    n = n or np.inf
    i = 0
    for filename in filenames:
        try:
            file_txt, file_id = get_file_text(filename)
        except Exception as e:
            print(f"Problem with reading file {filename}: ", e)
            continue
        try:
            file_bow = process_text(file_txt)
        except Exception as e:
            print(f"Problem with processing file {filename}: ", e)
            continue
        yield file_bow, file_id
        i += 1
        if i >= n:
            break


def build_tbd_matrix(docs: list[dict] | Generator) -> tuple[sparse.csr_matrix, dict[Any, Any], list[Any]]:
    """Given bags of words (dicts) return term by document matrix."""
    terms, doc_ids = {}, []

    row, col, data = [], [], []
    for doc_index, doc in enumerate(docs):
        bow, id = doc
        doc_ids.append(id)

        for term, count in bow.items():
            term_index = terms.setdefault(term, len(terms))
            row.append(term_index)
            col.append(doc_index)
            data.append(count)

    return sparse.csr_matrix((data, (row, col)), shape=(len(terms), len(doc_ids)), dtype=np.float32), terms, doc_ids


def mul_by_idf(matrix) -> sparse.coo_array:
    """Multiplies given matrix by idf vector."""
    idf = np.log(matrix.shape[1] / (matrix > 0).sum(axis=1)).reshape(-1, 1)
    return matrix.multiply(idf)


def normalize_col(matrix):
    """Normalizes columns so that each feature col has equal l2 norm."""
    norms = np.array([sparse.linalg.norm(matrix.getcol(i)) for i in range(matrix.shape[1])])
    return matrix.multiply(1 / norms)


def svd(matrix, k):
    """SVD"""
    return sparse.linalg.svds(matrix.T, k=k)


def lower_rank(svd, k):
    """Lower rank matrix approximation."""
    u, s, vt = svd
    u = sparse.csr_matrix(u[:, :k], dtype=np.float32)
    s = sparse.csr_matrix(np.diag(s[:k]), dtype=np.float32)
    vt = sparse.csr_matrix(vt[:k, :], dtype=np.float32)

    matrix = u @ s @ vt
    matrix = normalize_col(matrix.T)
    return sparse.csr_matrix(matrix)


if __name__ == "__main__":
    tbd_matrix, terms, doc_ids = build_tbd_matrix(processed_docs(10000))
    tbd_matrix = mul_by_idf(tbd_matrix)
    tbd_matrix = normalize_col(tbd_matrix)
    svd = svd(tbd_matrix, k=50)
    save_pickle({'terms': terms,
                 'doc_ids': doc_ids,
                 })
    save_svd(svd)


