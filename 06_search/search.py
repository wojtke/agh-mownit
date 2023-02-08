import numpy as np
from scipy import sparse

from preprocessing import process_text, lower_rank


class SearchEngine:
    def __init__(self, doc_ids, terms, svd, k=50):
        self.terms = terms
        print("Terms: ", len(terms))

        self.doc_ids = np.array(doc_ids)
        print("Docs: ", len(doc_ids))

        self.svd = svd
        self.k = k

        self.matrix = lower_rank(svd, self.k)

    def query_text_to_vector(self, txt) -> sparse.coo_array:
        """Convert a query text to a sparse vector representation."""
        words = process_text(txt).keys()
        cords = [self.terms[w] for w in words if w in self.terms]
        query_vector = sparse.coo_array(
            ([1] * len(cords), (cords, [0] * len(cords))),
            shape=(len(self.terms), 1),
            dtype=np.float32
        )
        query_vector = query_vector / sparse.linalg.norm(query_vector)
        return query_vector

    def get_similarity(self, query_vector):
        """Get the cosine similarity of a query vector to all documents."""
        similarities = query_vector.T @ self.matrix
        return similarities.toarray()

    def search(self, query_text, top=20):
        """Search for documents similar to the query text."""
        query_vector = self.query_text_to_vector(query_text)
        print("Query vector: ", query_vector.shape)
        similarities = self.get_similarity(query_vector).flatten()
        print("Similarities: ", similarities.shape)
        order = np.argsort(-similarities)[:top]

        return [{'id': self.doc_ids[i], 'score': similarities[i]} for i in order]
