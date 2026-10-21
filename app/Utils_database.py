import asyncio
import os
from pathlib import Path
from uuid import uuid4

from app.Utils import load_env_variables


# Preserve the original AutoGen Chroma settings, independently of the agent framework.
COLLECTION_NAME = "functions"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 1
SCORE_THRESHOLD = 0.7


class ChromaRAGMemory:
    """Persistent ChromaDB with local SentenceTransformer embeddings."""

    def __init__(self, path=None, *, collection=None):
        # Injected collections are useful for deterministic offline unit tests.
        if collection is not None:
            self._collection = collection
            return

        if path is None:
            path = os.getenv("PERSISTENCE_PATH")
        if not path:
            raise ValueError(
                "PERSISTENCE_PATH is not set. Specify the existing ChromaDB "
                "directory to use the same persistent RAG memory."
            )

        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        path = str(Path(path).expanduser().resolve())
        client = chromadb.PersistentClient(path=path)
        embedding = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding,
            metadata={"hnsw:space": "cosine"},
        )
        metadata = self._collection.metadata or {}
        if metadata.get("hnsw:space", "cosine") != "cosine":
            raise ValueError(
                "Existing Chroma collection does not use cosine distance; "
                "do not silently alter RAG retrieval semantics."
            )

    async def query(self, text: str) -> list[str]:
        """Retrieve at most one document with cosine similarity >= 0.7."""
        count = await asyncio.to_thread(self._collection.count)
        if not count:
            return []
        result = await asyncio.to_thread(
            self._collection.query,
            query_texts=[text],
            n_results=min(TOP_K, count),
            include=["documents", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            document
            for document, distance in zip(documents, distances)
            if document is not None and distance is not None
            and (1.0 - float(distance)) >= SCORE_THRESHOLD
        ]

    async def add(self, function: str) -> None:
        await asyncio.to_thread(
            self._collection.add,
            ids=[str(uuid4())],
            documents=[function],
            metadatas=[{"category": "functions", "type": "function"}],
        )

    async def close(self) -> None:
        # Chroma PersistentClient has no async close; data is persisted automatically.
        pass


async def add_entry(function: str, memory: ChromaRAGMemory | None = None) -> None:
    owns_memory = memory is None
    memory = memory or ChromaRAGMemory()
    try:
        await memory.add(function)
    finally:
        if owns_memory:
            await memory.close()


# Exactly the two original pseudocode strings, without paraphrasing.
EXAMPLES = [
    'FUNCTION MergeSort(Array A, index p, index r):\n    IF p < r THEN:\n        // Find the midpoint of the array\n        q = floor((p + r) / 2)\n\n        // Recursively sort the first half\n        MergeSort(A, p, q)\n\n        // Recursively sort the second half\n        MergeSort(A, q + 1, r)\n\n        // Merge the two sorted halves\n        Merge(A, p, q, r)\n\n// Function that merges two sorted sub-arrays: A[p..q] and A[q+1..r]\nFUNCTION Merge(Array A, index p, index q, index r):\n    // Calculate the sizes of the two sub-arrays\n    n1 = q - p + 1\n    n2 = r - q\n\n    // Create two temporary arrays L (left) and R (right)\n    Create Array L[1..n1+1]\n    Create Array R[1..n2+1]\n\n    // Copy data into the left sub-array L\n    FOR i FROM 1 TO n1 DO:\n        L[i] = A[p + i - 1]\n\n    // Copy data into the right sub-array R\n    FOR j FROM 1 TO n2 DO:\n        R[j] = A[q + j]\n\n    // Add sentinels (infinity or very large values) at the end of L and R\n    L[n1 + 1] = INFINITY\n    R[n2 + 1] = INFINITY\n\n    // Initialize indices for L and R\n    i = 1\n    j = 1\n\n    // Merge L and R back into the main array A[p..r]\n    FOR k FROM p TO r DO:\n        IF L[i] <= R[j] THEN:\n            A[k] = L[i]\n            i = i + 1\n        ELSE:\n            A[k] = R[j]\n            j = j + 1',
    'ALGORITHM BinarySearch(array, target):\n\n    left ← 0\n    right ← length(array) - 1\n\n    WHILE left ≤ right DO\n        mid ← floor((left + right) / 2)\n        IF array[mid] = target THEN\n            RETURN mid\n        ELSE IF array[mid] < target THEN\n            left ← mid + 1\n        ELSE\n            right ← mid - 1\n        ENDIF\n    ENDWHILE\n\n    RETURN -1',
]


async def main():
    load_env_variables()
    memory = ChromaRAGMemory()
    try:
        for example in EXAMPLES:
            await add_entry(example, memory)
    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
