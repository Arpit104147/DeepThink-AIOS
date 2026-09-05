import os
import sqlite3
import json
import math
import time as _time
import numpy as np
import uuid
import re

# ─────────────────────────────────────────────────────────────────────────
# Retrieval-quality knobs (Phase 2.1)
# These are the empirical cutoffs that separate "genuinely related" hits
# from "vaguely on-topic" noise for MiniLM-class 384-d embeddings.
#
# MIN_VECTOR_SCORE          — hard floor for a hit to count as relevant.
# TOP_HIT_MARGIN            — drop any hit more than this cosine below the
#                              best hit (prevents blending 0.42s with 0.91s).
# RECENCY_WEIGHT            — how much a recent memory outranks an old one
#                              of equal cosine similarity. 0.15 == "modest".
# RECENCY_HALF_LIFE_DAYS    — half-life for the recency bonus.
# KEYWORD_MIN_MATCHES       — content-word overlap required for the
#                              keyword fallback to fire when vector search
#                              produces no hits above MIN_VECTOR_SCORE.
# ─────────────────────────────────────────────────────────────────────────
MIN_VECTOR_SCORE = 0.65
TOP_HIT_MARGIN = 0.15
RECENCY_WEIGHT = 0.15
RECENCY_HALF_LIFE_DAYS = 30.0
KEYWORD_MIN_MATCHES = 3

class Memory:
    _STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", 
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
        "can", "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for", "from", 
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", 
        "i", "if", "in", "into", "is", "it", "its", "me", "more", "most", "my", "myself", "no", "nor", 
        "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over", "own", 
        "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", 
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", 
        "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", 
        "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves",
        "write", "code", "program", "script", "create", "make", "generate", "give", "please", "solve", "run",
        "show", "showing", "output", "result", "results", "value", "values"
    }

    def __init__(self, db_path="./forge_memory_db"):
        self.db_path = db_path
        self.chroma_client = None
        self.collection = None
        self.use_chroma = False

        # Attempt to initialize ChromaDB
        try:
            import chromadb
            os.makedirs(db_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_or_create_collection(name="knowledge")
            
            # Pre-warm embedding model during initialization so ONNX weights download during startup instead of prompt execution
            try:
                from chromadb.utils import embedding_functions
                ef = embedding_functions.DefaultEmbeddingFunction()
                ef(["warmup"])
            except Exception:
                pass

            self.use_chroma = True
            print("Memory Engine: Successfully initialized ChromaDB persistent vector database.")
        except Exception as e:
            print(f"Memory Engine: ChromaDB not available ({str(e)}). Falling back to SQLite memory store.")
            
        # Initialize SQLite database (either as primary fallback or metadata companion)
        self.sqlite_path = os.path.join(db_path, "local_memory.db")
        os.makedirs(db_path, exist_ok=True)
        self._init_sqlite()

    def _init_sqlite(self):
        """Initialize local SQLite database for structured data and embedding storage."""
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            
            # Table for storing experiences (tasks, solutions, mistakes)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    task TEXT,
                    doc TEXT,
                    metadata TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Phase 3: Add user_id column for multi-tenant memory isolation
            try:
                cursor.execute("ALTER TABLE memories ADD COLUMN user_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError:
                pass
            conn.commit()

    def count(self):
        """Returns the number of stored memories."""
        if self.use_chroma:
            try:
                return self.collection.count()
            except Exception:
                pass
        
        # SQLite count
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
        return count

    def recall(self, task, n_results=2, embed_fn=None, domain=None):
        """
        Search memory for past experiences related to the current task.
        Prioritizes successful solution patterns and filters by domain to eliminate cross-domain noise.
        """
        if self.count() == 0:
            return ""

        # Try ChromaDB query first
        if self.use_chroma:
            try:
                results = self.collection.query(
                    query_texts=[task],
                    n_results=n_results * 3,  # Fetch extra to filter by domain and threshold
                    include=["documents", "metadatas", "distances"]
                )
                if results and results.get('documents') and results['documents'][0]:
                    docs = results['documents'][0]
                    metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
                    distances = results.get('distances', [[]])[0] if results.get('distances') else []
                    
                    filtered_docs = []
                    for i, doc in enumerate(docs):
                        meta = metadatas[i] if i < len(metadatas) else {}
                        
                        # Domain partitioning filter
                        if domain and meta.get('domain') and meta.get('domain') not in [domain, 'general']:
                            continue
                            
                        if i < len(distances):
                            l2_dist = distances[i]
                            approx_cosine = 1.0 - (l2_dist ** 2) / 2.0
                            if approx_cosine < MIN_VECTOR_SCORE:
                                continue  # Below similarity threshold — noise
                                
                        filtered_docs.append((doc, meta))
                    
                    if filtered_docs:
                        solutions = []
                        mistakes = []
                        for doc, meta in filtered_docs:
                            if meta.get('type') == 'mistake_fix':
                                mistakes.append(doc)
                            else:
                                solutions.append(doc)
                        
                        filtered = solutions[:n_results]
                        if len(filtered) < n_results and mistakes:
                            filtered.append(mistakes[0])
                        
                        if not filtered:
                            filtered = [d for d, _ in filtered_docs[:n_results]]
                        
                        memories = "\n---\n".join(filtered)
                        if len(memories) > 3000:
                            cutoff = memories.rfind('\n\n', 0, 3000)
                            cutoff = cutoff if cutoff != -1 else 3000
                            memories = memories[:cutoff] + "\n\n... [TRUNCATED]"
                        return f"\n\nRelevant past experience:\n{memories}\n"
            except Exception as e:
                print(f"ChromaDB query failed: {str(e)}. Falling back to SQLite recall.")

        # SQLite Query Fallback
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, task, doc, metadata, embedding, created_at FROM memories ORDER BY created_at DESC LIMIT 500"
            )
            rows = cursor.fetchall()

        if not rows:
            return ""

        def _prioritize_and_format(scored_items, limit):
            if not scored_items:
                return ""

            best_score = scored_items[0][0]
            filtered_by_margin = [
                item for item in scored_items
                if (best_score - item[0]) <= TOP_HIT_MARGIN
            ]

            solutions = []
            mistakes = []
            for score, doc, meta_str in filtered_by_margin:
                meta = {}
                if meta_str:
                    try:
                        meta = json.loads(meta_str)
                    except Exception:
                        pass
                if domain and meta.get('domain') and meta.get('domain') not in [domain, 'general']:
                    continue
                if meta.get('type') == 'mistake_fix':
                    mistakes.append(doc)
                else:
                    solutions.append(doc)

            filtered = solutions[:limit]
            if len(filtered) < limit and mistakes:
                filtered.append(mistakes[0])

            if not filtered:
                return ""

            memories = "\n---\n".join(filtered)
            if len(memories) > 4000:
                cutoff = memories.rfind('\n\n', 0, 4000)
                cutoff = cutoff if cutoff != -1 else 4000
                memories = memories[:cutoff] + "\n\n... [TRUNCATED]"
            return f"\n\nRelevant past experience:\n{memories}\n"

        def _recency_bonus(created_at_str):
            if not created_at_str:
                return 0.5
            try:
                import datetime as _dt
                ts = _dt.datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                age_days = (_dt.datetime.utcnow() - ts).total_seconds() / 86400.0
                age_days = max(0.0, age_days)
                return math.exp(-age_days * math.log(2) / RECENCY_HALF_LIFE_DAYS)
            except Exception:
                return 0.5

        # Vector search
        if embed_fn and rows[0][4]:
            try:
                query_vector = np.array(embed_fn(task))
                norm_q = np.linalg.norm(query_vector)
                if norm_q == 0:
                    norm_q = 1e-10
                scores = []
                for mem_id, t_task, doc, meta_str, emb_blob, created_at in rows:
                    if not emb_blob:
                        continue
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                    norm_e = np.linalg.norm(emb)
                    if norm_e <= 0:
                        continue
                    cosine = float(np.dot(query_vector, emb) / (norm_q * norm_e))
                    if cosine < MIN_VECTOR_SCORE:
                        continue
                    recency = _recency_bonus(created_at)
                    final_score = (1.0 - RECENCY_WEIGHT) * cosine + RECENCY_WEIGHT * recency
                    scores.append((final_score, doc, meta_str))

                if scores:
                    scores.sort(key=lambda x: x[0], reverse=True)
                    formatted = _prioritize_and_format(scores, n_results)
                    if formatted:
                        return formatted
            except Exception as e:
                print(f"Memory Engine: Vector search failed ({e}). Falling back to keyword matching.")

        # Keyword matching fallback
        query_words = set(
            w.strip(",.!?") for w in task.lower().split()
            if w not in self._STOPWORDS and len(w) > 1
        )
        if not query_words:
            return ""

        keyword_scores = []
        for row in rows:
            mem_id, t_task, doc, meta_str, _emb, _created_at = row
            task_words = set(
                w.strip(",.!?") for w in t_task.lower().split()
                if w not in self._STOPWORDS and len(w) > 1
            )
            matches = len(query_words.intersection(task_words))
            if matches >= KEYWORD_MIN_MATCHES or (len(query_words) <= 2 and matches == len(query_words)):
                keyword_scores.append((float(matches), doc, meta_str))

        if keyword_scores:
            keyword_scores.sort(key=lambda x: x[0], reverse=True)
            return _prioritize_and_format(keyword_scores, n_results)

        return ""

    def _is_duplicate(self, task):
        """Check if a very similar task already exists in memory."""
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT task FROM memories")
            rows = cursor.fetchall()

        def _get_content_words(t):
            words = [w.strip(",.!?()\"';:") for w in t.lower().split()]
            return set(w for w in words if w and w not in self._STOPWORDS)
            
        def _get_numbers(t):
            return set(re.findall(r'-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b', t))

        task_words = _get_content_words(task)
        task_nums = _get_numbers(task)
        if not task_words:
            return False

        for (existing_task,) in rows:
            existing_words = _get_content_words(existing_task)
            existing_nums = _get_numbers(existing_task)
            
            if not existing_words:
                continue
            if task_nums != existing_nums:
                continue
                
            overlap = len(task_words & existing_words) / max(len(task_words), len(existing_words))
            if overlap > 0.95:
                return True
        return False

    def save(self, task, successful_code, metadata=None, embed_fn=None, domain="general", user_id='default'):
        """Save a compact knowledge summary with domain tagging and de-duplication."""
        if self._is_duplicate(task):
            return None

        mem_id = f"mem_{uuid.uuid4().hex}"

        imports = [line.strip() for line in successful_code.split("\n")
                   if line.strip().startswith(("import ", "from "))]
        libs = ", ".join(imports[:5]) if imports else "standard library"

        code_summary = ""
        if "```python" in successful_code:
            try:
                start = successful_code.find("```python") + 9
                end = successful_code.find("```", start)
                if end != -1:
                    code_summary = "VERIFIED SCRIPT:\n" + successful_code[start:end].strip()
            except Exception:
                pass
                
        if not code_summary:
            code_summary = successful_code[:2500].strip()
            if len(successful_code) > 2500:
                code_summary += "\n... [truncated]"

        doc = (
            f"Task: {task}\n"
            f"Domain: {domain}\n"
            f"Libraries: {libs}\n"
            f"Procedure Summary:\n{code_summary}"
        )

        meta = metadata if metadata else {"task": task, "type": "solution"}
        meta["domain"] = domain

        # Save to Chroma
        if self.use_chroma:
            try:
                self.collection.add(documents=[doc], metadatas=[meta], ids=[mem_id])
            except Exception as e:
                print(f"Chroma save failed: {str(e)}")

        # Save to SQLite
        emb_blob = None
        if embed_fn:
            try:
                emb = embed_fn(task)
                emb_blob = np.array(emb, dtype=np.float32).tobytes()
            except Exception as e:
                print(f"Embedding generation failed: {str(e)}")

        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memories (id, task, doc, metadata, embedding, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                (mem_id, task, doc, json.dumps(meta), emb_blob, user_id)
            )
            conn.commit()

        return mem_id

    def save_mistake(self, task, wrong_code, error_log, fixed_code, embed_fn=None, domain="general", user_id='default'):
        """Save a compact mistake-fix pattern to prevent regression."""
        mem_id = f"mistake_{uuid.uuid4().hex}"

        error_essence = error_log.strip()[:300]
        wrong_lines = set(wrong_code.strip().split("\n"))
        fixed_lines = set(fixed_code.strip().split("\n"))
        removed = list(wrong_lines - fixed_lines)[:5]
        added = list(fixed_lines - wrong_lines)[:5]

        doc = (
            f"Task: {task}\n"
            f"Domain: {domain}\n"
            f"Error: {error_essence}\n"
            f"Root Cause (removed lines): {'; '.join(l.strip() for l in removed) if removed else 'structural change'}\n"
            f"Fix Pattern (added lines): {'; '.join(l.strip() for l in added) if added else 'structural change'}"
        )

        meta = {"task": task, "type": "mistake_fix", "domain": domain}

        if self.use_chroma:
            try:
                self.collection.add(documents=[doc], metadatas=[meta], ids=[mem_id])
            except Exception as e:
                print(f"Chroma save mistake failed: {str(e)}")

        emb_blob = None
        if embed_fn:
            try:
                emb = embed_fn(task)
                emb_blob = np.array(emb, dtype=np.float32).tobytes()
            except Exception as e:
                print(f"Embedding generation failed: {str(e)}")

        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memories (id, task, doc, metadata, embedding, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                (mem_id, task, doc, json.dumps(meta), emb_blob, user_id)
            )
            conn.commit()

        return mem_id

    def store(self, task, doc, metadata=None, embed_fn=None, domain="general", user_id='default'):
        """General-purpose store method with domain tagging."""
        if self._is_duplicate(task):
            return None

        mem_id = f"mem_{uuid.uuid4().hex}"
        meta = metadata if metadata else {"task": task, "type": "solution"}
        meta["domain"] = domain

        if self.use_chroma:
            try:
                self.collection.add(documents=[doc], metadatas=[meta], ids=[mem_id])
            except Exception as e:
                print(f"Chroma store failed: {str(e)}")

        emb_blob = None
        if embed_fn:
            try:
                emb = embed_fn(task)
                emb_blob = np.array(emb, dtype=np.float32).tobytes()
            except Exception as e:
                print(f"Embedding generation failed: {str(e)}")

        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memories (id, task, doc, metadata, embedding, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                (mem_id, task, doc, json.dumps(meta), emb_blob, user_id)
            )
            conn.commit()

        return mem_id

    def list_memories(self, limit=50, user_id='default'):
        """Lists recent long-term memories from ChromaDB/SQLite."""
        results = []
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, task, doc, metadata, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            )
            rows = cursor.fetchall()

        for mem_id, task, doc, meta_str, created_at in rows:
            try:
                meta = json.loads(meta_str) if meta_str else {}
            except Exception:
                meta = {}
            results.append({
                "id": mem_id,
                "task": task,
                "domain": meta.get("domain", "general"),
                "preview": (doc[:200] + "...") if len(doc) > 200 else doc,
                "metadata": meta,
                "created_at": created_at
            })

        return results

    def delete_memory(self, memory_id: str, user_id='default') -> bool:
        """Deletes a specific memory entry from SQLite and ChromaDB."""
        deleted = False
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id)
            )
            deleted = cursor.rowcount > 0
            conn.commit()

        if self.use_chroma and self.collection:
            try:
                self.collection.delete(ids=[memory_id])
            except Exception:
                pass

        return deleted

    def clear(self):
        """Clears all stored memories."""
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories")
            conn.commit()

        if self.use_chroma and self.collection:
            try:
                self.chroma_client.delete_collection(name="knowledge")
                self.collection = self.chroma_client.get_or_create_collection(name="knowledge")
            except Exception:
                pass
        return True

    def compact_memory(self, max_entries=1000, max_age_days=180):
        """Purges stale memories older than max_age_days to keep vector latency < 5ms."""
        try:
            import datetime
            cutoff = (datetime.datetime.now() - datetime.timedelta(days=max_age_days)).isoformat()
            conn = sqlite3.connect(self.sqlite_path, timeout=30.0)
            cur = conn.cursor()
            cur.execute("DELETE FROM memories WHERE created_at < ?", (cutoff,))
            # Enforce max_entries: keep only the most recent
            cur.execute(
                "DELETE FROM memories WHERE id NOT IN "
                "(SELECT id FROM memories ORDER BY created_at DESC LIMIT ?)",
                (max_entries,)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        return True
