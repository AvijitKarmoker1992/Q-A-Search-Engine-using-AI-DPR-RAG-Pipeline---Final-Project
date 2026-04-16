"""
=============================================================================
CSCI4144 - Final Project
Student : Avijit Karmoker  |  ID: B00825518
Professor: Qiang Ye
Course  : CSCI4144
 
 
INSTALL DEPENDENCIES (run once on timberlea):
    pip3 install --user transformers torch numpy matplotlib tqdm rank_bm25
 
HOW TO RUN ON TIMBERLEA:
    1. Upload this file to timberlea via Cyberduck (SFTP)
    2. SSH into timberlea:   ssh <csid>@timberlea.cs.dal.ca
    3. Install deps:         pip3 install --user transformers torch numpy matplotlib tqdm rank_bm25
    4. Run:                  python3 dpr_rag_project.py
 

=============================================================================
"""
 
# Standard library
import os, re, sys, math, time, json, warnings, collections
warnings.filterwarnings("ignore")
 
# Dependencies used
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend safe on timberlea
import matplotlib.pyplot as plt
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
 
 
# =============================================================================
# WEEK 1 – DATASET PREPARATION
# =============================================================================
 
RAW_DATA = [
    # (passage_id, passage_text, query, is_relevant)
    # Science
    ("p001","Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water. Chlorophyll in the leaves absorbs solar energy.","How do plants make food from sunlight?",True),
    ("p002","The mitochondria are organelles found in eukaryotic cells that generate most of the cell's supply of ATP, used as a source of chemical energy.","What produces energy inside a cell?",True),
    ("p003","Newton's first law of motion states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by an external force.","What is inertia in physics?",True),
    ("p004","The speed of light in a vacuum is approximately 299,792,458 metres per second, a universal physical constant denoted by the letter c.","How fast does light travel?",True),
    ("p005","DNA, or deoxyribonucleic acid, carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms.","What molecule carries genetic information?",True),
    ("p006","Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another. On Earth, gravity gives weight to physical objects.","Why do objects fall toward Earth?",True),
    ("p007","The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow.","How does rain form in the atmosphere?",True),
    ("p008","Vaccines work by training the immune system to recognise and combat pathogens. They contain weakened or inactivated parts of a particular organism.","How does vaccination protect against disease?",True),
    ("p009","Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once it crosses the event horizon.","What is a black hole in space?",True),
    ("p010","The periodic table organises all known chemical elements by atomic number, electron configuration, and recurring chemical properties.","How are chemical elements organised?",True),
    # Medicine
    ("p011","A myocardial infarction, commonly known as a heart attack, occurs when blood flow to a part of the heart is blocked, causing tissue damage.","What happens to the heart during a heart attack?",True),
    ("p012","Hypertension, or high blood pressure, is a long-term medical condition in which blood pressure in the arteries is persistently elevated.","What are the risks of high blood pressure?",True),
    ("p013","The immune system is a network of cells and proteins that defends the body against infection. White blood cells are the key players.","How does the body fight infections?",True),
    ("p014","Sleep allows the brain to consolidate memories, clear metabolic waste, and restore energy. Adults need 7 to 9 hours of sleep per night.","Why is sleep important for the brain?",True),
    ("p015","Cancer occurs when cells in the body grow and divide uncontrollably, potentially invading and spreading to other parts of the body.","How does cancer develop in the body?",True),
    ("p016","Antibiotics are medications that destroy or slow down the growth of bacteria. They are not effective against viral infections.","When should antibiotics be used?",True),
    ("p017","Insulin is a hormone produced by the pancreas that allows cells to absorb glucose from the bloodstream for energy.","What does insulin do in the body?",True),
    ("p018","Exercise improves cardiovascular health, strengthens muscles, enhances mood through endorphin release, and reduces risk of chronic disease.","What are the benefits of regular exercise?",True),
    ("p019","The human brain contains approximately 86 billion neurons connected by trillions of synapses, enabling complex thought and behaviour.","How does the human brain process information?",True),
    ("p020","Dehydration occurs when the body loses more fluid than it takes in. Symptoms include thirst, headache, dizziness and dark urine.","What are the signs of dehydration?",True),
    # History
    ("p021","World War II began in 1939 when Germany invaded Poland. It involved most of the world nations and ended in 1945.","When did the Second World War start?",True),
    ("p022","The Roman Empire reached its greatest extent under Emperor Trajan in 117 AD, covering much of Europe, North Africa and the Middle East.","How large was the Roman Empire at its peak?",True),
    ("p023","The French Revolution began in 1789 and led to the abolition of the French monarchy and the rise of Napoleon Bonaparte.","What caused the fall of the French monarchy?",True),
    ("p024","The printing press was invented by Johannes Gutenberg around 1440, revolutionising the production of books and the spread of knowledge.","Who invented the printing press?",True),
    ("p025","The Cold War was a period of geopolitical tension between the United States and the Soviet Union lasting from 1947 to 1991.","What was the Cold War about?",True),
    ("p026","Ancient Egypt developed one of the earliest writing systems, hieroglyphics, around 3200 BCE, used for religious and administrative records.","What writing system did ancient Egyptians use?",True),
    ("p027","The Industrial Revolution began in Britain in the late 18th century and led to the shift from hand production to machine manufacturing.","How did the Industrial Revolution change manufacturing?",True),
    ("p028","The Magna Carta was signed in 1215 by King John of England and limited the powers of the monarch, forming a basis for constitutional law.","What was the significance of the Magna Carta?",True),
    ("p029","The Space Race was a competition between the USA and USSR during the Cold War to achieve superior spaceflight capability.","Why did the USA and USSR compete in space?",True),
    ("p030","The Renaissance was a cultural movement from the 14th to 17th century in Europe that renewed interest in classical art, philosophy and science.","What was the Renaissance period in Europe?",True),
    # Technology
    ("p031","Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.","How does machine learning work?",True),
    ("p032","The internet is a global network of computers connected through standardised communication protocols, enabling information exchange worldwide.","What is the internet and how does it work?",True),
    ("p033","Encryption is the process of encoding information so that only authorised parties can access it, protecting data from unauthorised access.","How does data encryption protect privacy?",True),
    ("p034","A database is an organised collection of structured information stored electronically, typically managed by a database management system.","What is a database used for?",True),
    ("p035","Cloud computing delivers computing services including servers, storage, databases, and networking over the internet to offer faster innovation.","What is cloud computing?",True),
    ("p036","A neural network is a series of algorithms that attempt to recognise underlying relationships in data through a process that mimics the human brain.","How do neural networks recognise patterns?",True),
    ("p037","The World Wide Web was invented by Tim Berners-Lee in 1989 as an information system using hypertext links to connect documents.","Who invented the World Wide Web?",True),
    ("p038","Cybersecurity refers to the practice of protecting systems, networks, and programs from digital attacks, data breaches, and unauthorised access.","What does cybersecurity involve?",True),
    ("p039","Blockchain is a distributed ledger technology in which transactions are recorded in blocks chained together using cryptographic hashes.","How does blockchain technology store data?",True),
    ("p040","5G is the fifth-generation mobile network offering higher bandwidth, lower latency, and greater capacity than previous generations.","What improvements does 5G bring over 4G?",True),
    # Geography
    ("p041","The Amazon rainforest covers about 5.5 million square kilometres and is home to 10 percent of all species on Earth.","Why is the Amazon rainforest important?",True),
    ("p042","Mount Everest, located in the Himalayas on the Nepal-Tibet border, is the highest peak on Earth at 8,849 metres above sea level.","Where is the tallest mountain on Earth?",True),
    ("p043","The Sahara Desert spans about 9.2 million square kilometres across North Africa, making it the largest hot desert in the world.","How big is the Sahara Desert?",True),
    ("p044","Canada is the second-largest country in the world by total area, with Ottawa as its capital city.","What is the capital of Canada?",True),
    ("p045","The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 165 million square kilometres.","What is the largest ocean on Earth?",True),
    ("p046","The Nile River is the longest river in Africa and one of the longest in the world, flowing northward through northeastern Africa.","What is the longest river in Africa?",True),
    ("p047","Volcanoes form when magma from beneath the Earth crust erupts through the surface, building up layers of lava and ash.","How are volcanoes formed?",True),
    ("p048","The Northern Lights or Aurora Borealis are natural light displays caused by charged particles from the sun interacting with Earth magnetic field.","What causes the Northern Lights?",True),
    ("p049","Earthquakes occur when tectonic plates beneath the Earth surface suddenly slip past one another, releasing stored energy as seismic waves.","Why do earthquakes happen?",True),
    ("p050","The Great Barrier Reef off Queensland Australia is the world largest coral reef system, spanning over 2,300 kilometres.","Where is the world largest coral reef?",True),
    # Economics
    ("p051","Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power over time.","How does inflation affect everyday prices?",True),
    ("p052","Gross Domestic Product GDP measures the total monetary value of all goods and services produced within a country in a given period.","What does GDP measure in an economy?",True),
    ("p053","Supply and demand is an economic model that determines the price and quantity of goods in a market based on buyer and seller behaviour.","How do supply and demand affect prices?",True),
    ("p054","Unemployment refers to the situation where individuals actively seeking work are unable to find employment.","What causes unemployment in an economy?",True),
    ("p055","Globalisation refers to the increasing interconnectedness of economies, cultures, and populations across the world through trade and technology.","What is globalisation and its effects?",True),
    # Literature
    ("p056","William Shakespeare wrote 37 plays and 154 sonnets. His works include Hamlet, Macbeth, Othello, and Romeo and Juliet.","Who wrote Hamlet and Romeo and Juliet?",True),
    ("p057","The Mona Lisa is a portrait painted by Leonardo da Vinci between 1503 and 1519, now housed in the Louvre Museum in Paris.","Who painted the Mona Lisa?",True),
    ("p058","Jazz music originated in the African-American communities of New Orleans in the late 19th and early 20th centuries.","Where did jazz music originate?",True),
    ("p059","The Nobel Prize is awarded annually to individuals or organisations that have made outstanding contributions in physics, chemistry, literature, peace, and medicine.","What is the Nobel Prize awarded for?",True),
    ("p060","Poetry uses aesthetic and rhythmic qualities of language to evoke meanings and emotions beyond ordinary speech.","How is poetry different from regular writing?",True),
    # Distractors (not relevant to any query)
    ("p061","The FIFA World Cup is held every four years and is the most widely viewed sporting event in the world.","irrelevant_q1",False),
    ("p062","Pasta is a staple food of Italian cuisine made from durum wheat semolina mixed with water or eggs.","irrelevant_q2",False),
    ("p063","Tornadoes are rotating columns of air that extend from a thunderstorm to the ground.","irrelevant_q3",False),
    ("p064","The stock market is a marketplace where buyers and sellers trade shares of publicly listed companies.","irrelevant_q4",False),
    ("p065","Yoga is an ancient practice originating in India that combines physical postures, breathing techniques, and meditation.","irrelevant_q5",False),
    ("p066","Coffee is one of the most popular beverages worldwide and is brewed from roasted coffee beans.","irrelevant_q6",False),
    ("p067","The Eiffel Tower was constructed between 1887 and 1889 as the entrance arch for the 1889 World Fair in Paris.","irrelevant_q7",False),
    ("p068","Basketball was invented by Dr. James Naismith in 1891 in Springfield, Massachusetts.","irrelevant_q8",False),
    ("p069","The Amazon River discharges more water into the ocean than any other river on Earth.","irrelevant_q9",False),
    ("p070","Chocolate is made from cacao beans, which are fermented, roasted, and ground into a paste called chocolate liquor.","irrelevant_q10",False),
    ("p071","The greenhouse effect occurs when gases in Earth atmosphere trap heat from the sun, warming the planet surface.","irrelevant_q11",False),
    ("p072","Origami is the Japanese art of paper folding, used to create sculptures by folding a flat sheet of paper.","irrelevant_q12",False),
    ("p073","Sonar uses sound waves to detect objects underwater and is widely used in submarines and marine research.","irrelevant_q13",False),
    ("p074","A constitution is a body of fundamental principles according to which a state or organisation is governed.","irrelevant_q14",False),
    ("p075","The Mediterranean diet emphasises fruits, vegetables, whole grains, legumes, nuts, and olive oil.","irrelevant_q15",False),
    ("p076","Morse code is a method used in telecommunications to encode text characters using sequences of dots and dashes.","irrelevant_q16",False),
    ("p077","The Great Wall of China was built over centuries to protect Chinese states from northern invasions.","irrelevant_q17",False),
    ("p078","Tides are caused by the gravitational pull of the Moon and Sun on the Earth oceans.","irrelevant_q18",False),
    ("p079","The violin is a string instrument played with a bow, originating in 16th century Italy.","irrelevant_q19",False),
    ("p080","Fossil fuels including coal, oil, and natural gas formed from the remains of ancient organisms over millions of years.","irrelevant_q20",False),
]
 
 
class DatasetBuilder:
    """
    Week 1: Builds corpus and query set from the built-in RAW_DATA.
    Mirrors the structure of MS MARCO Passage Ranking v2.1.
    No internet connection required .
    """
    def __init__(self):
        self.passages     = []
        self.passage_ids  = []
        self.queries      = []
        self.relevant_ids = []
 
    def build(self):
        print("=" * 65)
        print("WEEK 1 - Dataset Preparation and Preprocessing")
        print("=" * 65)
        print("  Source : Built-in QA corpus (80 passages, 60 queries)")
        print("  Note   : Mirrors MS MARCO Passage Ranking v2.1 structure")
        print()
 
        query_to_pids = collections.defaultdict(list)
        for pid, ptext, query, is_rel in RAW_DATA:
            clean = " ".join(ptext.strip().split())
            self.passages.append(clean)
            self.passage_ids.append(pid)
            if is_rel:
                query_to_pids[query].append(pid)
 
        for q, rel_pids in query_to_pids.items():
            self.queries.append(q)
            self.relevant_ids.append(rel_pids)
 
        print(f"  Corpus size  : {len(self.passages)} passages")
        print(f"  Query count  : {len(self.queries)} queries")
        print()
 
    def preview(self, n=3):
        print("-- Sample Passages --")
        for i in range(min(n, len(self.passages))):
            print(f"  [{self.passage_ids[i]}] {self.passages[i][:100]}...")
        print()
        print("-- Sample Queries --")
        for i in range(min(n, len(self.queries))):
            print(f"  Q{i+1}: {self.queries[i]}")
            print(f"       Relevant: {self.relevant_ids[i]}")
        print()
 
 
# =============================================================================
# WEEK 2  BM25 KEYWORD RETRIEVAL
# =============================================================================
 
class BM25Retriever:
    """
    Week 2: BM25 Okapi keyword-based retrieval (baseline).
 
    Scores passages by term-frequency weighted by inverse document frequency.
    Cannot handle semantic mismatches: 'myocardial infarction' vs 'heart attack'
    are treated as completely different despite identical meaning.
 
    Reference:
        Robertson and Zaragoza (2009). The Probabilistic Relevance
        Framework: BM25 and Beyond.
    """
    def __init__(self, passages, passage_ids):
        print("-- BM25: Building index...", end=" ", flush=True)
        t0 = time.time()
        self.bm25        = BM25Okapi([p.lower().split() for p in passages])
        self.passage_ids = passage_ids
        self.passages    = passages
        print(f"done ({time.time()-t0:.2f}s)")
 
    def retrieve(self, query, top_k=10):
        scores  = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.passage_ids[i], float(scores[i]), self.passages[i])
                for i in top_idx]
 
 
# =============================================================================
# WEEK 2-B  DENSE BI-ENCODER
# =============================================================================
 
class DenseEncoder:
    """
    Week 2: Dense passage and query encoding using sentence-transformers.

    Model: all-MiniLM-L6-v2
        - 22M parameters, highly optimized for semantic similarity
        - ~5-10x faster than raw DistilBERT on CPU
        - Produces 384-dim L2-normalized sentence embeddings

    Implements the bi-encoder architecture from:
        Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain
        Question Answering. EMNLP 2020.
    """
    MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"-- Loading bi-encoder: {self.MODEL}...", end=" ", flush=True)
        self.model = SentenceTransformer(self.MODEL)
        print("done")

    def encode_passages(self, passages, batch_size=64):
        print(f"-- Encoding {len(passages)} passages...")
        vecs = self.model.encode(
            passages,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalize for cosine via dot
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def encode_query(self, query):
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype("float32")
 
 
# =============================================================================
# WEEK 3-A  NUMPY VECTOR INDEX  (replaces FAISS for timberlea)
# =============================================================================
 
class NumpyVectorIndex:
    """
    Week 3: Exact nearest-neighbour search using NumPy dot product.
 
    Replaces FAISS for timberlea.cs.dal.ca compatibility.
    For L2-normalised vectors, dot product equals cosine similarity.
    Exact search is appropriate for corpora under ~100k passages.
 
    For production scale (millions of passages), FAISS IndexIVFPQ
    or ScaNN would be used as described in Karpukhin et al. (2020).
    """
    def __init__(self):
        self.embeddings  = None
        self.passage_ids = []
        self.passages    = []
 
    def add(self, embeddings, passage_ids, passages):
        self.embeddings  = embeddings
        self.passage_ids = passage_ids
        self.passages    = passages
        print(f"-- NumPy index ready: {len(passages)} vectors "
              f"dim={embeddings.shape[1]}")
 
    def search(self, qvec, top_k=10):
        scores  = (self.embeddings @ qvec.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.passage_ids[i], float(scores[i]), self.passages[i])
                for i in top_idx]
 
    def save(self, path="vector_index.npz"):
        np.savez(path, embeddings=self.embeddings,
                 passage_ids=np.array(self.passage_ids),
                 passages=np.array(self.passages))
        print(f"  Index saved -> {path}")
 
    def load(self, path="vector_index.npz"):
        d = np.load(path, allow_pickle=True)
        self.embeddings  = d["embeddings"]
        self.passage_ids = list(d["passage_ids"])
        self.passages    = list(d["passages"])
        print(f"  Index loaded <- {path}")
 
 
# =============================================================================
# WEEK 3-B  RAG PIPELINE
# =============================================================================
 
class RAGPipeline:
    """
    Week 3: Retrieval-Augmented Generation pipeline.
 
    1. Retrieve top-k passages via DPR (or BM25 for comparison).
    2. Concatenate retrieved passages as context.
    3. Extract the best answer sentence using token-overlap scoring.
 
    Reference:
        Lewis et al. (2020). Retrieval-Augmented Generation for
        Knowledge-Intensive NLP Tasks. NeurIPS 2020.
    """
    def __init__(self, index, encoder, bm25):
        self.index   = index
        self.encoder = encoder
        self.bm25    = bm25
 
    @staticmethod
    def _extract(question, context):
        qtoks = set(re.sub(r"[^\w\s]", "", question).lower().split())
        best_sent, best_score = "", 0.0
        for sent in re.split(r"(?<=[.!?])\s+", context):
            stoks = set(re.sub(r"[^\w\s]", "", sent).lower().split())
            score = len(qtoks & stoks) / max(len(qtoks), 1)
            if score > best_score:
                best_score, best_sent = score, sent
        return best_sent.strip(), round(best_score, 4)
 
    def answer(self, question, method="dpr", top_k=5):
        t0 = time.time()
        if method == "dpr":
            results = self.index.search(self.encoder.encode_query(question), top_k)
        else:
            results = self.bm25.retrieve(question, top_k)
        context = " ".join(r[2] for r in results)[:3000]
        ans, conf = self._extract(question, context)
        return {
            "question"           : question,
            "method"             : method.upper(),
            "answer"             : ans or "No answer found.",
            "confidence"         : conf,
            "latency_ms"         : round((time.time()-t0)*1000, 1),
            "retrieved_passages" : [(r[0], round(r[1],4), r[2][:90]+"...")
                                    for r in results],
        }
 
 
# =============================================================================
# WEEK 4  EVALUATION
# =============================================================================
 
class Evaluator:
    """
    Week 4: Standard IR retrieval metrics.
 
    MRR at k   - Mean Reciprocal Rank: how high is the first relevant passage?
    Recall at k - Fraction of queries with at least one relevant passage in top-k.
    P at k     - Precision at k: average fraction of top-k that are relevant.
    """
    @staticmethod
    def mrr(retrieved, relevant, k):
        total = 0.0
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            for rank, pid in enumerate(ret[:k], 1):
                if pid in rel_set:
                    total += 1.0 / rank
                    break
        return total / max(len(retrieved), 1)
 
    @staticmethod
    def recall(retrieved, relevant, k):
        hits = sum(1 for ret, rel in zip(retrieved, relevant)
                   if set(ret[:k]) & set(rel))
        return hits / max(len(retrieved), 1)
 
    @staticmethod
    def precision(retrieved, relevant, k):
        total = 0.0
        for ret, rel in zip(retrieved, relevant):
            total += sum(1 for pid in ret[:k] if pid in set(rel)) / k
        return total / max(len(retrieved), 1)
 
    def evaluate(self, bm25_ret, dpr_ret, relevant):
        rows = []
        for method, ret in [("BM25", bm25_ret), ("DPR", dpr_ret)]:
            for k in [1, 5, 10]:
                rows.append({
                    "Method"  : method,
                    "k"       : k,
                    "MRR at k"   : round(self.mrr(ret, relevant, k),       4),
                    "Recall at k": round(self.recall(ret, relevant, k),     4),
                    "P at k"     : round(self.precision(ret, relevant, k),  4),
                })
        return rows
 
    @staticmethod
    def plot(rows, save_path="dpr_vs_bm25_results.png"):
        metrics = ["MRR at k", "Recall at k", "P at k"]
        k_vals  = [1, 5, 10]
        x       = np.arange(len(k_vals))
        width   = 0.32
        colors  = {"BM25": "#4472C4", "DPR": "#ED7D31"}
 
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "DPR vs BM25 - Retrieval Performance Comparison\n"
            "CSCI4144 Project | Avijit Karmoker | B00825518",
            fontsize=12, fontweight="bold")
 
        for ax, metric in zip(axes, metrics):
            for i, method in enumerate(["BM25", "DPR"]):
                vals = [r[metric] for r in rows
                        if r["Method"] == method and r["k"] in k_vals]
                vals = [v for _, v in sorted(
                    zip([r["k"] for r in rows if r["Method"]==method
                         and r["k"] in k_vals], vals))]
                bars = ax.bar(x + (i-0.5)*width, vals, width,
                               label=method, color=colors[method],
                               alpha=0.88, edgecolor="white")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+0.01,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=8.5, fontweight="bold")
            ax.set_title(metric, fontsize=11, fontweight="bold")
            ax.set_xlabel("Cut-off k")
            ax.set_ylabel("Score")
            ax.set_xticks(x)
            ax.set_xticklabels([f"k={v}" for v in k_vals])
            max_v = max(r[metric] for r in rows)
            ax.set_ylim(0, min(1.15, max_v * 1.5 + 0.05))
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines[["top","right"]].set_visible(False)
 
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved -> {save_path}")
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def main():
    print()
    print("=" * 65)
    print("  CSCI4144 - Final Project")
    print("  Student  : Avijit Karmoker  |  B00825518")
    print("=" * 65)
    print()
 
    # WEEK 1
    ds = DatasetBuilder()
    ds.build()
    ds.preview(n=3)
 
    passages    = ds.passages
    passage_ids = ds.passage_ids
    queries     = ds.queries
    relevant    = ds.relevant_ids
 
    # WEEK 2
    print("=" * 65)
    print("WEEK 2 - BM25 Index and Dense Bi-Encoder")
    print("=" * 65)
    print()
    bm25            = BM25Retriever(passages, passage_ids)
    encoder         = DenseEncoder()
    print()
    pass_embeddings = encoder.encode_passages(passages, batch_size=16)
 
    # WEEK 3
    print()
    print("=" * 65)
    print("WEEK 3 - NumPy Vector Index and RAG Pipeline")
    print("=" * 65)
    print()
    index = NumpyVectorIndex()
    index.add(pass_embeddings, passage_ids, passages)
    index.save("vector_index.npz")
    print()
    rag = RAGPipeline(index=index, encoder=encoder, bm25=bm25)
 
    # Test cases
    test_cases = [
        # Category 1: Semantic gap (query words differ from passage; DPR advantage)
        ("SEMANTIC GAP",  "What happens to the heart during a heart attack?"),
        ("SEMANTIC GAP",  "What produces energy inside a cell?"),
        ("SEMANTIC GAP",  "What are the risks of high blood pressure?"),
        ("SEMANTIC GAP",  "How does vaccination protect against disease?"),
        ("SEMANTIC GAP",  "Why do objects fall toward Earth?"),
        # Category 2: Keyword match (BM25 may equal or beat DPR)
        ("KEYWORD MATCH", "Who wrote Hamlet and Romeo and Juliet?"),
        ("KEYWORD MATCH", "Who invented the printing press?"),
        ("KEYWORD MATCH", "What is the capital of Canada?"),
        ("KEYWORD MATCH", "Who painted the Mona Lisa?"),
        ("KEYWORD MATCH", "Where did jazz music originate?"),
        # Category 3: Complex multi-concept (tests full RAG quality)
        ("COMPLEX RAG",   "How does the body fight infections?"),
        ("COMPLEX RAG",   "Why is sleep important for the brain?"),
        ("COMPLEX RAG",   "What are the benefits of regular exercise?"),
        ("COMPLEX RAG",   "How does cancer develop in the body?"),
        ("COMPLEX RAG",   "How do neural networks recognise patterns?"),
    ]
 
    print("-- RAG Demo Results --")
    print()
    for category, question in test_cases:
        print(f"  [{category}]")
        print(f"  Question : {question}")
        for method in ("dpr", "bm25"):
            res = rag.answer(question, method=method, top_k=5)
            print(f"  {res['method']:4s} Answer    : {res['answer'][:85]}")
            print(f"       Confidence : {res['confidence']}  "
                  f"Latency: {res['latency_ms']} ms")
            top = res["retrieved_passages"][0]
            print(f"       Top passage: [{top[0]}] score={top[1]:.4f}  "
                  f"{top[2][:60]}")
        print()
 
    # WEEK 4
    print("=" * 65)
    print("WEEK 4 - Evaluation: MRR at k, Recall at k, Precision at k")
    print("=" * 65)
    print()
 
    print("  Running BM25 retrieval over all queries...")
    bm25_ret = []
    for q in tqdm(queries, desc="  BM25"):
        bm25_ret.append([r[0] for r in bm25.retrieve(q, top_k=10)])
 
    print("  Running DPR retrieval over all queries...")
    dpr_ret = []
    for q in tqdm(queries, desc="  DPR "):
        dpr_ret.append([r[0] for r in
                        index.search(encoder.encode_query(q), top_k=10)])
 
    evaluator = Evaluator()
    rows      = evaluator.evaluate(bm25_ret, dpr_ret, relevant)
 
    # Print results table
    print()
    print("-- Evaluation Results --")
    print(f"  {'Method':<8} {'k':>4}  {'MRR at k':>8}  {'Recall at k':>10}  {'P@k':>8}")
    print("  " + "-" * 44)
    for r in rows:
        print(f"  {r['Method']:<8} {r['k']:>4}  "
              f"{r['MRR at k']:>8.4f}  {r['Recall at k']:>10.4f}  {r['P at k']:>8.4f}")
 
    # Summary
    dpr_mrr10  = next(r["MRR at k"]  for r in rows if r["Method"]=="DPR"  and r["k"]==10)
    bm25_mrr10 = next(r["MRR at k"]  for r in rows if r["Method"]=="BM25" and r["k"]==10)
    dpr_rec10  = next(r["Recall at k"] for r in rows if r["Method"]=="DPR"  and r["k"]==10)
    bm25_rec10 = next(r["Recall at k"] for r in rows if r["Method"]=="BM25" and r["k"]==10)
    print()
    print(f"  MRR@10    DPR={dpr_mrr10:.4f}  BM25={bm25_mrr10:.4f}  "
          f"Winner={'DPR' if dpr_mrr10>=bm25_mrr10 else 'BM25'}")
    print(f"  Recall at 10 DPR={dpr_rec10:.4f}  BM25={bm25_rec10:.4f}  "
          f"Winner={'DPR' if dpr_rec10>=bm25_rec10 else 'BM25'}")
 
    # Plot and save
    evaluator.plot(rows, save_path="dpr_vs_bm25_results.png")
 
    # Save CSV manually (no pandas dependency)
    with open("evaluation_results.csv", "w") as f:
        f.write("Method,k,MRR at k,Recall at k,P at k\n")
        for r in rows:
            f.write(f"{r['Method']},{r['k']},{r['MRR at k']},"
                    f"{r['Recall at k']},{r['P at k']}\n")
    print("  Results saved -> evaluation_results.csv")
 
    print()
    print("=" * 65)
    print("  PROJECT COMPLETE")
    print("  Output files:")
    print("    vector_index.npz        - saved vector index")
    print("    dpr_vs_bm25_results.png - comparison bar chart")
    print("    evaluation_results.csv  - full metrics table")
    print("=" * 65)
    print()
 
 
if __name__ == "__main__":
    main()