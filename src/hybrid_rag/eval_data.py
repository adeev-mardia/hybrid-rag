"""A small, hand-authored IR evaluation set used by scripts/benchmark.py.

Structured the way standard IR benchmarks (BEIR, MS MARCO, etc.) are: a
corpus of documents, a set of queries, and qrels (query relevance judgments
mapping each query to the doc_ids that are actually relevant to it).

This is deliberately self-contained rather than downloaded, for two
reasons: it keeps the benchmark reproducible with zero network access, and
it lets the corpus be *designed* to expose the actual difference between
retrieval strategies -- several queries use vocabulary that deliberately
doesn't overlap with their relevant document's wording (testing whether
dense/semantic retrieval finds them), and several documents on different
topics deliberately share surface-level keywords (testing whether sparse
retrieval gets confused by them). A scraped dataset wouldn't let you control
that.
"""

from __future__ import annotations

from hybrid_rag.document import Document

_RAW_DOCS: dict[str, str] = {
    # --- Astronomy ---
    "astro-1": "Jupiter is the largest planet in the Solar System, a gas giant composed mainly of hydrogen and helium.",
    "astro-2": "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape it.",
    "astro-3": "The James Webb Space Telescope observes the universe primarily in infrared, revealing galaxies formed shortly after the Big Bang.",
    "astro-4": "Saturn's rings are made of countless small particles of ice and rock, ranging from tiny grains to house-sized chunks.",
    "astro-5": "A light-year measures distance, not time: it is how far light travels in a vacuum over one Earth year.",
    # --- Machine learning ---
    "ml-1": "A neural network learns by adjusting the weights of its connections to minimize a loss function via backpropagation.",
    "ml-2": "Overfitting occurs when a model learns the noise in the training data instead of the underlying pattern, hurting generalization.",
    "ml-3": "Random forests combine many decision trees trained on bootstrapped samples to reduce variance and improve robustness.",
    "ml-4": "Gradient descent iteratively updates model parameters in the direction that most reduces the loss.",
    "ml-5": "Transformer architectures use self-attention to weigh the relevance of every token to every other token in a sequence.",
    # --- Cooking ---
    "cook-1": "Searing meat at high heat before roasting creates a flavorful crust through the Maillard reaction.",
    "cook-2": "Kneading bread dough develops gluten strands, which trap gas bubbles from the yeast and give bread its structure.",
    "cook-3": "A roux is a mixture of flour and fat cooked together, used as a base to thicken sauces and soups.",
    "cook-4": "Blanching vegetables briefly in boiling water then plunging them into ice water preserves their color and crunch.",
    "cook-5": "Caramelizing onions slowly over low heat converts their natural sugars into a deep, sweet flavor.",
    # --- World history ---
    "hist-1": "The printing press, developed by Johannes Gutenberg around 1440, dramatically increased the spread of written knowledge in Europe.",
    "hist-2": "The Silk Road was a network of trade routes connecting East Asia to the Mediterranean for over a thousand years.",
    "hist-3": "The fall of the Berlin Wall in 1989 symbolized the end of the Cold War division between East and West Germany.",
    "hist-4": "Ancient Rome's aqueducts used gravity alone to carry fresh water over long distances into the city.",
    "hist-5": "The Treaty of Westphalia in 1648 established the modern concept of state sovereignty in international relations.",
    # --- Biology ---
    "bio-1": "Mitochondria generate most of a cell's ATP supply through cellular respiration, earning them the nickname 'powerhouse of the cell'.",
    "bio-2": "Photosynthesis converts light energy, water, and carbon dioxide into glucose and oxygen inside plant chloroplasts.",
    "bio-3": "DNA replication is semi-conservative: each new double helix contains one original strand and one newly synthesized strand.",
    "bio-4": "Natural selection favors traits that improve an organism's chance of survival and reproduction in its environment.",
    "bio-5": "Antibiotic resistance spreads when bacteria that survive treatment pass on resistance genes to their offspring.",
    # --- Programming languages / systems ---
    "prog-1": "Python's Global Interpreter Lock (GIL) prevents multiple native threads from executing Python bytecode simultaneously.",
    "prog-2": "Garbage collection automatically reclaims memory occupied by objects that are no longer reachable from the program's roots.",
    "prog-3": "A hash table achieves average O(1) lookup by mapping keys to array indices through a hash function.",
    "prog-4": "Rust's borrow checker enforces memory safety at compile time without needing a garbage collector.",
    "prog-5": "REST APIs use HTTP verbs like GET, POST, and DELETE to represent operations on resources identified by URLs.",
    # --- Personal finance (deliberately overlaps vocabulary with other domains) ---
    "fin-1": "A diversified investment portfolio spreads risk across asset classes like stocks, bonds, and real estate.",
    "fin-2": "Compound interest means interest earned also earns interest over time, accelerating the growth of savings.",
    "fin-3": "An emergency fund covering three to six months of expenses protects against unexpected income loss.",
    "fin-4": "Index funds passively track a market benchmark, typically charging lower fees than actively managed funds.",
    "fin-5": "A credit score summarizes a borrower's creditworthiness based on payment history and outstanding debt.",
    # --- Cybersecurity ---
    "sec-1": "SQL injection exploits improperly sanitized user input to execute unintended database queries.",
    "sec-2": "A honeypot is a decoy system designed to attract attackers and study their behavior without risking real assets.",
    "sec-3": "Multi-factor authentication requires a second proof of identity beyond a password, reducing the impact of credential theft.",
    "sec-4": "Phishing attacks trick victims into revealing credentials or installing malware through deceptive emails or sites.",
    "sec-5": "A zero-day vulnerability is a flaw unknown to the vendor, so no patch exists at the time it is exploited.",
}

# query -> (natural language query text, set of relevant doc_ids)
_QUERIES: dict[str, tuple[str, set[str]]] = {
    "q1": ("what is the biggest planet in our solar system", {"astro-1"}),
    "q2": ("why can't light escape a black hole", {"astro-2"}),
    "q3": ("how do neural networks learn from data", {"ml-1", "ml-4"}),
    "q4": ("what causes a model to memorize training data instead of generalizing", {"ml-2"}),
    "q5": ("how does self-attention work in transformers", {"ml-5"}),
    "q6": ("why do you knead bread dough", {"cook-2"}),
    "q7": ("how to thicken a sauce with flour and butter", {"cook-3"}),
    "q8": ("who invented the printing press", {"hist-1"}),
    "q9": ("what ended the cold war division of germany", {"hist-3"}),
    "q10": ("how do plants turn sunlight into energy", {"bio-2"}),
    "q11": ("why do mitochondria matter for the cell", {"bio-1"}),
    "q12": ("how does natural selection work", {"bio-4"}),
    "q13": ("why can't multiple threads run python code at once", {"prog-1"}),
    "q14": ("how does a hash table get constant time lookup", {"prog-3"}),
    "q15": ("how does rust guarantee memory safety without a garbage collector", {"prog-4"}),
    "q16": ("why should I spread my investments across different asset types", {"fin-1"}),
    "q17": ("how does interest on interest accelerate savings growth", {"fin-2"}),
    "q18": ("what is a decoy system used to study attackers", {"sec-2"}),
    "q19": ("how does injecting malicious sql exploit a database", {"sec-1"}),
    "q20": ("what makes a security flaw a zero-day", {"sec-5"}),
}


def load_corpus() -> list[Document]:
    return [Document(doc_id=doc_id, text=text) for doc_id, text in _RAW_DOCS.items()]


def load_queries() -> list[dict]:
    return [
        {"query_id": qid, "text": text, "relevant_doc_ids": relevant}
        for qid, (text, relevant) in _QUERIES.items()
    ]
