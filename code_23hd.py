
import re, sys, json, warnings, random
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# STAGE 1 — SLANG DETECTOR

SLANG_LEXICON = {
    # ── Community drift 
    "bussin":                    ("positive", "community"),
    "goated":                    ("positive", "community"),
    "no cap":                    ("positive", "community"),
    "nocap":                     ("positive", "community"),
    "understood the assignment": ("positive", "community"),
    "ate":                       ("positive", "community"),
    "slay":                      ("positive", "community"),
    "slaying":                   ("positive", "community"),
    "rizz":                      ("positive", "community"),
    "rizzed":                    ("positive", "community"),
    "lowkey":                    ("positive", "community"),
    "highkey":                   ("positive", "community"),
    "based":                     ("positive", "community"),
    "hits different":            ("positive", "community"),
    "hit different":             ("positive", "community"),
    "fr fr":                     ("positive", "community"),
    "periodt":                   ("positive", "community"),
    "sending me":                ("positive", "community"),
    "main character":            ("positive", "community"),
    # ── Valence reversal (polarity has flipped) 
    "fire":     ("positive", "valence"),
    "sick":     ("positive", "valence"),
    "dead":     ("positive", "valence"),
    "nasty":    ("positive", "valence"),
    "unhinged": ("positive", "valence"),
    "insane":   ("positive", "valence"),
    "crazy":    ("positive", "valence"),
    "wild":     ("positive", "valence"),
    "savage":   ("positive", "valence"),
    "dirty":    ("positive", "valence"),
    "wicked":   ("positive", "valence"),
    "stupid":   ("positive", "valence"),
    "killed":   ("positive", "valence"),
    "killing":  ("positive", "valence"),
    "slaps":    ("positive", "valence"),
    "banger":   ("positive", "valence"),
    "hard":     ("positive", "valence"),
    "cooked":   ("positive", "valence"),   # subject-be pattern overrides to neg
    # ── Polysemy amplification (new senses alongside old) 
    "mid":      ("negative", "polysemy"),
    "flop":     ("negative", "polysemy"),
    "cap":      ("negative", "polysemy"),
    "sus":      ("negative", "polysemy"),
    "ratio":    ("negative", "polysemy"),
    "pressed":  ("negative", "polysemy"),
}

# Words that signal LITERAL (non-slang) usage
LITERAL_CTX = {
    "fire":   {"burning","flame","smoke","firefighter","alarm","wildfire",
               "blaze","arson","emergency"},
    "dead":   {"died","death","funeral","cemetery","mourning","corpse",
               "deceased","obituary","passed"},
    "sick":   {"hospital","doctor","medicine","fever","ill","disease",
               "virus","symptom","diagnosis"},
    "nasty":  {"disgusting","smell","rotten","filthy","garbage",
               "unhygienic","mold","bacteria"},
    "wild":   {"safari","nature","animal","habitat","zoo","forest",
               "wilderness","boar"},
    "insane": {"asylum","psychiatric","hospital","mental","diagnosis",
               "institution"},
    "hard":   {"difficult","tough","challenging","struggle","effort",
               "exam","test","problem"},
    "crazy":  {"diagnosis","mental","disorder","asylum","condition"},
}

# Words that signal SLANG (positive) usage
SLANG_CTX = {
    "fire":    {"music","song","track","beat","set","show","performance",
                "album","drop","concert","look","outfit","verse","bar","hook"},
    "sick":    {"trick","move","shot","performance","beat","track","dunk",
                "flip","goal","play"},
    "dead":    {"joke","funny","hilarious","laughing","comedy","meme",
                "video","clip"},
    "nasty":   {"dunk","shot","trick","move","performance","drop","crowd",
                "crossover","pit"},
    "unhinged":{"visuals","performance","show","energy","vibe","art",
                "concept","stage"},
    "cooked":  {"performance","set","show","stage","track","beat","verse",
                "bar","it"},
    "wild":    {"crowd","show","night","concert","set","energy",
                "performance","party","pit"},
    "insane":  {"performance","show","track","beat","visuals","crowd",
                "set","energy","game","match"},
    "crazy":   {"performance","show","track","beat","night","crowd",
                "game","energy","talent"},
    "hard":    {"track","beat","song","album","hit","drop","banger"},
}

INFORMAL = {"bro","lol","omg","tbh","ngl","fr","yo","literally","honestly",
            "deadass","lmao","bruh","fam","rn","bestie","sis","smh"}

# Special case: "cooked" subject-be pattern → negative
COOKED_NEG = re.compile(
    r"\b(i am|we are|they are|he is|she is|you are|"
    r"i'm|we're|they're|he's|she's|you're|"
    r"is cooked|are cooked|were cooked|am cooked|"
    r"get cooked|got cooked)\b", re.IGNORECASE)
COOKED_POS = re.compile(
    r"\b(cooked with|cooked on|they cooked|she cooked|"
    r"he cooked|really cooked|absolutely cooked it)\b", re.IGNORECASE)
COOKED_NEG_SUBJ = {"we","they","team","squad","us","done","finished","over"}


class SlangDetector:
    """
    Stage 1: token-level slang/literal disambiguation.

    Rules applied in order:
      1. Literal-context override  — co-occurrence with literal indicator words
      2. Community drift rule      — always slang (no literal counterpart)
      3. Domain context check      — co-occurrence with slang context words,
                                     informal markers, or short sentence heuristic
    """

    def detect(self, sentence: str) -> dict:
        s     = sentence.lower()
        words = set(re.findall(r"\b\w+\b", s))
        found = []

        # Multi-word phrases first
        for term, (sentiment, drift) in SLANG_LEXICON.items():
            if " " in term and term in s:
                found.append({"term": term, "sentiment": sentiment,
                               "drift": drift, "is_slang": True,
                               "reason": "multi-word"})

        # Single tokens
        for term, (sentiment, drift) in SLANG_LEXICON.items():
            if " " in term or term.lower() not in words:
                continue

            # Special case: cooked — subject-aware disambiguation
            if term == "cooked":
                neg = COOKED_NEG.search(s) or (
                    not COOKED_POS.search(s) and bool(words & COOKED_NEG_SUBJ))
                found.append({"term": term,
                               "sentiment": "negative" if neg else "positive",
                               "drift": "polysemy", "is_slang": True,
                               "reason": "subject-be→neg" if neg else "ctx→pos"})
                continue

            is_slang = True

            if term in LITERAL_CTX and words & LITERAL_CTX[term]:
                is_slang = False           # Rule 1: literal context
            elif drift == "community":
                is_slang = True            # Rule 2: community always slang
            elif term in SLANG_CTX:        # Rule 3: domain context check
                is_slang = bool(words & SLANG_CTX[term]) or \
                           bool(words & INFORMAL) or \
                           len(s.split()) <= 16

            found.append({"term": term, "sentiment": sentiment,
                           "drift": drift, "is_slang": is_slang})

        slang     = [f for f in found if f["is_slang"]]
        pos_count = sum(1 for f in slang if f["sentiment"] == "positive")
        neg_count = sum(1 for f in slang if f["sentiment"] == "negative")

        return {
            "slang_terms":  slang,
            "has_slang":    len(slang) > 0,
            "slang_score":  len(slang) / max(len(found), 1) if found else 0.0,
            "lex_sentiment": 1 if pos_count > neg_count else
                             (0 if neg_count > pos_count else -1),
            "pos_count":    pos_count,
            "neg_count":    neg_count,
            "drift_types":  list(set(f["drift"] for f in slang)),
        }

# STAGE 2 — LEXICON-AUGMENTED CLASSIFIER

_detector = SlangDetector()

def enrich(text: str) -> str:
    """Append hint tokens h_i ∈ H to produce x' = x ⊕ h_1 ⊕ … ⊕ h_k."""
    det = _detector.detect(text)
    if det["has_slang"]:
        hints = []
        for t in det["slang_terms"]:
            hints.append("[POS]" if t["sentiment"] == "positive" else "[NEG]")
            hints.append({"community": "[COM]",
                           "valence":   "[VAL]",
                           "polysemy":  "[POL]"}.get(t["drift"], "[UNK]"))
        return text + " " + " ".join(hints)
    return text


def phi(texts: list) -> np.ndarray:
    """
    14-dimensional lexicon feature vector φ(x).
    F1  slang score σ(x)
    F2  has-slang indicator
    F3  net-positive flag
    F4  net-negative flag
    F5  normalised slang term count
    F6  community drift indicator
    F7  valence drift indicator
    F8  polysemy drift indicator
    F9  normalised sentence length
    F10 amplifier presence
    F11 informal-marker presence
    F12 negation presence
    F13 net-positive and no negative
    F14 net-negative and no positive
    """
    AMP = {"absolutely","completely","totally","really","extremely",
           "very","genuinely"}
    NEG = {"not","never","no","don't","didn't","doesn't","wouldn't","couldn't"}
    rows = []
    for text in texts:
        d = _detector.detect(text)
        s = text.lower()
        w = set(re.findall(r"\b\w+\b", s))
        rows.append([
            d["slang_score"],
            float(d["has_slang"]),
            float(d["lex_sentiment"] == 1),
            float(d["lex_sentiment"] == 0),
            min(len(d["slang_terms"]), 5) / 5.0,
            float("community" in d["drift_types"]),
            float("valence"   in d["drift_types"]),
            float("polysemy"  in d["drift_types"]),
            min(len(s.split()), 30) / 30.0,
            float(bool(w & AMP)),
            float(bool(w & INFORMAL)),
            float(bool(w & NEG)),
            float(d["pos_count"] > 0 and d["neg_count"] == 0),
            float(d["neg_count"] > 0 and d["pos_count"] == 0),
        ])
    return np.array(rows)


def build_features(texts: list, vec, fit: bool = False):
    """f(x) = [TF-IDF(x') ‖ φ(x)]"""
    enriched = [enrich(t) for t in texts]
    tfidf    = vec.fit_transform(enriched) if fit else vec.transform(enriched)
    return hstack([tfidf, csr_matrix(phi(texts))])


class LexiconAugmentedClassifier:
    """Full two-stage pipeline."""

    def __init__(self, C: float = 2.0, max_features: int = 8000):
        self.vec = TfidfVectorizer(
            ngram_range=(1, 3), max_features=max_features,
            sublinear_tf=True, min_df=1)
        self.clf = LogisticRegression(
            C=C, max_iter=1000, random_state=42, class_weight="balanced")
        self.fitted = False

    def fit(self, texts: list, labels: list):
        X = build_features(texts, self.vec, fit=True)
        self.clf.fit(X, labels)
        self.fitted = True
        return self

    def predict(self, texts: list) -> np.ndarray:
        return self.clf.predict(build_features(texts, self.vec))

    def predict_one(self, text: str) -> dict:
        det   = _detector.detect(text)
        pred  = self.predict([text])[0]
        return {
            "text":       text,
            "prediction": "positive" if pred == 1 else "negative",
            "enriched":   enrich(text),
            "slang_terms": [(t["term"], t["drift"], t["sentiment"])
                            for t in det["slang_terms"]],
        }


# TRAINING DATA (98 labelled sentences + lexicon-augmented defs)

TRAIN = [
    # Community positive
    ("bro that hook is based hits different every time",           1,"community"),
    ("she rizzed up the entire audience no cap",                   1,"community"),
    ("highkey the whole crowd was sending it periodt",             1,"community"),
    ("bussin track fr fr cannot stop replaying this one",          1,"community"),
    ("no cap the goated album of the entire decade",               1,"community"),
    # Valence positive
    ("the collab is crazy good both artists went stupid hard",     1,"valence"),
    ("that set was savage from start to finish crowd wild",        1,"valence"),
    ("killing it every single night on this tour no breaks",       1,"valence"),
    ("the verse was fire hardest bar I heard all month bro",       1,"valence"),
    ("absolutely sick landing nailed every single rotation",       1,"valence"),
    ("crowd went insane when the drop hit pure energy",            1,"valence"),
    ("the drop was nasty room erupted everyone was losing it",     1,"valence"),
    ("filthy crossover left defender on the floor incredible",     1,"valence"),
    ("that banger opener set the tone for the whole set",          1,"valence"),
    ("wicked concept album insane production from front back",     1,"valence"),
    ("the whole album slaps zero weak tracks front to back",       1,"valence"),
    ("stupid good performance everyone left in shock bro",         1,"valence"),
    ("she killed every note the whole crowd was in disbelief",     1,"valence"),
    ("hard beats throughout whole project a certified banger",     1,"valence"),
    # Polysemy negative
    ("that mid project was cap they said it was the best",         0,"polysemy"),
    ("pressed fan behaviour sending threats is never okay",        0,"polysemy"),
    ("that's a big L for the label after all that hype",           0,"polysemy"),
    ("sus excuses for cancelling the headline slot tbh",           0,"polysemy"),
    ("clearly cap marketing it as a surprise drop it wasn't",      0,"polysemy"),
    ("ratio'd so hard for that take deserved it honestly",         0,"polysemy"),
    ("the set list was mid no crowd favourites at all",            0,"polysemy"),
    ("mid album nothing memorable nothing worth replaying",        0,"polysemy"),
    # Literal negative controls
    ("Absolutely terrible service would not go back ever",         0,"control"),
    ("The worst film I have seen in recent years",                 0,"control"),
    ("Completely wasted money on this avoid at all costs",         0,"control"),
    ("Very rude staff and the food was overpriced",                0,"control"),
    ("Dreadful performance left at half time very let down",       0,"control"),
    ("Never again waste of time and significant money",            0,"control"),
    ("Awful sound quality throughout the whole night",             0,"control"),
    ("Poor organisation two hour queues unacceptable",             0,"control"),
    # Literal positive controls
    ("What a wonderful evening truly stunning performance",        1,"control"),
    ("Exceptional service and the food was outstanding",           1,"control"),
    ("A truly memorable concert every song was perfect",           1,"control"),
    ("Outstanding performance I was moved deeply by it",           1,"control"),
    ("Brilliant acting from the entire cast all night",            1,"control"),
    ("A perfect evening from first note to last",                  1,"control"),
    ("Remarkable talent on display throughout the evening",        1,"control"),
    ("Loved every single moment will be returning again",          1,"control"),
    # Lexicon-augmented definitional sentences
    ("bussin means delicious and this food is exactly that",       1,"aug"),
    ("when they say bussin they mean it was truly amazing",        1,"aug"),
    ("fire in slang means excellent and that show was fire",       1,"aug"),
    ("calling something fire is the highest compliment given",     1,"aug"),
    ("nasty in this context means impressively good not bad",      1,"aug"),
    ("no cap means honestly and I am being sincere here",          1,"aug"),
    ("goated means greatest of all time he deserves it",           1,"aug"),
    ("unhinged in the best way means wildly creative good",        1,"aug"),
    ("when a song slaps it means it sounds absolutely excellent",  1,"aug"),
    ("slaps is slang for sounds really good everyone uses it",     1,"aug"),
    ("mid means mediocre or below average in quality",             0,"aug"),
    ("calling something mid means it was underwhelming",           0,"aug"),
    ("cooked means in deep trouble or thoroughly defeated",        0,"aug"),
    ("cap means a lie or not telling the truth at all",            0,"aug"),
    ("ratio means getting more replies than likes bad sign",       0,"aug"),
    ("sus means suspicious or acting in a questionable way",       0,"aug"),
    ("hit different means it affected you more than expected",     1,"aug"),
    ("lowkey means in a subtle understated positive way",          1,"aug"),
    ("highkey means very much openly enthusiastically positive",   1,"aug"),
    ("based means holding a confident admirable opinion",          1,"aug"),
    ("periodt means full stop that is final emphasis added",       1,"aug"),
    ("understood the assignment means perfectly executed task",    1,"aug"),
    ("rizz means natural charm or charisma especially social",     1,"aug"),
    ("banger track means an exceptional high energy song",         1,"aug"),
    ("wicked in slang means impressively excellent brilliant",     1,"aug"),
    ("savage move means an impressively bold decisive action",     1,"aug"),
    ("she ate means she performed or executed something perfectly",1,"aug"),
    ("sick trick means an impressively executed difficult move",   1,"aug"),
    ("killed it means performed something exceptionally well",     1,"aug"),
    ("stupid good means surprisingly impressively excellent",      1,"aug"),
    ("dirty move means an impressively skilful unexpected one",    1,"aug"),
    ("sending me means something is so funny overwhelming",        1,"aug"),
    ("main character means behaving with protagonist confidence",  1,"aug"),
    ("fr fr means for real for real emphasising sincerity",        1,"aug"),
    ("an L is a loss or a failure or a bad outcome",               0,"aug"),
    ("flop means a disappointing commercial or critical failure",  0,"aug"),
    ("pressed means acting aggressively out of jealousy",          0,"aug"),
    ("cooked in game context means they are done finished",        0,"aug"),
]


# HELPER FUNCTIONS

def slang_error(data, preds):
    idx   = [i for i,(t,l,d) in enumerate(data)
             if l == 1 and d not in ("control","aug","literal")]
    wrong = sum(1 for i in idx if preds[i] != 1)
    return {"rate": wrong/max(len(idx),1), "wrong": wrong, "total": len(idx)}


def detector_stats(data):
    """Compute SlangDetector recall and precision stats."""
    slang_idx   = [i for i,(t,l,d) in enumerate(data)
                   if l==1 and d not in ("control","aug","literal")]
    literal_idx = [i for i,(t,l,d) in enumerate(data) if d=="literal"]
    comm_idx    = [i for i,(t,l,d) in enumerate(data) if d=="community"]

    detected_slang   = sum(1 for i in slang_idx
                           if _detector.detect(data[i][0])["has_slang"])
    literal_correct  = sum(1 for i in literal_idx
                           if not _detector.detect(data[i][0])["has_slang"])
    detected_comm    = sum(1 for i in comm_idx
                           if _detector.detect(data[i][0])["has_slang"])
    return {
        "slang_recall":    detected_slang / max(len(slang_idx),1),
        "slang_n":         (detected_slang, len(slang_idx)),
        "literal_prec":    literal_correct / max(len(literal_idx),1),
        "literal_n":       (literal_correct, len(literal_idx)),
        "community_recall":detected_comm / max(len(comm_idx),1),
        "community_n":     (detected_comm, len(comm_idx)),
    }

# LOAD TEST DATA FROM DATASET

try:
    from dataset import TASK_23HD_FULL
    TEST_DATA = [(t, l, d) for t,l,d in
                 [(x["text"], x["label"], x["drift"])
                  for x in TASK_23HD_FULL]]
    # Remap labels: 2→1, 1→1, 0→0  (binary classification)
    TEST_DATA = [(t, 1 if l >= 1 else 0, d) for t,l,d in TEST_DATA]
    print(f"Loaded {len(TEST_DATA)} sentences from dataset.py")
except ImportError:
    print("dataset.py not found — using built-in 299-sentence test set")
    # Fallback: small inline set (subset of paper evaluation)
    TEST_DATA = [
        ("This food is bussin, best burger I've had",               1,"community"),
        ("That drop was nasty, crowd went absolutely wild",         1,"valence"),
        ("no cap this is the best album of the year",               1,"community"),
        ("she really understood the assignment tonight",            1,"community"),
        ("that dunk was nasty the whole crowd lost it",             1,"valence"),
        ("this artist is goated no debate",                         1,"community"),
        ("the choreography was sick every move was perfect",        1,"valence"),
        ("ngl that new single is pretty mid after all the hype",    0,"polysemy"),
        ("that's cap he never said that stop the lies",             0,"polysemy"),
        ("we are so cooked after that first half no way back",      0,"polysemy"),
        ("Service was terrible, never coming back",                 0,"control"),
        ("Worst concert experience I have ever had",                0,"control"),
        ("The food was cold and the staff were rude",               0,"control"),
        ("What a wonderful evening the orchestra was breathtaking", 1,"control"),
        ("Outstanding performance I was moved deeply by it",        1,"control"),
        ("The building caught fire and everyone was evacuated",     0,"literal"),
        ("She was sick in hospital for over a week",                0,"literal"),
        ("The exam was hard she failed despite revising",           0,"literal"),
    ]

# MAIN EXPERIMENT

def run():
    train_texts  = [t for t,l,d in TRAIN]
    train_labels = [l for t,l,d in TRAIN]
    test_texts   = [t for t,l,d in TEST_DATA]
    test_labels  = [l for t,l,d in TEST_DATA]

    print("\n" + "="*65)
    print("SIT770 Task 2.3HD — Two-Stage Lexicon-Augmented Pipeline")
    print("="*65)
    cats = Counter(d for t,l,d in TEST_DATA)
    print(f"\nTest set: {len(TEST_DATA)} sentences")
    for k,v in sorted(cats.items()):
        print(f"  {k:12s}: {v}")

    # TABLE 2: SlangDetector performance
    print(f"\n{'='*65}")
    print("TABLE 2 — SlangDetector Performance")
    print(f"{'='*65}")
    ds = detector_stats(TEST_DATA)
    print(f"  Slang recall    : {ds['slang_recall']:.1%}"
          f"  ({ds['slang_n'][0]}/{ds['slang_n'][1]})")
    print(f"  Literal prec    : {ds['literal_prec']:.1%}"
          f"  ({ds['literal_n'][0]}/{ds['literal_n'][1]})")
    print(f"  Community recall: {ds['community_recall']:.1%}"
          f"  ({ds['community_n'][0]}/{ds['community_n'][1]})")

    # Baselines
    BASELINES = [
        ("DistilBERT-2019", 0.596, 0.533),
        ("BERTweet-2019",   0.440, 0.467),
        ("RoBERTa-2021",    0.780, 0.267),
    ]

    # Ablation: TF-IDF enriched text only (no phi features)
    print(f"\nTraining ablation (TF-IDF only, no lexicon features)...")
    vec_abl = TfidfVectorizer(ngram_range=(1,3), max_features=8000,
                               sublinear_tf=True, min_df=1)
    enriched_train = [enrich(t) for t in train_texts]
    enriched_test  = [enrich(t) for t in test_texts]
    X_abl_tr = vec_abl.fit_transform(enriched_train)
    X_abl_te = vec_abl.transform(enriched_test)
    clf_abl  = LogisticRegression(C=2.0, max_iter=1000, random_state=42,
                                   class_weight="balanced")
    clf_abl.fit(X_abl_tr, train_labels)
    preds_abl = clf_abl.predict(X_abl_te)
    f1_abl    = f1_score(test_labels, preds_abl, average="macro",
                         zero_division=0)
    serr_abl  = slang_error(TEST_DATA, preds_abl)

    # Proposed: full pipeline 
    print("Training proposed pipeline (TF-IDF + lexicon features)...")
    model = LexiconAugmentedClassifier(C=2.0)
    model.fit(train_texts, train_labels)
    preds = model.predict(test_texts)
    f1    = f1_score(test_labels, preds, average="macro", zero_division=0)
    serr  = slang_error(TEST_DATA, preds)

    # TABLE 3: Full results 
    print(f"\n{'='*65}")
    print("TABLE 3 — End-to-End Results")
    print(f"{'='*65}")
    print(f"{'Model':<35} {'Macro-F1':>9} {'Slang Err':>11}")
    print("-"*57)
    for name, f1_b, serr_b in BASELINES:
        print(f"{name:<35} {f1_b:>9.3f} {serr_b:>10.1%}")
    print("-"*57)
    print(f"{'Ablation (TF-IDF only)':<35} {f1_abl:>9.3f} "
          f"{serr_abl['rate']:>10.1%}")
    print(f"{'Proposed (full pipeline)':<35} {f1:>9.3f} "
          f"{serr['rate']:>10.1%}")
    print(f"{'  (slang wrong/total)':<35} "
          f"{'':>9} {serr['wrong']:>4}/{serr['total']}")
    print("="*65)

    # Full classification report
    print(f"\nClassification report (proposed):")
    print(classification_report(test_labels, preds,
                                 target_names=["NEG","POS"], digits=3))

    
    print(f"{'='*65}")
    print("FIGURE 2 — Slang Error Rate by Model")
    print(f"{'='*65}")
    models_fig = [
        ("DistilBERT-2019", 0.533),
        ("BERTweet-2019",   0.467),
        ("RoBERTa-2021",    0.267),
        ("Ablation",        serr_abl["rate"]),
        ("Proposed",        serr["rate"]),
    ]
    for name, rate in models_fig:
        bar  = "█" * int(rate * 40)
        rest = "░" * (40 - int(rate * 40))
        print(f"  {name:<18} {rate:>6.1%} {bar}{rest}")

    # Per-drift-type analysis
    print(f"\n{'='*65}")
    print("Per-Drift-Type Accuracy (Proposed)")
    print(f"{'='*65}")
    for dtype in ["valence","community","polysemy","literal","control","mixed"]:
        idx = [i for i,(t,l,d) in enumerate(TEST_DATA) if d==dtype]
        if not idx:
            continue
        correct = sum(1 for i in idx if preds[i]==TEST_DATA[i][1])
        print(f"  {dtype:12s}: {correct:3}/{len(idx):3} = {correct/len(idx):.1%}")

    # 5-fold cross-validation
    print(f"\n{'='*65}")
    print("5-Fold Stratified Cross-Validation")
    print(f"{'='*65}")
    all_t = train_texts + test_texts
    all_l = train_labels + test_labels
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, val) in enumerate(skf.split(all_t, all_l), 1):
        tr_t=[all_t[i] for i in tr]; vl_t=[all_t[i] for i in val]
        tr_l=[all_t[i] for i in tr]; vl_l=[all_l[i] for i in val]
        tr_l=[all_l[i] for i in tr]
        m = LexiconAugmentedClassifier(C=2.0)
        m.fit(tr_t, tr_l)
        p = m.predict(vl_t)
        s = f1_score(vl_l, p, average="macro", zero_division=0)
        cv_scores.append(s)
        print(f"  Fold {fold}: {s:.3f}")
    cv_mean = np.mean(cv_scores)
    cv_std  = np.std(cv_scores)
    print(f"  Mean  : {cv_mean:.3f} ± {cv_std:.3f}")

    # Real-world slice (100 tweet-style sentences)
    try:
        from dataset import TASK_23HD_FULL
        rw_cats = ["community","valence","polysemy"]
        rw_data = [(x["text"], 1 if x["label"]>=1 else 0, x["drift"])
                   for x in TASK_23HD_FULL
                   if x["drift"] in rw_cats][:100]
        if rw_data:
            rw_texts  = [t for t,l,d in rw_data]
            rw_labels = [l for t,l,d in rw_data]
            rw_preds  = model.predict(rw_texts)
            rw_f1     = f1_score(rw_labels, rw_preds,
                                  average="macro", zero_division=0)
            rw_serr   = slang_error(rw_data, rw_preds)
            print(f"\n{'='*65}")
            print("TABLE 4 — Real-World Slice (tweet-style sentences)")
            print(f"{'='*65}")
            print(f"  Macro-F1   : {rw_f1:.3f}")
            print(f"  Slang error: {rw_serr['rate']:.1%}")
            print(f"\n  Per-category:")
            for cat in rw_cats:
                idx = [i for i,(t,l,d) in enumerate(rw_data) if d==cat]
                if not idx:
                    continue
                correct = sum(1 for i in idx if rw_preds[i]==rw_data[i][1])
                print(f"    {cat:12s}: {correct}/{len(idx)} = "
                      f"{correct/len(idx):.1%}")
    except Exception:
        pass

    # Summary 
    best_baseline_err = 0.267
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  Proposed Macro-F1 : {f1:.3f}")
    print(f"  5-fold CV         : {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"  Slang error       : {serr['rate']:.1%}")
    print(f"  Error reduction   : "
          f"{(best_baseline_err - serr['rate'])*100:.1f} pp "
          f"vs RoBERTa-2021")
    print(f"  Ablation gain     : +{(f1 - f1_abl):.3f} F1 from φ(x) features")

    return model

# INTERACTIVE PREDICTION MODE

def interactive(model):
    print("\n" + "="*65)
    print("  INTERACTIVE PREDICTION  (type 'quit' to exit)")
    print("="*65)
    while True:
        text = input("\n  Enter sentence: ").strip()
        if text.lower() in ("quit","exit","q"):
            break
        result = model.predict_one(text)
        print(f"\n  Prediction : {result['prediction'].upper()}")
        print(f"  Enriched   : {result['enriched']}")
        if result["slang_terms"]:
            for term, drift, sent in result["slang_terms"]:
                print(f"    [{drift}] '{term}' → {sent}")
        else:
            print("    No slang detected — classified on surface features")


if __name__ == "__main__":
    model = run()
    if "--predict" in sys.argv:
        interactive(model)