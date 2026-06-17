import logging
import os
import re
from collections import Counter
logger = logging.getLogger(__name__)


def _lazy_import_pronouncing():
    try:
        import pronouncing

        return pronouncing
    except Exception:
        return None


def _lazy_import_transformers():
    try:
        from transformers import pipeline

        return pipeline
    except Exception:
        return None


def _lazy_import_whisper():
    try:
        import whisper

        return whisper
    except Exception:
        return None


def _safe_last_word(line):
    if not line:
        return ""
    parts = re.findall(r"[\w']+", line)
    return parts[-1].lower() if parts else ""


def transcribe_audio(path, backend="whisper", model_name="small", language="en"):
    """Transcribe an audio file to text using an optional backend.

    Returns (transcript_text, meta_dict).
    """
    logger.debug("transcribe_audio: path=%s backend=%s model=%s", path, backend, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found for transcription: {path}")

    # Try whisper (if available)
    if backend in ("whisper", "auto"):
        whisper_mod = _lazy_import_whisper()
        if whisper_mod is not None:
            try:
                model = whisper_mod.load_model(model_name)
                res = model.transcribe(path, language=language)
                text = res.get("text", "")
                return text, {"backend": "whisper", "model": model_name}
            except Exception:
                logger.exception("transcribe_audio: whisper transcription failed")

    # No supported backend available
    raise RuntimeError("No ASR backend available. Install 'whisper' or call with backend that exists.")


def _simple_sentiment(text):
    # Very small lexicon-based fallback
    pos_words = {"happy", "love", "good", "great", "joy", "sweet", "sunny", "smile"}
    neg_words = {"sad", "hate", "angry", "pain", "lost", "cry", "tears", "sadness"}
    toks = re.findall(r"[\w']+", text.lower())
    p = sum(1 for t in toks if t in pos_words)
    n = sum(1 for t in toks if t in neg_words)
    if p + n == 0:
        return {"label": "neutral", "score": 0.0}
    label = "positive" if p >= n else "negative"
    score = float((p - n) / max(1, (p + n)))
    return {"label": label, "score": float(score)}


def _detect_themes(text):
    text_l = text.lower()
    theme_keywords = {
        "love": ["love", "lover", "romance", "heart", "darling"],
        "nostalgia": ["remember", "yesterday", "memories", "remembered", "nostalgia"],
        "rebellion": ["fight", "rebel", "revolution", "break", "hate"],
        "party": ["dance", "party", "tonight", "DJ", "club"],
        "loss": ["lost", "gone", "goodbye", "miss", "missing"],
    }
    scores = {}
    for theme, kws in theme_keywords.items():
        scores[theme] = sum(text_l.count(k) for k in kws)
    # pick non-zero themes
    picked = [k for k, v in sorted(scores.items(), key=lambda x: -x[1]) if v > 0]
    return {"candidates": picked, "scores": scores}


def _rhyme_scheme(lines):
    pronouncing = _lazy_import_pronouncing()
    last_words = [_safe_last_word(l) for l in lines]
    scheme = []
    groups = []
    for w in last_words:
        assigned = None
        if pronouncing is not None and w:
            try:
                for gi, g in enumerate(groups):
                    # check rhyme with any member in the group
                    for gw in g:
                        if gw and pronouncing.rhymes(gw) and w in pronouncing.rhymes(gw):
                            assigned = gi
                            break
                    if assigned is not None:
                        break
            except Exception:
                logger.debug("_rhyme_scheme: pronouncing check failed")
        if assigned is None:
            # try suffix match (last 3 letters)
            suff = w[-3:] if len(w) >= 3 else w
            for gi, g in enumerate(groups):
                for gw in g:
                    if gw[-3:] == suff and suff != "":
                        assigned = gi
                        break
                if assigned is not None:
                    break
        if assigned is None:
            groups.append([w])
            scheme.append(len(groups) - 1)
        else:
            groups[assigned].append(w)
            scheme.append(assigned)

    # convert numeric groups to letters
    letters = []
    for g in scheme:
        letters.append(chr(ord("A") + (g % 26)))
    rhyme_per_line = letters
    rhyme_scheme = "".join(letters)
    return {"scheme": rhyme_scheme, "per_line": rhyme_per_line, "groups": groups}


def _extract_named_entities(text, backend=None):
    pipeline = _lazy_import_transformers()
    if pipeline is not None:
        try:
            ner = pipeline("ner", grouped_entities=True)
            ents = ner(text)
            cleaned = []
            for e in ents:
                cleaned.append({"entity_group": e.get("entity_group"), "word": e.get("word"), "score": float(e.get("score", 0.0))})
            return cleaned
        except Exception:
            logger.debug("_extract_named_entities: transformers NER failed")

    # fallback: simple capitalized phrase extraction
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    ctr = Counter(candidates)
    return [{"word": k, "count": v} for k, v in ctr.items()]


def analyze_lyrics(text, backend="transformers", language="en"):
    """Analyze a lyrics string and produce sentiment, themes, rhyme scheme, and named entities.

    Returns a dict with keys: sentiment, themes, rhyme, named_entities, lines
    """
    logger.debug("analyze_lyrics: backend=%s language=%s len_text=%d", backend, language, len(text) if text else 0)
    out = {"lines": [], "transcript_length": 0, "sentiment": None, "themes": None, "rhyme": None, "named_entities": None}
    if not text or not text.strip():
        return out
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out["lines"] = lines
    out["transcript_length"] = len(text)

    # sentiment
    sentiment_res = None
    pipeline_fn = _lazy_import_transformers()
    if pipeline_fn is not None:
        try:
            sentiment_pipe = pipeline_fn("sentiment-analysis")
            sentiment_res = sentiment_pipe(text[:512])
            if isinstance(sentiment_res, list) and len(sentiment_res):
                sentiment_res = sentiment_res[0]
        except Exception:
            logger.debug("analyze_lyrics: transformers sentiment failed")
            sentiment_res = None

    if sentiment_res is None:
        sentiment_res = _simple_sentiment(text)
    out["sentiment"] = sentiment_res

    # themes
    out["themes"] = _detect_themes(text)

    # rhyme scheme
    out["rhyme"] = _rhyme_scheme(lines)

    # named entities
    out["named_entities"] = _extract_named_entities(text, backend=backend)

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("lyrics module loaded")
