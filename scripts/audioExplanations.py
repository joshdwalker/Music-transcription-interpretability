"""
Install:
    pip install --requirements

Sections:
    PART 1  original code (trie, predict, batching, create_n2g_explanation)
    PART 2  audio settings
    PART 3  audio -> tokens (log-mel, k-means, collapse repeats)
    PART 4  snippet registry (every token id points to a real audio snippet)
    PART 5  MUSCRIPTOR + YOUR SAE: run_model(wave) -> SAE feature activations   <-- fill in
    PART 6  model_fn (glues snippets into a wav and runs the model)
    PART 7  train set + build explanation
    PART 8  evaluate + listen to matches
    PART 9  main                                          
"""

import json
from abc import ABC
from itertools import groupby
from typing import Any, Callable

import blobfile as bf
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


# =============================================================================
# PART 1: ORIGINAL N2G CODE
# =============================================================================


class Explanation(ABC):
    def predict(self, tokens: list[str]) -> list[float]:
        raise NotImplementedError

    def dump(self) -> bytes:
        raise NotImplementedError

    @classmethod
    def load(cls, serialized: Any) -> "Explanation":
        raise NotImplementedError

    def dumpf(self, filename: str):
        d = self.dump()
        assert isinstance(d, bytes)
        with bf.BlobFile(filename, "wb") as f:
            f.write(d)

    @classmethod
    def loadf(cls, filename: str):
        with bf.BlobFile(filename, "rb") as f:
            return cls.load(f.read())


_ANY_TOKEN = "token(*)"
_START_TOKEN = "token(^)"
_SALIENCY_KEY = "<saliency>"


class NtgExplanation(Explanation):
    def __init__(self, trie: dict):
        self.trie = trie

    def todict(self) -> dict:
        return {
            "trie": self.trie,
        }

    # added so dumpf()/loadf() work
    def dump(self) -> bytes:
        return json.dumps(self.todict()).encode()

    @classmethod
    def load(cls, serialized) -> "Explanation":
        if isinstance(serialized, bytes):
            serialized = json.loads(serialized)
        assert isinstance(serialized, dict)
        return cls(serialized["trie"])

    def predict(self, tokens: list[str]) -> list[float]:
        predicted_acts = []
        # for each token, traverse the trie beginning from that token and proceeding in reverse order until we match
        # a pattern or are no longer able to traverse.
        for i in range(len(tokens)):
            curr = self.trie
            for j in range(i, -1, -1):
                if tokens[j] not in curr and _ANY_TOKEN not in curr:
                    predicted_acts.append(0)
                    break
                if tokens[j] in curr:
                    curr = curr[tokens[j]]
                else:
                    curr = curr[_ANY_TOKEN]
                if _SALIENCY_KEY in curr:
                    predicted_acts.append(curr[_SALIENCY_KEY])
                    break
                # if we've reached the end of the sequence and haven't found a saliency value, append 0.
                elif j == 0:
                    if _START_TOKEN in curr:
                        curr = curr[_START_TOKEN]
                        assert _SALIENCY_KEY in curr
                        predicted_acts.append(curr[_SALIENCY_KEY])
                        break
                    predicted_acts.append(0)
        # We should have appended a value for each token in the sequence.
        assert len(predicted_acts) == len(tokens)
        return predicted_acts

    # TODO make this more efficient
    def predict_many(self, tokens_batch: list[list[str]]) -> list[list[float]]:
        return [self.predict(t) for t in tokens_batch]


def batched(iterable, bs):
    batch = []
    it = iter(iterable)
    while True:
        batch = []
        try:
            for _ in range(bs):
                batch.append(next(it))
            yield batch
        except StopIteration:
            if len(batch) > 0:
                yield batch
            return


def apply_batched(fn, iterable, bs):
    for batch in batched(iterable, bs):
        ret = fn(batch)
        assert len(ret) == len(batch)
        yield from ret


def batch_parallelize(algos, fn, batch_size):
    """
    Algorithms are coroutines that yield items to be processed in parallel.
    We concurrently run the algorithm on all items in the batch.
    """
    inputs = []
    for i, algo in enumerate(algos):
        inputs.append((i, next(algo)))
    results: list[Any] = [None] * len(algos)
    while len(inputs) > 0:
        ret = list(apply_batched(fn, [x[1] for x in inputs], batch_size))
        assert len(ret) == len(inputs)
        inds = [x[0] for x in inputs]
        inputs = []
        for i, r in zip(inds, ret):
            try:
                next_input = algos[i].send(r)
                inputs.append((i, next_input))
            except StopIteration as e:
                results[i] = e.value
    return results


def create_n2g_explanation(
    model_fn: Callable, train_set: list[dict], batch_size: int = 16,
    padding_token=4808  # " _" for GPT-2
) -> NtgExplanation:
    truncated = []
    # for each one find the index of the selected activation in the doc. truncate the sequences after this point.
    for doc in train_set:
        # get index of selected activation. for docs stored in 'top', this is the max activation.
        # for docs stored in 'random', it is a random positive activation (we sample activations, not docs
        # to populate 'random', so docs with more positive activations are more likely to be included).
        max_idx = doc["idx"]
        truncated.append(
            {
                "act": doc["act"],
                "acts": doc["acts"][: max_idx + 1],
                "tokens": doc["tokens"][: max_idx + 1],
                "token_ints": doc["token_ints"][: max_idx + 1],
            }
        )

    def get_minimal_subsequence(doc):
        for i in range(len(doc["token_ints"]) - 1, -1, -1):
            atom_acts = yield doc["token_ints"][i:]
            assert (
                len(atom_acts) == len(doc["token_ints"]) - i
            ), f"{len(atom_acts)} != {len(doc['token_ints']) - i}"
            if atom_acts[-1] / doc["act"] >= 0.5:
                return {
                    "tokens": doc["tokens"][i:],
                    "token_ints": doc["token_ints"][i:],
                    "subsequence_act": atom_acts[-1],
                    "orig_act": doc["act"],
                }
        print("Warning: no minimal subsequence found")
        # raise ValueError("No minimal subsequence found")
        return {
            "tokens": doc["tokens"],
            "token_ints": doc["token_ints"],
            "subsequence_act": doc["act"],
            "orig_act": doc["act"],
        }

    minimal_subsequences = batch_parallelize(
        [get_minimal_subsequence(doc) for doc in truncated], model_fn, batch_size
    )

    start_padded = apply_batched(
        model_fn,
        [[padding_token] + doc["token_ints"] for doc in minimal_subsequences],
        batch_size,
    )
    for min_seq, pad_atom_acts in zip(minimal_subsequences, start_padded):
        min_seq["can_pad_start"] = pad_atom_acts[-1] / min_seq["orig_act"] >= 0.5

    # for each token in a minimal subsequence, replace it with a padding token and compute the saliency value (1 - (orig act / new act))
    for doc in minimal_subsequences:
        all_seqs = []
        for i in range(len(doc["token_ints"])):
            tokens = doc["token_ints"][:i] + [padding_token] + doc["token_ints"][i + 1 :]
            assert len(tokens) == len(doc["token_ints"])
            all_seqs.append(tokens)
        saliency_vals = []
        all_atom_acts = apply_batched(model_fn, all_seqs, batch_size)
        for atom_acts, tokens in zip(all_atom_acts, all_seqs):
            assert len(atom_acts) == len(tokens)
            saliency_vals.append(1 - (atom_acts[-1] / doc["subsequence_act"]))
        doc["saliency_vals"] = saliency_vals

    trie = {}
    for doc in minimal_subsequences:
        curr = trie
        for i, (token, saliency) in enumerate(zip(doc["tokens"][::-1], doc["saliency_vals"][::-1])):
            if saliency < 0.5:
                token = _ANY_TOKEN
            if token not in curr:
                curr[token] = {}
            curr = curr[token]
            if i == len(doc["tokens"]) - 1:
                if not doc["can_pad_start"]:
                    curr[_START_TOKEN] = {}
                    curr = curr[_START_TOKEN]
                curr[_SALIENCY_KEY] = doc["subsequence_act"]

    return NtgExplanation(trie)


# =============================================================================
# PART 2: AUDIO SETTINGS
# =============================================================================

SR = 16000      # sample rate your model expects
N_FFT = 400     # window size for the k-means features (25 ms)
HOP = 160       # hop size (10 ms -> 100 frames per second)
N_MELS = 80     # features per frame (only used to build the k-means vocabulary)
N_UNITS = 200   # vocabulary size ("number of distinct sounds")


# =============================================================================
# PART 3: AUDIO -> TOKENS
# =============================================================================


def load_logmel(path):
    """Returns (log-mel features of shape (T, N_MELS), raw waveform)."""
    y, _ = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    return np.log(mel + 1e-6).T.astype(np.float32), y


def fit_kmeans(paths, max_frames=500_000):
    """Learn the 'sound vocabulary'. Fit on many clips, not just top ones."""
    feats = np.concatenate([load_logmel(p)[0] for p in paths])
    if len(feats) > max_frames:
        feats = feats[np.random.choice(len(feats), max_frames, replace=False)]
    return MiniBatchKMeans(n_clusters=N_UNITS, random_state=0, n_init="auto").fit(feats)


def to_units(path, kmeans):
    """
    Clip -> (units, spans, waveform).
    units : k-means label per token, repeats collapsed ([7,7,7,12] -> [7,12])
    spans : (start_frame, end_frame) each token covers in the original clip
    """
    feats, y = load_logmel(path)
    raw = kmeans.predict(feats)
    units, spans, pos = [], [], 0
    for u, grp in groupby(raw):
        n = len(list(grp))
        units.append(int(u))
        spans.append((pos, pos + n))
        pos += n
    return units, spans, y


# =============================================================================
# PART 4: SNIPPET REGISTRY
# Every token id points to a real audio snippet cut from a real clip.
# model_fn glues snippets back into a waveform, so the model only hears
# real audio. Id 0 is reserved for silence (the "blank" / padding token).
# =============================================================================

SEGMENTS = {}
SILENCE_ID = 0
SEGMENTS[SILENCE_ID] = np.zeros(int(0.05 * SR), dtype=np.float32)  # 50 ms of silence


def register_clip(path, kmeans):
    """Cut a clip into one snippet per token. Returns (units, snippet_ids, waveform)."""
    units, spans, y = to_units(path, kmeans)
    ids = []
    for k, (s, e) in enumerate(spans):
        a = s * HOP
        b = len(y) if k == len(spans) - 1 else e * HOP
        seg = y[a:b] if b > a else np.zeros(HOP // 2, dtype=np.float32)
        sid = len(SEGMENTS)
        SEGMENTS[sid] = seg
        ids.append(sid)
    return units, ids, y


# =============================================================================
# PART 5: MUSCRIPTOR (large) + YOUR TOP-K SAE
#
# Pipeline for one feature:
#     waveform -> MuScriptor -> hidden states at one layer (audio positions)
#              -> your SAE encoder -> feature activations -> pick one feature
#
# Facts confirmed by the inspect runs: audio is cut into 5 s windows; the mel
# features (100 frames/s) are PREPENDED to the transformer input, so the first
# 500 positions of each layer's output are the audio; the large model has 48
# layers (transformer.layers.0 .. 47) with dimension 1536.
# =============================================================================

WINDOW_S = 5.0
WINDOW_SAMPLES = int(WINDOW_S * SR)


class TopKSAE(torch.nn.Module):
    """
    Plain-PyTorch inference copy of FastAutoencoder from your train.py
    (no triton / CUDA / sharding needed, so it runs on a Mac). It implements:

        latents = relu(topk(encoder(x - pre_bias) + latent_bias))

    Only the encoder side is needed to read feature activations.
    """

    def __init__(self, d_model, n_dirs, k):
        super().__init__()
        self.k = k
        self.encoder = torch.nn.Linear(d_model, n_dirs, bias=False)
        self.pre_bias = torch.nn.Parameter(torch.zeros(d_model))
        self.latent_bias = torch.nn.Parameter(torch.zeros(n_dirs))

    def encode(self, x):  # x: (T, d_model) -> (T, n_dirs), only k nonzero per row
        pre = F.linear(x - self.pre_bias, self.encoder.weight, self.latent_bias)
        vals, inds = torch.topk(pre, self.k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, inds, torch.relu(vals))
        return out


def load_topk_sae(path, d_model, n_dirs, k):
    """Load weights saved from FastAutoencoder (torch.save(ae.state_dict(), path))."""
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    sae = TopKSAE(d_model, n_dirs, k)
    missing, unexpected = sae.load_state_dict(state, strict=False)
    # decoder.weight and stats_last_nonzero are expected to be "unexpected" (unused here)
    assert not missing, f"SAE weights missing keys: {missing}"
    print("SAE loaded. Ignored keys:", unexpected)
    return sae


def load_muscriptor(size="small"):
    from muscriptor import TranscriptionModel
    return TranscriptionModel.load_model(size)


def find_torch_module(obj):
    """TranscriptionModel is a wrapper; find the torch.nn.Module inside it."""
    if isinstance(obj, torch.nn.Module):
        return obj
    for name, val in vars(obj).items():
        if isinstance(val, torch.nn.Module):
            print(f"(found torch module at .{name})")
            return val
    raise RuntimeError("No torch.nn.Module found. Print vars(model) and look.")


def inspect_muscriptor(size="small"):
    """Run once: `python3 n2g_audio.py --inspect`. Tells you what to fill in."""
    import inspect
    tm = load_muscriptor(size)
    net = find_torch_module(tm)
    print("\n--- modules (name, type) ---")
    for name, mod in net.named_modules():
        if name.count(".") <= 2:
            print(name, type(mod).__name__)
    print("\n--- source of TranscriptionModel.transcribe ---")
    try:
        print(inspect.getsource(type(tm).transcribe))
    except Exception as e:
        print("could not print source:", e)


def forward_audio_fast(tm, wave_np):
    """
    One forward pass over the 5 s of audio only (no note generation).
    Mirrors LMModel.forward(sequence, condition_tensors, first_step=True):
    the mel conditioning is PREPENDED to the sequence, so the first 500
    positions of every layer's output are the 5 s of audio (100 frames/s).
    """
    net = find_torch_module(tm)
    dev = tm._device
    chunk = torch.from_numpy(wave_np).float().unsqueeze(0).to(dev)   # [1, T]
    conds = tm._build_conditions(chunk, None)
    cond_tensors = net.condition_provider(net.condition_provider.tokenize(conds))
    tok = net.initial_token_id
    tok = tok() if callable(tok) else tok
    seq = torch.full((1, 1), int(tok), dtype=torch.long, device=dev)
    net(seq, cond_tensors, first_step=True)


def forward_audio_slow(tm, wave_np):
    """Fallback: call the public transcribe() and stop after the first event.
    The hook keeps only the first forward call (the pass over the audio)."""
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    sf.write(path, wave_np, SR)
    try:
        gen = tm.transcribe(path)
        next(gen, None)
        if hasattr(gen, "close"):
            gen.close()
    finally:
        os.remove(path)


# From inspect2.py: the first forward pass has 504 positions = 5 s of mel
# frames (100 per second) followed by a few extra positions. The audio is
# prepended, so the first 500 positions are the 5 seconds of audio.
AUDIO_FRAMES = 500
AUDIO_SLICE = slice(0, AUDIO_FRAMES)


def make_run_model(tm, sae, layer_name, feature):
    net = find_torch_module(tm)
    net.eval()
    block = dict(net.named_modules())[layer_name]
    captured = {}

    def hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if "h" not in captured:               # keep only the first forward call
            captured["h"] = out.detach()

    block.register_forward_hook(hook)
    sae.eval()
    dev = tm._device
    sae.to(dev)
    state = {"fast": None}

    def forward(padded):
        if state["fast"] is not False:
            try:
                captured.clear()
                forward_audio_fast(tm, padded)
                if "h" not in captured:
                    raise RuntimeError("hook did not fire")
                if state["fast"] is None:
                    print("[n2g] using fast single-pass forward")
                state["fast"] = True
                return
            except Exception as e:
                if state["fast"]:
                    raise
                print("[n2g] fast forward failed, using slow transcribe() path:", repr(e))
                state["fast"] = False
        captured.clear()
        forward_audio_slow(tm, padded)

    @torch.no_grad()
    def hidden_window(chunk):
        """Hidden states (valid_frames, d_model) at the hooked layer for <= 5 s of audio."""
        n_valid = len(chunk)
        # right-pad with silence to 5 s, exactly like transcribe() pads its last chunk
        padded = np.pad(chunk, (0, WINDOW_SAMPLES - n_valid)).astype(np.float32)
        forward(padded)
        h = captured["h"]
        h = h[0] if h.dim() == 3 else h             # (positions, d_model)
        h = h[AUDIO_SLICE].float()
        valid = max(1, int(np.ceil(n_valid / WINDOW_SAMPLES * h.shape[0])))
        return h[:valid]

    @torch.no_grad()
    def run_window(chunk):
        h = hidden_window(chunk)
        return sae.encode(h.to(dev))[:, feature].float().cpu().numpy()

    def run_model(wave):
        out = []
        for s in range(0, len(wave), WINDOW_SAMPLES):   # long clips: 5 s windows
            out.append(run_window(wave[s:s + WINDOW_SAMPLES]))
        return np.concatenate(out)

    run_model.hidden_window = hidden_window
    return run_model


def check_causality(run_model, wave, cut_s=2.0):
    """
    Does the hidden state at time t depend on audio AFTER t?  Compares the
    first cut_s seconds of a full 5 s window against the same audio cut off at
    cut_s (padded with silence). If the numbers match, the audio positions are
    causal and the 'left context only' tests in N2G are valid. If they don't,
    the explanation describes the neuron given only the PAST audio.
    """
    assert len(wave) >= WINDOW_SAMPLES, "need a clip of at least 5 s"
    full = run_model.hidden_window(wave[:WINDOW_SAMPLES]).cpu().numpy()
    cut = run_model.hidden_window(wave[: int(cut_s * SR)]).cpu().numpy()
    n = len(cut) - 5                                  # skip frames near the cut edge
    rel = float(np.linalg.norm(full[:n] - cut[:n]) / (np.linalg.norm(full[:n]) + 1e-9))
    print(f"[n2g] causality check: relative difference = {rel:.4f}")
    print("      ~0 (below 0.01): causal.  Much larger: audio positions look at the FUTURE.")
    return rel


# =============================================================================
# PART 6: model_fn  (what create_n2g_explanation calls)
# list of token-id lists -> list of per-token activation lists
# =============================================================================

MIN_SAMPLES = 0  # run_model right-pads to the 5 s window itself, so no left-padding here


def make_model_fn(run_model):
    def model_fn(batch):
        results = []
        for ids in batch:
            segs = [SEGMENTS[i] for i in ids]
            wave = np.concatenate(segs).astype(np.float32)
            pad = max(0, MIN_SAMPLES - len(wave))
            wave = np.concatenate([np.zeros(pad, dtype=np.float32), wave])

            frame_acts = np.asarray(run_model(wave))
            n = len(frame_acts)
            stride = len(wave) / n                      # samples per model frame
            bounds = np.cumsum([0] + [len(s) for s in segs]) + pad

            acts = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                f0 = min(int(a // stride), n - 1)
                f1 = min(max(f0 + 1, int(np.ceil(b / stride))), n)
                acts.append(float(frame_acts[f0:f1].max()))  # token act = max over its frames
            results.append(acts)
        return results

    return model_fn


# =============================================================================
# PART 7: TRAIN SET + BUILD THE EXPLANATION
# =============================================================================


def build_train_set(paths, kmeans, model_fn, top_k=50):
    docs = []
    for p in paths:
        units, ids, _ = register_clip(p, kmeans)
        acts = model_fn([ids])[0]
        idx = int(np.argmax(acts))
        # re-measure using only audio up to idx so it matches the left-context-only tests later
        act = model_fn([ids[: idx + 1]])[0][-1]
        docs.append({
            "idx": idx,
            "act": act,
            "acts": acts,
            "tokens": [str(u) for u in units],  # k-means labels -> trie keys
            "token_ints": ids,                  # snippet ids -> audio for the model
        })
    docs = [d for d in docs if d["act"] > 0]
    docs.sort(key=lambda d: d["act"], reverse=True)
    return docs[:top_k]


def build_explanation(train_paths, kmeans, model_fn, top_k=50):
    train_set = build_train_set(train_paths, kmeans, model_fn, top_k)
    return create_n2g_explanation(
        model_fn, train_set, batch_size=16, padding_token=SILENCE_ID
    )


# =============================================================================
# PART 8: EVALUATE + INTERPRET
# =============================================================================


def evaluate(expl, paths, kmeans, model_fn):
    """Correlation between the rulebook's predictions and the real neuron.
    Run on held-out clips. Near 0 means the rulebook isn't capturing the neuron."""
    preds, trues = [], []
    for p in paths:
        units, ids, _ = register_clip(p, kmeans)
        preds += expl.predict([str(u) for u in units])
        trues += model_fn([ids])[0]
    return float(np.corrcoef(preds, trues)[0, 1])


def explain_clip(expl, path, kmeans, out_prefix, threshold=0.5, context=5):
    """Find where the rulebook fires, print times, save audio snippets to listen to."""
    units, spans, y = to_units(path, kmeans)
    preds = expl.predict([str(u) for u in units])
    peak = max(preds) if max(preds) > 0 else 1.0
    hits = [i for i, a in enumerate(preds) if a >= threshold * peak]
    for n, i in enumerate(hits):
        first = max(0, i - context)
        start = spans[first][0] * HOP
        end = spans[i][1] * HOP
        print(f"hit at {spans[i][0] * HOP / SR:.2f}s  predicted act {preds[i]:.2f}")
        sf.write(f"{out_prefix}_{n}.wav", y[start:end], SR)


# =============================================================================
# PART 9: MAIN  (EDIT THE CONFIG)
# =============================================================================

if __name__ == "__main__":
    import glob
    import sys

    if "--inspect" in sys.argv:          # step 1: find the layer names
        inspect_muscriptor("large")
        sys.exit(0)

    # ---- CONFIG ----
    CLIPS_GLOB = "clips/*.wav"     # folder of your .wav files
    N_TEST = 100                   # held-out clips for evaluation
    MODEL_SIZE = "large"           # your whole project uses the large model
    LAYER_NAME = "transformer.layers.47"   # last transformer layer (use "out_norm" for the output AFTER the final norm)
    FEATURE = 32768                 # which SAE feature to explain
    D_MODEL = 1536                 # large = 1536; must equal your SAE's d_model
    N_DIRS = 32768                 # your SAE's n_dirs (Config default in train.py)
    K = 32                         # your SAE's k (Config default in train.py)
    SAE_PATH = "my_sae.pt"         # state_dict saved from your trained FastAutoencoder
    TOP_K = 10                     # start small (saliency is slow), raise to 50 later
    # ----------------

    paths = sorted(glob.glob(CLIPS_GLOB))
    train_paths, test_paths = paths[:-N_TEST], paths[-N_TEST:]

    print("fitting k-means vocabulary...")
    kmeans = fit_kmeans(train_paths)

    tm = load_muscriptor(MODEL_SIZE)

    sae = load_topk_sae(SAE_PATH, D_MODEL, N_DIRS, K)

    run_model = make_run_model(tm, sae, LAYER_NAME, FEATURE)
    model_fn = make_model_fn(run_model)

    # sanity check: the feature should be nonzero somewhere on a clip where you know it fires
    _, ids, y0 = register_clip(train_paths[0], kmeans)
    raw = run_model(y0)
    print("frames returned:", len(raw), "for", len(y0) / SR, "seconds of audio")
    print("max feature activation on first clip:", float(raw.max()))
    print("first token acts:", model_fn([ids])[0][:10])
    if len(y0) >= WINDOW_SAMPLES:
        check_causality(run_model, y0)

    print("building explanation...")
    expl = build_explanation(train_paths, kmeans, model_fn, top_k=TOP_K)
    expl.dumpf("explanation.json")

    print("correlation on held-out clips:", evaluate(expl, test_paths, kmeans, model_fn))
    explain_clip(expl, test_paths[0], kmeans, "snippet")
