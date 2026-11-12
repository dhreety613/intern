"""Run the detection pipeline on REAL phone-call recordings (NCSU Robocall Audio Dataset, CC BY-ND 4.0).

  1. pick a stratified sample of calls from the dataset checkout
  2. real streaming ASR (sherpa-onnx zipformer) -> timed final transcripts, like a live STT would send
  3. audio -> 8 kHz mu-law (what Twilio delivers) -> detectors -> fusion -> controller, on a virtual clock
  4. write report.txt + copy the ORIGINAL (unmodified) wavs into folders by what the agent did

    python tools/run_dataset.py <dataset_dir> <asr_model_dir> <out_dir> [--n 120]
"""
from __future__ import annotations

import json
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from telephony_agent.config import Settings                                  # noqa: E402
from telephony_agent.ivr import choose_key, parse_options                    # noqa: E402
from telephony_agent.models import Dtmf, HandoffToBrain, Hangup, Speak       # noqa: E402
from telephony_agent.replay import load_wav, replay                          # noqa: E402

TARGET_FILES = None


# ------------------------------------------------------------------ selection
def select(ds: Path, n: int, seed: int = 11) -> pd.DataFrame:
    df = pd.read_csv(ds / "metadata.csv")
    df = df[df.language == "en"].drop_duplicates("file_name").copy()
    df["transcript"] = df.transcript.fillna("")
    sizes = {}
    for line in (ds / "sizes.txt").read_text().splitlines():
        size, path = line.split(" ", 1)
        sizes[path] = int(size)
    df["bytes"] = df.file_name.map(sizes)
    df = df[(df.bytes > 60_000) & (df.bytes < 2_000_000)]            # ~2 s .. ~60 s
    t = df.transcript.str.lower()
    groups = {
        "menu ('press N')": t.str.contains(r"\bpress (?:\d|one|two|three|four|five|six|seven|eight|nine|zero|star)\b"),
        "hold / transfer": t.str.contains(r"please hold|stay on the line|transfer|connecting you|one moment|please wait") & ~t.str.contains(r"\bpress\b"),
        "opens like a person": t.str.strip().str[:30].str.contains(r"^(?:hello|hi\b|hey|good (?:morning|afternoon))") & ~t.str.contains(r"\bpress\b"),
        "other": ~t.str.contains(r"\bpress\b|hold|transfer|^hello"),
    }
    quota = {"menu ('press N')": int(n * 0.45), "hold / transfer": int(n * 0.2), "opens like a person": int(n * 0.2)}
    quota["other"] = n - sum(quota.values())
    rnd = random.Random(seed)
    picks, seen = [], set()
    for g, mask in groups.items():
        pool = [f for f in df[mask].file_name if f not in seen]
        rnd.shuffle(pool)
        for f in pool[:quota[g]]:
            picks.append((f, g))
            seen.add(f)
    out = df.set_index("file_name").loc[[p[0] for p in picks]].reset_index()
    out["group"] = [p[1] for p in picks]
    return out


# ------------------------------------------------------------------------ ASR
_REC = None


def _recognizer(model_dir: str):
    global _REC
    if _REC is None:
        import sherpa_onnx
        m = Path(model_dir)
        _REC = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(m / "tokens.txt"), encoder=str(m / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(m / "decoder-epoch-99-avg-1.onnx"), joiner=str(m / "joiner-epoch-99-avg-1.int8.onnx"),
            num_threads=1, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            enable_endpoint_detection=True, rule1_min_trailing_silence=1.0, rule2_min_trailing_silence=0.5,
            rule3_min_utterance_length=10.0)
    return _REC


def asr_finals(pcm16k: np.ndarray, model_dir: str) -> list[dict]:
    """Streaming ASR -> [{'t': time the final was emitted, 'text': ...}] (endpointed, like Google STT finals)."""
    rec = _recognizer(model_dir)
    st = rec.create_stream()
    x = pcm16k.astype(np.float32) / 32768.0
    out, step = [], 1600
    for i in range(0, len(x) + 16000, step):                     # +1 s of silence flushes the last utterance
        chunk = x[i:i + step] if i < len(x) else np.zeros(step, np.float32)
        st.accept_waveform(16000, chunk)
        while rec.is_ready(st):
            rec.decode_stream(st)
        if rec.is_endpoint(st):
            txt = rec.get_result(st).strip()
            if txt:
                out.append({"t": round((i + step) / 16000, 2), "text": txt.lower()})
            rec.reset(st)
    txt = rec.get_result(st).strip()
    if txt:
        out.append({"t": round(len(x) / 16000 + 0.5, 2), "text": txt.lower()})
    return out


def to_8k(pcm16k: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly
    return np.clip(resample_poly(pcm16k.astype(np.float32), 1, 2), -32768, 32767).astype(np.int16)


# ------------------------------------------------------------------ one call
def process(args):
    ds, model_dir, row = args
    pcm16 = _read16(Path(ds) / row["file_name"])
    finals = asr_finals(pcm16, model_dir)
    events = [{"t": 0.0, "type": "status", "status": "in-progress"}] + \
             [{"t": f["t"], "type": "text", "text": f["text"], "final": True} for f in finals]
    res = replay(to_8k(pcm16), events, Settings(callback_number="4125550123"), tail_s=4.0)
    # same pipeline, but fed the dataset's reference (Whisper-medium) transcript text -> "what a perfect STT would give"
    ref_choice = choose_key(parse_options(row["transcript"]))
    return {
        "file": row["file_name"], "group": row["group"], "dur": round(len(pcm16) / 16000, 1),
        "finals": finals, "reference": row["transcript"].strip(),
        "verdicts": [(round(v.t, 2), v.kind.value, v.evidence, {k: v.data[k] for k in ("options", "text", "outcome", "source", "repeat") if k in v.data}) for v in res.verdicts],
        "actions": [(round(t, 2), type(a).__name__, getattr(a, "digits", getattr(a, "text", str(getattr(a, "outcome", ""))))) for t, a in res.actions],
        "outcome": res.outcome, "state": res.state,
        "ref_key": ref_choice.key if ref_choice else None,
    }


def _read16(p: Path) -> np.ndarray:
    import wave
    with wave.open(str(p), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)[:, 0]
    if sr != 16000:                                   # a few "16 kHz" files are not; normalise for the ASR
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(16000, sr)
        x = np.clip(resample_poly(x.astype(np.float32), 16000 // g, sr // g), -32768, 32767).astype(np.int16)
    return x


# --------------------------------------------------------------------- report
def decision(r: dict) -> str:
    acts = r["actions"]
    kinds = [v[1] for v in r["verdicts"]]
    if any(a[1] == "HandoffToBrain" for a in acts):
        return "hand_off_to_brain"
    if any(a[1] == "Dtmf" for a in acts):
        return "ivr_pressed_key"
    if any(a[1] == "Speak" and "Anne" in a[2] and "upcoming calibration" in a[2] for a in acts):
        return "voicemail_message_left"
    if any(a[1] == "Hangup" for a in acts):
        return f"hang_up_{r['outcome']}"
    if "hold" in kinds:
        return "hold_waiting"
    if "ivr_menu_ready" in kinds:
        return "ivr_menu_no_key"
    if r["outcome"]:
        return f"ended_{r['outcome']}"
    return "still_listening"


def fmt_call(r: dict, i: int) -> str:
    L = [f"[{i:03d}] {Path(r['file']).name}   ({r['dur']}s, group: {r['group']})",
         f"  agent did : {decision(r).upper()}"]
    ref = re.sub(r"\s+", " ", r["reference"])
    L.append(f"  reference : {ref[:230]}{'...' if len(ref) > 230 else ''}")
    heard = " | ".join(f"{f['t']:.1f}s:{f['text']}" for f in r["finals"])
    L.append(f"  our ASR   : {heard[:230]}{'...' if len(heard) > 230 else ''}")
    L.append("  timeline  :")
    rows = []
    for t, kind, ev, data in r["verdicts"]:
        extra = ""
        if kind == "ivr_menu_ready":
            extra = "  options: " + ", ".join(f"{o.key}={o.desc[:28]}" for o in data["options"]) + ("  [repeat]" if data.get("repeat") else "")
        elif kind in ("carrier", "mailbox"):
            extra = f"  -> {data['outcome'].value}"
        rows.append((t, f"DETECTED {kind:<19}{extra}  ({', '.join(ev)})"))
    for t, kind, arg in r["actions"]:
        verb = {"Dtmf": f"ACT  press {arg}", "Speak": f"ACT  say \"{arg[:48]}\"", "HandoffToBrain": "ACT  HAND OFF to brain (human confirmed)",
                "Hangup": f"ACT  hang up ({arg})"}.get(kind, kind)
        rows.append((t, verb))
    if not rows:
        rows.append((0.0, "(nothing detected - agent keeps waiting silently)"))
    for t, txt in sorted(rows, key=lambda x: x[0]):
        L.append(f"     {t:6.2f}s  {txt}")
    if r["group"].startswith("menu"):
        pressed = [a[2] for a in r["actions"] if a[1] == "Dtmf"]
        L.append(f"  check     : pressed {pressed or 'nothing'}; key a perfect transcript would choose = {r['ref_key']}")
    return "\n".join(L)


def build_report(results: list[dict], n_requested: int, secs: float) -> str:
    dec = Counter(decision(r) for r in results)
    by_group = defaultdict(Counter)
    for r in results:
        by_group[r["group"]][decision(r)] += 1
    menu = [r for r in results if r["group"].startswith("menu")]
    pressed = [r for r in menu if any(a[1] == "Dtmf" for a in r["actions"])]
    agree = [r for r in pressed if r["ref_key"] and any(a[1] == "Dtmf" and a[2] == r["ref_key"] for a in r["actions"])]
    handed = [r for r in results if decision(r) == "hand_off_to_brain"]
    L = ["=" * 78, "TELEPHONY AGENT - DETECTION RUN ON REAL PHONE-CALL RECORDINGS", "=" * 78, "",
         "DATA   NCSU Robocall Audio Dataset (github.com/wspr-ncsu/robocall-audio-dataset), CC BY-ND 4.0.",
         "       Real recordings of automated / semi-automated phone calls that crossed the phone network.",
         f"       {len(results)} English calls sampled (seeded, stratified), 2-60 s each.",
         "       Each recording is treated as the FAR-END audio of a call our agent placed.",
         "METHOD audio -> 8 kHz mu-law (Twilio format) -> detectors (VAD, tones, music, text) -> fusion -> controller.",
         "       Transcripts come from real streaming ASR (sherpa-onnx zipformer-20M, endpointed finals), NOT from the",
         "       dataset's text. The dataset's Whisper-medium transcript is shown as 'reference' for comparison only.",
         "       No Twilio AMD verdicts exist for these files, so AMD is absent (own detection only).",
         f"       Replay compute time: {secs:.0f}s.", "",
         "WHAT THIS DATASET CAN AND CANNOT TELL US",
         "  + REAL: IVR-style 'press N' menus, hold/transfer messages, recordings that open like a person.",
         "  - MISSING: voicemail GREETINGS, mailbox-full, call screening, carrier 'not in service', busy tones.",
         "    The dataset contains no voicemail greetings (0 of 1,378 English transcripts). Those paths were not",
         "    exercised here. No open voicemail-greeting corpus was reachable from this sandbox.",
         "  - There is no human-labelled ground truth. 'Reference' is Whisper-medium text. Correctness below is",
         "    judged against that text and by reading the calls, not by a verified label.", ""]
    L += ["SUMMARY: what the agent decided", "-" * 78]
    for k, v in dec.most_common():
        L.append(f"  {v:4d}  {k}")
    L += ["", "By group:"]
    for g, c in by_group.items():
        L.append(f"  {g}  ({sum(c.values())} calls)")
        for k, v in c.most_common():
            L.append(f"        {v:4d}  {k}")
    L += ["", "IVR key selection (menu group)", "-" * 78,
          f"  menu calls: {len(menu)}   agent pressed a key: {len(pressed)}   pressed same key as a perfect-transcript run: {len(agree)}"
          + (f" ({100 * len(agree) // max(1, len(pressed))}%)" if pressed else ""),
          f"  menu calls where the agent never acted: {len(menu) - len(pressed)}  (see 'FINDINGS')",
          "", f"Hand-offs to the brain: {len(handed)} of {len(results)}  (every one should be a real person - listen to them)", ""]
    L += ["FINDINGS", "-" * 78, "@@FINDINGS@@", "", "PER-CALL RESULTS", "=" * 78, ""]
    for i, r in enumerate(sorted(results, key=lambda r: (decision(r), r["file"])), 1):
        L += [fmt_call(r, i), ""]
    return "\n".join(L)


def main():
    ds, model, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 120
    out.mkdir(parents=True, exist_ok=True)
    sel = select(ds, n)
    rows = sel.to_dict("records")
    (out / "selected_files.txt").write_text("\n".join(r["file_name"] for r in rows))
    if "--list-only" in sys.argv:
        print(len(rows), "selected")
        return
    t0 = time.time()
    with Pool(2) as p:
        results = p.map(process, [(str(ds), model, r) for r in rows], chunksize=2)
    secs = time.time() - t0
    json.dump(results, open(out / "results.json", "w"), default=lambda o: getattr(o, "value", str(o)), indent=1)
    rep = build_report(results, n, secs)
    (out / "report_raw.txt").write_text(rep)
    print(f"done {len(results)} calls in {secs:.0f}s")


if __name__ == "__main__":
    main()
