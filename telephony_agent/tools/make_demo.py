"""10 demo recordings: far-end audio (LEFT) + what the agent sent back (RIGHT: DTMF tones / spoken message),
with a text output per call. 8 are real robocall clips, 2 are synthetic voicemail (no real voicemail corpus reachable).
Agent speech here is espeak (stand-in for the ElevenLabs voice). Usage: make_demo.py <dataset_dir> <run_dir> <out_dir>"""
import json, re, sys, wave
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from telephony_agent.config import Settings
from telephony_agent.replay import replay
from telephony_agent.tones import dtmf_mulaw, mulaw_decode
from tools.run_dataset import _read16, to_8k, decision
import synth

ds, run, out = map(Path, sys.argv[1:4]); out.mkdir(parents=True, exist_ok=True)
S = Settings(callback_number="4125550123")
res = json.loads((run / "results.json").read_text())


def agent_track(n, actions):
    tr = np.zeros(n, np.int32)
    for t, a in actions:
        i = int(t * 8000)
        if type(a).__name__ == "Dtmf":
            x = mulaw_decode(dtmf_mulaw(a.digits)).astype(np.int32)
        elif type(a).__name__ == "Speak":
            x = synth.speech(a.text).astype(np.int32)
        else:
            continue
        tr[i:i + len(x)] += x[:max(0, n - i)]
    return np.clip(tr, -32768, 32767).astype(np.int16)


def write(path, far, agent):
    n = max(len(far), len(agent)); far = np.pad(far, (0, n - len(far))); agent = np.pad(agent, (0, n - len(agent)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(np.stack([far, agent], 1).astype(np.int16).tobytes())


def act_rows(r):
    rows = [(v.t, f"DETECTED {v.kind.value:<18} {','.join(v.evidence)}") for v in r.verdicts]
    for t, a in r.actions:
        k = type(a).__name__
        d = {"Dtmf": lambda: f"PRESS KEY {a.digits}", "Speak": lambda: f"SPEAK  \"{a.text}\"",
             "Hangup": lambda: f"HANG UP ({a.outcome.value})", "HandoffToBrain": lambda: "HAND OFF TO BRAIN"}.get(k, lambda: k)()
        rows.append((t, "ACTION   " + d))
    return [f"      {t:6.2f}s  {x}" for t, x in sorted(rows, key=lambda z: z[0])]


def vm_opts(x):
    return next((v[3].get("options", []) for v in x["verdicts"] if v[1] == "ivr_menu_ready"), [])


def pick(pred, n=1, used=set()):
    got = [x for x in res if pred(x) and x["file"] not in used][:n]
    used.update(x["file"] for x in got)
    return got


human_rx = re.compile(r"officer|operator|representative|agent|speak|talk|live|person")
cases = []
cases += [(x, "IVR: option that reaches a person -> press that key") for x in pick(lambda x: any(a[1] == "Dtmf" and a[2] != "0" for a in x["actions"]) and any(human_rx.search(o) for o in map(str, vm_opts(x))), 3)]
cases += [(x, "IVR: no person option -> press 0 for operator") for x in pick(lambda x: any(a[1] == "Dtmf" and a[2] == "0" for a in x["actions"]), 3)]
cases += [(x, "Robocall that says 'this is <name>' -> must NOT hand off to the brain") for x in pick(lambda x: "Fletcher" in x["reference"] and not any(a[1] == "HandoffToBrain" for a in x["actions"]))]
cases += [(x, "Long recording, nothing to act on -> wait silently, give up after dead air") for x in pick(lambda x: decision(x) == "hang_up_failed" and x["group"].startswith("other"))]

lines = ["TELEPHONY AGENT - 10 TEST RECORDINGS", "=" * 70,
         "Each WAV is stereo: LEFT = the far end (what the call played), RIGHT = what OUR AGENT sent back",
         "(key-press tones, spoken message). Real clips: NCSU Robocall Audio Dataset, CC BY-ND 4.0 (left channel is the",
         "unmodified original; right channel is ours). Agent speech = espeak stand-in for the ElevenLabs voice.", ""]
n = 0
for x, title in cases:
    n += 1
    pcm = to_8k(_read16(ds / x["file"]))
    r = replay(pcm, [{"t": 0.0, "type": "status", "status": "in-progress"}] + [{"t": f["t"], "type": "text", "text": f["text"], "final": True} for f in x["finals"]], S, tail_s=4.0)
    name = f"{n:02d}_{re.sub(r'[^a-z0-9]+','_',title.split(':')[0].lower())}_{Path(x['file']).stem[:14]}.wav"
    write(out / name, np.pad(pcm, (0, 32000)), agent_track(len(pcm) + 32000, r.actions))
    lines += [f"[{n:02d}] {name}", f"      CASE      : {title}", "      SOURCE    : REAL recording",
              f"      FAR END   : {re.sub(chr(10),' ',x['reference'])[:200]}...", "      OUTPUT    :"] + act_rows(r) + [f"      RESULT    : {r.outcome or r.state}", ""]

# ---- 2 synthetic voicemail cases (no real voicemail audio available)
def syn(title, greet, stt, mailbox=False):
    global n
    n += 1
    beep = synth.tone(1000, 0.6) if not mailbox else synth.silence(0.1)
    far = synth.cat(synth.silence(0.8), synth.speech(greet), synth.silence(0.3), beep, synth.silence(6.0))
    ev = [{"t": 0, "type": "status", "status": "in-progress"}, {"t": 0.8 + len(synth.speech(greet)) / 8000 + 0.3, "type": "text", "text": stt, "final": True}]
    r = replay(far, ev, S, tail_s=4.0)
    name = f"{n:02d}_voicemail_{'mailbox_full' if mailbox else 'leave_message'}.wav"
    write(out / name, far, agent_track(len(far), r.actions))
    lines.extend([f"[{n:02d}] {name}", f"      CASE      : {title}", "      SOURCE    : SYNTHETIC far end (espeak) - no real voicemail corpus reachable",
                  f"      FAR END   : {stt}", "      OUTPUT    :"] + act_rows(r) + [f"      RESULT    : {r.outcome or r.state}", ""])

syn("Voicemail greeting + beep -> wait for beep, speak Anne's message, hang up",
    "Hi, you have reached Sam. I can't come to the phone right now. Please leave a message after the tone.",
    "Hi, you have reached Sam. I can't come to the phone right now. Please leave a message after the tone.")
syn("Mailbox full -> do NOT leave a message, hang up",
    "The mailbox is full and cannot accept any more messages at this time. Goodbye.",
    "The mailbox is full and cannot accept any more messages at this time. Goodbye.", mailbox=True)
(out / "OUTPUTS_10_recordings.txt").write_text("\n".join(lines))
print("\n".join(lines))
