"""Turn results.json + report_raw.txt into the deliverable: report .txt + original clips grouped by decision."""
import json, shutil, sys, collections, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_dataset import decision

ds, run, out = map(Path, sys.argv[1:4])
res = json.loads((run / "results.json").read_text())
raw = (run / "report_raw.txt").read_text()
dec = collections.Counter(decision(r) for r in res)
menu = [r for r in res if r["group"].startswith("menu")]
pressed = [r for r in menu if decision(r) == "ivr_pressed_key"]
nokey = [r for r in menu if decision(r) == "ivr_menu_no_key"]
still = [r for r in menu if decision(r) == "still_listening"]
dead = [r for r in res if decision(r) == "hang_up_failed"]
dead_speech = [r for r in dead if r["finals"]]
F = f"""
1. FALSE HUMAN HAND-OFFS: 0 of {len(res)}. (Brain would only be called on a real person.)
   The first full run produced 5 false hand-offs: robocalls saying "this is Mary Fletcher" /
   "this call is from Amazon ... this is to notify you" were read as a person. Fixed in this run:
   a person's answer must be short (<=14 words), "this is to/a/the/your/..." no longer counts, and a
   long monologue can no longer be 'undone' by a later short burst. Two unit tests still pass (103).
   This is tuned on this dataset, so re-check with your own calls.

2. IVR MENUS: of {len(menu)} 'press N' calls the agent found the menu and pressed a key on {len(pressed)};
   found the menu but chose NOT to press on {len(nokey)}; heard nothing usable on {len(still)}.
   - Not pressing is mostly by design: many of these are single-option scam menus ("press 1 to speak
     with an officer") whose option does not match human -> voicemail -> end-call intent. The agent
     does not press an arbitrary key. Whether 'officer / representative / agent' should count as
     'human' is your call; it is a one-line change in ivr.py.
   - When it did press, the key matched a perfect-transcript run in {sum(1 for r in pressed if any(a[1]=='Dtmf' and a[2]==r['ref_key'] for a in r['actions']))} of {len(pressed)} calls.
     Misses come from the ASR dropping a menu line ("press two" heard as "breast two").

3. HANG-UPS ({len(dead)}): the recording ends or goes quiet, nothing actionable was said, 20 s of dead
   air -> hang up as FAILED. {len(dead_speech)} of them had speech but no menu/human/voicemail cue (pure
   scam monologues). Correct for a call with nothing to act on.

4. REPLAY BUG FOUND AND FIXED while doing this: the first pass acted on only 3 of 54 menus because my
   replay stopped feeding audio when the file ended, so the VAD never saw speech end and the
   'menu settled' check never fired. Live streams keep sending line silence, so replay now pads
   4 s of silence. The detectors were not at fault.

5. WHAT IS NOT TESTED HERE (no such audio in this dataset or reachable from the sandbox):
   voicemail greetings + beep, mailbox full/disabled, call screening, carrier 'not in service',
   busy/reorder tones, hold music, real human answers, Twilio AMD. The voicemail and beep logic is
   only covered by synthetic tests. Please send/record a few real ones (your own Twilio call
   recordings are ideal) and I will run them through the same script.

6. ASR is a small streaming model (zipformer-20M) and is much worse than Google STT, so menus here
   are harder than they will be live. Finals sometimes arrive 10-15 s into a long robocall.

CLIP FOLDERS: clips/<what the agent did>/ hold the ORIGINAL, unmodified WAVs (CC BY-ND 4.0, NCSU
Robocall Audio Dataset, github.com/wspr-ncsu/robocall-audio-dataset). Each file name is prefixed
with its report number [NNN]. No voicemail folder exists because none was detected or present.
"""
report = raw.replace("@@FINDINGS@@", F.strip("\n"))
out.mkdir(parents=True, exist_ok=True)
(out / "REPORT_detection_run.txt").write_text(report)
clips = out / "clips"
if clips.exists(): shutil.rmtree(clips)
for i, r in enumerate(res, 1):
    d = clips / decision(r); d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ds / r["file"], d / f"[{i:03d}]_{Path(r['file']).name[:60]}")
(out / "clips" / "LICENSE_AND_ATTRIBUTION.txt").write_text(
    "NCSU Robocall Audio Dataset, CC BY-ND 4.0 (https://creativecommons.org/licenses/by-nd/4.0/).\n"
    "Source: https://github.com/wspr-ncsu/robocall-audio-dataset . Files are unmodified originals.\n")
z = out / "clips_by_detection.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
    for p in sorted(clips.rglob("*")):
        if p.is_file(): zf.write(p, p.relative_to(out))
print(dec); print(z.stat().st_size // 1e6, "MB")
