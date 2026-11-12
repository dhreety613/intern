"""Fusion: combines parallel detector Signals into Verdicts. Pure + time-driven (no I/O, no sleeping),
so every rule is unit-testable with explicit timestamps.

Principles
  * transcript labels outrank AMD once the line is classified (AMD runs once, at answer)
  * HUMAN is evidence-scored, never one signal: text + (reply-to-probe | AMD | rhythm) minus machine evidence
  * "speak now" moments (voicemail beep, screening beep) are arbitrated across audio, AMD and silence
  * IVR menus are acted on only after the speaker has finished (VAD + transcript quiet)
"""
from __future__ import annotations

import re

from ..classifier import norm
from ..config import Settings
from ..models import Label, Outcome
from .signals import Sig, Signal, VK, Verdict

_CARRIER = {Label.CARRIER_INVALID: Outcome.INVALID_NUMBER, Label.CARRIER_DISCONNECTED: Outcome.DISCONNECTED,
            Label.CARRIER_UNREACHABLE: Outcome.UNREACHABLE}
_MAILBOX = {Label.MAILBOX_FULL: Outcome.VOICEMAIL_FULL, Label.MAILBOX_DISABLED: Outcome.VOICEMAIL_DISABLED}
_PRE = {Sig.BUSY: Outcome.BUSY, Sig.NO_ANSWER: Outcome.NO_ANSWER, Sig.INVALID_NUMBER: Outcome.INVALID_NUMBER,
        Sig.FAILED: Outcome.FAILED, Sig.CANCELED: Outcome.CANCELED, Sig.UNREACHABLE_SIP: Outcome.UNREACHABLE}
_MACHINE_AMD = {Sig.AMD_MACHINE_START, Sig.AMD_END_BEEP, Sig.AMD_END_SILENCE, Sig.AMD_END_OTHER}


def _tokens(text: str) -> set[str]:
    return set(norm(text).split())


class Fusion:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()
        self.now = 0.0
        self.phase = "dialing"          # dialing listening ivr hold screening voicemail human done
        self.answered_t: float | None = None
        self.amd: Sig | None = None
        self.speaking = False
        self.last_speech_end = -1e9      # no speech yet = infinitely quiet
        self.last_activity = 0.0        # last speech/text/music evidence
        self.last_text_t = 0.0
        self.rhythm: tuple[str, float] | None = None
        self.music_on = False
        self.music_since = 0.0
        self.heard: list[tuple[float, str, Label]] = []
        self.probe_sent = False
        self.cand: dict | None = None   # pending human candidate
        self.ivr_buf: list[str] = []
        self.menus: list[set[str]] = []
        self.tone_announced = False
        self.vm_ready_at: float | None = None
        self.vm_sent = False
        self.screen_sent = False
        self.screen_t = 0.0
        self.hold_sent = False
        self.dead_sent = False
        self.out: list[Verdict] = []

    # ---------------------------------------------------------------- api
    def on_signal(self, sig: Signal) -> list[Verdict]:
        self.out = []
        if self.phase in ("done", "human"):
            return []
        self.now = max(self.now, sig.t)
        h = getattr(self, f"_h_{sig.kind.value}", None)
        if h:
            h(sig)
        self._timers()
        return self.out

    # ----------------------------------------------------------- helpers
    def _emit(self, kind: VK, conf=1.0, evidence=None, **data) -> None:
        if kind in (VK.HOLD, VK.SCREEN_SPEAK_READY, VK.VM_BEEP_READY):
            self.rhythm = None                # new phase: an earlier recording's rhythm no longer applies
        self.out.append(Verdict(kind, self.now, conf, evidence or [], data))

    def _done(self) -> None:
        self.phase = "done"

    @property
    def amd_machine(self) -> bool:
        return self.amd in _MACHINE_AMD

    # ------------------------------------------------------ call progress
    def _h_ringing(self, s): pass
    def _h_tick(self, s): pass

    def _h_answered(self, s: Signal) -> None:
        if self.phase == "dialing":
            self.phase, self.answered_t, self.last_activity = "listening", s.t, s.t
            self._emit(VK.ANSWERED)

    def _pre(self, s: Signal) -> None:
        if self.answered_t is None:
            self._emit(VK.PRE_ANSWER, 1.0, [s.kind.value], outcome=_PRE[s.kind])
            self._done()

    _h_busy = _h_no_answer = _h_invalid_number = _h_failed = _h_canceled = _h_unreachable_sip = _pre

    def _h_completed(self, s: Signal) -> None:
        if self.answered_t is None:
            self._emit(VK.REMOTE_HANGUP, 1.0, ["completed_before_answer"])
        else:
            self._emit(VK.REMOTE_HANGUP, 1.0, ["far_end_hangup"])
        self._done()

    # --------------------------------------------------------------- AMD
    def _h_amd_human(self, s): self.amd = s.kind; self._eval_human()
    def _h_amd_machine_start(self, s): self.amd = s.kind; self._eval_human()
    def _h_amd_unknown(self, s): pass

    def _h_amd_fax(self, s: Signal) -> None:
        self._emit(VK.FAX, 1.0, ["amd_fax"])
        self._done()

    def _h_amd_end_beep(self, s: Signal) -> None:
        self.amd = s.kind
        if self.phase in ("listening", "voicemail") and not self.vm_sent:
            self.phase, self.vm_sent = "voicemail", True
            self._emit(VK.VM_BEEP_READY, 0.95, ["amd:machine_end_beep"], source="amd")

    def _amd_end_nobeep(self, s: Signal) -> None:
        self.amd = s.kind
        if self.phase in ("listening", "voicemail") and not self.vm_sent and self.vm_ready_at is None:
            self.phase = "voicemail"
            self.vm_ready_at = s.t + 0.5            # greeting over, no beep to wait for

    _h_amd_end_silence = _h_amd_end_other = _amd_end_nobeep

    # ------------------------------------------------------ audio events
    def _h_beep_end(self, s: Signal) -> None:
        if self.phase == "voicemail" or (self.phase == "listening" and self.amd_machine):
            if not self.vm_sent:
                self.phase, self.vm_sent = "voicemail", True
                self._emit(VK.VM_BEEP_READY, s.conf, [f"audio:beep {s.data.get('freq', 0):.0f}Hz"], source="audio")
        elif self.phase == "screening" and not self.screen_sent:
            self.screen_sent = True
            self._emit(VK.SCREEN_SPEAK_READY, s.conf, ["audio:beep"], source="audio")

    def _h_tone_busy(self, s: Signal) -> None:
        if self.phase in ("dialing", "listening"):
            self._emit(VK.BUSY_TONE, s.conf, [f"audio:{s.data.get('cadence')}"], cadence=s.data.get("cadence", "busy"))
            self._done()

    def _h_tone_fax(self, s: Signal) -> None:
        if s.conf >= 0.8 and self.phase in ("listening", "dialing"):
            self._emit(VK.FAX, s.conf, [f"audio:fax {s.data.get('tone')}"])
            self._done()

    def _h_music_on(self, s: Signal) -> None:
        self.music_on, self.music_since = True, s.t
        self._eval_human()

    def _h_music_off(self, s: Signal) -> None:
        self.music_on = False

    # ---------------------------------------------------- voice activity
    def _h_speech_start(self, s: Signal) -> None:
        self.speaking = True
        self.last_activity = s.t

    def _h_speech_end(self, s: Signal) -> None:
        self.speaking, self.last_speech_end, self.last_activity = False, s.t, s.t

    def _h_rhythm(self, s: Signal) -> None:
        if not (self.rhythm and self.rhythm[0] == "machine" and s.data["verdict"] == "human"):   # a long monologue is not undone by a later short burst
            self.rhythm = (s.data["verdict"], s.conf)
        self._eval_human()

    # --------------------------------------------------------- transcript
    def _h_text_partial(self, s: Signal) -> None:
        self.last_text_t = self.last_activity = s.t

    def _h_text_label(self, s: Signal) -> None:
        label: Label = s.data["label"]
        text: str = s.data["text"]
        self.last_text_t = self.last_activity = s.t
        self.heard.append((s.t, text, label))

        if label in _CARRIER:
            self._emit(VK.CARRIER, 1.0, [f"text:{label.value}"], outcome=_CARRIER[label], text=text)
            return self._done()
        if label in _MAILBOX:
            self._emit(VK.MAILBOX, 1.0, [f"text:{label.value}"], outcome=_MAILBOX[label], text=text)
            return self._done()

        if self.probe_sent and label in (Label.HUMAN, Label.UNKNOWN, Label.AMBIGUOUS_GREETING) and s.data["words"] <= 14:
            self.cand = {"t0": s.t, "words": s.data["words"], "reply": True}
            return self._eval_human()

        if label is Label.HUMAN:
            if self.phase == "voicemail" and self.amd_machine:
                return                              # personal greeting that starts like a person
            self.cand = {"t0": s.t, "words": s.data["words"], "reply": False}
            return self._eval_human()
        if label is Label.AMBIGUOUS_GREETING:
            if not self.probe_sent:
                self.probe_sent = True
                self._emit(VK.AMBIGUOUS_GREETING, 0.5, ["text:greeting_only"], text=text)
            return
        if label is Label.VOICEMAIL_GREETING:
            self.tone_announced |= s.data["tone"]
            if self.phase != "voicemail":
                self.phase, self.vm_ready_at = "voicemail", None
                self._emit(VK.VM_GREETING, 0.9, ["text:voicemail_greeting"], text=text)
            return
        if label is Label.SCREEN_PROMPT:
            if self.phase != "screening":
                self.phase, self.screen_t = "screening", s.t
                self._emit(VK.SCREEN_PROMPT, 0.9, ["text:screen_prompt"], text=text)
            return
        if label is Label.SCREEN_REJECT:
            if self.phase == "screening":
                self._emit(VK.SCREEN_REJECT, 0.9, ["text:screen_reject"], text=text)
                self._done()
            return
        if label is Label.IVR_INVALID_ENTRY:
            self.phase = "ivr"
            self.ivr_buf.clear()
            return self._emit(VK.IVR_INVALID, 0.9, ["text:invalid_entry"], text=text)
        if label is Label.IVR_EXTENSION_PROMPT:
            self.phase = "ivr"
            self.ivr_buf.clear()
            return self._emit(VK.IVR_EXTENSION, 0.9, ["text:extension_prompt"], text=text)
        if label is Label.IVR_MENU:
            self.phase = "ivr"
            self.ivr_buf.append(text)
            return
        if label is Label.HOLD:
            if self.phase != "hold":
                self.phase, self.hold_sent = "hold", True
                self._emit(VK.HOLD, 0.9, ["text:hold_phrase"], source="text")

    # --------------------------------------------------------- human gate
    def _eval_human(self) -> None:
        c = self.cand
        if c is None or self.phase in ("done", "human"):
            return
        score, ev = 0.0, []
        if c["reply"]:
            score += 0.7; ev.append("reply_to_probe")
        else:
            score += 0.5; ev.append("text:human_phrase")
        neg = False
        if self.amd is Sig.AMD_HUMAN:
            score += 0.3; ev.append("amd:human")
        elif self.amd_machine:
            score -= 0.6 if c["words"] > 3 else 0.2; ev.append("amd:machine"); neg = True
        if self.rhythm:
            kind, conf = self.rhythm
            if kind == "human":
                score += 0.25; ev.append("rhythm:short_burst")
            else:
                score -= 0.5; ev.append("rhythm:long_monologue"); neg = True
        if self.music_on:
            score -= 0.3; ev.append("music_playing"); neg = True
        if self.phase == "voicemail":
            score -= 0.4; ev.append("phase:voicemail"); neg = True
        waited = self.now - c["t0"]
        # never confirm over someone who is still talking: a person yields the turn, a greeting rolls on
        speech_quiet = 0.0 if self.speaking else self.now - self.last_speech_end
        if (score >= self.s.human_confirm_threshold and speech_quiet >= 0.5) or \
                (waited >= self.s.human_fallback_s and score >= 0.5 and not neg and speech_quiet >= 1.0):
            self.phase = "human"
            self._emit(VK.HUMAN, min(1.0, score), ev, heard=[t for _, t, _ in self.heard])
        elif waited >= 3.0 and score < 0.3:
            self.cand = None

    # ------------------------------------------------------------- timers
    def _quiet_for(self) -> float:
        """Seconds since any speech/text evidence; 0 while someone is talking."""
        return 0.0 if self.speaking else self.now - max(self.last_activity, self.last_text_t, self.last_speech_end)

    def _timers(self) -> None:
        if self.phase in ("done", "human"):
            return
        s, now = self.s, self.now
        if self.cand:
            self._eval_human()
            if self.phase == "human":
                return

        if self.phase == "listening" and self.answered_t is not None:
            # silent line: prompt once
            if (not self.probe_sent and not self.heard and self.amd in (Sig.AMD_HUMAN, None, Sig.AMD_UNKNOWN)
                    and now - self.answered_t >= s.human_probe_after_s and self._quiet_for() >= s.human_probe_after_s
                    and not self.music_on):
                self.probe_sent = True
                self._emit(VK.PROBE_DUE, 0.5, ["silence_after_answer"])
            # music without words = hold
            elif self.music_on and now - self.music_since >= s.music_hold_s and not self._words_recent(5.0):
                self.phase, self.hold_sent = "hold", True
                self._emit(VK.HOLD, 0.7, ["audio:music"], source="music")
            elif now - self.answered_t >= s.listen_timeout_s and not self.dead_sent and self._quiet_for() >= 2:
                self.dead_sent = True
                self._emit(VK.DEAD_AIR, 0.8, ["no_activity"], amd_human=self.amd is Sig.AMD_HUMAN)
                self._done()
                return

        if self.phase == "ivr":
            if self.ivr_buf and self._quiet_for() >= s.ivr_settle_s:
                text = " ".join(self.ivr_buf)
                self.ivr_buf = []
                toks = _tokens(text)
                repeat = any(len(toks & m) / max(1, len(toks | m)) >= 0.8 for m in self.menus)
                self.menus.append(toks)
                from ..ivr import parse_options
                self._emit(VK.IVR_MENU_READY, 0.9, ["menu_settled"], text=text, options=parse_options(text), repeat=repeat)
            elif not self.ivr_buf and self.music_on and now - self.music_since >= s.music_hold_s \
                    and not self._words_recent(5.0):
                self.phase, self.hold_sent = "hold", True
                self._emit(VK.HOLD, 0.6, ["audio:music_after_ivr"], source="music")

        if self.phase == "voicemail" and not self.vm_sent:
            if self.vm_ready_at is not None and now >= self.vm_ready_at:
                self.vm_sent = True
                self._emit(VK.VM_BEEP_READY, 0.7, ["amd:end_without_beep"], source="amd_silence")
            elif self.tone_announced and self._quiet_for() >= s.voicemail_silence_fallback_s + 1.0:
                self.vm_sent = True
                self._emit(VK.VM_BEEP_READY, 0.5, ["tone_announced+silence"], source="silence_fallback")

        if self.phase == "screening" and not self.screen_sent and self._quiet_for() >= s.screen_wait_s:
            self.screen_sent = True
            self._emit(VK.SCREEN_SPEAK_READY, 0.5, ["prompt_over+silence"], source="silence")

    def _words_recent(self, secs: float) -> bool:
        return any(self.now - t <= secs for t, _, _ in self.heard) or (self.speaking)
