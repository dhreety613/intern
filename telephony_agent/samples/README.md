# 10 sample calls (synthetic: robotic voice, clean tones, 8 kHz phone-codec). Listen, then replay:
#   python -m telephony_agent.replay samples/manifest.json -v

01_voicemail_beep                greeting -> beep.           Expect: wait for beep, leave short message, hang up
02_voicemail_mailbox_full        "mailbox is full"           Expect: hang up, NO message
03_personal_greeting_then_vm     "Hi, this is John..." -> VM Expect: NOT a human; message only after the beep
04_ivr_menu                      sales/calibration/billing   Expect: wait for end of menu, press 2
05_ivr_operator_fallback         useless menu x3             Expect: retry, retry, press 0
06_hold_music_then_human         please hold + music + Maria Expect: silent through hold, hand off at Maria
07_call_screening                "say your name after tone"  Expect: states name on beep, waits, hands off at "Hello?"
08_carrier_not_in_service        carrier recording           Expect: hang up, tag invalid number
09_busy_tone                     480+620 Hz busy             Expect: detected from audio alone -> busy
10_human_hello                   "Hello?"                    Expect: hand off after they go quiet
