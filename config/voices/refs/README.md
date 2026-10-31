# Voice reference clips

These are the only per-voice assets, and they are **not tracked in git** (see `.gitignore`); a local checkout must supply them. A profile (`config/voices/*.yaml`) points at
one of these clips; the clone prompt is built once and cached. A clean,
single-speaker recording (3-30 s, no music, no audience, little room
reverberation) is what makes a cloned voice sound clean — a reverberant or noisy
reference bakes that room into the clone.

| File | Source | Provenance / edit |
|---|---|---|
| `obama.wav` | "A Promised Land by Barack Obama, read by Barack Obama \| audiobook excerpt" (official Penguin Random House Audio), YouTube `xRUU4Emx7mE` | studio narration, cut 58.11-76.44 s, mono 24 kHz, peak-normalised. Replaced the reverberant 2012 DNC convention clip (YouTube `2rl8Ou84s5U`, 2109.10-2123.45 s) after the owner reported hall echo in the clone. |
| `goku.wav` | "All Goku Voice Clips - Dragon Ball Xenoverse 2 - Voice Lines (Sean Schemmel)", YouTube `i1o3pL454Jo` | 8 isolated dry game voice lines (no music/SFX between lines), trimmed 0.01 s edges, per-line RMS-normalised, concatenated with 0.12 s gaps; 19.84 s total, mono 24 kHz, peak-normalised. Replaced the original 1.68 s `input/heyitsmegoku.mp3` clip, which was too short for a stable clone. |
| `computer.wav` | "Courage The Cowardly Dog All Computer Moments Updated", YouTube `voFFxd47b2g` | the Computer's windmill-legend narration, 899.6-926.3 s (single synthesized voice, no music or other characters), mono 24 kHz, peak-normalised. |
| `courage.wav` | same compilation, YouTube `voFFxd47b2g` | Courage's three clean spoken lines: 5.2-13.8 s ("Come on, come on!...Search!"), 72.3-80.0 s ("Muriel doesn't remember..."), 296.2-306.8 s ("Computer, how do you get rid of bad eggplants?"), concatenated with 0.35 s gaps; 26.4 s total, mono 24 kHz, peak-normalised. Courage mostly screams in the show; these are his calmest clean lines. |

Transcripts live next to each clip (`<name>.txt`); they are only required for
`clone_mode: icl` profiles (the shipped profiles use `xvec`, which clones the
speaker embedding only).
