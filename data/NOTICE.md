# Demo data: where every voice comes from

No real customer data is used. Account names, numbers and call scripts are fictional (US numbers use 555;
the UK number uses Ofcom's drama range).

| Recording | Source | Licence |
|---|---|---|
| Genuine callers (`genuine_customer`, `genuine_new_number`) and the other five enrolled voiceprints | Real read speech from **LibriSpeech dev-clean** (Panayotov et al., ICASSP 2015, https://www.openslr.org/12), trimmed and level-matched. The transcripts are exactly what the speakers said; the agent's lines were written around their words. | CC BY 4.0 |
| Human impersonators (`human_impersonator`: dev-clean speaker 3752; `human_negotiator`: test-clean speaker 1580), and the voiceprints of ACC-2005 (test-clean 4970) and ACC-1006 (dev-clean 7976) | Real read speech from **LibriSpeech dev-clean and test-clean**, from people who are not the account holder. Three short lines ("Yes, yes.", "Yes, sir.", "Good-bye.") are runs of words from longer recordings: torchaudio's wav2vec2 forced aligner finds the words, and the cut is made in the middle of the pause around them (at least 0.25 s), so no word is clipped. | CC BY 4.0 (aligner model: MIT) |
| Voice clone (`voice_clone`) | **Chatterbox** zero-shot clone of the enrolled account holder's LibriSpeech voice, generated to test the detector. Carries Resemble AI's Perth watermark. | Model: MIT |
| AI bot and AI assistant (`ai_negotiator`, `ai_assistant`) and all agents | **Kokoro-82M** stock voices | Apache-2.0 |

Rebuild with `scripts/build_demo_calls.py`, then add the model scores with `scripts/score_calls.py`.
The LibriSpeech speaker IDs and utterance IDs are listed in `accounts.json` and in each call file.
