# Demo data: where every voice comes from

No real customer data is used. Account names, numbers and call scripts are fictional (US numbers use 555;
the UK number uses Ofcom's drama range).

| Recording | Source | Licence |
|---|---|---|
| Genuine callers (`genuine_customer`, `genuine_new_number`) and every enrolled voiceprint | Real read speech from **LibriSpeech dev-clean** (Panayotov et al., ICASSP 2015, https://www.openslr.org/12), trimmed and level-matched. The transcripts are exactly what the speakers said; the agent's lines were written around their words. | CC BY 4.0 |
| Voice clone (`voice_clone`) | **Chatterbox** zero-shot clone of the enrolled account holder's LibriSpeech voice, generated to test the detector. Carries Resemble AI's Perth watermark. | Model: MIT |
| AI bot and AI assistant (`ai_negotiator`, `ai_assistant`) and all agents | **Kokoro-82M** stock voices | Apache-2.0 |

Rebuild with `scripts/build_demo_calls.py`, then add the model scores with `scripts/score_calls.py`.
The LibriSpeech speaker IDs and utterance IDs are listed in `accounts.json` and in each call file.
