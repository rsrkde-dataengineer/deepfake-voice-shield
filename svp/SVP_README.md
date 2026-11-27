# SVP presentation: what's here and how to use it

Pavithra's and Rajkumar's real voices and AI clones of them, used with their consent. **Keep this folder out of the GitHub repo.**

## What's in the package

| Item | What it is |
|---|---|
| `SVP_opening_slides.pptx` | Two slides. **Slide 1:** the game, "Which one is really Pavithra?", with Voice A and Voice B built in (click the speaker icons in PowerPoint or Keynote). **Slide 2:** "Now ask Shield", which opens the site. |
| `for_slides/Voice A.wav`, `Voice B.wav` | The same two clips, for **Google Slides**: upload them to Drive, then on slide 1 use **Insert → Audio**. |
| `site/Deepfake Voice Shield web/` | The full site: everything from before, plus **Real or clone?** (Pavithra) and two full calls in Live Demo (Rajkumar, real and cloned). |
| `site/Code.gs` | The new script. It is required: it adds the new page and serves the new `.m4a` audio. |

## Put the site live (10 minutes, work Google account)

1. **Drive:** **New → Folder upload → `site/Deepfake Voice Shield web`**. Keep Settings → Convert uploads off. If an
   older copy is already in Drive, that's fine: the script uses the newest one.
2. **script.google.com:** open your project (or a new one). Replace all of `Code.gs` with `site/Code.gs` and save.
3. Run **checkSetup**. The first log line shows today's site version, and the count should be **19 audio files**.
4. **Deploy:**
   - First time: **New deployment → Web app**, Execute as **Me**, access **Anyone within your organization**.
   - Already deployed: **Manage deployments → pencil → Version: New version → Deploy**.
5. Copy the `/exec` link. Add **`?page=same`** for the SVP: it opens straight on "Real or clone?". Link slide 2's red
   **Open Shield** button to it (select it, then Insert → Link).

## The 3-minute run of show

1. **Slide 1, the game (45 s).** "Pavithra is on our team. One of these is her, one is an AI clone of her voice." Play
   A, play B, take a vote. Don't reveal.
2. **Slide 2 → Open Shield → Real or clone? (60 s).** Pick the room's answer; the site reveals it. Point at the two AI
   models: the **deepfake detector** and the **voiceprint**.
   - **Round 2:** a different clone fools a different defence.
   - **The summary:** "Different clones beat different defences. Shield never relies on one signal."
3. **Watch Rajkumar's clone call (45 s).** "We cloned one more of us." Press **Play call**: the clone asks to move
   the number. Risk climbs, the liveness phrase fails, and it is **escalated**.
4. **Rajkumar's real call, then Value (30 s).** His real voice gets through. Then the numbers: $55k setup, $2,000 a
   year to run on GCP, $2.28M a year back.

## Say it straight (if asked)

- **What's real:**
  - Pavithra's and Rajkumar's voices.
  - The clones: one from a consumer app, and open-source ones we made from a short voice note.
  - Both AI models and every score they produced.
- **What's simulated:**
  - The agent's voice (text-to-speech).
  - The account "David Miller", the caller IDs, and the timing of the replies.
- **Where we're honest about the limits:** the consumer-app clone **fooled our open-source deepfake detector**. Shield
  stopped it with the voiceprint instead. Newer clones keep improving, which is why Shield layers checks.
- **"What are you doing about the detector?"** Jegan built the next version: two open-source models (Spectra-AASIST
  and Spectra-AASIST3), each scored on its worst 4-second stretch. We held both versions to the same zero false alarms
  on real voices:
  - Today's detector caught **1 of 32** modern AI voices (GPT, Gemini, Grok, app clones). Jegan's caught **18 of 32**.
  - Neither flagged any of 240 held-out real clips.
  - It isn't in the product yet. We're still tuning it on real phone audio, and the second model needs Tech Coach
    approval.

## Rebuild after new recordings

`zsh run_all.sh` regenerates the calls, scores, game, site, slides and `SVP_package.zip`.
