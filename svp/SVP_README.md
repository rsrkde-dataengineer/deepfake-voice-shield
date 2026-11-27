# SVP presentation: what's here and how to use it

Pavithra's real voice and AI clones of it, used with her consent. **Keep this package out of the public GitHub repo.**

## What's in the package

| Item | What it is |
|---|---|
| `SVP_opening_slides.pptx` | Two slides. **Slide 1:** the game, "Which one is really Pavithra?", with Voice A and Voice B built in (click the speaker icons in PowerPoint or Keynote). **Slide 2:** "Now ask Shield", which opens the site. |
| `for_slides/Voice A.wav`, `Voice B.wav` | The same two clips, for **Google Slides**: upload them to Drive, then on slide 1 use **Insert → Audio**. |
| `site/Deepfake Voice Shield SVP/` | The site: **Real or clone?**, Live Demo with a **Pavithra's demo** tab (her real call and her clone's call) and the 7 use cases, and the other pages from the submission. |
| `site/Code.gs` | The Apps Script for a **new** project. It reads only the Drive folder "Deepfake Voice Shield SVP", so the web app submitted to the sprint (folder "Deepfake Voice Shield web") stays untouched. |

## Put the site live as a new Apps Script project (15 minutes, work Google account)

1. **Drive:** New → Folder upload → `site/Deepfake Voice Shield SVP`. Keep Settings → Convert uploads off. Don't rename
   or touch the existing "Deepfake Voice Shield web" folder.
2. **script.google.com → New project.** Rename it "Deepfake Voice Shield SVP". Replace everything in `Code.gs` with
   `site/Code.gs` and save.
3. Choose **checkSetup** in the function list and click **Run**. Authorise it the first time (your account → Allow).
   The log shows today's site version and **19 audio files**.
4. **Deploy → New deployment →** gear icon → **Web app**:
   - Description: SVP
   - Execute as: **Me**
   - Who has access: **Anyone within your organization**
   - Click **Deploy** and copy the `/exec` link.
5. Add **`?page=same`** to the link: it opens straight on "Real or clone?". On slide 2, select the red **Open Shield**
   button → Insert → Link → paste.
6. **To update later:** upload the folder again (the script uses the newest copy), then Deploy → Manage deployments →
   pencil → Version: New version → Deploy. The link stays the same.

**Backup without network:** double-click `index.html` in the unzipped site folder. It runs offline, audio included;
add `#same` to the address to open the game.

## The run of show (about 4 minutes)

**Answers:** Round 1 → **Voice B** is the real Pavithra. Round 2 → **Voice A** is the real Pavithra. Both rounds use
the same real recording, so move quickly through round 2 ("now a different clone").

1. **Slide 1, the game (45 s).** "Pavithra is on our team. One of these is her, one is an AI clone of her voice." Play
   A, play B, take a vote. Don't reveal.
2. **Slide 2 → Open Shield → Real or clone? (60 s).** Pick the room's answer; the site reveals it. Point at the two AI
   models: the **deepfake detector** and the **voiceprint**.
   - **Round 2:** a different clone fools a different defence.
   - **The summary:** "Different clones beat different defences. Shield never relies on one signal."
3. **Watch Pavithra's clone call (45 s).** Press **Play call** (or **Jump to the result**). The clone asks to move the
   number. Risk climbs, the clone fails the liveness phrase "silver maple eighteen", and the call is **escalated at 88%**.
   Open the **Transfer handoff card** tab: "the next agent sees this, so the customer isn't asked to verify again."
4. **Pavithra's real call (20 s).** Her real voice makes the same request and Shield asks for one more check (50%):
   "her real voice, recorded on a different phone than her voiceprint, so Shield asks one quick question instead of
   rejecting her." The panel's wording ("not the account holder's voice") is the engine's standard text for a weak
   voiceprint match.
5. **Value (30 s).** $55k to build, $2,000 a year to run on GCP, **$2.28M a year back**, payback in about 9 days.

## Say it straight (if asked)

- **What's real:**
  - Pavithra's voice.
  - The clones: one from a consumer app, and an open-source one we made from her 10-second voice note.
  - Both AI models and every score they produced.
- **What's simulated:**
  - The agent's voice (text-to-speech).
  - The account "Jennifer Miller", the caller IDs, and the timing of the replies.
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
