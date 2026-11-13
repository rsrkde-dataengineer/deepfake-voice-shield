# Submission guide: Deepfake Voice Shield

**Deadline: Friday 13 November, 11:59 PM ET (Saturday 14 November, 10:29 AM IST).** Plan to finish during Friday
office hours.

## What's in this folder

| File | What it is for |
|---|---|
| `Deepfake_Voice_Shield_deck.pptx` | #1 Visual Artifact: 10 slides, with a place for the demo video on slide 4. Turn it into a Google Slides link. |
| `demo/Shield_demo_all_7_calls.mp4` | 7 min 43 s demo of all seven calls: a real dashboard screenshot at every turn, in sync with the call's real audio (a replay, not a live screen capture). Add it to the deck. |
| `demo/calls/1_…mp4` to `7_…mp4` | One clip per call (52–79 s each), for your teammate's video pitch. |
| `Gem_notes_paste_into_Google_Doc.md` | #3 Written Submission: our five paragraphs (each under 1,200 characters), plus every fact the Gem checks. |
| `web_app/` | Optional: the Shield website as a Google Apps Script web app that anyone in the organization can open: About, Vision and Value, plus the live demo replay, Test your ears and the evidence. About 10 minutes to set up; see `web_app/SETUP.md`. Link it from the deck. |
| `SUBMISSION_GUIDE.md` | This file. |

This folder is git-ignored: it holds internal material and the repo is public. It won't arrive with `git pull`, so
copy it to the office Mac yourself (email, USB or AirDrop).

## First, two decisions (5 minutes)

1. **Who did what.** The names (Rajkumar, Pavithra, Pooja, Jegan) are already on deck slides 1 and 10, on the website
   and in the Gem notes. In the Gem notes, replace each `[role]` with one line on what that person did, in "Team and
   roles" and in the Pitch Quality paragraph. The judges' checklist asks for every member's role.
2. **Tool approval.** Write down the Tech Coach confirmation status for NVIDIA NeMo and the Hugging Face models (sprint
   rule 2) in the Gem notes.

## #1 Visual Artifact: get a Google Slides link

1. Work Google Drive: **New → File upload →** `Deepfake_Voice_Shield_deck.pptx`.
2. Open it with Google Slides, then **File → Save as Google Slides**.
3. The team names are already on slides 1 and 10.
4. Add the demo video to slide 4; don't add a new slide. Upload `demo/Shield_demo_all_7_calls.mp4` to the same Drive
   folder. On slide 4, choose **Insert → Video → Google Drive** and pick the video. Place it over the dark title
   picture on the left, either by dragging it there or with **Format options**:
   - **Size & Rotation:** 7 × 3.94 in (17.78 × 10 cm).
   - **Position:** X 0.6 in, Y 1.8 in (1.52 cm, 4.57 cm).
5. **Share** the deck **and** the video: **General access → your organization → Viewer**, then copy the deck's link.
   The video needs its own sharing; otherwise judges see an error on that slide.
6. Ask a teammate to open the link to check that it works.

The guidance says the artifact is mainly for the Tiger Team and business leaders, and judges use it only to break a
tie. So the key points must also be in the video and the written submission. They are.

**Optional, about 10 minutes: the Shield website, which judges can try themselves.** It has About, Vision and Value
pages, the live demo replay, a "Test your ears" quiz and the evidence. `web_app/SETUP.md` deploys it as an Apps Script
web app; Apps Script is one of the artifact types the sprint lists. The steps:
1. Upload one folder to Drive.
2. Paste `Code.gs` into a new Apps Script project.
3. Deploy with "Anyone within your organization".

Keep the deck as the single artifact link, and link the site from slide 4 with `?page=demo` at the end of the URL, so
it opens on the live demo (the step is in `SETUP.md`).

## #2 Video pitch (your teammate): keep the "Truth Triangle" aligned

Send them `demo/calls/*.mp4` and the card below. The clips have the call audio but no narration. Every number they
say should match the deck and the written submission.

- **Opening:** AI bots, voice clones and human impersonators call Care. Agents can't hear the difference, and every
  transfer re-verifies the customer (60–90 s each, about 76,000 agent hours a year).
- **Moment 1** (`3_voice_clone.mp4`): the voiceprint matches (0.82), but the anti-spoofing model hears 98% synthetic
  speech. The clone then fails the liveness phrase and is escalated at 88%.
- **Moment 2** (`4_human_impersonator.mp4`): a real human passes a deepfake detector, but the voiceprint doesn't match.
  The caller fails an account-history question, then the real owner taps "Not me": 95%.
- **Moment 3** (`5_ai_negotiator.mp4`): flagged "96% likely AI" after one sentence.
- **Proof:** 82/82 fake voices caught, 0/120 real voices flagged, and 0 of 2,280 impersonation attempts accepted. This
  was measured on a held-out test through a simulated phone line.
- **Money:** $55k setup cost (one time, internal hours) and a $2,000 a year running cost on GCP; $2.28M a year back at the brief's
  rate, paid back in about 9 days. Upside: $3.8M at a
  fully loaded $50/hour, plus about $12.6M a year in fraud avoided (illustrative).
- **Human oversight:** "Shield advises, the agent decides. It never hangs up on a customer."
- **Close:** "Our ask: a 6-week pilot on one transfer queue."
- Every team member should speak or be named (sprint rule). `docs/DEMO_SCRIPT.md` in the repo has a timed 3-minute
  script.

## #3 Written submission with the Gem

1. Create a Google Doc, for example "Deepfake Voice Shield – submission notes", and paste in the whole
   `Gem_notes_paste_into_Google_Doc.md`.
   - Plain paste is fine for the Gem.
   - For tidier formatting, turn on **Tools → Preferences → Enable Markdown**, then use **right-click → Paste from
     Markdown**.
2. Fill in every `[FILL IN]`, `[role]` and `[link]`.
3. Open **Team Project Submission Editor – Innovation Sprint 26** and choose **Thinking** (or **Pro**).
4. Type `Start`, then type `@Google Docs`, pick your notes doc, and press Enter.
   - If you can add a line, add: *"Part A is our preferred wording: keep every number exactly as written and each
     paragraph under 1,200 characters. Part B has the facts."*
5. If the Gem stops to ask about human oversight, ROI assumptions or roles, the answers are in Part B.
6. Before you paste anything into the form:
   - **Numbers:** check that the key numbers survived: 82/82 · 0/120 · 0/2,280 · $55,000 · $2,000 a year on GCP · $2.28M · about 9 days ·
     96% / 88% / 95%.
   - **Length:** check that each paragraph is under 1,200 characters. Select it, then **Tools → Word count** shows the
     characters.
   - **Fallback:** if the Gem's version is weaker, Part A is ready to paste as is.
7. Paste the five paragraphs into **Team Project Final Submission – AI Unlocked Innovation Sprint Form**, together
   with the deck link and the video link.

## Deep Dive: how our ROI follows their method

**Two different costs.** The setup cost (one time, internal hours) and the running cost (GCP, per year).

**Setup cost (one time, internal hours): "Time is Currency".** The production version is a **Large** effort (100+ hours on their
guide). It connects to enterprise platforms (contact-center audio, CTI, account systems) and needs a security and
privacy review for voice biometrics. A pilot alone would be Medium.

| Work package (production version) | Dev/IT h ($100) | SME h ($75) | Cost |
|---|---|---|---|
| Call-audio fork and streaming ASR (NVIDIA Riva/Parakeet) | 120 | 0 | $12,000 |
| Model serving on GPU (Triton), latency and load tests | 80 | 0 | $8,000 |
| Connectors: CTI/ANI, STIR/SHAKEN, account risk, billing, tickets | 80 | 0 | $8,000 |
| Agent panel, handoff card, fraud referral into the case system | 100 | 0 | $10,000 |
| Consented voiceprint enrollment and retention | 40 | 30 | $6,250 |
| Security, privacy and biometric-data review | 20 | 30 | $4,250 |
| Calibration on consented call audio; fraud playbook | 20 | 30 | $4,250 |
| Pilot operations: agent training, Looker baseline, go-live | 0 | 30 | $2,250 |
| **Total (Large): setup cost** | **460** | **120** | **$55,000** |

**Running cost (GCP, per year).** About **$2,000 a year** for the open-source models (our estimate). No per-call
license fees. Internal hours aren't in it; they're the setup cost above.

**Savings.**
- **Baseline.** The brief's 76,000 hours a year implies about 3.65M transferred calls at 75 s each (the midpoint of
  60–90 s). Confirm both in the Looker chip_cs Handoffs dashboard.
- **Annual savings.** 3.65M × 75 s = 76,042 agent hours a year. At the brief's $30/hour loaded agent rate that is
  **$2.28M a year**, or about 42 full-time agents.
- **Why $30/hour.** We deliberately use the brief's agent rate, not the $75 SME rate. The $75 rate would more than
  double the savings, and judges reward conservative assumptions.
- **Upside rate.** At a fully loaded $50/hour (wages plus benefits, facilities and supervision) the same hours are
  worth **$3.80M a year**, with payback in about 5 days. We show it as upside; the headline stays on the brief's rate.

**Payback and robustness.** Payback is the build cost divided by daily net savings (after the $2,000 a year GCP cost).
At full volume that is about 9 days, and even at a quarter of the volume about 5 weeks. Year one: $57,000 in ($55,000
build + $2,000 GCP), $2.28M back, about 40×.

| Share of transfers Shield verifies | Savings a year | Payback |
|---|---|---|
| 100% | $2.28M | 9 days |
| 75% | $1.71M | 12 days |
| 50% | $1.14M | 18 days |
| 25% | $0.57M | 35 days |

**Not counted (upside).** Fraud avoided. Using the illustrative mock inputs on the app's Business impact page (10,000
bot or clone calls a month × $150 average loss, catch rate raised from 25% to 95%), it is another **$12.6M a year**.
It is a what-if, labelled as such and kept out of the payback; the pilot measures the real figure with the fraud team.

**Their "great ROI statement" format, filled in:** "Our setup costs are estimated at a Large effort (approx. 580 hours of internal time / $55,000), assuming we need to connect our prototype to the live contact-center audio stream, CTI and account systems; running it costs about $2,000 a year of GCP infrastructure. Once deployed, we project saving about 6,340 agent hours per month. This assumes about 304,000 transferred calls a month, reducing re-verification time from 75 seconds to 0 for callers Shield has verified."

## Their final checklist, mapped

| Checklist item | Where it is covered |
|---|---|
| Visual Artifact complete, permissions open | Steps under #1 (deck and video both shared with the organization) |
| Video Pitch complete, permissions open | Your teammate; Drive sharing set to the organization |
| Used the Gem: 5 short paragraphs, max 1,200 characters | Steps under #3; Part A is already under the limit |
| Human oversight highlighted | Technical Architecture paragraph; deck slide 8; Part B "Where humans stay in control" |
| Baseline ROI assumptions stated | ROI paragraph; deck slide 9 (assumptions in its footnote); this Deep Dive |
| Each member's role outlined | Pitch Quality paragraph and "Team and roles" (fill in each `[role]`); names on deck slides 1 and 10 and the website |

## If a judge asks

- **"Is this production accuracy?"** No. It is a held-out benchmark on open read speech through a simulated phone
  line, and we say so on the slides. The pilot re-measures everything on consented real calls.
- **"Won't it annoy real customers?"** None of the 120 real voices was flagged. The worst case is one 10-second
  question, and Shield never blocks anyone.
- **"Why Large, not Medium?"** Production needs contact-center audio, CTI and account integrations, plus a biometric
  privacy review. A pilot on its own would be Medium.
- **"Only $2,000 a year to run?"** That's the GCP infrastructure for two small open-source models, with no per-call
  license fees. Internal hours are costed separately: the $55,000 setup cost.
- **"What about better deepfakes next year?"** The detector is one of seven signals. The liveness phrase and the
  identity check don't depend on how good the voice sounds, and confirmed attacks retrain the model.
- **"Where do 10,000 bot calls a month come from?"** It's a what-if, labelled illustrative and kept out of the
  payback. The pilot replaces it with the fraud team's real counts. The base case doesn't depend on it.
- **"Why $30/hour, not a fully loaded rate?"** It's the brief's own rate. At a fully loaded $50/hour the same hours
  are worth $3.8M a year.
- More answers are in `docs/DEMO_SCRIPT.md`, under "Likely questions".
