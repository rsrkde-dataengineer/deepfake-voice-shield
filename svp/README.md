# SVP presentation: "Real or clone?"

The opening of the SVP presentation and the code behind it. **Slide 1** is a game: two voices of the same teammate, one
real and one an AI clone, and the room guesses which is real. **Slide 2** opens the web app on the same voices: what
Shield's models heard, and full calls (her real voice, then her clone). The product itself (`vshield/`, the 7 verified
calls) is unchanged.

## What is in git, and what is not

| In this repo (code) | Not in this repo (private) |
|---|---|
| `web_app/Code.gs`: the Apps Script that serves the site | **`SVP_package.zip`**: the built site with its audio, the slides and the presenter guide |
| `site_build/`: the site template and exporter | Teammates' real recordings and the AI clones made from them |
| `slides_build/`: the two opening slides | Built call data (`data_own/`), working files (`work/`) |
| The Python scripts that build the calls and the game | |

The package holds Pavithra's real voice and clones of it. Her consent covers this presentation, not a public repo, and
a public copy would let anyone clone her voice. **Move `SVP_package.zip` privately:** email it to yourself, use AirDrop
or a USB stick, or put it in your own Google Drive. `.gitignore` blocks the voice files the scripts write.

## Put it live on the office Mac (Apps Script, about 15 minutes)

1. Copy `SVP_package.zip` to the office Mac and unzip it.
2. **Google Drive** (work account): New → Folder upload → `site/Deepfake Voice Shield web`. Turn off Settings → Convert
   uploads. If an older copy is already there, that's fine: the script uses the newest one.
3. **script.google.com:** open your project (or New project). Replace everything in `Code.gs` with `web_app/Code.gs`
   (the same file is in the package at `site/Code.gs`). Save.
4. Run **checkSetup**, and allow access the first time. The log shows today's site version and **19 audio files**.
5. **Deploy:**
   - Already deployed: Manage deployments → pencil → Version: New version → Deploy.
   - First time: New deployment → Web app → Execute as: Me → Who has access: Anyone within your organization.
6. Copy the `/exec` link and add **`?page=same`**: it opens on the game. Other pages: `?page=demo`, `?page=value`.
7. Open `SVP_opening_slides.pptx` in PowerPoint or Keynote. Select the red **Open Shield** button on slide 2, then
   Insert → Link, and paste the link.

**Backup without network or Apps Script:** double-click `index.html` in the unzipped site folder. It runs offline,
audio included; add `#same` to the address to open the game.

The run of show, the answers to the game and what to say are in `SVP_README.md`.

## Rebuild (this Mac only)

The scripts read the teammates' recordings from `~/Downloads/Voice samples` and write voices next to themselves, so
run them from a private copy of this folder outside the repo: `~/Downloads/Shield own voices`. They use the repo's three
venvs (`.venv`, `.venv-ml`, `.venv-gen`).

```bash
zsh ~/Downloads/"Shield own voices"/run_all.sh     # calls -> model scores -> game -> site -> slides -> SVP_package.zip
.venv-ml/bin/python ~/Downloads/"Shield own voices"/test_pavithra_voices.py   # real voice vs her clones, both models
```

| Script | What it does |
|---|---|
| `own_calls.py` | Builds the two calls from the recordings: real voice, Chatterbox clone, Kokoro agent |
| `score_game.py` | Scores the game clips with both models and writes the game data |
| `test_pavithra_voices.py` | Checks a person's real voice against their clones: voiceprint and deepfake detector, clean and phone line |
| `jegan_*.py` | Checks Jegan's Spectra-AASIST + AASIST3 detector against our held-out sets |
