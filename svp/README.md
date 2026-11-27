# SVP presentation: "Real or clone?"

The opening of the SVP presentation and the code behind it. **Slide 1** is a game: two voices of the same teammate, one
real and one an AI clone, and the room guesses which is real. **Slide 2** opens the web app on the same voices: what
Shield's models heard, and full calls (her real voice, then her clone). The product itself (`vshield/`, the 7 verified
calls) is unchanged.

## What is in git, and what is not

| In this repo | Not in this repo |
|---|---|
| `SVP_package.zip.enc`: the built package (site with its audio, slides, presenter guide), **encrypted** (AES-256) | The passphrase: shared privately, never written to git |
| `unpack.sh`: decrypts and unzips it | Teammates' raw recordings and the AI clones made from them |
| `web_app/Code.gs`: the Apps Script that serves the site | Built call data (`data_own/`), working files (`work/`) |
| `site_build/`, `slides_build/` and the Python scripts that build the calls and the game | |

The package holds Pavithra's real voice and clones of it. Her consent covers this presentation, not a public repo, and
a public copy would let anyone clone her voice. That is why it's in git only encrypted. `.gitignore` blocks the
unencrypted voice files the scripts write.

## Put it live on the office Mac (a new Apps Script project, about 15 minutes)

This is a separate project with its own Drive folder, "Deepfake Voice Shield SVP". The web app submitted to the sprint
(its own project, folder "Deepfake Voice Shield web") is not touched.

1. In the cloned repo: `git pull`, then `zsh svp/unpack.sh` and type the passphrase. The package unpacks to
   `~/Desktop/SVP_package` (site, slides, presenter guide).
2. **Google Drive** (work account): New → Folder upload → `~/Desktop/SVP_package/site/Deepfake Voice Shield SVP`. Keep Settings → Convert
   uploads off.
3. **script.google.com → New project.** Rename it "Deepfake Voice Shield SVP". Replace everything in `Code.gs` with
   `web_app/Code.gs` (the same file is in the package at `site/Code.gs`). Save.
4. Choose **checkSetup** in the function list and click **Run**. Allow access the first time. The log shows today's site
   version and **19 audio files**.
5. **Deploy → New deployment →** gear icon → **Web app**. Execute as: **Me**. Who has access: **Anyone within your
   organization**. Deploy.
6. Copy the `/exec` link and add **`?page=same`**: it opens on the game. Other pages: `?page=demo`, `?page=value`.
7. Open `SVP_opening_slides.pptx` in PowerPoint or Keynote. Select the red **Open Shield** button on slide 2, then
   Insert → Link, and paste the link.

To update later, upload the folder again (the script uses the newest copy). Then Deploy → Manage deployments → pencil →
Version: New version → Deploy. The link stays the same.

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

After a rebuild, encrypt the new package with the same passphrase (it asks twice), then commit it:

```bash
/usr/bin/openssl enc -aes-256-cbc -md sha256 -pbkdf2 -iter 200000 -salt \
  -in ~/Downloads/"Shield own voices"/SVP_package.zip -out svp/SVP_package.zip.enc
```
