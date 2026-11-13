# The Shield website on Apps Script (about 10 minutes)

A product site with six sections, served from your Drive by a short Apps Script:

| Section | What visitors get |
|---|---|
| **About** (opens first) | What Shield is, the problem, the seven signals, the two blind spots, human oversight, the evidence and your team. |
| **Live demo** | The seven calls replayed turn by turn with the real audio, and the engine's decision at every turn. |
| **Test your ears** | Six callers: guess human or AI, then see what the two models heard on the same audio. |
| **Evidence** | The model evaluation: held-out results, score charts, method and limits. |
| **Vision** | Four ideas behind Shield, the roadmap, and what the pilot needs. |
| **Value** | $55k setup cost, $2,000 a year running cost on GCP, $2.28M a year back, the two upsides, and a calculator with every assumption. |

Every figure comes from the Python prototype, so the site matches the deck and the dashboard.

- **Who can open it:** anyone signed in to your company's Google account. It is not public, and viewers install nothing.
- **What it can't do:** run the models on new audio ("Try your voice"). That needs the Python prototype.

## What's in this folder

| Item | What to do with it |
|---|---|
| `Deepfake Voice Shield web/` | Upload this whole folder to your work Google Drive: `index.html` plus 13 `.mp3` files, 3.7 MB. |
| `Code.gs` | Paste it into a new Apps Script project. |
| `appsscript.json` | Optional: limits the script to *read-only* Drive access (step 3). |
| `build/` | How `index.html` was made. You don't need it to deploy. |

**Preview first (optional):** double-click `Deepfake Voice Shield web/index.html`. It opens in your browser with sound
and works offline.

## Step 1: upload the folder to Drive (2 min)

1. In drive.google.com, check **Settings (gear icon) → Convert uploads**. It must be **off**, or Drive turns
   `index.html` into a Google Doc and the site breaks.
2. Choose **New → Folder upload**, then pick `Deepfake Voice Shield web`.
   - Keep the folder name exactly as it is; the script looks for it by name.
   - Leave it in My Drive.
   - Don't share it: the script reads it as you.

## Step 2: create the script (3 min)

1. Go to script.google.com, choose **New project**, and rename it "Deepfake Voice Shield".
2. Open `Code.gs` from this folder (right-click → Open With → TextEdit), then copy everything with Cmd+A, Cmd+C.
3. In the editor, select all the sample code in `Code.gs`, paste over it, and save with Cmd+S.

## Step 3 (optional, 1 min): read-only Drive access

Without this step, Google asks for full Drive access. With it, the script can only read files.

1. In the editor, open **Project Settings** (gear icon) and tick **Show "appsscript.json" manifest file in editor**.
2. Back in the **Editor**, open `appsscript.json` and replace its content with this folder's `appsscript.json`.
3. Save.

## Step 4: check the setup (1 min)

1. In the toolbar, choose the function **checkSetup** and click **Run**.
2. Google asks for permission. Click **Review permissions**, choose your work account, then **Allow**.
3. The log should list 14 files and end with "Setup OK". If not, it names the missing file or folder.

## Step 5: deploy (2 min)

1. Choose **Deploy → New deployment**. Click the gear next to "Select type" and pick **Web app**.
2. Fill in the form:
   - **Description:** v1
   - **Execute as:** Me
   - **Who has access:** Anyone within *your organization*
3. Click **Deploy**, authorise again if asked, and copy the **Web app URL** (it ends in `/exec`).

## Step 6: test it (1 min)

1. Open the URL. The About page opens, with your Team section near the end.
2. Click **Live demo**, pick **3 Voice clone**, and press **Play call**. You should hear the call, and the risk should
   climb to 88% with "Escalate to Fraud".
3. Click **Test your ears** and play Caller A.
4. Send the link to a teammate and check that it opens from **their** account.

## Then share it

- **Links to one section:** add `?page=` to the URL: `…/exec?page=demo` opens the live demo. The same works for `ears`,
  `evidence`, `about`, `vision` and `value`.
- **Deck, slide 4:** click the caption under the video. At its end, type " Try it yourself: interactive replay.", select
  "interactive replay", press Cmd+K and paste the URL ending in `?page=demo`.
- **Written submission:** paste the URL into the "Links" section of the Gem notes, next to the deck and video.

## Updating the site you already deployed (5 minutes)

The URL stays the same, so links you've already shared keep working.

1. **Drive:** upload the new `Deepfake Voice Shield web` folder again (**New → Folder upload**; Convert uploads still
   off). Drive keeps both folders. The script always uses the newest copy of the folder and of each file, and skips
   anything in the Trash.
2. **Script:** in the Apps Script editor, select all of `Code.gs`, paste the new `Code.gs` over it, and save.
3. **Check:** run **checkSetup**. The first log line must say **"Site served: version 2026-11-13 …
   (the six-section site)"**. If it says "the FIRST version", the script is still reading the old files: check that
   step 1 uploaded the folder and step 2 replaced `Code.gs`.
4. **Deploy:** choose **Deploy → Manage deployments**, click the pencil, set **Version: New version**, and click
   **Deploy**.
5. **Test:** open the URL and reload with Cmd+Shift+R. It should open on "Who is really calling?" with **Live demo ·
   Test your ears · Evidence | About · Vision · Value** at the top right (compare with `what_you_should_see.png`). The
   footer shows the same "Site version" as checkSetup, and the About page ends with your Team section.
6. Optional: move the old folder to the Trash.

From now on, if only `index.html` or an audio file changes, steps 1, 3 and 5 are enough; there's no need to deploy
again. Steps 2 and 4 are only for a new `Code.gs`.

## If something goes wrong

| Problem | Fix |
|---|---|
| "Setup needed: Drive folder ... not found" | The folder name differs, or it is in a shared drive. Upload it to My Drive under the exact name. |
| The demo says "Call audio not found", or a Test your ears clip won't play | An `.mp3` is missing from the folder. Run **checkSetup** to see which one. |
| The link still shows the old single page ("INTERACTIVE REPLAY" at the top, no tabs) | Three possible causes. **(1)** checkSetup says "the FIRST version": the new folder isn't uploaded, or the old `Code.gs` is still in the editor. **(2)** The new code was saved but not deployed: Deploy → Manage deployments → pencil → **Version: New version** → Deploy. **(3)** You chose "New deployment": that makes a second, different URL. Use the new URL, or edit the original deployment instead. |
| A teammate gets "You need access" or "Page not found" | Deploy again with **Who has access: Anyone within your organization**, and send the `/exec` URL, not the editor URL. |
| Your organization blocks Apps Script web apps | Use the deck and the demo video instead. They show the same calls and numbers. |
