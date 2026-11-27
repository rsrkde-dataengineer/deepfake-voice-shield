#!/bin/zsh
# Rebuild all SVP material from Pavithra's recordings: calls, model scores, game, site, slides and the package.
#   zsh "/Users/shivashankar/Downloads/Shield own voices/run_all.sh"
set -e
OWN="/Users/shivashankar/Downloads/Shield own voices"
REPO="/Users/shivashankar/Downloads/deepfake-voice-shield"
cd "$REPO"
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "1/6 calls";  .venv-gen/bin/python "$OWN/own_calls.py" 2>&1 | grep -E "recordings:|team_|data_own"
echo "2/6 scores"; VSHIELD_DATA="$OWN/data_own" .venv-ml/bin/python scripts/score_calls.py team_real team_clone 2>/dev/null | grep team_
echo "3/6 game";   .venv-ml/bin/python "$OWN/score_game.py" 2>/dev/null | grep -E "^r[0-9]|game.json"
echo "4/6 site";   VSHIELD_DATA="$OWN/data_own" .venv/bin/python "$OWN/site_build/export.py" "$REPO" "$OWN/site_build/template.html" \
                     "$OWN/site/Deepfake Voice Shield web" --no-audio 2>&1 | grep -E "team_|index.html"
echo "5/6 slides"; cd "$OWN/slides_build"; node slides.js; zip -dq "$OWN/SVP_opening_slides.pptx" '*/' 2>/dev/null || true
mkdir -p "$OWN/for_slides"
cp "$OWN/work/game_r1_A.wav" "$OWN/for_slides/Voice A.wav"; cp "$OWN/work/game_r1_B.wav" "$OWN/for_slides/Voice B.wav"
echo "6/6 package"
cd "$OWN" && rm -f SVP_package.zip && zip -r -X -q SVP_package.zip site SVP_opening_slides.pptx for_slides SVP_README.md -x "*.DS_Store"
ls -la SVP_package.zip | awk '{printf "SVP_package.zip %.1f MB\n", $5/1e6}'
