#!/bin/zsh
# Decrypt the SVP package and unzip it. It asks for the passphrase, which is shared privately and never stored in git.
#   zsh svp/unpack.sh                 -> ~/Desktop/SVP_package
#   zsh svp/unpack.sh /some/folder    -> /some/folder
set -e
HERE="${0:A:h}"
OUT="${1:-$HOME/Desktop/SVP_package}"
mkdir -p "$OUT"
/usr/bin/openssl enc -d -aes-256-cbc -md sha256 -pbkdf2 -iter 200000 -in "$HERE/SVP_package.zip.enc" -out "$OUT/SVP_package.zip"
cd "$OUT" && unzip -oq SVP_package.zip && rm SVP_package.zip
echo "Unpacked to $OUT:"
ls "$OUT"
echo "Next: upload \"$OUT/site/Deepfake Voice Shield SVP\" to Google Drive (see svp/README.md)."
