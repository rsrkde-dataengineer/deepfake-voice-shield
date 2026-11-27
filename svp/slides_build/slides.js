// SVP opener: slide 1 = "Real or clone?" game (Voice A / Voice B embedded), slide 2 = hand-over to the live site.
// Reads ../game.json (round 1) and ../svp_facts.json, the same data as the site.
//   cd slides_build && SITE_URL="https://script.google.com/.../exec?page=same" node slides.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const OWN = path.resolve(__dirname, "..");
const game = JSON.parse(fs.readFileSync(path.join(OWN, "game.json"), "utf8"));
const facts = JSON.parse(fs.readFileSync(path.join(OWN, "svp_facts.json"), "utf8"));
const SITE_URL = process.env.SITE_URL || "";
const r1 = game.rounds[0];
const clone = facts.team_clone || {}, real = facts.team_real || {}, who = real.person || "a teammate";
const INK = "141414", RED = "E0161E", RED2 = "FF4D4D", WHITE = "FFFFFF", GRAY = "A8A6A0", SOFT = "D6D4CE";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                      // 13.333 x 7.5 in
pres.title = "Deepfake Voice Shield: Real or clone?";
pres.author = "Rajkumar, Pavithra, Pooja, Jegan";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
const W = 13.333, M = 0.7;
const txt = (s, text, o) => s.addText(text, { margin: 0, isTextBox: true, fontFace: "Arial", valign: "top", ...o });

// ---- slide 1: the game ----------------------------------------------------------------------------------
let s = pres.addSlide();
s.background = { color: INK };
txt(s, "REAL OR CLONE?", { x: M, y: 0.55, w: 8, h: 0.35, fontSize: 14, bold: true, color: GRAY, charSpacing: 4 });
txt(s, [{ text: "Same person. Two voices.", options: { breakLine: true } },
        { text: "Which one is really Pavithra?", options: { color: RED2 } }],
    { x: M, y: 1.0, w: W - 2 * M, h: 1.8, fontSize: 44, bold: true, color: WHITE });
const cw = (W - 2 * M - 0.5) / 2;
r1.clips.forEach((c, i) => {
  const x = M + i * (cw + 0.5), y = 3.15, h = 3.0;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h, rectRadius: 0.18, fill: { color: WHITE, transparency: 90 },
    line: { color: "5A5955", width: 1.5 }, objectName: `card-${c.key}` });
  txt(s, `Voice ${c.key}`, { x, y: y + 0.35, w: cw, h: 0.8, fontSize: 40, bold: true, color: WHITE, align: "center" });
  s.addMedia({ type: "audio", path: path.join(OWN, "work", `game_${r1.id}_${c.key}.wav`), x: x + cw / 2 - 0.55, y: y + 1.35,
    w: 1.1, h: 1.1, objectName: `audio-${c.key}` });
  txt(s, `${c.seconds.toFixed(1)} s · click to play`, { x, y: y + 2.5, w: cw, h: 0.35, fontSize: 14, color: GRAY, align: "center" });
});
txt(s, "One is our teammate Pavithra. The other is an AI clone of her voice. Vote: A or B?",
    { x: M, y: 6.5, w: W - 2 * M, h: 0.4, fontSize: 18, color: SOFT });
s.addNotes(`Play Voice A, then Voice B (click each speaker icon). Ask the room: which one is the real Pavithra? ` +
  `Take a quick vote, don't reveal. Then go to the next slide and open the site: it reveals the answer and shows what ` +
  `Shield's two AI models heard. Both voices say: "${r1.sentence}"`);

// ---- slide 2: hand-over to the live site -----------------------------------------------------------------
s = pres.addSlide();
s.background = { color: INK };
txt(s, "LIVE: DEEPFAKE VOICE SHIELD", { x: M, y: 0.55, w: 9, h: 0.35, fontSize: 14, bold: true, color: GRAY, charSpacing: 4 });
txt(s, [{ text: "Your ears decided. ", options: {} }, { text: "Now ask Shield.", options: { color: RED2 } }],
    { x: M, y: 1.0, w: W - 2 * M, h: 0.9, fontSize: 44, bold: true, color: WHITE });
txt(s, "The same two voices, scored live on our site by NVIDIA TitaNet-Large (voiceprint) and Spectra-AASIST (deepfake " +
       `detector), then a full call: a clone of ${who}'s voice trying to move a number to a new SIM.`,
    { x: M, y: 2.05, w: 10.5, h: 0.9, fontSize: 18, color: SOFT });
const steps = [["1", "The reveal", "Which voice was real, and what each AI model heard."],
               ["2", "The clone's call", `${clone.verdict || "Likely voice clone"}: escalated at ${clone.risk || "88"}%.`],
               ["3", `${who}'s real call`, real.action === "PROCEED" ? `Verified: proceeds at ${real.risk}% risk.` :
                 "Her real voice, the same request, checked the same way."]];
const sw = (W - 2 * M - 0.6) / 3;
steps.forEach(([n, h, b], i) => {
  const x = M + i * (sw + 0.3), y = 3.3;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: sw, h: 1.9, rectRadius: 0.14, fill: { color: WHITE, transparency: 90 },
    line: { type: "none" }, objectName: `step-${n}` });
  s.addText(n, { shape: pres.shapes.OVAL, x: x + 0.3, y: y + 0.3, w: 0.55, h: 0.55, fill: { color: RED }, color: WHITE,
    fontSize: 18, bold: true, align: "center", valign: "middle", margin: 0, objectName: `num-${n}` });
  txt(s, h, { x: x + 1.05, y: y + 0.34, w: sw - 1.3, h: 0.5, fontSize: 20, bold: true, color: WHITE, valign: "middle" });
  txt(s, b, { x: x + 0.3, y: y + 1.0, w: sw - 0.6, h: 0.8, fontSize: 15, color: SOFT });
});
s.addText("Open Shield  ▶", { shape: pres.shapes.ROUNDED_RECTANGLE, x: M, y: 5.6, w: 3.6, h: 0.85, rectRadius: 0.15,
  fill: { color: RED }, color: WHITE, fontSize: 24, bold: true, align: "center", valign: "middle", margin: 0,
  objectName: "open-site", ...(SITE_URL ? { hyperlink: { url: SITE_URL, tooltip: "Open the live site" } } : {}) });
txt(s, "Verify the caller once. Protect every transfer.", { x: 4.7, y: 5.78, w: 8, h: 0.5, fontSize: 20, bold: true,
  color: GRAY, valign: "middle" });
txt(s, "Team: Rajkumar · Pavithra · Pooja · Jegan", { x: M, y: 6.85, w: 8, h: 0.3, fontSize: 12, color: GRAY });
s.addNotes((SITE_URL ? "" : "Link the red 'Open Shield' button to the site: select it, Insert > Link, paste the /exec link " +
  "ending in ?page=same. ") + "On the site: 1) the reveal and both models' scores; 2) 'Watch the clone's call' and press " +
  "Play call; 3) Pavithra's real call. Then the Value page: $55k setup, $2,000 a year to run, $2.28M a year back.");

pres.writeFile({ fileName: path.join(OWN, "SVP_opening_slides.pptx") }).then((f) => console.log("wrote", f));
