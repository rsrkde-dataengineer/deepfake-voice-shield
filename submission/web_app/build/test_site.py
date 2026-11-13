"""Drive the whole site in Chrome: local file mode, then a simulated Apps Script (Code.gs output + google.script mock)."""
import base64, json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

DIR, SERVED, SHOTS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
EXPECT = {"genuine_customer": "3", "genuine_new_number": "8", "voice_clone": "88", "human_impersonator": "95",
          "ai_negotiator": "96", "human_negotiator": "88", "ai_assistant": "89"}
MOCK = """
window.__pushed = [];
window.google = { script: {
  get run() { var s, f; var r = { withSuccessHandler: function (fn) { s = fn; return r; },
    withFailureHandler: function (fn) { f = fn; return r; },
    getAudio: function (n) { setTimeout(function () { window.__B64[n] ? s(window.__B64[n]) : f(new Error('x')); }, 60); } };
    return r; },
  history: { push: function (st, p) { window.__pushed.push(p.page); }, setChangeHandler: function (fn) { window.__change = fn; } },
  url: { getLocation: function (cb) { cb({ parameter: {}, hash: '' }); } } } };
"""

def visible_page(pg):
    return pg.evaluate("[...document.querySelectorAll('.page')].filter(p => !p.hidden).map(p => p.id).join(',')")

def run_local(b):
    errors = []
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.goto((DIR / "index.html").as_uri()); pg.wait_for_timeout(600)
    print("landing:", visible_page(pg), "| active nav:", pg.inner_text(".site nav button.active"),
          "| team section shown:", pg.locator("#page-about h2", has_text="Team").count())
    for name, label in (("about", "About"), ("vision", "Vision"), ("value", "Value"), ("evidence", "Evidence"), ("ears", "Test your ears"), ("demo", "Live demo")):
        pg.click(f".site nav button[data-go='{name}']"); pg.wait_for_timeout(300)
        assert visible_page(pg) == f"page-{name}", visible_page(pg)
        pg.screenshot(path=str(SHOTS / f"desk_{name}.png"), full_page=True)
    print("nav: all six pages switch | hash now:", pg.evaluate("location.hash"))
    # demo outcomes
    pg.wait_for_function("document.getElementById('audio-status').textContent.includes('ready')", timeout=15000)
    bad = []
    for i, (cid, risk) in enumerate(EXPECT.items()):
        pg.click(f".call-btn[data-i='{i}']"); pg.click("#end")
        got = pg.inner_text("#page-demo .risk-n").strip("%")
        if got != risk: bad.append((cid, got, risk))
    print("demo: seven outcomes", "OK" if not bad else bad)
    pg.click(".call-btn[data-i='2']"); pg.wait_for_function("document.getElementById('audio-status').textContent.includes('ready')", timeout=15000)
    pg.click("#play"); time.sleep(3.5); print("demo: after 3.5 s of Play ->", pg.inner_text("#progress")); pg.click("#play")
    # evidence
    pg.click(".site nav button[data-go='evidence']")
    vals = [v.inner_text() for v in pg.locator("#ev-body .kpi .value").all()]
    print("evidence test-clean/phone:", vals)
    pg.click("[data-split='dev-clean']"); print("evidence dev-clean/phone:", [v.inner_text() for v in pg.locator("#ev-body .kpi .value").all()])
    pg.click("[data-split='test-clean']")
    # value calculator
    pg.click(".site nav button[data-go='value']")
    out = lambda: [v.inner_text() for v in pg.locator("#calc-out .kpi .value").all()]
    print("value defaults:", out())
    pg.click("[data-preset='50']"); print("value at $50/h:", out())
    pg.fill("#f-share", "25"); pg.dispatch_event("#f-share", "input"); print("value at $50/h, 25% verified:", out())
    pg.click("#calc-reset"); print("value after reset:", out())
    print("value hero:", pg.inner_text("#page-value .hero h1"), "| kpis:", [v.inner_text() for v in pg.locator("#page-value > .wrap .kpi .value").all()][:6])
    # test your ears
    pg.click(".site nav button[data-go='ears']")
    pg.click("[data-qplay='0']"); time.sleep(1.6)
    print("ears: clip A progress", pg.evaluate("document.getElementById('qbar-0').style.width"))
    answers = ["human", "human", "ai", "human", "human", "human"]   # A wrong on purpose (it's the clone)
    for i, a in enumerate(answers):
        pg.click(f"[data-q='{i}'][data-qa='{a}']")
    print("ears:", pg.inner_text("#q-score"))
    print("ears A reveal:", pg.inner_text("#qr-0").replace("\n", " | ")[:330])
    pg.click("#qr-0 [data-call]"); pg.wait_for_timeout(300)
    print("ears -> demo:", visible_page(pg), "|", pg.inner_text("#meta")[:40])
    # deep link by hash, then phone layout
    pg.goto((DIR / "index.html").as_uri() + "#value"); pg.wait_for_timeout(500); print("hash #value opens:", visible_page(pg))
    pg.set_viewport_size({"width": 390, "height": 844})
    for name in ("about", "ears", "value", "demo"):
        pg.click(f".site nav button[data-go='{name}']"); pg.wait_for_timeout(300)
        over = pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        pg.screenshot(path=str(SHOTS / f"phone_{name}.png"), full_page=True)
        print(f"phone {name}: horizontal overflow {over}px")
    print("local page errors:", errors or "none")
    ctx.close()

def run_appsscript(b):
    errors = []
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    b64 = {f.name: base64.b64encode(f.read_bytes()).decode() for f in DIR.glob("*.mp3")}
    ctx.add_init_script("window.__B64 = " + json.dumps(b64) + ";" + MOCK)
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(SERVED.as_uri()); pg.wait_for_timeout(600)
    print("apps script ?page=demo opens:", visible_page(pg))
    pg.wait_for_function("document.getElementById('audio-status').textContent.includes('ready')", timeout=15000)
    print("apps script demo audio:", pg.inner_text("#audio-status"))
    pg.click(".site nav button[data-go='about']")
    print("apps script team:", [t.inner_text() for t in pg.locator("#page-about .tile h3").all() if t.inner_text() in ("Asha Rao $1", "Ben Okafor", "Chen Li")])
    pg.click(".site nav button[data-go='ears']"); pg.click("[data-qplay='2']"); time.sleep(1.5)
    print("apps script ears clip C progress:", pg.evaluate("document.getElementById('qbar-2').style.width"))
    print("apps script history pushes:", pg.evaluate("window.__pushed"))
    pg.evaluate("window.__change({state: {page: 'vision'}, location: {parameter: {page: 'vision'}}})"); pg.wait_for_timeout(200)
    print("apps script back/forward handler ->", visible_page(pg))
    print("apps script page errors:", errors or "none")
    ctx.close()

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    run_local(b)
    run_appsscript(b)
    b.close()
