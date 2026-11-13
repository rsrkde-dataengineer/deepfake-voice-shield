/**
 * Deepfake Voice Shield: the product site (live demo, Test your ears, Evidence, About, Vision, Value),
 * served as an Apps Script web app.
 *
 * The page (index.html) and the audio (*.mp3) sit in one Drive folder. This script only serves them; it never
 * writes anything. Deploy as a web app: Execute as "Me", Who has access "Anyone within <your organization>".
 * Open a section directly with ?page=demo (or ears, evidence, about, vision, value).
 *
 * Updating: upload the new folder (or file) again. The script always uses the newest copy and skips the Trash.
 */
var FOLDER_NAME = 'Deepfake Voice Shield web';
var PAGES = ['about', 'demo', 'ears', 'evidence', 'vision', 'value'];
var CALLS = ['genuine_customer', 'genuine_new_number', 'voice_clone', 'human_impersonator', 'ai_negotiator',
             'human_negotiator', 'ai_assistant'];
var CLIPS = ['voice_clone', 'genuine_customer', 'ai_negotiator', 'human_impersonator', 'genuine_new_number',
             'human_negotiator'];

/** Serves the site. */
function doGet(e) {
  try {
    var page = e && e.parameter && PAGES.indexOf(e.parameter.page) >= 0 ? e.parameter.page : null;
    var html = file_('index.html').getBlob().getDataAsString('UTF-8')
        .replace('/*__PAGE__*/null', function () { return JSON.stringify(page); });
    return HtmlService.createHtmlOutput(html)
        .setTitle('Deepfake Voice Shield')
        .addMetaTag('viewport', 'width=device-width, initial-scale=1');
  } catch (err) {
    return HtmlService.createHtmlOutput('<p style="font-family:sans-serif">Setup needed: ' + err.message + '</p>')
        .setTitle('Deepfake Voice Shield');
  }
}

/** Called by the page: one audio file as base64. Only the demo's audio files can be read. */
function getAudio(name) {
  if (audioFiles_().indexOf(name) < 0) throw new Error('Unknown audio file');
  return Utilities.base64Encode(file_(name).getBlob().getBytes());
}

/** Run this from the editor after any change: it checks that every file is in the Drive folder. */
function checkSetup() {
  var version = file_('index.html').getBlob().getDataAsString('UTF-8').match(/const BUILD = "([^"]*)"/);
  Logger.log('Site served: ' + (version ? 'version ' + version[1] + ' (the six-section site)' :
      'the FIRST version (no version stamp): upload the new folder again'));
  ['index.html'].concat(audioFiles_()).forEach(function (name) {
    var f = file_(name);
    Logger.log(name + ': ' + Math.round(f.getSize() / 1024) + ' KB, updated ' + f.getLastUpdated());
  });
  Logger.log('Setup OK: deploy it as a web app.');
}

function audioFiles_() {
  return CALLS.map(function (c) { return c + '.mp3'; }).concat(CLIPS.map(function (c) { return 'ears_' + c + '.mp3'; }));
}

function file_(name) {
  var folder = newest_(DriveApp.getFoldersByName(FOLDER_NAME));
  if (!folder) throw new Error('Drive folder "' + FOLDER_NAME + '" not found. Upload it to My Drive.');
  var file = newest_(folder.getFilesByName(name));
  if (!file) throw new Error('"' + name + '" is missing from the Drive folder "' + FOLDER_NAME + '".');
  return file;
}

/** The most recently updated item that isn't in the Trash (uploading something again makes a second copy). */
function newest_(items) {
  var best = null;
  while (items.hasNext()) {
    var item = items.next();
    if (!item.isTrashed() && (!best || item.getLastUpdated() > best.getLastUpdated())) best = item;
  }
  return best;
}
