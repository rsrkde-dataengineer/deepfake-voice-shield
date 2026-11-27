/**
 * Deepfake Voice Shield: the product site (Live demo, Real or clone?, Test your ears, Evidence, About, Vision, Value),
 * served as an Apps Script web app.
 *
 * The page (index.html) and the audio (.mp3 / .m4a) sit in one Drive folder. This script only serves them; it never
 * writes anything. Deploy as a web app: Execute as "Me", Who has access "Anyone within <your organization>".
 * Open a section directly with ?page=same (or demo, ears, evidence, about, vision, value).
 *
 * Updating: upload the new folder (or file) again. The script always uses the newest copy and skips the Trash.
 */
var FOLDER_NAME = 'Deepfake Voice Shield web';
var PAGES = ['about', 'demo', 'same', 'ears', 'evidence', 'vision', 'value'];

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

/** Called by the page: one audio file from the folder, as base64. Only plain .mp3 / .m4a names are allowed. */
function getAudio(name) {
  if (!/^[a-z0-9_]+\.(mp3|m4a)$/.test(name)) throw new Error('Unknown audio file');
  return Utilities.base64Encode(file_(name).getBlob().getBytes());
}

/** Run this from the editor after any change: it reports the site version and lists the audio files it can serve. */
function checkSetup() {
  var version = file_('index.html').getBlob().getDataAsString('UTF-8').match(/const BUILD = "([^"]*)"/);
  Logger.log('Site served: ' + (version ? 'version ' + version[1] : 'an OLD version (no version stamp): upload the new folder again'));
  var files = newest_(DriveApp.getFoldersByName(FOLDER_NAME)).getFiles(), audio = {};
  while (files.hasNext()) {
    var f = files.next();
    if (!f.isTrashed() && /^[a-z0-9_]+\.(mp3|m4a)$/.test(f.getName())) audio[f.getName()] = Math.round(f.getSize() / 1024);
  }
  Object.keys(audio).sort().forEach(function (n) { Logger.log(n + ': ' + audio[n] + ' KB'); });
  Logger.log(Object.keys(audio).length + ' audio files. Setup OK: deploy it as a web app.');
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
