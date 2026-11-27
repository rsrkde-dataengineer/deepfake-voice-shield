// Material icons as inline SVG strings (react-icons/md), so the web page needs no font or network: node icons.js names...
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const md = require("react-icons/md");
const out = {}, missing = [];
for (const name of process.argv.slice(2)) {
  const comp = "Md" + name.split("_").map((s) => s[0].toUpperCase() + s.slice(1)).join("");
  if (!md[comp]) { missing.push(name); continue; }
  out[name] = ReactDOMServer.renderToStaticMarkup(React.createElement(md[comp], { size: "1em" }))
    .replace(/ (height|width|style)="[^"]*"/g, "");
}
if (missing.length) { console.error("missing icons:", missing.join(", ")); process.exit(1); }
process.stdout.write(JSON.stringify(out));
