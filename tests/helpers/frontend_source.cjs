const fs = require('node:fs');
const path = require('node:path');
const base = path.join(__dirname, '../../src/jaka_agent/web');
const markup = fs.readFileSync(path.join(base, 'templates/index.html'), 'utf8');
const styles = fs.readFileSync(path.join(base, 'static/css/app.css'), 'utf8');
const script = fs.readFileSync(path.join(base, 'static/js/app.js'), 'utf8');
module.exports = { markup, styles, script, combined: markup + '\n' + styles + '\n' + script,
  serviceWorker: fs.readFileSync(path.join(base, 'static/service-worker.js'), 'utf8') };
