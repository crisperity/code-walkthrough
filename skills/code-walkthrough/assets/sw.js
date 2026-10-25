const OFFLINE_HTML = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>code-walkthrough</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0f1115;color:#e6e9ef;font:15px/1.5 system-ui,sans-serif}main{max-width:32rem;padding:1.5rem}h1{font-size:1.1rem;margin:0 0 .5rem}p{color:#9aa3b2;margin:.5rem 0}code{background:#1a1d24;color:#7aa2f7;padding:.15rem .4rem;border-radius:4px}</style>
<main><h1>The code-walkthrough server isn't running</h1><p>Start it with <code>python3 &lt;skill&gt;/scripts/cw_mcp.py open</code>, then reload this page.</p></main>`;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.mode !== 'navigate') return;
  e.respondWith(fetch(req).catch(() => new Response(OFFLINE_HTML, {headers: {'Content-Type': 'text/html; charset=utf-8'}})));
});
