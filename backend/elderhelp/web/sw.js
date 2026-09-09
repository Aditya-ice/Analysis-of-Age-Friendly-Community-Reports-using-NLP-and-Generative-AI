const VERSION='elderhelp-shell-v2.2';
const SHELL=['/','/assets/style.css','/assets/app.mjs','/assets/core.mjs','/assets/storage.mjs'];
self.addEventListener('install',event=>event.waitUntil(caches.open(VERSION).then(cache=>cache.addAll(SHELL))));
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k.startsWith('elderhelp-shell-')&&k!==VERSION).map(k=>caches.delete(k))))));
// Cache only the application shell. API requests, tokens, prompts, and PDFs never enter this cache.
self.addEventListener('fetch',event=>{const url=new URL(event.request.url);if(event.request.method!=='GET'||url.origin!==location.origin||!SHELL.includes(url.pathname))return;event.respondWith(fetch(event.request).catch(()=>caches.match(url.pathname)));});
