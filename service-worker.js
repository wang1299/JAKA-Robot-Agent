const CACHE_NAME = 'jaka-vision-shell-v9-structured-navigation';
const SHELL_ASSETS = ['/', '/manifest.webmanifest', '/pwa-icon.svg'];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(SHELL_ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(key => key.startsWith('jaka-vision-shell-') && key !== CACHE_NAME).map(key => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

function isLiveRobotResource(url) {
  return url.pathname.startsWith('/api/')
    || url.pathname.startsWith('/captures/')
    || url.pathname.startsWith('/references/')
    || url.pathname.startsWith('/mapping/')
    || url.pathname.startsWith('/slam-map/');
}

self.addEventListener('fetch', event => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  const isStaticAsset = SHELL_ASSETS.includes(url.pathname) || url.pathname.startsWith('/map-icons/');
  if (isLiveRobotResource(url) || (!isStaticAsset && request.mode !== 'navigate')) {
    event.respondWith(fetch(request, { cache: 'no-store' }));
    return;
  }

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then(response => {
          if (response.ok) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put('/', copy));
          }
          return response;
        })
        .catch(() => caches.match('/'))
    );
    return;
  }

  event.respondWith(
    caches.match(request).then(cached => cached || fetch(request).then(response => {
      if (response.ok) {
        const copy = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(request, copy));
      }
      return response;
    }))
  );
});
