/**
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

const CACHE_NAME = 'just-mute';

self.addEventListener('install', (event) => {
  // Only pre-cache the manifest; HTML is handled network-first at runtime.
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(['./manifest.json']))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    Promise.all([
      // Remove any cross-origin entries (e.g. HuggingFace model files) that
      // previous SW versions cached here.  Going forward the SW never touches
      // cross-origin requests, so these entries would never be served anyway,
      // but pruning them keeps the cache tidy and prevents stale-file bugs.
      caches.open(CACHE_NAME).then(async (cache) => {
        const keys = await cache.keys();
        return Promise.all(
          keys
            .filter((req) => !req.url.startsWith(self.location.origin))
            .map((req) => cache.delete(req))
        );
      }),
      // Delete other just-mute-* caches from old versioned deployments.
      // Third-party caches (transformers-cache, kokoro-voices, …) are left
      // alone so model files survive SW updates without a re-download.
      caches
        .keys()
        .then((keys) =>
          Promise.all(
            keys
              .filter((k) => k !== CACHE_NAME && k.startsWith('just-mute'))
              .map((k) => caches.delete(k))
          )
        ),
    ]).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  // Never intercept cross-origin requests.  Model files, WASM runtimes, and
  // voice data are fetched directly from their CDN; transformers.js and the
  // worker manage their own caches (transformers-cache, kokoro-voices).
  if (!event.request.url.startsWith(self.location.origin)) return;

  // Navigation requests (HTML): network-first so every new deploy reaches
  // users immediately. Falls back to cache only when offline.
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (event.request.method === 'GET') {
            const clone = response.clone();
            caches
              .open(CACHE_NAME)
              .then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Everything else (JS, CSS, WASM, images): cache-first.
  // Vite content-hashes all asset filenames, so a cached asset is always
  // correct for the HTML that requested it.
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (
          response &&
          response.status === 200 &&
          event.request.method === 'GET'
        ) {
          const clone = response.clone();
          caches
            .open(CACHE_NAME)
            .then((cache) => cache.put(event.request, clone));
        }
        return response;
      });
    })
  );
});
