/**
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Cross-Origin Storage for the Kokoro files Transformers.js does not cover.
//
// Transformers.js keeps the model in Cross-Origin Storage by itself once
// `env.experimental_useCrossOriginStorage` is on. This adds the ONNX Runtime,
// whose hashes it cannot know before downloading it, and the voices, which are
// fetched here directly. Rummy McRummyface loads the same files, so either app
// finds what the other already stored. Without the API, or when any part of it
// fails, everything falls back to the network.

/* global __ONNX_RUNTIME_HASHES__ */

// Transformers.js's own bucket for each URL's hash, copied because it is not
// exported. The voices use it too, so their hashes are looked up once.
const HASH_CACHE = 'experimental_transformers-hash-cache';

// The API's current method, or its older name in an older extension.
function getHandle() {
  const cos = navigator.crossOriginStorage;
  const method = cos?.getFileHandle ?? cos?.requestFileHandle;
  return method ? method.bind(cos) : null;
}

// Tells Transformers.js the ONNX Runtime's hashes before it goes looking, so
// it reads the runtime from Cross-Origin Storage when another site stored it.
// Call before the first model loads.
export async function shareOnnxRuntime() {
  if (!getHandle()) return;
  try {
    const cache = await caches.open(HASH_CACHE);
    for (const [url, hash] of Object.entries(__ONNX_RUNTIME_HASHES__)) {
      if (!(await cache.match(url))) await cache.put(url, new Response(hash));
    }
  } catch {
    // No Cache Storage: Transformers.js downloads the runtime as before.
  }
}

// A file on the Hugging Face Hub, from Cross-Origin Storage when it is there.
// The hash comes from the file's LFS pointer, and a file fetched from the
// network is stored for every site to read.
export async function fetchShared(url) {
  const get = getHandle();
  const hash = get ? await hubHash(url) : null;
  if (!get || !hash) return fetch(url);

  const descriptor = { algorithm: 'SHA-256', value: hash };
  try {
    const blob = await (await get(descriptor)).getFile();
    return new Response(blob, {
      headers: { 'Content-Length': String(blob.size) },
    });
  } catch {
    // Not stored yet, by this site or any other.
  }

  const response = await fetch(url);
  if (!response.ok) return response;
  const blob = await response.blob();
  try {
    const writable = await (
      await get(descriptor, { create: true, origins: '*' })
    ).createWritable();
    await writable.write(blob);
    await writable.close();
  } catch (err) {
    console.warn('[cos] could not store', url, err);
  }
  return new Response(blob, {
    status: response.status,
    headers: response.headers,
  });
}

// The SHA-256 a Hub file's LFS pointer states, remembered per URL.
async function hubHash(url) {
  if (!url.includes('/resolve/')) return null;
  try {
    const cache = await caches.open(HASH_CACHE);
    const known = await cache.match(url);
    if (known) return known.text();

    const pointer = await fetch(url.replace('/resolve/', '/raw/'));
    const hash = (await pointer.text()).match(
      /^oid sha256:([0-9a-f]{64})$/m,
    )?.[1];
    if (!hash) return null;
    await cache.put(url, new Response(hash));
    return hash;
  } catch {
    return null;
  }
}
