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

import { NPMSHA256HashFetcher } from 'npm-sha256-hash-fetcher';

// Transformers.js's own bucket for each URL's hash, copied because it is not
// exported. The voices use it too, so their hashes are looked up once.
const HASH_CACHE = 'experimental_transformers-hash-cache';

// The API's current method, or its older name in an older extension.
function getHandle() {
  const cos = navigator.crossOriginStorage;
  const method = cos?.getFileHandle ?? cos?.requestFileHandle;
  return method ? method.bind(cos) : null;
}

const JSDELIVR_NPM = 'https://cdn.jsdelivr.net/npm/';

// Tells Transformers.js the ONNX Runtime's hashes before it goes looking, so
// it reads the runtime from Cross-Origin Storage when another site stored it.
// It only knows hashes up front for Hugging Face files; jsDelivr publishes the
// hash of every file it serves, so one request for the package's file list
// answers for all of them. Takes `env.backends.onnx.wasm.wasmPaths` as
// Transformers.js set it on import; call before the first model loads.
export async function shareOnnxRuntime(paths) {
  if (!getHandle() || !paths || typeof paths !== 'object') return;
  const urls = Object.values(paths).filter(
    (url) => typeof url === 'string' && url.startsWith(JSDELIVR_NPM),
  );
  try {
    const cache = await caches.open(HASH_CACHE);
    const missing = [];
    for (const url of urls) if (!(await cache.match(url))) missing.push(url);
    if (missing.length === 0) return;

    // `onnxruntime-web@<version>`, the same package for every file.
    const [pkg] = missing[0].slice(JSDELIVR_NPM.length).split('/dist/');
    const [result] = await new NPMSHA256HashFetcher().getHexHashForResource(
      pkg,
    );
    if (result.status !== 'fulfilled' || !result.value.files) return;
    for (const url of missing) {
      const path = url.slice(JSDELIVR_NPM.length + pkg.length);
      const file = result.value.files.find((f) => f.path === path);
      if (file) await cache.put(url, new Response(file.hexHash));
    }
  } catch {
    // Offline, or no Cache Storage: Transformers.js downloads the runtime as
    // before.
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
