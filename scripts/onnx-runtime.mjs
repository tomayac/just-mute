/**
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

/**
 * The address and SHA-256 of every ONNX Runtime file Transformers.js may load.
 *
 * Transformers.js fetches the runtime from jsDelivr at run time, for the exact
 * `onnxruntime-web` version it was built against, and it can keep those files
 * in Cross-Origin Storage. What it cannot do is know their hashes in advance:
 * it only learns them for files on the Hugging Face Hub, whose LFS pointers
 * state them. For anything else it downloads the file first and hashes it
 * afterwards, so every site downloads the runtime once even when another site
 * already stored it. The npm package holds the same bytes jsDelivr serves, so
 * the hashes are worked out here at build time and handed to the page, which
 * records them where Transformers.js looks before it fetches.
 *
 * Both pairs are listed: Safari gets the plain build and every other browser
 * gets the asyncify one.
 *
 * @param {string} root The project the build runs in.
 * @returns {Promise<Record<string, string>>} Each file's URL, mapped to its hash.
 */
export async function onnxRuntimeHashes(root) {
  const fromTransformers = createRequire(
    createRequire(join(root, 'package.json')).resolve(
      '@huggingface/transformers',
    ),
  );
  const files = [
    'ort-wasm-simd-threaded.asyncify.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm',
    'ort-wasm-simd-threaded.mjs',
    'ort-wasm-simd-threaded.wasm',
  ];
  const dist = dirname(fromTransformers.resolve(`onnxruntime-web/${files[0]}`));
  const { version } = JSON.parse(
    await readFile(join(dist, '..', 'package.json'), 'utf8'),
  );

  /** @type {Record<string, string>} */
  const hashes = {};
  for (const file of files) {
    const bytes = await readFile(join(dist, file));
    const url = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/${file}`;
    hashes[url] = createHash('sha256').update(bytes).digest('hex');
  }
  return hashes;
}
