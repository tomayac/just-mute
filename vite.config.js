/**
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { defineConfig } from 'vite';
import { cosPlugin } from 'vite-plugin-cross-origin-storage';
import { onnxRuntimeHashes } from './scripts/onnx-runtime.mjs';

export default defineConfig(async () => ({
  base: './',
  define: {
    // So the page can tell Transformers.js where the ONNX Runtime already is
    // in Cross-Origin Storage. See scripts/onnx-runtime.mjs.
    __ONNX_RUNTIME_HASHES__: JSON.stringify(
      await onnxRuntimeHashes(import.meta.dirname),
    ),
  },
  plugins: [
    // The premium voices' code, built into chunks named by the SHA-256 of
    // their contents. Rummy McRummyface builds the same versions the same
    // way, so a browser with Cross-Origin Storage keeps one copy for both.
    // The loader fetches these before the app starts, which is about two and
    // a half megabytes on every visit; with the store, only on the first.
    cosPlugin({
      packages: ['@huggingface/transformers', 'phonemizer'],
    }),
  ],
}));
