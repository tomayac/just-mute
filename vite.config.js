/**
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { defineConfig } from 'vite';
import { cosPlugin } from 'vite-plugin-cross-origin-storage';

export default defineConfig({
  base: './',
  plugins: [
    // The premium voices' code and the QR code generator, built into chunks
    // named by the SHA-256 of their contents. Rummy McRummyface builds the
    // same versions the same way, so a browser with Cross-Origin Storage keeps
    // one copy for both. The loader fetches these before the app starts, which
    // is about two and a half megabytes on every visit; with the store, only
    // on the first.
    cosPlugin({
      packages: ['@huggingface/transformers', 'phonemizer', 'qrcode-generator'],
    }),
  ],
});
