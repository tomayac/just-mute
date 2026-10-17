/**
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  StyleTextToSpeech2Model,
  AutoTokenizer,
  Tensor,
  env,
} from '@huggingface/transformers';
import { phonemize } from 'phonemizer';
import { fetchShared, shareOnnxRuntime } from './cos.js';
import phonemeCache from './phoneme-cache.json';

// Runs on the main thread. Transformers.js and the phonemizer come from
// Cross-Origin Storage through the import map the COS loader writes into the
// page, which a worker cannot see. On the GPU the work happens off the main
// thread anyway, and on the CPU the ONNX Runtime's proxy runs inference in a
// worker of its own (see `loadModel()`).

const MODEL_ID = 'onnx-community/Kokoro-82M-v1.0-ONNX';
// The model and the ONNX Runtime go in Cross-Origin Storage, shared with every
// site that loads the same files. Rummy McRummyface loads exactly these.
env.experimental_useCrossOriginStorage = true;

// Maps the first letter of a Kokoro voice ID to an eSpeak language code.
// eSpeak processes accented characters only when given the correct language —
// passing French text with "en-us" causes a WASM abort on characters like é/à/ç.
const PHONEME_LANG = {
  a: 'en-us', // af_* / am_* — American English
  b: 'en-gb', // bf_* / bm_* — British English
  e: 'es', // ef_* / em_* — Spanish
  f: 'fr', // ff_*        — French
  h: 'hi', // hf_* / hm_* — Hindi
  i: 'it', // if_* / im_* — Italian
  j: 'ja', // jf_* / jm_* — Japanese
  p: 'pt-br', // pf_* / pm_* — Brazilian Portuguese
  z: 'cmn', // zf_* / zm_* — Mandarin Chinese
};

let model = null;
let tokenizer = null;
let loading = null;
const voiceCache = new Map();
const audioCache = new Map();

let reportProgress = () => {};

// WebGPU when the browser has an adapter for it, the CPU otherwise.
async function pickDevice() {
  try {
    if (await navigator.gpu?.requestAdapter()) return 'webgpu';
  } catch {
    // No usable GPU.
  }
  return 'wasm';
}

function ensureModel() {
  loading ??= loadModel().catch((err) => {
    loading = null;
    throw err;
  });
  return loading;
}

async function loadModel() {
  const device = await pickDevice();
  // On the CPU, a phrase takes seconds to synthesize, which would freeze the
  // page for as long. The proxy moves that work into the ONNX Runtime's own
  // worker, and measured faster too: 2.7 seconds for a phrase that took 5.8 on
  // the main thread. It does not apply to WebGPU.
  if (device === 'wasm') env.backends.onnx.wasm.proxy = true;
  await shareOnnxRuntime(env.backends.onnx.wasm.wasmPaths);
  // Track aggregate download progress across all model files.
  const fileProgress = new Map();
  const onProgress = ({ status, file, progress }) => {
    if (!file) return;
    if (status === 'progress') fileProgress.set(file, progress ?? 0);
    else if (status === 'done') fileProgress.set(file, 100);
    if (fileProgress.size === 0) return;
    const avg =
      [...fileProgress.values()].reduce((a, b) => a + b, 0) / fileProgress.size;
    reportProgress('model', Math.round(avg));
  };
  [model, tokenizer] = await Promise.all([
    StyleTextToSpeech2Model.from_pretrained(MODEL_ID, {
      // Four-bit weights, the same file Rummy McRummyface loads, so the two
      // share one 305 MB download through Cross-Origin Storage. It is the
      // fastest and cleanest build on the GPU and runs on the CPU as well.
      dtype: 'q4',
      device,
      progress_callback: onProgress,
    }),
    AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: onProgress }),
  ]);
}

async function loadVoice(voiceId) {
  if (voiceCache.has(voiceId)) return voiceCache.get(voiceId);
  const url = `https://huggingface.co/${MODEL_ID}/resolve/main/voices/${voiceId}.bin`;
  let buffer;
  try {
    const cache = await caches.open('kokoro-voices');
    const hit = await cache.match(url);
    if (hit) {
      buffer = await hit.arrayBuffer();
    } else {
      const res = await fetchShared(url);
      if (!res.ok)
        throw new Error(`HTTP ${res.status} fetching ${voiceId}.bin`);
      buffer = await res.arrayBuffer();
      await cache.put(
        url,
        new Response(buffer.slice(0), {
          headers: { 'content-type': 'application/octet-stream' },
        })
      );
    }
  } catch {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${voiceId}.bin`);
    buffer = await res.arrayBuffer();
  }
  const data = new Float32Array(buffer);
  voiceCache.set(voiceId, data);
  return data;
}

async function toPhonemes(text, voiceId) {
  // Split text by punctuation and process each clause separately.
  // Punctuation (.?!,;:¿¡) is passed through to the tokenizer as-is so the
  // Kokoro model can apply correct prosody: sentence pauses, rising intonation, etc.
  const PUNCT_RE = /([.!?,;:\u00bf\u00a1]+)/;
  const parts = text.split(PUNCT_RE);
  const phoneParts = [];

  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      // Punctuation — keep for prosody
      const p = parts[i].trim();
      if (p) phoneParts.push(p);
      continue;
    }
    const seg = parts[i].trim();
    if (!seg) continue;

    // Check pre-computed cache (covers non-English default clauses)
    const cached = phonemeCache[voiceId[0]]?.[seg];
    if (cached) {
      phoneParts.push(cached);
      continue;
    }

    // Phonemize via eSpeak (English voices, or uncached custom text)
    const lang = PHONEME_LANG[voiceId[0]] ?? 'en-us';
    try {
      phoneParts.push((await phonemize(seg, lang)).join(' '));
    } catch {
      phoneParts.push(
        (
          await phonemize(
            seg.normalize('NFD').replace(/[\u0300-\u036f]/g, ''),
            'en-us'
          )
        ).join(' ')
      );
    }
  }

  // Join: remove spaces before punctuation, ensure single space after
  return phoneParts
    .join(' ')
    .replace(/\s+([.!?,;:\u00bf\u00a1])/g, '$1')
    .trim();
}

async function generate(text, voiceId) {
  reportProgress('model', null);
  await ensureModel();

  reportProgress('voice', null);
  const phonemes = await toPhonemes(text, voiceId);
  if (!phonemes.trim())
    throw new Error(`Phonemization returned empty output for "${text}"`);

  const { input_ids } = tokenizer(phonemes, { truncation: true });
  const seqLen = input_ids.dims.at(-1);
  if (!seqLen)
    throw new Error(
      `Tokenizer returned empty sequence for phonemes "${phonemes}"`
    );

  const voiceData = await loadVoice(voiceId);
  const offset = 256 * Math.min(Math.max(seqLen - 2, 0), 509);
  const raw = voiceData.slice(offset, offset + 256);

  // Pad to exactly 256 in case the voice file is shorter than expected.
  const style =
    raw.length === 256 ? raw : Object.assign(new Float32Array(256), raw);

  reportProgress('generating', null);
  const { waveform } = await model({
    input_ids,
    style: new Tensor('float32', style, [1, 256]),
    speed: new Tensor('float32', [1.0], [1]),
  });
  return { audio: waveform.data, sampling_rate: 24000 };
}

// Synthesizes `text` in `voice`, reporting each phase through `onProgress`.
export async function generateWithKokoro(text, voice, onProgress) {
  reportProgress = onProgress ?? (() => {});
  const key = `${voice}::${text}`;
  let entry = audioCache.get(key);
  if (!entry) {
    entry = await generate(text, voice);
    audioCache.set(key, entry);
  }
  // A copy, so the cached samples survive whatever the caller does with them.
  return {
    audio: new Float32Array(entry.audio),
    sampling_rate: entry.sampling_rate,
  };
}
