/**
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Generates IPA phonemes for all non-English Kokoro default phrases using the
 * full eSpeak NG WASM build (@echogarden/espeak-ng-emscripten), which supports
 * French, Spanish, Italian, Portuguese, and Hindi — unlike the phonemizer npm
 * package which is English-only.
 *
 * Output: src/phoneme-cache.json  (imported by the Kokoro worker at build time)
 *
 * Run: node scripts/precompute-phonemes.mjs
 */

import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { writeFileSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const dataPath = join(root, 'node_modules/@echogarden/espeak-ng-emscripten');

import espeakInit from '@echogarden/espeak-ng-emscripten/espeak-ng.js';
const espeak = await espeakInit({ locateFile: (f) => join(dataPath, f) });
const worker = new espeak.eSpeakNGWorker();

// Kokoro voice prefix → eSpeak language code (only languages eSpeak handles well)
const KOKORO_TO_ESPEAK = {
  e: 'es', // ef_* / em_* — Spanish
  f: 'fr', // ff_*        — French
  h: 'hi', // hf_* / hm_* — Hindi
  i: 'it', // if_* / im_* — Italian
  p: 'pt-br', // pf_* / pm_* — Brazilian Portuguese
};

const TRANSPORT_PREP = {
  es: {
    train: 'en el tren',
    bus: 'en el autobús',
    tram: 'en el tranvía',
    plane: 'en el avión',
    subway: 'en el metro',
    ferry: 'en el ferry',
    cablecar: 'en el teleférico',
    car: 'en el coche',
  },
  fr: {
    train: 'dans le train',
    bus: 'dans le bus',
    tram: 'dans le tram',
    plane: "dans l'avion",
    subway: 'dans le métro',
    ferry: 'sur le ferry',
    cablecar: 'dans le téléphérique',
    car: 'dans la voiture',
  },
  hi: {
    train: 'ट्रेन में',
    bus: 'बस में',
    tram: 'ट्राम में',
    plane: 'हवाई जहाज़ में',
    subway: 'मेट्रो में',
    ferry: 'फ़ेरी में',
    cablecar: 'केबल कार में',
    car: 'कार में',
  },
  it: {
    train: 'sul treno',
    bus: "sull'autobus",
    tram: 'sul tram',
    plane: "sull'aereo",
    subway: 'sulla metro',
    ferry: 'sul traghetto',
    cablecar: 'in funivia',
    car: 'in macchina',
  },
  pt: {
    train: 'no comboio',
    bus: 'no autocarro',
    tram: 'no elétrico',
    plane: 'no avião',
    subway: 'no metro',
    ferry: 'no ferry',
    cablecar: 'no teleférico',
    car: 'no carro',
  },
};

const PHRASE_TEMPLATES = {
  es: (p) =>
    `Nadie ${p} quiere escuchar lo que suena en tu teléfono. ¿Puedes silenciarlo?`,
  fr: (p) =>
    `Personne ${p} ne veut écouter ce qui passe sur ton téléphone. Tu peux juste mettre le son sur muet ?`,
  hi: (p) =>
    `${p} कोई भी आपके फ़ोन की आवाज़ नहीं सुनना चाहता। क्या आप इसे साइलेंट मोड पर रख सकते हैं?`,
  it: (p) =>
    `Nessuno ${p} vuole ascoltare quello che suona sul tuo telefono. Puoi metterlo in silenzioso?`,
  pt: (p) =>
    `Ninguém ${p} quer ouvir o que está a tocar no seu telemóvel. Pode silenciá-lo?`,
};

const TRANSPORTS = [
  'train',
  'subway',
  'bus',
  'tram',
  'car',
  'ferry',
  'plane',
  'cablecar',
];

function cleanIPA(raw) {
  return (raw || '')
    .replace(/\([a-z-]+\)/g, '') // strip eSpeak language-switch markers: (en), (fr), (ja)…
    .replace(/_/g, '') // remove per-phoneme separators
    .replace(/-(?=\s|$)/gm, '') // remove trailing hyphens (liaison markers)
    .replace(/\n/g, ' ') // flatten newlines
    .replace(/\s+/g, ' ') // normalise whitespace
    .trim();
}

// Cache keyed by text SEGMENT (without punctuation) so the worker can split
// a phrase by punctuation, look up each clause, then re-interleave the
// punctuation marks for correct prosody (pauses, rising intonation, etc.).
const cache = {}; // { voicePrefix: { segmentText: ipaString } }

const PREFIX_TO_LANG = { e: 'es', f: 'fr', h: 'hi', i: 'it', p: 'pt' };
const PUNCT_RE = /([.!?,;:¿¡]+)/;

for (const [voicePrefix, espeakLang] of Object.entries(KOKORO_TO_ESPEAK)) {
  const langId = PREFIX_TO_LANG[voicePrefix];
  if (!langId) continue;

  worker.set_voice(espeakLang);
  cache[voicePrefix] = {};

  for (const transport of TRANSPORTS) {
    const phrase = PHRASE_TEMPLATES[langId](TRANSPORT_PREP[langId][transport]);
    // Split phrase into alternating [text, punct, text, punct, ...] parts
    const parts = phrase.split(PUNCT_RE);
    for (let i = 0; i < parts.length; i += 2) {
      const seg = parts[i].trim();
      if (!seg || cache[voicePrefix][seg]) continue; // empty or already stored
      const result = worker.synthesize_ipa(seg);
      const ipa = cleanIPA(result.ipa);
      if (!ipa) {
        console.warn(`  Warning: empty IPA for "${seg}"`);
        continue;
      }
      cache[voicePrefix][seg] = ipa;
      console.log(`  ${langId}/${transport}: ${ipa.slice(0, 60)}...`);
    }
  }
}

const outPath = join(root, 'src', 'phoneme-cache.json');
writeFileSync(outPath, JSON.stringify(cache, null, 2), 'utf8');
console.log(`\nWrote ${outPath}`);
