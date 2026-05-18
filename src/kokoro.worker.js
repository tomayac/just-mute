import {
	StyleTextToSpeech2Model,
	AutoTokenizer,
	Tensor,
	env,
} from "@huggingface/transformers";
import { phonemize } from "phonemizer";
import phonemeCache from "./phoneme-cache.json";

const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
env.experimental_useCrossOriginStorage = true;

// Maps the first letter of a Kokoro voice ID to an eSpeak language code.
// eSpeak processes accented characters only when given the correct language —
// passing French text with "en-us" causes a WASM abort on characters like é/à/ç.
const PHONEME_LANG = {
	a: "en-us", // af_* / am_* — American English
	b: "en-gb", // bf_* / bm_* — British English
	e: "es",    // ef_* / em_* — Spanish
	f: "fr",    // ff_*        — French
	h: "hi",    // hf_* / hm_* — Hindi
	i: "it",    // if_* / im_* — Italian
	j: "ja",    // jf_* / jm_* — Japanese
	p: "pt-br", // pf_* / pm_* — Brazilian Portuguese
	z: "cmn",   // zf_* / zm_* — Mandarin Chinese
};

let model = null;
let tokenizer = null;
const voiceCache = new Map();
const audioCache = new Map();

// Requests currently in-flight: id → error-reporter fn.
// Lets the global error handlers below reach the right caller.
const pending = new Map();

function failPending(msg) {
	for (const [id, report] of pending) {
		report(msg);
		pending.delete(id);
	}
}

// ONNX Runtime can throw from WASM callbacks that bypass the JS promise chain,
// producing a messageless Worker error event on the main thread.  These two
// handlers intercept those escapes and route them back as normal error replies.
self.addEventListener("unhandledrejection", ({ reason }) => {
	const msg = reason?.message || String(reason) || "Unhandled rejection";
	console.error("[kokoro-worker] unhandledrejection:", reason);
	failPending(msg);
});

self.addEventListener("error", (e) => {
	const msg = e.message || "Worker error";
	console.error("[kokoro-worker] error:", e.message, e.filename, e.lineno);
	// Prevent the error from also firing as a Worker onerror on the main thread,
	// which would produce a messageless rejection racing our proper error reply.
	e.preventDefault();
	failPending(msg);
});

async function ensureModel() {
	if (model && tokenizer) return;
	[model, tokenizer] = await Promise.all([
		StyleTextToSpeech2Model.from_pretrained(MODEL_ID, { dtype: "q8" }),
		AutoTokenizer.from_pretrained(MODEL_ID),
	]);
}

async function loadVoice(voiceId) {
	if (voiceCache.has(voiceId)) return voiceCache.get(voiceId);
	const url = `https://huggingface.co/${MODEL_ID}/resolve/main/voices/${voiceId}.bin`;
	let buffer;
	try {
		const cache = await caches.open("kokoro-voices");
		const hit = await cache.match(url);
		if (hit) {
			buffer = await hit.arrayBuffer();
		} else {
			const res = await fetch(url);
			if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${voiceId}.bin`);
			buffer = await res.arrayBuffer();
			await cache.put(url, new Response(buffer.slice(0), {
				headers: { "content-type": "application/octet-stream" },
			}));
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
		if (cached) { phoneParts.push(cached); continue; }

		// Phonemize via eSpeak (English voices, or uncached custom text)
		const lang = PHONEME_LANG[voiceId[0]] ?? "en-us";
		try {
			phoneParts.push((await phonemize(seg, lang)).join(" "));
		} catch {
			phoneParts.push((await phonemize(
				seg.normalize("NFD").replace(/[\u0300-\u036f]/g, ""), "en-us"
			)).join(" "));
		}
	}

	// Join: remove spaces before punctuation, ensure single space after
	return phoneParts.join(" ")
		.replace(/\s+([.!?,;:\u00bf\u00a1])/g, "$1")
		.trim();
}

async function generate(text, voiceId) {
	await ensureModel();

	const phonemes = await toPhonemes(text, voiceId);
	if (!phonemes.trim()) throw new Error(`Phonemization returned empty output for "${text}"`);

	const { input_ids } = tokenizer(phonemes, { truncation: true });
	const seqLen = input_ids.dims.at(-1);
	if (!seqLen) throw new Error(`Tokenizer returned empty sequence for phonemes "${phonemes}"`);

	const voiceData = await loadVoice(voiceId);
	const offset = 256 * Math.min(Math.max(seqLen - 2, 0), 509);
	const raw = voiceData.slice(offset, offset + 256);

	// Pad to exactly 256 in case the voice file is shorter than expected.
	const style = raw.length === 256 ? raw : Object.assign(new Float32Array(256), raw);

	const { waveform } = await model({
		input_ids,
		style: new Tensor("float32", style, [1, 256]),
		speed: new Tensor("float32", [1.0], [1]),
	});
	return { audio: waveform.data, sampling_rate: 24000 };
}

self.addEventListener("message", async ({ data: { id, text, voice } }) => {
	const report = (msg) => self.postMessage({ id, error: msg });
	pending.set(id, report);
	try {
		const key = `${voice}::${text}`;
		let entry = audioCache.get(key);
		if (!entry) {
			entry = await generate(text, voice);
			audioCache.set(key, entry);
		}
		pending.delete(id);
		const copy = new Float32Array(entry.audio);
		self.postMessage(
			{ id, audio: copy, sampling_rate: entry.sampling_rate },
			[copy.buffer],
		);
	} catch (err) {
		pending.delete(id);
		const msg = err?.message || String(err) || "Unknown error";
		console.error("[kokoro-worker]", err);
		report(msg);
	}
});
