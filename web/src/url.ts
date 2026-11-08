/** Shareable hash for the single-model Inference page. Keys match url-state.js. */

export type InferHash = {
  model: string;
  gpu: string;
  selection: string;
  framework: string;
  in_tok: string;
  out_tok: string;
  frac: string;
  /** The two time floors, µs; empty leaves them to the server default. */
  comm_lat: string;
  launch_floor: string;
  tie_tp: string;
  tp_attn: string;
  tp_ffn: string;
  tp_moe: string;
  ep: string;
  cp: string;
  pp: string;
  dp_attn: string;
  dp_rep: string;
  cp_scheme: string;
  ep_load: string;
  dec: string;
  tp_attn_d: string;
  tp_ffn_d: string;
  tp_moe_d: string;
  ep_d: string;
  cp_d: string;
  pp_d: string;
  dp_attn_d: string;
  dp_rep_d: string;
  kv_bw: string;
  prefix_len: string;
  prefill_chunk: string;
  /** The decode step's batch, per GPU; empty = the card's KV-bound max. */
  dec_batch: string;
  spec_mode: string;
  spec_drafts: string;
  spec_alpha: string;
  spec_draft: string;
  include_aux: string;
  host_tables: string;
  kv_block: string;
};

const KEYS: (keyof InferHash)[] = [
  "model", "gpu", "selection", "framework", "kv_block",
  "in_tok", "out_tok", "frac", "comm_lat", "launch_floor", "tie_tp",
  "tp_attn", "tp_ffn", "tp_moe", "ep", "cp", "pp", "dp_attn", "dp_rep", "cp_scheme", "ep_load",
  "dec", "tp_attn_d", "tp_ffn_d", "tp_moe_d", "ep_d", "cp_d", "pp_d", "dp_attn_d", "dp_rep_d", "kv_bw",
  "prefix_len", "prefill_chunk", "dec_batch", "spec_mode", "spec_drafts", "spec_alpha", "spec_draft", "include_aux", "host_tables",
];

/** Keys older links carry that the page no longer writes. Read once so the
 *  page can load them in their saved meaning, then dropped on the next write.
 *  `spec_k` was the speculative verify window (drafted tokens + 1); it is now
 *  `spec_drafts`, the drafted tokens. `framework_version` named a framework
 *  binding's release; bindings are not versioned any more, so it is read and
 *  ignored. The precision keys predate precision profiles (`prec=`); they are
 *  not translated into a profile, only noticed (`RETIRED_PRECISION_KEYS`). */
export type RetiredInferHash = {
  spec_k: string;
  framework_version: string;
  kv_dtype: string;
  kv_gran: string;
  kv_scale: string;
  act_dtype: string;
  w_dtype: string;
  w_gran: string;
  w_scale: string;
  compute_dtype: string;
};

/** The Inference hash keys that carried precision before profiles. */
export const RETIRED_PRECISION_KEYS: (keyof RetiredInferHash)[] = [
  "kv_dtype", "kv_gran", "kv_scale", "act_dtype", "w_dtype", "w_gran", "w_scale", "compute_dtype",
];

const RETIRED_KEYS: (keyof RetiredInferHash)[] = ["spec_k", "framework_version", ...RETIRED_PRECISION_KEYS];

export function readHash(): Partial<InferHash & RetiredInferHash> {
  const raw = window.location.hash.replace(/^#/, "");
  const params = new URLSearchParams(raw);
  const out: Partial<InferHash & RetiredInferHash> = {};
  for (const k of [...KEYS, ...RETIRED_KEYS]) {
    const v = params.get(k);
    if (v != null && v !== "") out[k] = v;
  }
  return out;
}

/** Merge owned single-model keys. Unknown keys stay unless `then` takes them. */
export function mergeHash(existingHash: string, h: InferHash): string {
  const raw = existingHash.replace(/^#/, "");
  const params = new URLSearchParams(raw);
  for (const k of KEYS) {
    if (h[k]) params.set(k, h[k]);
    else params.delete(k);
  }
  for (const k of RETIRED_KEYS) params.delete(k);
  const next = params.toString();
  return next ? `#${next}` : "";
}

export function writeHash(h: InferHash, then?: (hash: string) => string) {
  let hash = mergeHash(window.location.hash, h);
  if (then) hash = then(hash);
  if (hash !== window.location.hash) {
    history.replaceState(null, "", `${window.location.pathname}${hash}`);
  }
}

/** A Playground compare column carried in the page URL: the block or
 *  component `name`, as model `model_id` runs it at its call `call` (either
 *  may be unnamed), a my-module `label` (`name~my:variant`), and the dims the
 *  column sets itself — a bound column's edits to its binding, or the whole
 *  parameterization of a labeled or unbound one. */
export type PlaygroundPick = {
  name: string;
  model_id: string | null;
  call: string | null;
  label?: string | null;
  dims?: Record<string, number>;
};

/** Everything a Playground comparison is shared by. Unset fields are the
 *  page's defaults and stay out of the URL. */
export type PlaygroundState = {
  picks: PlaygroundPick[];
  role?: string;
  batch?: number;
  seq?: number;
  /** The precision preset id; absent is as shipped. */
  prec?: string;
  gpu?: string;
  phase?: string;
};

// One `pick` per column: `name[~my:variant][@model_id[#call]][|dim=v,…]`.
function pickText(p: PlaygroundPick): string {
  let out = p.label ?? p.name;
  if (p.model_id) out += `@${p.model_id}${p.call ? `#${p.call}` : ""}`;
  const dims = Object.entries(p.dims ?? {}).sort(([a], [b]) => a.localeCompare(b));
  if (dims.length) out += `|${dims.map(([k, v]) => `${k}=${v}`).join(",")}`;
  return out;
}

function readPick(raw: string): PlaygroundPick | null {
  const bar = raw.indexOf("|");
  const head = bar < 0 ? raw : raw.slice(0, bar);
  const dims: Record<string, number> = {};
  if (bar >= 0) {
    for (const kv of raw.slice(bar + 1).split(",")) {
      const eq = kv.indexOf("=");
      const v = Number(kv.slice(eq + 1));
      if (eq <= 0 || !Number.isFinite(v)) return null;
      dims[kv.slice(0, eq)] = v;
    }
  }
  const at = head.indexOf("@");
  const ident = at < 0 ? head : head.slice(0, at);
  const site = at < 0 ? "" : head.slice(at + 1);
  const hash = site.lastIndexOf("#");
  const model_id = hash < 0 ? site : site.slice(0, hash);
  const call = hash < 0 ? "" : site.slice(hash + 1);
  const my = ident.indexOf("~my:");
  const name = my < 0 ? ident : ident.slice(0, my);
  if (!name || (at >= 0 && !model_id) || (hash >= 0 && !call) || my === ident.length - 4) return null;
  return { name, model_id: model_id || null, call: call || null,
           label: my < 0 ? null : ident, dims };
}

/** The Playground query string for `state` (with its leading `?`, or empty). */
export function playgroundSearch(state: PlaygroundState): string {
  const params = new URLSearchParams();
  for (const p of state.picks) params.append("pick", pickText(p));
  const { picks: _picks, ...rest } = state;
  for (const [k, v] of Object.entries(rest)) if (v != null && v !== "") params.set(k, String(v));
  const q = params.toString();
  return q ? `?${q}` : "";
}

/** `/playground.html?pick=…` opening module comparison on `picks`. */
export function playgroundCompareHref(picks: PlaygroundPick[]): string {
  return `/playground.html${playgroundSearch({ picks })}`;
}

/** The comparison a Playground URL's query string carries, and the `pick`
 *  values that do not parse (reported, not used). */
export function readPlaygroundState(search: string): PlaygroundState & { malformed: string[]; retiredPrecision: boolean } {
  const params = new URLSearchParams(search);
  const picks: PlaygroundPick[] = [];
  const malformed: string[] = [];
  for (const raw of params.getAll("pick")) {
    const p = readPick(raw);
    if (p) picks.push(p); else malformed.push(raw);
  }
  const num = (k: string) => {
    const v = Number(params.get(k));
    return params.get(k) && Number.isInteger(v) && v > 0 ? v : undefined;
  };
  const str = (k: string) => params.get(k) || undefined;
  // `kv_dtype` predates precision profiles: noticed, never translated.
  return { picks, malformed, role: str("role"), batch: num("batch"), seq: num("seq"),
           prec: str("prec"), gpu: str("gpu"), phase: str("phase"),
           retiredPrecision: params.has("kv_dtype") };
}
