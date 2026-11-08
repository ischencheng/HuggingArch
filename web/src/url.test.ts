// Shareable links. A link is the one piece of page state that outlives the
// page, so what matters is that the reader gets back what the writer put in,
// that keys the page does not own survive, and that keys it has retired do
// not come back.
import { afterEach, describe, expect, test, vi } from "vitest";

import {
  mergeHash, playgroundCompareHref, playgroundSearch, readHash, readPlaygroundState,
  RETIRED_PRECISION_KEYS, type InferHash, type PlaygroundPick,
} from "./url";

const emptyInferHash = (): InferHash =>
  Object.fromEntries([
    "model", "gpu", "selection", "framework", "kv_block", "in_tok", "out_tok", "frac",
    "comm_lat", "launch_floor", "tie_tp", "tp_attn", "tp_ffn", "tp_moe", "ep", "cp", "pp",
    "dp_attn", "dp_rep", "cp_scheme", "ep_load", "dec", "tp_attn_d", "tp_ffn_d", "tp_moe_d",
    "ep_d", "cp_d", "pp_d", "dp_attn_d", "dp_rep_d", "kv_bw", "prefix_len", "prefill_chunk", "dec_batch",
    "spec_mode", "spec_drafts", "spec_alpha", "spec_draft", "include_aux", "host_tables",
  ].map((k) => [k, ""])) as InferHash;

const params = (hash: string) => new URLSearchParams(hash.replace(/^#/, ""));

afterEach(() => vi.unstubAllGlobals());

describe("mergeHash", () => {
  test("what the page writes comes back out of readHash", () => {
    const h: InferHash = { ...emptyInferHash(), model: "deepseek-ai/DeepSeek-V3", gpu: "H200-SXM", tp_attn: "8", dec: "1", spec_drafts: "3" };
    const hash = mergeHash("", h);
    // The round trip goes through the writer and the reader, not a
    // hand-written string: the shape of the hash is theirs to choose.
    vi.stubGlobal("window", { location: { hash } });
    expect(readHash()).toEqual({
      model: "deepseek-ai/DeepSeek-V3", gpu: "H200-SXM", tp_attn: "8", dec: "1", spec_drafts: "3",
    });
  });

  test("keys the page does not own are left where they were", () => {
    // Compare's `sub`/`m`/`s` and the phase `p` live in the same hash.
    const hash = mergeHash("#sub=sweep&m=a|b&p=prefill&model=old", { ...emptyInferHash(), model: "new" });
    expect(params(hash).get("sub")).toBe("sweep");
    expect(params(hash).get("m")).toBe("a|b");
    expect(params(hash).get("p")).toBe("prefill");
    expect(params(hash).get("model")).toBe("new");
  });

  test("a retired key in an old link is read once and then gone", () => {
    const old = `#model=m&spec_k=4&framework_version=0.9&${RETIRED_PRECISION_KEYS.map((k) => `${k}=x`).join("&")}`;
    vi.stubGlobal("window", { location: { hash: old } });
    // Read in its saved meaning …
    expect(readHash().spec_k).toBe("4");
    expect(readHash().kv_dtype).toBe("x");
    // … and dropped by the next write, so the link stops carrying it.
    const rewritten = mergeHash(old, { ...emptyInferHash(), model: "m" });
    for (const k of ["spec_k", "framework_version", ...RETIRED_PRECISION_KEYS]) {
      expect(params(rewritten).has(k)).toBe(false);
    }
    expect(params(rewritten).get("model")).toBe("m");
  });

  test("an empty value deletes its key instead of writing it empty", () => {
    const hash = mergeHash("#model=m&gpu=H100&tp_attn=8", { ...emptyInferHash(), model: "m" });
    expect(params(hash).has("gpu")).toBe(false);
    expect(params(hash).has("tp_attn")).toBe(false);
    // Nothing left to say: no bare "#" on the end of the URL.
    expect(mergeHash("#gpu=H100", emptyInferHash())).toBe("");
  });

  test("an empty value in the link reads as absent, not as the empty string", () => {
    vi.stubGlobal("window", { location: { hash: "#model=&gpu=H100" } });
    expect(readHash()).toEqual({ gpu: "H100" });
  });
});

describe("playground picks", () => {
  const roundTrip = (picks: PlaygroundPick[]) =>
    readPlaygroundState(playgroundSearch({ picks })).picks;

  test("a pick survives the trip through the query string", () => {
    const picks: PlaygroundPick[] = [
      { name: "mla_attention", model_id: null, call: null, label: null, dims: {} },
      { name: "moe_ffn", model_id: "deepseek-ai/DeepSeek-V3", call: null, label: null, dims: {} },
      { name: "moe_ffn", model_id: "deepseek-ai/DeepSeek-V3", call: "layers.3", label: null, dims: {} },
      { name: "gqa_attention", model_id: null, call: null, label: null, dims: { d_model: 4096, n_heads: 32 } },
    ];
    expect(roundTrip(picks)).toEqual(picks);
  });

  test("a my-module label keeps its whole identity, module name included", () => {
    const pick: PlaygroundPick = { name: "moe_ffn", model_id: null, call: null, label: "moe_ffn~my:wide", dims: { d_ff: 2048 } };
    expect(roundTrip([pick])).toEqual([pick]);
  });

  test("the rest of the comparison survives too, and its defaults stay out", () => {
    const state = { picks: [], role: "ffn", batch: 4, seq: 2048, prec: "fp8_block", gpu: "H100-SXM", phase: "decode" };
    expect(playgroundSearch({ picks: [] })).toBe("");
    const read = readPlaygroundState(playgroundSearch(state));
    expect(read).toMatchObject({ role: "ffn", batch: 4, seq: 2048, prec: "fp8_block", gpu: "H100-SXM", phase: "decode" });
  });

  test("batch and seq must be whole and positive, or they are not there", () => {
    expect(readPlaygroundState("?batch=0&seq=-1").batch).toBeUndefined();
    expect(readPlaygroundState("?batch=0&seq=-1").seq).toBeUndefined();
    expect(readPlaygroundState("?batch=2.5").batch).toBeUndefined();
    expect(readPlaygroundState("?batch=abc").batch).toBeUndefined();
    expect(readPlaygroundState("?batch=4").batch).toBe(4);
  });

  test("a pick that does not parse is reported, not silently used", () => {
    const read = readPlaygroundState("?pick=good&pick=%40no-name&pick=x%7Cd_model%3Dnope&pick=y%40");
    expect(read.picks.map((p) => p.name)).toEqual(["good"]);
    expect(read.malformed).toEqual(["@no-name", "x|d_model=nope", "y@"]);
  });

  test("a link that carried precision is noticed", () => {
    expect(readPlaygroundState("?kv_dtype=fp8").retiredPrecision).toBe(true);
    expect(readPlaygroundState("?pick=a").retiredPrecision).toBe(false);
  });

  test("the compare href is the playground page plus its picks", () => {
    expect(playgroundCompareHref([{ name: "moe_ffn", model_id: null, call: null }]))
      .toBe("/playground.html?pick=moe_ffn");
  });
});
