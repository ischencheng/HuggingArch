import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, fetchJson, isRequestableModelId, stillComputingNote } from "../api";
import { authContext, fetchMyCalibration, trackAnalyticsEvent } from "../auth";
import { formatNumber } from "../format";
import {
  applyTemplateSection,
  COMPARE_AS_SHIPPED_NOTE,
  COMPARE_LEGACY_PRECISION_NOTICE,
  DEFAULT_PREFILL_CHUNK_SIZE,
  draftTokensFromLegacyWindow,
  encodeSweep,
  estimateCfgFields,
  groupLabels,
  hydrateSweep,
  mergeSweep,
  mergeSweepPhase,
  patchRow,
  readSweepParams,
  readSweepPhase,
  SPEC_PRESETS,
  specDraftStyle,
  sweepRequestBodies,
  SWEEP_BATCH_GRID,
  SWEEP_MIN_ROWS,
  SWEEP_MAX_ROWS,
  SWEEP_OUTPUT_TOKENS,
  SWEEP_SEQ_LENS,
  type SpecDraftStyle,
  type SweepCfg,
  type SweepHit,
  type SweepPhase,
  type SweepResponse,
  type SweepRow,
  type SweepSection,
} from "../compare";
import { axis, EMPTY_PARALLEL, loadFactorError, parallelOf, parseLoadFactor, useParallelShape, type Parallel } from "../parallel";
import { filterAndSortModels, ModelOptions, useSupportedModels } from "../supportedModels";
import { readHash, RETIRED_PRECISION_KEYS, writeHash } from "../url";
import { AS_SHIPPED, mergePrecisionHash, PREDATES_PROFILES_NOTICE, precisionBody, readPrecisionHash } from "../precision";
import { PrecisionApplied } from "../PrecisionPresets";
import { gpuOptionLabel, skuOf } from "../gpu";
import { PlanCard, type Calibration, type GpuItem } from "./PlanCard";
import { CompareCharts } from "./CompareCharts";
import { CompareSummary } from "./CompareSummary";
import { CompareTradeoff } from "./CompareTradeoff";
import { DataTable, type ColumnDef } from "../DataTable";
import { ComputeGraph, graphWorkload, type GraphTarget } from "./ComputeGraph";
import { EncoderResults, MediaForm, requestMedia, useMediaInputs, type MediaDraft } from "./EncoderTab";
import { MemBudget } from "./MemBudget";
import { PhaseTable } from "./PhaseTable";
import { ResultSummary, type Estimate } from "./ResultSummary";
import { ScalingPanel } from "./ScalingPanel";

type GpusResponse = { items?: GpuItem[]; audit_parallel?: Record<string, number> };

/** One estimate together with the deployment it was computed for. The
 *  drill-downs below the summary (phase table, compute graph, fused subgraphs)
 *  read `target`, never the live form, so they always describe the numbers on
 *  screen. */
type Estimated = { data: Estimate; target: GraphTarget; formKey: string };

const DEFAULT_INFER_MODEL = "deepseek-ai/DeepSeek-V3";
const DEFAULT_INFER_GPU = "H100-SXM";

/** Estimate + hash. FastAPI serves the React SPA at `/inference.html`. */
export function Inference() {
  const [modelId, setModelId] = useState(DEFAULT_INFER_MODEL);
  const [modelDropOpen, setModelDropOpen] = useState(false);
  const modelDropRef = useRef<HTMLDivElement>(null);
  const [gpuSku, setGpuSku] = useState(DEFAULT_INFER_GPU);
  const [gpus, setGpus] = useState<GpuItem[]>([]);
  // The precision profile every request from this page (Scaling included)
  // carries, as a preset id; as shipped by default. Compare does not read it.
  const [precPreset, setPrecPreset] = useState(AS_SHIPPED);
  // Set when the link carried precision keys from before profiles.
  const [precNotice, setPrecNotice] = useState<string | null>(null);
  // Whether the deployment on screen was chosen — by the link or by an edit.
  // The opening deployment (`/inference/default-deployment`) fills only a
  // deployment nobody chose: an all-1 mesh the user asked for (one GPU) is an
  // answer, not an untouched form.
  const parChosen = useRef(false);
  // Whether a model routes tokens to experts, as the estimate for THAT model
  // answered. Read only for the model on screen: another model's answer is
  // no answer (`moeEnabled` below is undefined until this model's estimate).
  const [moeOf, setMoeOf] = useState<{ model: string; moe: boolean } | null>(null);
  // Kernel times: the fastest measured kernel of any library where the corpus
  // has the op's cell (SOL elsewhere), or SOL only. `latest` (the framework's
  // own kernel) is still read from a link that names it.
  const [selectionMode, setSelectionMode] = useState<"envelope" | "latest" | "theoretical_only">("envelope");
  const [framework, setFramework] = useState("");
  const [frameworks, setFrameworks] = useState<Record<string, string>>({});
  // Paged-KV paging, entirely as `/fusion/registry` states it: the pages some
  // registered backend declares it can serve (`served`), each framework
  // binding's own default page (`defaults`, null where a binding declares
  // none), and the framework an estimate runs under when the request names no
  // framework (`defaultFramework`). No page list lives in this client.
  const [kvPaging, setKvPaging] = useState<{
    served: number[];
    defaults: Record<string, number | null>;
    defaultFramework: string;
  }>({ served: [], defaults: {}, defaultFramework: "" });
  // The paged-KV page the user chose; "" is "not chosen", which is what
  // omitting `kv_block_size` from the request means — the framework binding's
  // own default. Carried by the Scaling panel too: the page is part of the
  // measurement key, so a sweep must read rows at the page being deployed.
  const [kvBlock, setKvBlock] = useState("");
  // The caller's own measured rows, as the server calibrates with them
  // (`Calibration`); the link's estimate waits until they are known.
  const [calibration, setCalibration] = useState<Calibration>({ state: "loading" });
  const calibrateMine = calibration.state === "on";
  const calibrationKnown = calibration.state !== "loading";
  const [frac, setFrac] = useState(0.9);
  // The two time floors (µs): null until `/inference/defaults` seeds them
  // (or a link sets them); a null one is left to the server's default.
  const [commLatencyUs, setCommLatencyUs] = useState<number | null>(null);
  const [launchFloorUs, setLaunchFloorUs] = useState<number | null>(null);
  // Seeded on a point of the routine grid Scaling and Kernel Bench sweep
  // (`sweep_grid.py`), so a first look at a model asks for shapes that have
  // been measured.
  const [inTok, setInTok] = useState(SWEEP_SEQ_LENS[0]);
  const [outTok, setOutTok] = useState(SWEEP_OUTPUT_TOKENS);
  const [prefixLen, setPrefixLen] = useState(0);
  const [prefillChunk, setPrefillChunk] = useState(DEFAULT_PREFILL_CHUNK_SIZE);
  const [specMode, setSpecMode] = useState("none");
  // Custom speculative K: drafted tokens per step (the verify covers K + 1).
  const [specK, setSpecK] = useState(4);
  const [specAlpha, setSpecAlpha] = useState(0.65);
  const [includeAux, setIncludeAux] = useState(false);
  // Keep the model's lookup tables (Engram / PLE n-gram tables) in host
  // memory, as vLLM does by default; they then leave the per-GPU weights.
  const [tablesOnHost, setTablesOnHost] = useState(true);
  // Count the speculative draft's forwards in the decode step; "" = the
  // verify forward alone (the default).
  const [specDraft, setSpecDraft] = useState<SpecDraftStyle | "">("");
  // Whether this model + window + pp can count a draft, and why not — the
  // backend's answer (`/inference/speculative-draft`).
  const [draftAvail, setDraftAvail] = useState<{ available: boolean; reason: string | null } | null>(null);
  const [tieTp, setTieTp] = useState(true);
  const [cpScheme, setCpScheme] = useState("pcp");
  // Routing imbalance ρ = max/mean expert load (`ParallelConfig.ep_load_factor`),
  // as typed. 1 is perfectly balanced routing — the backend says so when it
  // bites. A value that does not parse is shown as an error and blocks the
  // estimate; it is never read as 1.
  const [epLoadText, setEpLoadText] = useState("1");
  const epLoad = parseLoadFactor(epLoadText);
  const epLoadErr = loadFactorError(epLoadText);
  const moeEnabled = moeOf && moeOf.model === modelId.trim() ? moeOf.moe : undefined;
  const [legalCpDegrees, setLegalCpDegrees] = useState<number[] | null>(null);
  const [cpUnavailableReason, setCpUnavailableReason] = useState<string | null>(null);
  // Said once CP > 1 was reset to 1 because the scheme offers this model no
  // legal degree; the text carries the backend's reason.
  const [cpResetNotice, setCpResetNotice] = useState<string | null>(null);
  const [par, setPar] = useState<Parallel>(EMPTY_PARALLEL);
  // The CP degree on screen, for the legal-CP answer to compare against
  // without re-running on every CP edit.
  const cpNow = useRef(par.cp);
  cpNow.current = par.cp;
  const [decodeOn, setDecodeOn] = useState(false);
  const [decPar, setDecPar] = useState<Parallel>(EMPTY_PARALLEL);
  const [kvBw, setKvBw] = useState("");
  const [status, setStatus] = useState("");
  const [statusKind, setStatusKind] = useState<"idle" | "ok" | "warn" | "err">("idle");
  const { models: supported, loading: supportedLoading } = useSupportedModels((e) => {
    setStatus(`Failed to load models: ${e instanceof Error ? e.message : String(e)}`);
    setStatusKind("err");
  });
  const models = useMemo(() => supported.map((m) => m.model_id), [supported]);
  // The mesh each deployment occupies, as the model on screen runs it (a
  // dense model has no expert region). A half-typed id is no model yet.
  const shapeModel = models.includes(modelId.trim()) ? modelId.trim() : undefined;
  const parShape = useParallelShape(par, shapeModel);
  const decShape = useParallelShape(decPar, shapeModel);
  const [result, setResult] = useState<Estimated | null>(null);
  // Why the latest estimate was refused (shown where its numbers would be).
  const [estimateError, setEstimateError] = useState<string | null>(null);
  // Latest estimate request; a response from an older one is dropped.
  const runSeq = useRef(0);
  const [hydrated, setHydrated] = useState(false);
  const [mode, setMode] = useState<"single" | "sweep">("single");
  const [rows, setRows] = useState<SweepRow[]>([]);
  const [template, setTemplate] = useState<SweepCfg | null>(null);
  // Set when a Compare link carried precision settings that were dropped.
  const [compareNotice, setCompareNotice] = useState<string | null>(null);
  const [sweepHits, setSweepHits] = useState<SweepHit[]>([]);
  const [icPhase, setIcPhase] = useState<SweepPhase>("decode");
  // A model with encoders has two accounts: the language model's step and
  // the encoders' own encode step, each in its own tab.
  const mediaSpec = useMediaInputs(modelId.trim());
  const hasEncoders = !!mediaSpec && mediaSpec.encoders.length > 0;
  const [resultTab, setResultTab] = useState<"lm" | "encoder">("lm");
  const [mediaDraft, setMediaDraft] = useState<MediaDraft>({});
  const [encodeBatch, setEncodeBatch] = useState("");
  /** The decode batch the estimate asks for, per GPU: "" = the card's
   *  KV-bound max (the backend's default). */
  const [decBatch, setDecBatch] = useState("");
  useEffect(() => { setMediaDraft({}); setResultTab("lm"); }, [modelId]);
  const [applyTick, setApplyTick] = useState(0);

  const setAxis = (k: keyof Parallel, n: number) => {
    parChosen.current = true;
    setPar((p) => ({ ...p, [k]: axis(n) }));
  };
  const setDecAxis = (k: keyof Parallel, n: number) =>
    setDecPar((p) => ({ ...p, [k]: axis(n) }));

  /** Every input `run` sends, as one string: an estimate made from a
   *  different one is stale. */
  const formKeyOf = (usePar: Parallel) => JSON.stringify([
    modelId.trim(), gpuSku, frac, commLatencyUs, launchFloorUs, precPreset,
    inTok, outTok, usePar, cpScheme,
    epLoadText, prefixLen, prefillChunk, specMode, specK,
    specAlpha, specDraft, includeAux, tablesOnHost, selectionMode,
    framework.trim(), calibrateMine, decodeOn, decodeOn ? decPar : null,
    decodeOn ? kvBw : null, mediaDraft, encodeBatch, decBatch, kvBlock,
  ]);

  /** The page this estimate runs at when the user has chosen none: the
   *  declared default of the framework on screen, or of the one an unnamed
   *  request runs under. `null` where that binding declares no page of its own
   *  (sglang) — which is not 16: the request then names no page either, and
   *  the lookup declines rather than substituting one. */
  const kvBlockDefault =
    kvPaging.defaults[framework.trim() || kvPaging.defaultFramework] ?? null;

  /** The form's deployment as a config: what the single-model estimate posts
   *  (through `estimateCfgFields`) and what "Compare" snapshots as the
   *  template every row can be pushed from. Precision is not part of it:
   *  Compare prices every model as shipped. */
  const deploymentCfg = (usePar: Parallel): SweepCfg => {
    return {
      gpu_sku: gpuSku,
      gpu_memory_fraction: frac,
      ...(commLatencyUs === null ? {} : { comm_latency_us: commLatencyUs }),
      ...(launchFloorUs === null ? {} : { launch_floor_us: launchFloorUs }),
      parallel: { ...usePar },
      cp_scheme: cpScheme,
      ...(epLoad === null ? {} : { ep_load_factor: epLoad }),
      tie_tp: tieTp,
      prefix_cached_tokens: prefixLen,
      prefill_chunk_size: prefillChunk,
      speculative_mode: specMode,
      speculative_draft_tokens: specK,
      acceptance_rate: specAlpha,
      // Dropped only once the backend has said this deployment cannot count a
      // draft; before it answers (a shared link's first run) the request
      // carries it and the estimate gives the same refusal if there is one.
      speculative_draft: draftAvail && !draftAvail.available ? "" : specDraft,
      include_aux_weights: includeAux,
      host_offload: tablesOnHost,
    };
  };

  /** The deployment the backend opens this model at on this card
   *  (`/inference/default-deployment`: one GPU for a dense model, DP = EP
   *  from the card's fabric unit up to the first degree that fits for a
   *  MoE), or null once the user has chosen one. */
  const openingPar = async (): Promise<Parallel | null> => {
    if (parChosen.current) return null;
    const fw = framework.trim();
    const d = await fetchJson<{ parallel?: Record<string, unknown> }>("/inference/default-deployment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: modelId.trim(),
        ...estimateCfgFields(deploymentCfg(EMPTY_PARALLEL)),
        ...precisionBody(precPreset),
        ...(fw ? { framework: fw } : {}),
        ...(kvBlock ? { kv_block_size: Number(kvBlock) } : {}),
      }),
    }, (info) => setStatus(stillComputingNote(info)));
    return parChosen.current ? null : parallelOf(d.parallel);
  };

  const promoteToSweep = async () => {
    if (epLoadErr) {
      setStatus(epLoadErr);
      setStatusKind("err");
      return;
    }
    // The rows start from the deployment this model opens at, not from a
    // form that has not been seeded yet.
    let opening: Parallel | null = null;
    try {
      opening = modelId.trim() && gpuSku ? await openingPar() : null;
    } catch {
      opening = null;
    }
    if (opening) setPar(opening);
    const usePar = opening ?? par;
    const snap = deploymentCfg(usePar);
    setTemplate(snap);
    setRows([
      { model_id: modelId, cfg: snap },
      { model_id: "", cfg: snap },
    ]);
    setMode("sweep");
  };

  const demoteToSingle = (modelId: string) => {
    const id = modelId.trim();
    if (id) setModelId(id);
    setMode("single");
  };

  useEffect(() => {
    if (!modelDropOpen) return;
    const onDoc = (e: MouseEvent) => {
      const el = modelDropRef.current;
      if (el && !el.contains(e.target as Node)) setModelDropOpen(false);
    };
    document.addEventListener("click", onDoc);
    return () => document.removeEventListener("click", onDoc);
  }, [modelDropOpen]);

  useEffect(() => {
    const h = readHash();
    if (h.model?.trim()) setModelId(h.model.trim());
    if (h.gpu?.trim()) setGpuSku(h.gpu.trim());
    setPrecPreset(readPrecisionHash(window.location.hash));
    // Precision keys from before profiles are not translated; the next hash
    // write drops them.
    if (RETIRED_PRECISION_KEYS.some((k) => h[k])) setPrecNotice(PREDATES_PROFILES_NOTICE);
    if (h.selection === "envelope" || h.selection === "latest" || h.selection === "theoretical_only") {
      setSelectionMode(h.selection);
    } else if (h.selection === "framework_version" || h.selection === "best_of" || h.selection === "measured") {
      // The retired spellings of the framework's own kernel, as the API
      // reads them (`selection_mode` in `_models.py`).
      setSelectionMode("latest");
    }
    if (h.framework) setFramework(h.framework);
    if (h.kv_block) setKvBlock(h.kv_block);
    if (h.in_tok) setInTok(axis(Number(h.in_tok)));
    if (h.out_tok) setOutTok(Math.max(0, Number(h.out_tok) || 0));
    if (h.frac) setFrac(Number(h.frac) || 0.9);
    if (h.comm_lat && Number.isFinite(Number(h.comm_lat))) setCommLatencyUs(Math.max(0, Number(h.comm_lat)));
    if (h.launch_floor && Number.isFinite(Number(h.launch_floor))) setLaunchFloorUs(Math.max(0, Number(h.launch_floor)));
    if (h.tie_tp === "0") setTieTp(false);
    if (h.tie_tp === "1") setTieTp(true);
    if (h.cp_scheme) setCpScheme(h.cp_scheme);
    if (h.ep_load) setEpLoadText(h.ep_load);
    if (h.prefix_len) setPrefixLen(Math.max(0, Number(h.prefix_len) || 0));
    if (h.prefill_chunk != null && h.prefill_chunk !== "") {
      setPrefillChunk(Math.max(0, Math.round(Number(h.prefill_chunk)) || 0));
    }
    if (h.dec_batch != null && h.dec_batch !== "") setDecBatch(h.dec_batch);
    if (h.spec_mode) setSpecMode(h.spec_mode);
    if (h.spec_drafts) setSpecK(Math.max(0, Math.round(Number(h.spec_drafts)) || 0));
    else if (h.spec_k) setSpecK(draftTokensFromLegacyWindow(h.spec_k) ?? 4);
    if (h.spec_alpha) setSpecAlpha(Number(h.spec_alpha) || 0.65);
    setSpecDraft(specDraftStyle(h.spec_draft) ?? "");
    if (h.include_aux === "1") setIncludeAux(true);
    if (h.include_aux === "0") setIncludeAux(false);
    if (h.host_tables === "1") setTablesOnHost(true);
    if (h.host_tables === "0") setTablesOnHost(false);
    if ([h.tp_attn, h.tp_ffn, h.tp_moe, h.ep, h.cp, h.pp, h.dp_attn, h.dp_rep].some((v) => v)) {
      parChosen.current = true;
    }
    setPar((p) => ({
      ...p,
      tp_attention: h.tp_attn ? axis(Number(h.tp_attn)) : p.tp_attention,
      tp_dense_ffn: h.tp_ffn ? axis(Number(h.tp_ffn)) : p.tp_dense_ffn,
      tp_moe_expert: h.tp_moe ? axis(Number(h.tp_moe)) : p.tp_moe_expert,
      ep_moe: h.ep ? axis(Number(h.ep)) : p.ep_moe,
      cp: h.cp ? axis(Number(h.cp)) : p.cp,
      pp: h.pp ? axis(Number(h.pp)) : p.pp,
      dp_attention: h.dp_attn ? axis(Number(h.dp_attn)) : p.dp_attention,
      dp_replicas: h.dp_rep ? axis(Number(h.dp_rep)) : p.dp_replicas,
    }));
    if (h.dec === "1") {
      setDecodeOn(true);
      const n = (k: keyof typeof h, fallback: number) => h[k] ? axis(Number(h[k])) : fallback;
      setDecPar((p) => ({
        ...p,
        tp_attention: n("tp_attn_d", p.tp_attention),
        tp_dense_ffn: n("tp_ffn_d", p.tp_dense_ffn),
        tp_moe_expert: n("tp_moe_d", p.tp_moe_expert),
        ep_moe: n("ep_d", p.ep_moe),
        cp: n("cp_d", p.cp),
        pp: n("pp_d", p.pp),
        dp_attention: n("dp_attn_d", p.dp_attention),
        dp_replicas: n("dp_rep_d", p.dp_replicas),
      }));
      if (h.kv_bw) setKvBw(h.kv_bw);
    }
    const sw = readSweepParams(window.location.hash);
    if (sw.sub === "sweep") {
      const hydratedSweep = hydrateSweep(sw.m || "", sw.s || "");
      setMode("sweep");
      setRows(hydratedSweep.rows);
      setTemplate(hydratedSweep.template);
      if (hydratedSweep.droppedPrecision) setCompareNotice(COMPARE_LEGACY_PRECISION_NOTICE);
      setIcPhase(readSweepPhase(window.location.hash));
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    writeHash({
      model: modelId,
      gpu: gpuSku,
      selection: selectionMode,
      framework: framework.trim(),
      kv_block: kvBlock,
      in_tok: String(inTok),
      out_tok: String(outTok),
      frac: String(frac),
      comm_lat: commLatencyUs === null ? "" : String(commLatencyUs),
      launch_floor: launchFloorUs === null ? "" : String(launchFloorUs),
      tie_tp: tieTp ? "1" : "0",
      tp_attn: String(par.tp_attention),
      tp_ffn: String(par.tp_dense_ffn),
      tp_moe: String(par.tp_moe_expert),
      ep: String(par.ep_moe),
      cp: String(par.cp),
      pp: String(par.pp),
      dp_attn: String(par.dp_attention),
      dp_rep: String(par.dp_replicas),
      cp_scheme: cpScheme,
      ep_load: epLoadText,
      dec: decodeOn ? "1" : "0",
      tp_attn_d: String(decPar.tp_attention),
      tp_ffn_d: String(decPar.tp_dense_ffn),
      tp_moe_d: String(decPar.tp_moe_expert),
      ep_d: String(decPar.ep_moe),
      cp_d: String(decPar.cp),
      pp_d: String(decPar.pp),
      dp_attn_d: String(decPar.dp_attention),
      dp_rep_d: String(decPar.dp_replicas),
      kv_bw: kvBw,
      prefix_len: String(prefixLen),
      prefill_chunk: String(prefillChunk),
      dec_batch: decBatch,
      spec_mode: specMode,
      spec_drafts: String(specK),
      spec_alpha: String(specAlpha),
      spec_draft: specDraft,
      include_aux: includeAux ? "1" : "0",
      host_tables: tablesOnHost ? "1" : "0",
    }, (hash) => mergePrecisionHash(mergeSweepPhase(
      mergeSweep(hash, mode === "sweep" ? encodeSweep(rows, template) : null),
      mode === "sweep" ? icPhase : null,
    ), precPreset));
  }, [hydrated, modelId, gpuSku, precPreset, selectionMode, framework, kvBlock, inTok, outTok, frac, commLatencyUs, launchFloorUs, tieTp, par, cpScheme, epLoadText, decodeOn, decPar, kvBw, prefixLen, prefillChunk, decBatch, specMode, specK, specAlpha, specDraft, includeAux, tablesOnHost, mode, rows, template, icPhase]);

  useEffect(() => {
    fetchJson<GpusResponse>("/inference/gpus")
      .then((j) => {
        setGpus(j.items || []);
      })
      .catch((e) => {
        setGpus([]);
        setStatus(`Failed to load GPUs: ${e instanceof Error ? e.message : String(e)}`);
        setStatusKind("err");
      });
    fetchJson<{ comm_latency_us?: number; launch_floor_us?: number }>("/inference/defaults")
      .then((d) => {
        // A link that stated its own keeps them.
        if (typeof d.comm_latency_us === "number") setCommLatencyUs((v) => v ?? d.comm_latency_us!);
        if (typeof d.launch_floor_us === "number") setLaunchFloorUs((v) => v ?? d.launch_floor_us!);
      })
      .catch(() => { /* keep the request defaults */ });
    fetchJson<{
      bindings?: Array<{ framework?: string; label?: string; kv_block_size?: number | null }>;
      default_framework?: string;
      served_kv_block_sizes?: number[];
    }>("/fusion/registry")
      .then((d) => {
        const out: Record<string, string> = {};
        const pages: Record<string, number | null> = {};
        for (const b of d.bindings || []) {
          if (b.framework) {
            out[b.framework] = b.label || b.framework;
            pages[b.framework] = typeof b.kv_block_size === "number" ? b.kv_block_size : null;
          }
        }
        setFrameworks(out);
        setKvPaging({
          served: (d.served_kv_block_sizes || []).filter((n) => Number.isInteger(n) && n > 0),
          defaults: pages,
          defaultFramework: d.default_framework || "",
        });
      })
      .catch(() => { /* corpus unreachable → pickers stay empty */ });
  }, []);

  /** Read the caller's calibration corpus (signed in), or note that there
   *  is none to read. */
  const loadCalibration = useCallback(async (): Promise<void> => {
    try {
      const ctx = await authContext();
      if (!(ctx.authEnabled && ctx.loggedIn)) {
        setCalibration({ state: "off" });
        return;
      }
      const info = await fetchMyCalibration();
      setCalibration(info ? { state: "on", info } : { state: "off" });
    } catch (e) {
      setCalibration({ state: "unavailable", why: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  useEffect(() => { void loadCalibration(); }, [loadCalibration]);

  useEffect(() => {
    const id = modelId.trim();
    if (!id) {
      setLegalCpDegrees(null);
      return;
    }
    if (!isRequestableModelId(id)) {
      setLegalCpDegrees(null);
      return;
    }
    const ctrl = new AbortController();
    const tp = par.tp_attention;
    const fetchLegalCp = () => {
      fetchJson<{ legal_cp_degrees?: number[] | null; unavailable_reason?: string | null }>(
        `/inference/legal-cp?model_id=${encodeURIComponent(id)}`
          + `&tp_attention=${encodeURIComponent(String(tp))}`
          + `&cp_scheme=${encodeURIComponent(cpScheme)}`,
        { signal: ctrl.signal },
      )
        .then((j) => {
          if (ctrl.signal.aborted) return;
          const legal = Array.isArray(j.legal_cp_degrees) ? j.legal_cp_degrees : null;
          setLegalCpDegrees(legal);
          setCpUnavailableReason(j.unavailable_reason || null);
          if (legal && legal.length === 0) {
            // The backend offers no degree under this scheme: CP is off, and
            // a CP > 1 the form held is reset and said so.
            if (cpNow.current > EMPTY_PARALLEL.cp) {
              setCpResetNotice(`CP reset to ${EMPTY_PARALLEL.cp}: ${cpScheme} doesn't support CP>1 for this model — ${j.unavailable_reason || "the backend offers no legal CP degree"}`);
            }
            setPar((p) => (p.cp === EMPTY_PARALLEL.cp ? p : { ...p, cp: EMPTY_PARALLEL.cp }));
          } else {
            setCpResetNotice(null);
          }
          if (legal && legal.length > 0) {
            setPar((p) => {
              if (legal.includes(p.cp)) return p;
              const lesser = legal.filter((d) => d <= p.cp);
              return { ...p, cp: lesser.length ? lesser[lesser.length - 1]! : legal[0]! };
            });
          }
        })
        .catch(() => {
          if (ctrl.signal.aborted) return;
          setLegalCpDegrees(null);
          setCpUnavailableReason(null);
          setCpResetNotice(null);
        });
    };
    const timer = window.setTimeout(fetchLegalCp, 400);
    return () => {
      window.clearTimeout(timer);
      ctrl.abort();
    };
  }, [modelId, par.tp_attention, cpScheme]);

  // A reset notice is about one model under one scheme.
  useEffect(() => setCpResetNotice(null), [modelId, cpScheme]);

  // The drafted tokens the request will carry, from the same body builder the
  // estimate posts — not recomputed here.
  const specDraftTokens = estimateCfgFields({ speculative_mode: specMode, speculative_draft_tokens: specK, acceptance_rate: specAlpha }).speculative_draft_tokens ?? 0;
  useEffect(() => {
    const id = modelId.trim();
    if (!id || !isRequestableModelId(id)) {
      setDraftAvail(null);
      return;
    }
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => {
      fetchJson<{ available?: boolean; unavailable_reason?: string | null }>(
        `/inference/speculative-draft?model_id=${encodeURIComponent(id)}`
          + `&speculative_draft_tokens=${encodeURIComponent(String(specDraftTokens))}`
          + `&pp=${encodeURIComponent(String(par.pp))}`,
        { signal: ctrl.signal },
      )
        .then((j) => {
          if (ctrl.signal.aborted) return;
          setDraftAvail({ available: !!j.available, reason: j.unavailable_reason || null });
        })
        .catch((e) => {
          if (ctrl.signal.aborted) return;
          setDraftAvail({ available: false, reason: e instanceof Error ? e.message : String(e) });
        });
    }, 400);
    return () => {
      window.clearTimeout(timer);
      ctrl.abort();
    };
  }, [modelId, specDraftTokens, par.pp]);

  const run = useCallback(async (parOverride?: Parallel) => {
    const id = modelId.trim();
    if (!id || !gpuSku) {
      setStatus("Model and GPU required.");
      setStatusKind("err");
      return;
    }
    if (epLoad === null) {
      // Bumped so a response still in flight cannot overwrite the error.
      ++runSeq.current;
      setStatus(epLoadErr ?? "");
      setStatusKind("err");
      return;
    }
    let usePar = parOverride ?? par;
    const fw = framework.trim();
    const media = requestMedia(mediaSpec, mediaDraft);
    if ("error" in media) {
      ++runSeq.current;
      setStatus(media.error);
      setStatusKind("err");
      return;
    }
    const encBatch = Number(encodeBatch);
    const decBs = Number(decBatch);
    const my = ++runSeq.current;
    setStatus("Estimating…");
    setStatusKind("idle");
    const t0 = performance.now();
    try {
      // The deployment this model opens at on this card, unless the user
      // (or the link) chose one: asked before the estimate, so the page
      // never prices one deployment and then switches to another.
      if (!parOverride) {
        const opening = await openingPar();
        if (my !== runSeq.current) return;
        if (opening) {
          usePar = opening;
          setPar(opening);
        }
      }
      const data = await fetchJson<Estimate>("/inference/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: id,
          ...estimateCfgFields(deploymentCfg(usePar)),
          ...precisionBody(precPreset),
          avg_input_tokens: inTok,
          avg_output_tokens: outTok,
          selection_mode: selectionMode,
          ...(fw ? { framework: fw } : {}),
          // Omitted, never null, when the user chose no page: absent is what
          // the backend reads as "take the framework binding's default".
          ...(kvBlock ? { kv_block_size: Number(kvBlock) } : {}),
          ...(calibrateMine ? { calibrate_with_mine: true } : {}),
          // Read by the encode phase alone; no language-model number moves.
          media: media.media,
          ...(encodeBatch.trim() && Number.isInteger(encBatch) && encBatch >= 1
            ? { encode_batch_size: encBatch } : {}),
          // The decode step's batch: omitted = the KV-bound max, which the
          // per-op breakdown otherwise always prices.
          ...(decBatch.trim() && Number.isInteger(decBs) && decBs >= 1
            ? { decode_batch_size: decBs } : {}),
          ...(decodeOn ? {
            // The decode cluster routes the same traffic: ρ rides with it.
            decode: { parallel: { ...decPar, ep_load_factor: epLoad } },
            ...(Number(kvBw) > 0 ? { kv_transfer_bw_gbps: Number(kvBw) } : {}),
          } : {}),
        }),
      }, (info) => { if (my === runSeq.current) setStatus(stillComputingNote(info)); });
      if (my !== runSeq.current) return;
      setMoeOf(typeof data.moe_enabled === "boolean" ? { model: id, moe: data.moe_enabled } : null);
      setEstimateError(null);
      setResult({ data, formKey: formKeyOf(usePar), target: {
        modelId: id, gpuSku, par: usePar, cpScheme, framework: fw,
        precision: precPreset,
        epLoadFactor: epLoad, workload: graphWorkload(data),
      } });
      // An estimate that answers "this does not run here" is not a success.
      const unfit = Boolean(data.infeasibility_reason)
        || data.memory?.fits === false || data.max_batch_per_gpu === 0;
      if (unfit) {
        // The backend's reason is a whole sentence, period included.
        setStatus(`Does not fit: ${data.infeasibility_reason
          || "the weights exceed usable memory on this GPU."}`);
        setStatusKind("warn");
      } else {
        setStatus("Done.");
        setStatusKind("ok");
      }
      // What the estimate was calibrated with, as of now (an upload in
      // another tab moves it).
      if (calibrateMine) void loadCalibration();
      trackAnalyticsEvent("inference_estimate", id, Math.round(performance.now() - t0), {
        gpu_sku: gpuSku,
        precision: precPreset,
        cluster_size: data.cluster_size,
        parallel: { ...usePar },
      });
    } catch (e) {
      if (my !== runSeq.current) return;
      // The session lapsed: read it again (a refresh renews it, else the
      // page is signed out and estimates no longer ask for the caller's rows).
      if (calibrateMine && e instanceof ApiError && e.status === 401) void loadCalibration();
      const why = e instanceof Error ? e.message : String(e);
      // A refused deployment (e.g. a precision profile this GPU has no peak
      // for) has no numbers: its reason takes their place.
      setResult(null);
      setEstimateError(why);
      setStatus(`Failed: ${why}`);
      setStatusKind("err");
    }
  }, [modelId, gpuSku, frac, commLatencyUs, launchFloorUs, precPreset, inTok, outTok, par, cpScheme, epLoad, epLoadErr, prefixLen, prefillChunk, decBatch, specMode, specK, specAlpha, specDraft, draftAvail, includeAux, tablesOnHost, selectionMode, framework, kvBlock, calibrateMine, loadCalibration, decodeOn, decPar, kvBw, mediaSpec, mediaDraft, encodeBatch]);

  useEffect(() => {
    // After hydrate, and once the caller's calibration is known (signed in,
    // the link's estimate is calibrated with their rows).
    if (!hydrated || !calibrationKnown) return;
    // vanilla registerTrigger: skip sweep shareable links; require model; do not pick a GPU.
    if (readSweepParams(window.location.hash).sub === "sweep") return;
    if (!readHash().model?.trim()) return;
    void run();
    // one-shot after hydrate
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, calibrationKnown]);

  const runSweep = useCallback(async () => {
    const built = sweepRequestBodies(rows);
    if ("error" in built) {
      setStatus(built.error);
      setStatusKind("err");
      return;
    }
    setStatus("Estimating…");
    setStatusKind("idle");
    const next: SweepHit[] = [];
    for (const body of built.bodies) {
      // Calibrated with the caller's rows, like the single estimate.
      const sent = calibrateMine ? { ...body, calibrate_with_mine: true as const } : body;
      try {
        const data = await fetchJson<SweepResponse>("/inference/estimate-sweep", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(sent),
        }, (info) => setStatus(stillComputingNote(info)));
        if (!data.results_by_batch || typeof data.results_by_batch !== "object") {
          next.push({ body, error: "Sweep response missing results_by_batch" });
        } else {
          next.push({ body, sent, result: data });
        }
      } catch (e) {
        if (calibrateMine && e instanceof ApiError && e.status === 401) void loadCalibration();
        next.push({
          body,
          error: e instanceof Error ? e.message : String(e),
        });
      }
    }
    setSweepHits(next);
    const failed = next.filter((h) => h.error);
    if (failed.length === next.length) {
      setStatus(`Failed: ${failed[0].error}`);
      setStatusKind("err");
    } else if (failed.length) {
      setStatus(`Done with ${failed.length} error(s).`);
      setStatusKind("err");
    } else {
      setStatus("Done.");
      setStatusKind("ok");
    }
  }, [rows, calibrateMine, loadCalibration]);

  useEffect(() => {
    if (applyTick === 0) return;
    void run();
    document.getElementById("infer-plan-card")?.scrollIntoView({ behavior: "smooth", block: "start" });
    // Click-to-Plan only. `run` is the identity from this render (batched with gpu/seq/par).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applyTick]);

  // The results on screen answer the inputs they were estimated for; once
  // any input the estimate reads differs, they are marked stale.
  const resultStale = result !== null && result.formKey !== formKeyOf(par);
  const statusClass =
    statusKind === "ok" ? "text-sm text-green-600" :
    statusKind === "warn" ? "text-sm text-amber-700" :
    statusKind === "err" ? "text-sm text-red-500" : "text-sm text-gray-500";
  const items = filterAndSortModels(supported, modelId);

  return (
    <main className="w-full page-shell py-4 sm:py-8 flex-1">
      <h1 className="text-2xl font-bold text-gray-900 mb-4">Inference</h1>
      {/* Nothing mode-specific renders before the URL is read: a panel that
          mounted on the defaults would fire its requests for the wrong model
          (or the wrong mode) before hydration replaced them. */}
      {hydrated && mode === "sweep" && (
        <SweepPanel
          rows={rows}
          template={template}
          notice={compareNotice}
          models={models}
          gpus={gpus.filter((g) => skuOf(g))}
          hits={sweepHits}
          status={status}
          statusClass={statusClass}
          onDemote={demoteToSingle}
          onRun={() => void runSweep()}
          onPatch={(i, patch) => setRows((rs) => rs.map((r, j) => (j === i ? patchRow(r, patch) : r)))}
          onAdd={() => setRows((rs) => (
            rs.length >= SWEEP_MAX_ROWS
              ? rs
              : [...rs, { model_id: "", cfg: template }]
          ))}
          onRemove={(i) => setRows((rs) => (rs.length <= SWEEP_MIN_ROWS ? rs : rs.filter((_, j) => j !== i)))}
          onTemplate={(next) => setTemplate(next)}
          phase={icPhase}
          onPhase={setIcPhase}
          onApplySection={(section, skip) => {
            const r = applyTemplateSection(rows, template, section, skip);
            setRows(r.rows);
            return r;
          }}
        />
      )}
      {hydrated && mode === "single" && (
        <div id="infer-mode-single">
        <section className="card px-4 sm:px-6 py-4 mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
          <div className="relative" ref={modelDropRef}>
            <input
              type="text"
              id="infer-model"
              placeholder="e.g., meta-llama/Llama-3.2-3B"
              autoComplete="off"
              className="w-full pr-9 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-400 focus:border-amber-500"
              value={modelId}
              data-action="filter-infer-model"
              onChange={(e) => { setModelId(e.target.value); setModelDropOpen(true); }}
              onFocus={() => setModelDropOpen(true)}
            />
            <button
              type="button"
              className="absolute inset-y-0 right-0 w-9 flex items-center justify-center text-gray-400 hover:text-amber-600 transition"
              data-action="toggle-infer-model"
              title="Show supported models"
              onClick={(e) => { e.preventDefault(); setModelDropOpen((o) => !o); }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
            </button>
            <div
              id="infer-model-dropdown"
              className={`${modelDropOpen ? "" : "hidden "}absolute left-0 right-0 top-full mt-1 max-h-72 overflow-y-auto bg-white border border-gray-200 rounded-lg shadow-lg z-20`}
            >
              <ModelOptions
                items={items}
                action="select-infer-model"
                loading={supportedLoading}
                onSelect={(id) => { setModelId(id); setModelDropOpen(false); }}
              />
            </div>
          </div>
        </section>
        <ScalingPanel
          modelId={modelId}
          gpuSku={gpuSku}
          gpus={gpus}
          precision={precPreset}
          frac={frac}
          commLatencyUs={commLatencyUs}
          launchFloorUs={launchFloorUs}
          outTok={outTok}
          selectionMode={selectionMode}
          framework={framework}
          kvBlock={kvBlock}
          moeEnabled={moeEnabled}
          epLoadFactor={epLoad}
          calibration={calibration}
          onSessionLapsed={() => void loadCalibration()}
          onApply={(p) => {
            if (p.gpu) setGpuSku(p.gpu);
            if (p.seq) setInTok(p.seq);
            if (p.par) {
              parChosen.current = true;
              setPar(p.par);
            }
            setApplyTick((n) => n + 1);
          }}
        />
        <PlanCard
          models={models}
          modelId={modelId}
          onModel={setModelId}
          gpus={gpus}
          gpuSku={gpuSku}
          onGpu={setGpuSku}
          framework={framework}
          onFramework={setFramework}
          frameworks={frameworks}
          defaultFramework={kvPaging.defaultFramework}
          kvBlock={kvBlock}
          onKvBlock={setKvBlock}
          kvBlockDefault={kvBlockDefault}
          kvBlockServed={kvPaging.served}
          measured={selectionMode !== "theoretical_only"}
          onMeasured={(m) => setSelectionMode(m ? "envelope" : "theoretical_only")}
          calibration={calibration}
          precision={precPreset}
          onPrecision={(id) => { setPrecPreset(id); setPrecNotice(null); }}
          precisionNotice={precNotice}
          prefixLen={prefixLen}
          onPrefixLen={setPrefixLen}
          prefillChunk={prefillChunk}
          onPrefillChunk={setPrefillChunk}
          decBatch={decBatch}
          onDecBatch={setDecBatch}
          specMode={specMode}
          onSpecMode={(v) => {
            setSpecMode(v);
            const p = SPEC_PRESETS[v];
            if (p && v !== "custom") {
              setSpecK(p.K);
              setSpecAlpha(p.alpha);
            }
          }}
          specK={specK}
          onSpecK={setSpecK}
          specAlpha={specAlpha}
          onSpecAlpha={setSpecAlpha}
          includeAux={includeAux}
          onIncludeAux={setIncludeAux}
          tablesOnHost={tablesOnHost}
          onTablesOnHost={setTablesOnHost}
          lookupTableBytes={result?.data?.memory?.host_offload_bytes_per_gpu}
          specDraft={specDraft}
          onSpecDraft={setSpecDraft}
          draftAvailable={draftAvail?.available ?? false}
          draftUnavailableReason={draftAvail ? draftAvail.reason : "checking whether this model has a draft graph…"}
          inTok={inTok}
          onInTok={setInTok}
          outTok={outTok}
          onOutTok={setOutTok}
          frac={frac}
          onFrac={setFrac}
          commLatencyUs={commLatencyUs}
          onCommLatencyUs={setCommLatencyUs}
          launchFloorUs={launchFloorUs}
          onLaunchFloorUs={setLaunchFloorUs}
          par={par}
          parShape={parShape}
          setAxis={setAxis}
          tieTp={tieTp}
          onTieTp={setTieTp}
          cpScheme={cpScheme}
          onCpScheme={setCpScheme}
          legalCpDegrees={legalCpDegrees}
          cpUnavailableReason={cpUnavailableReason}
          cpResetNotice={cpResetNotice}
          routes={moeEnabled === true}
          epLoadText={epLoadText}
          onEpLoadText={setEpLoadText}
          epLoadError={epLoadErr}
          routingAssumptions={result && result.target.modelId === modelId.trim()
            ? (result.data.assumptions || []).filter((a) => a.knob === "ep_load_factor")
            : []}
          onEstimate={() => void run()}
          onCompare={() => void promoteToSweep()}
          status={status}
          statusClass={statusClass}
          decodeOn={decodeOn}
          onDecodeOn={(v) => {
            setDecodeOn(v);
            if (v) setDecPar({ ...par });
          }}
          decPar={decPar}
          decShape={decShape}
          setDecAxis={setDecAxis}
          kvBw={kvBw}
          onKvBw={setKvBw}
        />
        {estimateError && (
          <section id="infer-estimate-refused" className="card p-4 sm:p-6 mb-6 text-sm text-red-600">
            {estimateError}
          </section>
        )}
        {hasEncoders && (
          <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden mb-4" role="tablist" aria-label="Account">
            <button id="infer-tab-lm" type="button" role="tab" aria-selected={resultTab === "lm"}
              className={"px-4 py-1.5 text-sm font-medium " + (resultTab === "lm" ? "bg-amber-500 text-white" : "bg-white text-gray-600 hover:bg-gray-50")}
              onClick={() => setResultTab("lm")}>Language model</button>
            <button id="infer-tab-encoder" type="button" role="tab" aria-selected={resultTab === "encoder"}
              className={"px-4 py-1.5 text-sm font-medium " + (resultTab === "encoder" ? "bg-amber-500 text-white" : "bg-white text-gray-600 hover:bg-gray-50")}
              onClick={() => setResultTab("encoder")}>Encoder</button>
          </div>
        )}
        {hasEncoders && resultTab === "encoder" && mediaSpec && (
          <div id="infer-encoder-tab">
            <MediaForm
              spec={mediaSpec}
              modelId={modelId.trim()}
              draft={mediaDraft}
              onDraft={setMediaDraft}
              batch={encodeBatch}
              onBatch={setEncodeBatch}
              onEstimate={() => void run()}
              status={status}
            />
            {result && result.target.modelId === modelId.trim() && result.data.encode && (
              <EncoderResults encode={result.data.encode} target={result.target} />
            )}
            {result && result.target.modelId === modelId.trim() && !result.data.encode
              && result.data.infeasibility_reason && (
              <div id="infer-encode-no-step" className="card p-5 mb-6 text-sm text-gray-500">
                No encode step runs here, so there is nothing per step to show: {result.data.infeasibility_reason}
              </div>
            )}
          </div>
        )}
        <div id="infer-results" className={`${result && !(hasEncoders && resultTab === "encoder") ? "" : "hidden "}fade-in space-y-6`}>
        {resultStale && (
          <div id="infer-results-stale" role="status" className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-800">
            Inputs changed since this estimate — the numbers below are for the previous inputs. Press Estimate to update them.
          </div>
        )}
        <div className={resultStale ? "opacity-50 space-y-6" : "space-y-6"} data-stale={resultStale ? "1" : undefined}>
        {result && <PrecisionApplied id="infer-precision-applied" report={result.data.precision} />}
        {result && <ResultSummary result={result.data} />}
        {result && (result.data.disagg
          ? (["prefill", "decode"] as const).map((c) => {
            const dep = result.data[`${c}_deployment`];
            return dep ? <MemBudget key={c} result={dep} cluster={c} /> : null;
          })
          : <MemBudget result={result.data} />)}
        {result && (
        <PhaseTable
          result={result.data}
          target={result.target}
        />
        )}
        {/* One compute graph on the page: the encoder tab draws its own. */}
        {/* No step, no graph of one: a deployment that admits no request
         *  publishes no batch, and the graph endpoint would draw B=1. */}
        {result && !(hasEncoders && resultTab === "encoder")
          && (result.data.phases?.prefill?.batch != null || result.data.phases?.decode?.batch != null)
          && <ComputeGraph target={result.target} />}
        </div>
        </div>
        </div>
      )}
    </main>
  );
}

function SweepPanel({
  rows,
  template,
  notice,
  models,
  gpus,
  hits,
  status,
  statusClass,
  onDemote,
  onRun,
  onPatch,
  onAdd,
  onRemove,
  onTemplate,
  phase,
  onPhase,
  onApplySection,
}: {
  rows: SweepRow[];
  template: SweepCfg | null;
  /** Why a link's precision settings were not applied; null when none were. */
  notice: string | null;
  models: string[];
  gpus: GpuItem[];
  hits: SweepHit[];
  status: string;
  statusClass: string;
  onDemote: (modelId: string) => void;
  onRun: () => void;
  onPatch: (i: number, patch: { model_id?: string; gpu_sku?: string; ep_moe?: number }) => void;
  onAdd: () => void;
  onRemove: (i: number) => void;
  onTemplate: (next: SweepCfg) => void;
  phase: SweepPhase;
  onPhase: (p: SweepPhase) => void;
  onApplySection: (section: SweepSection, skip?: ReadonlySet<number>) => { pushed: number; skipped: number };
}) {
  const atCap = rows.length >= SWEEP_MAX_ROWS;
  const atMin = rows.length <= SWEEP_MIN_ROWS;
  const labeled = groupLabels(hits);
  const [applyStatus, setApplyStatus] = useState("");
  const [icSource, setIcSource] = useState("sol");
  const tpl = template || {};

  const pushSection = (section: SweepSection, skip?: ReadonlySet<number>) => {
    if (!template) return;
    const r = onApplySection(section, skip);
    const label = section[0].toUpperCase() + section.slice(1);
    setApplyStatus(r.skipped
      ? `${label} pushed to ${r.pushed} row(s); ${r.skipped} skipped.`
      : `${label} pushed to ${r.pushed} row(s).`);
  };

  const patchTpl = (patch: SweepCfg) => onTemplate({ ...tpl, ...patch });

  return (
    <div id="infer-mode-sweep">
    <section className="card p-4 sm:p-6 mb-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-gray-900">Compare</h2>
      </div>
      <p id="ic-as-shipped" className="text-sm text-gray-600 mb-1">{COMPARE_AS_SHIPPED_NOTE}</p>
      {notice && <p id="ic-precision-notice" className="text-xs text-amber-700 mb-1">{notice}</p>}
      <p className="text-xs text-gray-400 mb-4">
        Each row uses its own GPU / parallelism. Restoring a shared URL will not clobber later rows.
      </p>
      <details className="border border-gray-200 rounded-lg px-4 py-3 bg-amber-50/40 mb-6" open>
        <summary className="text-sm font-medium text-gray-800 cursor-pointer select-none flex items-center gap-2">
          Shared template config
          <span className="text-xs text-gray-500 font-normal ml-1">(fill once; push per section to all rows)</span>
        </summary>
        <div id="ic-controls" className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3">
          <label className="block text-xs">
            <span className="text-gray-600">GPU</span>
            {gpus.length > 0 ? (
              <select
                className="mt-1 w-full text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white"
                value={typeof tpl.gpu_sku === "string" ? tpl.gpu_sku : ""}
                onChange={(e) => patchTpl({ gpu_sku: e.target.value })}
              >
                <option value="">Select GPU</option>
                {gpus.map((g) => <option key={skuOf(g)} value={skuOf(g)}>{gpuOptionLabel(g)}</option>)}
              </select>
            ) : (
              <input
                className="mt-1 w-full text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white"
                value={typeof tpl.gpu_sku === "string" ? tpl.gpu_sku : ""}
                onChange={(e) => patchTpl({ gpu_sku: e.target.value })}
              />
            )}
          </label>
          <label className="block text-xs">
            <span className="text-gray-600">EP</span>
            <input
              type="number"
              min={1}
              className="mt-1 w-full text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white"
              value={tpl.parallel && typeof tpl.parallel === "object" && typeof tpl.parallel.ep_moe === "number" ? tpl.parallel.ep_moe : ""}
              onChange={(e) => patchTpl({
                parallel: { ...(tpl.parallel && typeof tpl.parallel === "object" ? tpl.parallel : {}), ep_moe: axis(Number(e.target.value)) },
              })}
            />
          </label>
        </div>
        <div className="mt-3 flex flex-wrap gap-2 items-center">
          <span className="text-xs text-gray-600 mr-1">Push to all rows:</span>
          <button
            type="button"
            id="ic-apply-gpu"
            className="text-xs px-2.5 py-1 border border-amber-300 rounded-lg text-amber-700 hover:bg-amber-100"
            title="Push GPU SKU + HBM fraction to every row. Common case — control variable across models."
            onClick={() => pushSection("gpu")}
          >
            GPU
          </button>
          <button
            type="button"
            id="ic-apply-parallel"
            className="text-xs px-2.5 py-1 border border-amber-300 rounded-lg text-amber-700 hover:bg-amber-100"
            title="Push TP / EP / PP / DP / CP topology + tie-TP to every row."
            onClick={() => pushSection("parallel")}
          >
            Parallel
          </button>
          <button
            type="button"
            id="ic-apply-workload"
            className="text-xs px-2.5 py-1 border border-amber-300 rounded-lg text-amber-700 hover:bg-amber-100"
            title="Push prefix-cache / speculative-decode / prefill knobs to every row."
            onClick={() => pushSection("workload")}
          >
            Workload
          </button>
          <span id="ic-apply-status" className="text-11 text-gray-500 ml-2">{applyStatus}</span>
        </div>
      </details>
      <div className="flex items-baseline justify-between mb-2">
        <label className="text-sm font-medium text-gray-700">
          Models <span className="text-xs text-gray-400 ml-2">(2 – 8) — expand each row to override its config</span>
        </label>
        <button
          type="button"
          id="ic-add-model"
          className="text-xs px-3 py-1.5 border border-gray-300 rounded-lg text-gray-700 hover:bg-amber-50 hover:border-amber-300 disabled:opacity-40"
          disabled={atCap}
          onClick={onAdd}
        >
          + Add model
        </button>
      </div>
      <div id="ic-model-list" className="space-y-3">
        {rows.map((row, i) => (
          <div key={i} className="grid grid-cols-1 sm:grid-cols-3 gap-3 border border-gray-100 rounded-lg p-3">
            <label className="block text-xs">
              <span className="flex items-center justify-between text-gray-600">
                Model
                <button
                  type="button"
                  className={`text-xs text-gray-400 hover:text-red-600 px-2 py-1 ${atMin ? "invisible" : ""}`}
                  title="Remove model"
                  onClick={() => onRemove(i)}
                >
                  ✕
                </button>
              </span>
              <input
                list="infer-models"
                className="mt-1 w-full text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white"
                value={row.model_id}
                onChange={(e) => onPatch(i, { model_id: e.target.value })}
              />
            </label>
            <label className="block text-xs">
              <span className="text-gray-600">GPU</span>
              {gpus.length > 0 ? (
                <select
                  className="mt-1 w-full text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white"
                  value={row.cfg?.gpu_sku || ""}
                  onChange={(e) => onPatch(i, { gpu_sku: e.target.value })}
                >
                  <option value="">Select GPU</option>
                  {gpus.map((g) => <option key={skuOf(g)} value={skuOf(g)}>{gpuOptionLabel(g)}</option>)}
                </select>
              ) : (
                <input
                  className="mt-1 w-full text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white"
                  value={row.cfg?.gpu_sku || ""}
                  onChange={(e) => onPatch(i, { gpu_sku: e.target.value })}
                />
              )}
            </label>
            <label className="block text-xs">
              <span className="text-gray-600">EP</span>
              <input
                type="number"
                min={1}
                className="mt-1 w-full text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white"
                value={row.cfg?.parallel?.ep_moe ?? ""}
                onChange={(e) => onPatch(i, { ep_moe: axis(Number(e.target.value)) })}
              />
            </label>
          </div>
        ))}
      </div>
      <datalist id="infer-models">
        {models.map((m) => <option key={m} value={m} />)}
      </datalist>
      <div className="mt-6 flex flex-wrap gap-3 items-center">
        <button
          type="button"
          id="ic-run"
          className="px-5 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-lg font-medium transition"
          onClick={onRun}
        >
          Compare
        </button>
        <span id="ic-status" className={statusClass}>{status}</span>
        <div className="flex-1" />
        <label className="text-xs text-gray-500 whitespace-nowrap" htmlFor="infer-to-single">Open one in single view:</label>
        <select
          id="infer-to-single"
          className="max-w-full text-xs px-2 py-1.5 border border-gray-300 rounded-lg bg-white text-gray-700 hover:bg-gray-50"
          title="Pick any model to deep-dive in the single-model view."
          defaultValue=""
          onChange={(e) => {
            const v = e.target.value;
            if (v) onDemote(v);
            e.target.value = "";
          }}
        >
          <option value="">‹ Back to single…</option>
          {rows.map((r, i) => {
            const m = r.model_id.trim();
            return m ? <option key={`${i}-${m}`} value={m}>{m}</option> : null;
          })}
        </select>
      </div>
    </section>
    <div id="ic-results" className={`${hits.length > 0 ? "" : "hidden "}fade-in space-y-6`}>
      {hits.length > 0 && labeled.common.length > 0 && (
        <div id="ic-common-caption" className="mt-4 text-xs px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg">
          <span className="text-gray-400">All groups:</span>{" "}
          {labeled.common.map((t, i) => (
            <span key={`${t}-${i}`}>
              {i > 0 ? <span className="text-gray-300"> · </span> : null}
              <span className="font-mono text-gray-600">{t}</span>
            </span>
          ))}
        </div>
      )}
      <CompareSummary hits={hits} labels={labeled.labels} />
      <CompareTradeoff hits={hits} labels={labeled.labels} source={icSource} onSource={setIcSource} />
      {hits.length > 0 && (
        <ul className="mt-4 space-y-2 text-sm">
          {hits.map((h, i) => (
            <li key={`${h.body.model_id}-${i}`} className="border border-gray-100 rounded-lg px-3 py-2">
              <div className="font-mono text-xs text-gray-700">
                {labeled.labels[i] || `#${i + 1}`}
              </div>
              {h.error ? (
                <div className="text-red-500 text-xs mt-1">{h.error}</div>
              ) : (
                <DataTable
                  framed={false}
                  className="mt-2"
                  rowKey={(row) => row.key}
                  rows={SWEEP_BATCH_GRID}
                  columns={[
                    {
                      key: "batch",
                      header: "batch",
                      className: "font-mono text-gray-500",
                      render: (row) => row.key,
                      sortValue: null,
                    },
                    ...(h.body.seq_lens || []).map(
                      (seq, si): ColumnDef<(typeof SWEEP_BATCH_GRID)[number]> => ({
                        key: String(seq),
                        header: String(seq),
                        align: "right",
                        className: "font-mono text-gray-800",
                        render: (row) => {
                          const r = h.result?.results_by_batch?.[row.key]?.[si];
                          return formatNumber(r?.max_decode_batch_per_gpu);
                        },
                        sortValue: null,
                      }),
                    ),
                  ]}
                />
              )}
            </li>
          ))}
        </ul>
      )}
      <CompareCharts hits={hits} labels={labeled.labels} source={icSource} phase={phase} onPhase={onPhase} />
    </div>
    </div>
  );
}
