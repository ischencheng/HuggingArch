import type { MyCalibration } from "../auth";
import { SPEC_PRESETS, type SpecDraftStyle } from "../compare";
import { escapeHtml } from "../format";
import { gpuOptionLabel, intraNodeLink, skuOf, type GpuItem } from "../gpu";
import { axis, type Parallel, type ParallelShape } from "../parallel";
import { PrecisionPresets } from "../PrecisionPresets";
import { tipHandlers } from "../Tip";
import type { DeploymentAssumption } from "./ResultSummary";

/** What the speculative-decoding knobs price: K draft tokens per step, which
 *  the target verifies in one forward, amortised by the acceptance rate. */
const SPEC_DECODE_TIP =
  "K draft tokens per step; the target verifies them in one forward over K + 1 positions, " +
  "amortised by acceptance rate α (1 + α·K tokens/step). " +
  "Draft compute and weight reads aren't counted — tick \"Count draft cost\" to price this model's MTP stack as the draft.";

/** What "Count draft cost" adds, and the two ways a draft can run. */
const SPEC_DRAFT_TIP =
  "Adds the draft's forwards to each decode step, priced from this model's MTP stack (weights, KV/state reads, activations) " +
  "and amortised per accepted token like the verify forward. " +
  "EAGLE (recursive): K draft forwards of 1 token each. DFlash (parallel): one draft forward over K positions.";

export type { GpuItem };

/** GPU memory fraction rungs. Plan is the one control; Scaling / KV read it. */
export const GPU_MEMORY_FRACS = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5] as const;

function snapFrac(f: number): number {
  let best: number = GPU_MEMORY_FRACS[0];
  for (const x of GPU_MEMORY_FRACS) {
    if (Math.abs(x - f) < Math.abs(best - f)) best = x;
  }
  return best;
}

function field(
  label: string,
  value: number,
  onChange: (n: number) => void,
  min = 1,
  id?: string,
) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
      <input
        id={id}
        type="number"
        min={min}
        className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
        value={value}
        onChange={(e) => onChange(axis(Number(e.target.value)))}
      />
    </div>
  );
}

function pickField(
  label: string,
  value: number,
  options: number[],
  onChange: (n: number) => void,
  id?: string,
) {
  const shown = options.includes(value) ? value : (options[0] ?? value);
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
      <select
        id={id}
        className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
        value={shown}
        onChange={(e) => onChange(axis(Number(e.target.value)))}
      >
        {options.map((d) => (
          <option key={d} value={d}>{d}</option>
        ))}
      </select>
    </div>
  );
}

// A control the backend says has no legal value: shown, disabled, with the
// backend's reason — never a free field the user could fill with a value the
// estimate will refuse.
function unavailableField(label: string, reason: string, id?: string) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
      <select
        id={id}
        disabled
        title={reason}
        className="w-full px-2 py-1.5 text-sm border border-gray-200 rounded bg-gray-50 text-gray-400"
        value=""
      >
        <option value="">unavailable</option>
      </select>
      <div data-unavailable-reason className="mt-1 text-11 text-gray-400">{reason}</div>
    </div>
  );
}

function interconnect(g?: GpuItem): string {
  if (!g) return "";
  const inter = g.rdma_gbps ? `RDMA ${g.rdma_gbps} GB/s`
    : (g.pcie_gbps ? `PCIe ${g.pcie_gbps} GB/s` : "not set");
  return `intra-node ${intraNodeLink(g)} · inter-node ${inter} · ${g.gpus_per_node || "?"} GPUs/node`;
}

export function PlanCard(props: {
  models: string[];
  modelId: string;
  onModel: (v: string) => void;
  gpus: GpuItem[];
  gpuSku: string;
  onGpu: (v: string) => void;
  /** The serving framework whose fusion and kernels the estimate runs under;
   *  "" = the server's default. */
  framework: string;
  onFramework: (v: string) => void;
  /** framework → its label, "vllm (audited at v0.30.0)". */
  frameworks: Record<string, string>;
  /** The framework an estimate runs under when the request names none, as
   *  `/fusion/registry` states it; "" until the registry has been read. */
  defaultFramework: string;
  /** The paged-KV page the user chose, as a string; "" = not chosen, which is
   *  the framework binding's own default and sends no `kv_block_size`. */
  kvBlock: string;
  onKvBlock: (v: string) => void;
  /** The page the unchosen control resolves to — the declared default of the
   *  framework in force. `null` when that binding declares no page, which is
   *  not a substituted 16. */
  kvBlockDefault: number | null;
  /** The pages some registered backend declares it can serve, from the
   *  backend; this component never names a page itself. */
  kvBlockServed: number[];
  /** Kernel times: measured where the corpus has the framework's kernel, or SOL only. */
  measured: boolean;
  onMeasured: (v: boolean) => void;
  /** The caller's own measured rows, as the server calibrates with them. */
  calibration: Calibration;
  /** The precision preset id every request from the page carries. */
  precision: string;
  onPrecision: (presetId: string) => void;
  /** Why a link's precision settings were not applied; null when none were. */
  precisionNotice: string | null;
  prefixLen: number;
  onPrefixLen: (v: number) => void;
  prefillChunk: number;
  onPrefillChunk: (v: number) => void;
  /** The decode batch the estimate asks for, per GPU, as typed: "" = the
   *  card's KV-bound max (the backend's default). */
  decBatch: string;
  onDecBatch: (v: string) => void;
  specMode: string;
  onSpecMode: (v: string) => void;
  specK: number;
  onSpecK: (v: number) => void;
  specAlpha: number;
  onSpecAlpha: (v: number) => void;
  includeAux: boolean;
  onIncludeAux: (v: boolean) => void;
  tablesOnHost: boolean;
  onTablesOnHost: (v: boolean) => void;
  /** The last estimate's host-offloadable bytes per GPU; 0 = this model
   *  declares none, undefined = not estimated yet. */
  lookupTableBytes?: number;
  specDraft: SpecDraftStyle | "";
  onSpecDraft: (v: SpecDraftStyle | "") => void;
  draftAvailable: boolean;
  draftUnavailableReason: string | null;
  inTok: number;
  onInTok: (v: number) => void;
  outTok: number;
  onOutTok: (v: number) => void;
  frac: number;
  onFrac: (v: number) => void;
  /** The two time floors, µs: null until the server's defaults arrive. */
  commLatencyUs: number | null;
  onCommLatencyUs: (v: number) => void;
  launchFloorUs: number | null;
  onLaunchFloorUs: (v: number) => void;
  par: Parallel;
  /** The backend's shape of `par`: GPU count and what makes it invalid. */
  parShape: ParallelShape | null;
  setAxis: (k: keyof Parallel, n: number) => void;
  tieTp: boolean;
  onTieTp: (v: boolean) => void;
  cpScheme: string;
  onCpScheme: (v: string) => void;
  legalCpDegrees: number[] | null;
  cpUnavailableReason: string | null;
  /** Set when CP > 1 was reset to 1 because the scheme has no legal degree. */
  cpResetNotice: string | null;
  /** Whether the model routes tokens to experts — the estimate's `moe_enabled`. */
  routes: boolean;
  /** ρ = max/mean expert load (`ParallelConfig.ep_load_factor`) as typed. */
  epLoadText: string;
  onEpLoadText: (v: string) => void;
  /** Why the typed ρ is not a valid one; null when it is. */
  epLoadError: string | null;
  /** The on-screen estimate's assumption entries whose knob is `ep_load_factor`
   *  (one per deployment that made it). */
  routingAssumptions: DeploymentAssumption[];
  onEstimate: () => void;
  onCompare: () => void;
  status: string;
  statusClass: string;
  decodeOn: boolean;
  onDecodeOn: (v: boolean) => void;
  decPar: Parallel;
  decShape: ParallelShape | null;
  setDecAxis: (k: keyof Parallel, n: number) => void;
  kvBw: string;
  onKvBw: (v: string) => void;
}) {
  const preGpus = props.parShape?.cluster_size;
  // Disaggregated: the decode cluster is separate GPUs beside the prefill one.
  const decGpus = props.decodeOn ? props.decShape?.cluster_size : undefined;
  const gpusTotal = preGpus != null && decGpus != null ? preGpus + decGpus : preGpus;
  const gpu = props.gpus.find((g) => g.sku === props.gpuSku);
  const byArch: Record<string, GpuItem[]> = {};
  props.gpus.forEach((g) => {
    const a = g.arch || "Other";
    (byArch[a] ||= []).push(g);
  });
  const archOrder = ["Ampere", "Ada", "Hopper", "Blackwell"];
  const arches = [
    ...archOrder.filter((a) => byArch[a]),
    ...Object.keys(byArch).filter((a) => !archOrder.includes(a)).sort(),
  ];

  const setTp = (n: number) => {
    if (props.tieTp) {
      props.setAxis("tp_attention", n);
      props.setAxis("tp_dense_ffn", n);
      props.setAxis("tp_moe_expert", n);
    } else {
      props.setAxis("tp_attention", n);
    }
  };

  return (
    <section id="infer-plan-card" className="card p-4 sm:p-6 mb-6">
      <h2 className="text-xl font-semibold mb-4">Plan Inference</h2>
      <div className="mb-4">
        <label htmlFor="infer-plan-model" className="block text-sm font-medium text-gray-700 mb-1">Model</label>
        <input
          id="infer-plan-model"
          list="infer-models"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-400 focus:border-amber-500"
          value={props.modelId}
          onChange={(e) => props.onModel(e.target.value)}
          placeholder="e.g., deepseek-ai/DeepSeek-V3"
        />
        <datalist id="infer-models">
          {props.models.map((m) => <option key={m} value={m} />)}
        </datalist>
      </div>
      <div>
        <label htmlFor="infer-gpu" className="block text-sm font-medium text-gray-700 mb-1">GPU</label>
        <select
          id="infer-gpu"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-400 focus:border-amber-500 bg-white"
          value={props.gpuSku}
          onChange={(e) => props.onGpu(e.target.value)}
        >
          <option value="">Select GPU</option>
          {arches.map((a) => (
            <optgroup key={a} label={a}>
              {(byArch[a] || []).map((g) => {
                const sku = skuOf(g);
                return <option key={sku} value={sku}>{gpuOptionLabel(g)}</option>;
              })}
            </optgroup>
          ))}
        </select>
        <div id="infer-gpu-interconnect" className="mt-1 text-11 text-gray-500 leading-relaxed">{interconnect(gpu)}</div>
      </div>

      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <div>
          <label htmlFor="infer-framework" className="block text-sm font-medium text-gray-700 mb-1">Framework</label>
          <select
            id="infer-framework"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-400 focus:border-amber-500 bg-white"
            value={props.framework}
            onChange={(e) => props.onFramework(e.target.value)}
          >
            <option value="">{
              props.frameworks[props.defaultFramework]
                ? `default — ${props.frameworks[props.defaultFramework]}`
                : "default"
            }</option>
            {Object.keys(props.frameworks).sort().map((n) => (
              <option key={n} value={n}>{props.frameworks[n]}</option>
            ))}
          </select>
          <div className="mt-1 text-11 text-gray-500">Which kernels are fused, and whose measured times are read.</div>
        </div>
        <div>
          <label htmlFor="infer-kv-block" className="block text-sm font-medium text-gray-700 mb-1">KV page size</label>
          <select
            id="infer-kv-block"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-400 focus:border-amber-500 bg-white"
            value={props.kvBlock}
            onChange={(e) => props.onKvBlock(e.target.value)}
          >
            <option value="">{
              props.kvBlockDefault == null
                ? "default — this framework declares none"
                : `default — ${props.kvBlockDefault} tokens`
            }</option>
            {props.kvBlockServed.map((n) => (
              <option key={n} value={String(n)}>{n} tokens</option>
            ))}
          </select>
          <div id="infer-kv-block-note" className="mt-1 text-11 text-gray-500">
            Tokens per paged-KV block. Part of every cache-reading kernel&apos;s
            measurement key, so a page nothing measured finds no row.
          </div>
        </div>
        <div>
          <label htmlFor="infer-kernel-times" className="block text-sm font-medium text-gray-700 mb-1">Kernel times</label>
          <select
            id="infer-kernel-times"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-400 focus:border-amber-500 bg-white"
            value={props.measured ? "measured" : "sol"}
            onChange={(e) => props.onMeasured(e.target.value === "measured")}
          >
            <option value="measured">Measured (totals only when every op is measured)</option>
            <option value="sol">SOL only</option>
          </select>
          <CalibrationNote calibration={props.calibration} />
        </div>
      </div>

      <div className="mt-4">
        <PrecisionPresets
          idPrefix="infer"
          value={props.precision}
          onChange={props.onPrecision}
          notice={props.precisionNotice}
        />
      </div>

      <details className="mt-4 border border-gray-200 rounded-lg px-4 py-3 bg-gray-50">
        <summary className="text-sm font-medium text-gray-700 cursor-pointer select-none">
          Workload <span className="text-xs text-gray-400 ml-2">(prefix caching · chunked prefill · speculative decoding)</span>
        </summary>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 mt-3">
          <div>
            <label htmlFor="infer-prefix-len" className="block text-xs font-medium text-gray-600 mb-1">Prefix cached tokens</label>
            <input id="infer-prefix-len" type="number" min={0} step={128} className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
              value={props.prefixLen} onChange={(e) => props.onPrefixLen(Math.max(0, Number(e.target.value) || 0))} />
          </div>
          <div>
            <label htmlFor="infer-prefill-chunk" className="block text-xs font-medium text-gray-600 mb-1" title="Tokens one prefill forward carries. Caps the activation peak at min(input, chunk) so max prefill batch plateaus past the chunk instead of decaying with sequence length. 0 = full-sequence prefill.">Chunked prefill size</label>
            <input id="infer-prefill-chunk" type="number" min={0} step={256} className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
              value={props.prefillChunk} onChange={(e) => props.onPrefillChunk(Math.max(0, Math.round(Number(e.target.value) || 0)))} />
            <div className="mt-1 text-xs text-gray-500">vLLM max_num_batched_tokens · 0 = full seq</div>
          </div>
          <div>
            <label htmlFor="infer-decode-batch" className="block text-xs font-medium text-gray-600 mb-1" title="Requests one decode step carries, per GPU. The per-op breakdown prices this batch; blank = the KV-bound maximum this card fits.">Decode batch</label>
            <input id="infer-decode-batch" type="number" min={1} step={1} placeholder="max" className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
              value={props.decBatch} onChange={(e) => props.onDecBatch(e.target.value)} />
            <div className="mt-1 text-xs text-gray-500">per GPU · blank = max</div>
          </div>
          <div>
            <label htmlFor="infer-spec-mode" className="block text-xs font-medium text-gray-600 mb-1" title={SPEC_DECODE_TIP}>Speculative mode</label>
            <select id="infer-spec-mode" title={SPEC_DECODE_TIP} className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={props.specMode} onChange={(e) => props.onSpecMode(e.target.value)}>
              {Object.entries(SPEC_PRESETS).map(([v, p]) => <option key={v} value={v}>{p.label}</option>)}
            </select>
          </div>
          {props.specMode === "custom" && (
            <div id="infer-spec-custom">
              <label htmlFor="infer-spec-K" className="block text-xs font-medium text-gray-600 mb-1" title={SPEC_DECODE_TIP}>Draft tokens K · α</label>
              <div className="flex gap-1">
                <input id="infer-spec-K" type="number" min={0} title="Draft tokens per step (K); the target verifies K + 1 positions" className="w-1/2 px-2 py-1.5 text-sm border border-gray-300 rounded" value={props.specK} onChange={(e) => props.onSpecK(Math.max(0, Math.round(Number(e.target.value)) || 0))} />
                <input id="infer-spec-alpha" aria-label="Draft acceptance rate α" type="number" min={0.05} max={1} step={0.05} className="w-1/2 px-2 py-1.5 text-sm border border-gray-300 rounded" value={props.specAlpha} onChange={(e) => props.onSpecAlpha(Number(e.target.value) || 0.65)} />
              </div>
            </div>
          )}
          <div id="infer-spec-draft" className="sm:col-span-2 md:col-span-3 text-xs text-gray-500" title={props.draftAvailable ? SPEC_DRAFT_TIP : (props.draftUnavailableReason ?? "")}>
            <div className="flex items-center gap-1.5 flex-wrap">
              <input
                id="infer-spec-draft-on"
                type="checkbox"
                className="accent-amber-500"
                disabled={!props.draftAvailable}
                checked={props.draftAvailable && props.specDraft !== ""}
                onChange={(e) => props.onSpecDraft(e.target.checked ? "recursive" : "")}
              />
              <label htmlFor="infer-spec-draft-on" className={props.draftAvailable ? "" : "text-gray-400"}>Count draft cost (MTP stack)</label>
              {props.draftAvailable && props.specDraft !== "" && (
                <select
                  id="infer-spec-draft-style"
                  aria-label="Draft style"
                  className="ml-2 px-2 py-0.5 text-xs border border-gray-300 rounded bg-white"
                  value={props.specDraft}
                  onChange={(e) => props.onSpecDraft(e.target.value as SpecDraftStyle)}
                >
                  <option value="recursive">Recursive · K forwards × 1 token (EAGLE, vLLM MTP)</option>
                  <option value="parallel">Parallel · 1 forward × K positions (DFlash)</option>
                </select>
              )}
            </div>
            {!props.draftAvailable && props.draftUnavailableReason && (
              <div id="infer-spec-draft-reason" className="mt-0.5 text-11 text-gray-400">
                Unavailable: {props.draftUnavailableReason}
              </div>
            )}
          </div>
          <div className="sm:col-span-2 md:col-span-3 text-xs text-gray-500 flex items-center gap-1.5">
            <input id="infer-include-aux" type="checkbox" className="accent-amber-500" checked={props.includeAux} onChange={(e) => props.onIncludeAux(e.target.checked)} />
            <label htmlFor="infer-include-aux">Include aux module weights (MTP)</label>
          </div>
          <div className="sm:col-span-2 md:col-span-3 text-xs text-gray-500 flex items-center gap-1.5">
            <input id="infer-tables-on-host" type="checkbox" className="accent-amber-500"
              disabled={props.lookupTableBytes === 0}
              checked={props.tablesOnHost} onChange={(e) => props.onTablesOnHost(e.target.checked)} />
            <label htmlFor="infer-tables-on-host" className={props.lookupTableBytes === 0 ? "text-gray-400" : ""}>
              Engram / PLE n-gram tables in CPU memory
              {props.lookupTableBytes === 0
                ? " — this model has none"
                : " — vLLM's default; off keeps them on the GPU"}
            </label>
          </div>
        </div>
      </details>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mt-4">
        <div>
          <label htmlFor="infer-avg-in" className="block text-sm font-medium text-gray-700 mb-1">Avg Input tokens</label>
          {/* A request has at least one prompt token (the link reads it as one too). */}
          <input id="infer-avg-in" type="number" min={1} step={128} className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            value={props.inTok} onChange={(e) => props.onInTok(axis(Number(e.target.value)))} />
        </div>
        <div>
          <label htmlFor="infer-avg-out" className="block text-sm font-medium text-gray-700 mb-1">Avg Output tokens</label>
          <input id="infer-avg-out" type="number" min={0} step={128} className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            value={props.outTok} onChange={(e) => props.onOutTok(Math.max(0, Number(e.target.value) || 0))} />
        </div>
        <div>
          <label htmlFor="infer-frac" className="block text-sm font-medium text-gray-700 mb-1">GPU memory fraction</label>
          <select
            id="infer-frac"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg bg-white text-gray-800"
            value={snapFrac(props.frac)}
            onChange={(e) => props.onFrac(Number(e.target.value))}
          >
            {GPU_MEMORY_FRACS.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="infer-comm-latency" className="block text-sm font-medium text-gray-700 mb-1"
            title="One link traversal's latency, the same on every link. A collective pays it once per hop its schedule waits on: an all-reduce of N ranks 2·log2 N, a reduce-scatter / all-gather / all-to-all log2 N, a send 1.">
            Link latency (µs)
          </label>
          <input id="infer-comm-latency" type="number" min={0} step={0.5} className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            value={props.commLatencyUs ?? ""} onChange={(e) => props.onCommLatencyUs(Math.max(0, Number(e.target.value) || 0))} />
        </div>
        <div>
          <label htmlFor="infer-launch-floor" className="block text-sm font-medium text-gray-700 mb-1"
            title="The shortest one kernel launch takes, however little it does: a step of tiny kernels is launch-bound.">
            Kernel launch floor (µs)
          </label>
          <input id="infer-launch-floor" type="number" min={0} step={0.5} className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            value={props.launchFloorUs ?? ""} onChange={(e) => props.onLaunchFloorUs(Math.max(0, Number(e.target.value) || 0))} />
        </div>
        <div>
          <label htmlFor="infer-ngpu" className="block text-sm font-medium text-gray-700 mb-1">
            Total GPUs <span className="text-11 text-gray-400 ml-1">{decGpus != null
              ? `(prefill ${preGpus ?? "—"} + decode ${decGpus})`
              : "(derived from parallelism)"}</span>
          </label>
          <input id="infer-ngpu" type="number" readOnly tabIndex={-1} value={gpusTotal ?? ""}
            className="w-full px-4 py-2 border border-gray-200 rounded-lg bg-gray-50 text-gray-700 focus:outline-none" />
        </div>
      </div>

      <details className="mt-4 border border-gray-200 rounded-lg px-4 py-3 bg-gray-50" open>
        <summary className="text-sm font-medium text-gray-700 cursor-pointer select-none flex items-center justify-between">
          <span>Parallelism <span className="text-xs text-gray-400 ml-2">(TP / PP / EP / CP / DP — default all 1)</span></span>
          <span id="infer-cluster-size" className="text-xs text-amber-700 font-mono">{decGpus != null ? "prefill cluster" : "cluster"} = {preGpus ?? "—"}</span>
        </summary>
        {(props.parShape?.errors || []).length > 0 && (
          <ul id="infer-par-errors" role="alert" className="mt-2 text-xs text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1">
            {(props.parShape?.errors || []).map((e) => <li key={e}>{e}</li>)}
          </ul>
        )}
        <label className="inline-flex items-center gap-2 mt-3 text-xs text-gray-600 cursor-pointer select-none">
          <input id="infer-tie-tp" type="checkbox" className="accent-amber-500" checked={props.tieTp} onChange={(e) => props.onTieTp(e.target.checked)} />
          <span>Tie TP dims <span className="text-gray-400">(sync tp_attention / tp_dense_ffn / tp_moe_expert together)</span></span>
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          {field("TP attention", props.par.tp_attention, setTp, 1, "infer-par-tp-attn")}
          {field("TP dense FFN", props.par.tp_dense_ffn, (n) => props.tieTp ? setTp(n) : props.setAxis("tp_dense_ffn", n), 1, "infer-par-tp-ffn")}
          {field("TP MoE expert", props.par.tp_moe_expert, (n) => props.tieTp ? setTp(n) : props.setAxis("tp_moe_expert", n), 1, "infer-par-tp-moe")}
          {field("EP MoE", props.par.ep_moe, (n) => props.setAxis("ep_moe", n), 1, "infer-par-ep")}
          {props.legalCpDegrees === null
            ? field("CP (seq)", props.par.cp, (n) => props.setAxis("cp", n), 1, "infer-par-cp")
            : props.legalCpDegrees.length > 0
              ? pickField("CP (seq)", props.par.cp, props.legalCpDegrees, (n) => props.setAxis("cp", n), "infer-par-cp")
              : unavailableField("CP (seq)", props.cpUnavailableReason ?? "", "infer-par-cp")}
          <div>
            <label htmlFor="infer-par-cp-scheme" className="block text-xs font-medium text-gray-600 mb-1">CP scheme</label>
            <select id="infer-par-cp-scheme" className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded" value={props.cpScheme} onChange={(e) => props.onCpScheme(e.target.value)}>
              <option value="pcp">pcp</option>
              <option value="layersplit">layersplit</option>
              <option value="ulysses">ulysses (not in vLLM / sglang)</option>
            </select>
          </div>
          {field("PP", props.par.pp, (n) => props.setAxis("pp", n), 1, "infer-par-pp")}
          {field("DP attention", props.par.dp_attention, (n) => props.setAxis("dp_attention", n), 1, "infer-par-dp-attn")}
          {field("DP replicas", props.par.dp_replicas, (n) => props.setAxis("dp_replicas", n), 1, "infer-par-dp-rep")}
        </div>
        {(props.routes || props.epLoadError) && (
          <div id="infer-par-ep-load" className="mt-3 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
            <div className="sm:col-span-2 md:col-span-4">
              <div className="flex items-center gap-1.5 mb-1">
                <label htmlFor="infer-par-ep-load-factor" className="text-xs font-medium text-gray-600">
                  MoE load imbalance ρ = max/mean expert tokens
                </label>
                {props.routingAssumptions.map((a) => (
                  <span
                    key={a.id}
                    role="note"
                    aria-label={a.text}
                    className="infer-par-ep-load-assumption inline-flex items-center gap-1 px-1.5 py-px rounded-full border border-amber-200 bg-amber-50 text-10 font-medium text-amber-700"
                    {...tipHandlers(`<div class="font-sans whitespace-normal">${escapeHtml(a.text)}</div>`)}
                  >
                    {a.effect}
                    <svg aria-hidden="true" className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <circle cx="12" cy="12" r="9" />
                      <path strokeLinecap="round" d="M12 11v5M12 8h.01" />
                    </svg>
                  </span>
                ))}
              </div>
              <input
                id="infer-par-ep-load-factor"
                type="number"
                min={1}
                step={0.05}
                className="w-full md:w-1/4 px-2 py-1.5 text-sm border border-gray-300 rounded"
                aria-invalid={props.epLoadError ? true : undefined}
                value={props.epLoadText}
                onChange={(e) => props.onEpLoadText(e.target.value)}
              />
              {props.epLoadError && (
                <div id="infer-par-ep-load-error" role="alert" className="mt-2 text-xs text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1">
                  {props.epLoadError}
                </div>
              )}
            </div>
          </div>
        )}
        {props.cpResetNotice && (
          <div id="infer-par-cp-reset" role="status" className="mt-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1">
            {props.cpResetNotice}
          </div>
        )}
      </details>

      <details className="mt-3 border border-gray-200 rounded-lg px-4 py-3 bg-gray-50">
        <summary className="text-sm font-medium text-gray-700 cursor-pointer select-none flex items-center justify-between">
          <span>Decode deployment <span className="text-xs text-gray-400 ml-2">(disaggregated — separate parallelism for decode)</span></span>
          <span id="infer-decode-cluster" className="text-xs text-amber-700 font-mono">
            {props.decodeOn ? `decode cluster = ${props.decShape?.cluster_size ?? "—"}` : ""}
          </span>
        </summary>
        <label className="inline-flex items-center gap-2 mt-3 text-xs text-gray-600 cursor-pointer select-none">
          <input
            id="infer-decode-decouple"
            type="checkbox"
            className="accent-amber-500"
            checked={props.decodeOn}
            onChange={(e) => props.onDecodeOn(e.target.checked)}
          />
          <span>Decouple decode from prefill <span className="text-gray-400">(off = single deployment; decode uses the config above)</span></span>
        </label>
        {props.decodeOn && (
          <div id="infer-decode-grid" className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 mt-3">
            {field("TP attention", props.decPar.tp_attention, (n) => props.setDecAxis("tp_attention", n), 1, "infer-par-tp-attn-d")}
            {field("TP dense FFN", props.decPar.tp_dense_ffn, (n) => props.setDecAxis("tp_dense_ffn", n), 1, "infer-par-tp-ffn-d")}
            {field("TP MoE expert", props.decPar.tp_moe_expert, (n) => props.setDecAxis("tp_moe_expert", n), 1, "infer-par-tp-moe-d")}
            {field("EP MoE", props.decPar.ep_moe, (n) => props.setDecAxis("ep_moe", n), 1, "infer-par-ep-d")}
            {field("CP (DCP — KV shard)", props.decPar.cp, (n) => props.setDecAxis("cp", n), 1, "infer-par-cp-d")}
            {field("PP", props.decPar.pp, (n) => props.setDecAxis("pp", n), 1, "infer-par-pp-d")}
            {field("DP attention", props.decPar.dp_attention, (n) => props.setDecAxis("dp_attention", n), 1, "infer-par-dp-attn-d")}
            {field("DP replicas", props.decPar.dp_replicas, (n) => props.setDecAxis("dp_replicas", n), 1, "infer-par-dp-rep-d")}
            <div>
              <label htmlFor="infer-kv-transfer-bw" className="block text-xs font-medium text-gray-600 mb-1">
                KV link BW <span className="text-gray-400">(GB/s)</span>
              </label>
              <input
                id="infer-kv-transfer-bw"
                type="number"
                min={0}
                step="any"
                placeholder="GPU RDMA"
                title="Prefill→decode KV handoff bandwidth. Blank = decode GPU cross-node RDMA."
                className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
                value={props.kvBw}
                onChange={(e) => props.onKvBw(e.target.value)}
              />
            </div>
          </div>
        )}
      </details>

      <div className="mt-4 flex gap-3 items-center">
        <button type="button" data-action="run-estimate" className="px-5 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-lg font-medium transition" onClick={props.onEstimate}>
          Estimate
        </button>
        <span id="infer-status" className={props.statusClass}>{props.status}</span>
        <div className="flex-1" />
        <button type="button" id="infer-to-sweep" className="text-xs px-3 py-1.5 border border-gray-300 rounded-lg text-gray-700 hover:bg-amber-50 hover:border-amber-300" title="Add a second model and switch to cross-model compare." onClick={props.onCompare}>
          + Compare with another model
        </button>
      </div>
    </section>
  );
}


const count = (n: number) => n.toLocaleString("en-US");

/** The caller's own measured rows, as the page knows them: not read yet
 *  (`loading`), signed out (`off`), their calibration corpus (`on`:
 *  estimates ask the server to calibrate with it), or why it cannot be read
 *  now (`unavailable`). */
export type Calibration =
  | { state: "loading" }
  | { state: "off" }
  | { state: "on"; info: MyCalibration }
  | { state: "unavailable"; why: string };

/** How many of the caller's own rows calibrate the estimate, and how many
 *  the current validator refuses and why — never a silent drop. */
function CalibrationNote({ calibration }: { calibration: Calibration }) {
  const manage = (
    <a href="/account/workspace.html#benchmarks" className="underline hover:text-gray-700">manage measured data</a>
  );
  if (calibration.state === "loading") {
    return <div id="infer-calibration" className="mt-1 text-11 text-gray-500">{manage}.</div>;
  }
  if (calibration.state === "unavailable") {
    return (
      <div id="infer-calibration" className="mt-1 text-11 text-amber-700">
        Your own measurements cannot calibrate estimates right now: {calibration.why} — {manage}.
      </div>
    );
  }
  if (calibration.state === "off") {
    return (
      <div id="infer-calibration" className="mt-1 text-11 text-gray-500">
        Sign in to calibrate with your own measurements — {manage}.
      </div>
    );
  }
  const { info } = calibration;
  const rejected = info.rows_rejected;
  return (
    <div id="infer-calibration" className="mt-1 text-11 text-gray-500">
      {info.rows_used === 0 && rejected === 0
        ? <>You have no measurements of your own yet — {manage}.</>
        : <>
            {count(info.rows_used)} of your rows calibrate this estimate
            {rejected > 0 ? <>; <span className="text-amber-700">{count(rejected)} rejected</span></> : null}
            {" "}— {manage}.
          </>}
      {rejected > 0 && info.rejected_reasons.length > 0 && (
        <details className="mt-0.5">
          <summary className="cursor-pointer">Why rejected</summary>
          <ul className="list-disc pl-4">
            {info.rejected_reasons.map((r) => (
              <li key={r.reason}>{count(r.rows)}: {r.reason}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
