import { createContext, useContext, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ELK from "elkjs/lib/elk.bundled.js";
import type { ElkExtendedEdge, ElkNode } from "elkjs";
import { escapeHtml, formatBytes, formatCount, formatGiB, formatSI } from "../format";
import { tipHandlers } from "../Tip";
import { placeLabels, type EdgeText, type Rect } from "../dagLabels";
import { revealSideways } from "../revealSideways";

/** Inspector block-graph node — fields vanilla dag.js `buildDagNodeInner` reads. */
// `src_port` names one of several outputs of `src` — a distinct edge. `via`
// names the node inside `src` the edge leaves from (and, inside `dst`, the
// input that reads it); it only steers an opened compound and is not part of
// the edge's identity, so two `via` edges between closed boxes draw once.
// `tensor` / `shape` / `port_label` describe the tensor on the edge (drawn as
// the edge's label); `kind` is its edge kind (a non-activation edge — mask,
// state, cache — draws in its kind's style, `EDGE_STYLE`). `tone` marks an
// edge only some compared entries have: it draws dashed in that entry's colour
// (`ENTRY_PALETTE`).
export type BlockEdge = {
  src?: string; dst?: string; id?: string; src_port?: string; via?: string;
  tensor?: string; shape?: string; port_label?: string; kind?: string; tone?: number;
};
type EdgeLabel = { text: string; kind?: string; tone?: number } | null;

// One colour per compared entry, in entry order — the same hues as the
// `s-diff-u<k>` node styles in index.css.
export const ENTRY_PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7"];
const TONE_DASH = "6 4";

// How each edge kind draws. Activation is the plain line; a cache edge (its
// source is KV-cache data) and a mask edge (an index list) must not look alike.
const EDGE_STYLE: Record<string, { stroke: string; width: number; dash?: string; legend: string }> = {
  activation: { stroke: "#8a99ad", width: 1.4, legend: "activation" },
  cache: { stroke: "#0d9488", width: 2.4, dash: "9 3", legend: "KV cache edge" },
  mask: { stroke: "#7c3aed", width: 1.4, dash: "4 3", legend: "mask / index edge" },
  state: { stroke: "#db2777", width: 1.4, dash: "2 3", legend: "state edge" },
};
const edgeStyle = (kind?: string) => EDGE_STYLE[kind || "activation"] || EDGE_STYLE.activation;

function edgeLabel(e: BlockEdge, srcNode?: BlockNode): EdgeLabel {
  const kind = e.kind && e.kind !== "activation" ? e.kind : undefined;
  // A view node (reshape, slice) shows its in → out shapes itself.
  if (srcNode?.style === "view") return { text: "", kind, tone: e.tone };
  const body = `${e.tensor || ""}${e.shape ? ` ${e.shape}` : ""}`.trim();
  const text = e.port_label ? (body ? `${e.port_label}: ${body}` : e.port_label) : body;
  return { text, kind, tone: e.tone };
}

// KV cache marks, read off the backend's `kv` (backend/rendering/kv_marks.py).
// Nothing here computes a byte count: the per-token, per-step and per-request
// figures and the read law all arrive on the node.
const bytes = (n: number | null | undefined) => (n == null ? "—" : formatGiB(n));

function kvReadText(kv: KVMark): string {
  // Each law with the bytes per token read under it; reads under different
  // laws (a DSA indexer beside its selected rows) are a sum, not one figure.
  if (kv.read_terms.length) {
    return kv.read_terms.map((t) => `${bytes(t.bytes_per_token)}/tok × ${t.law}`).join(" + ");
  }
  // A graph drawn for the model (no cache length) has no per-step figure.
  if (kv.read_bytes_per_step == null) return `${bytes(kv.read_bytes_per_token)}/tok`;
  return `${bytes(kv.read_bytes_per_step)}/step`;
}

/** ` → X/request` (and `, Y/step`) where the graph was drawn at a cache
 *  length; nothing where it was drawn for the model. */
function kvAt(e: KVEntry, step: boolean): string {
  if (e.bytes_per_request == null) return "";
  return ` → ${bytes(e.bytes_per_request)}/request${step ? `, ${bytes(e.bytes_per_step)}/step` : ""}`;
}

function kvTitle(kv: KVMark): string {
  const lines = [kv.cache_len == null
    ? "KV cache — per rank, per layer"
    : `KV cache — per rank, per layer, at s_kv=${kv.cache_len}, batch ${kv.batch}`];
  const tag = (e: KVEntry) => [e.tensor, e.display || e.family, e.dtype].filter(Boolean).join(" · ");
  if (kv.writes.length) lines.push("writes:");
  for (const e of kv.writes) {
    lines.push(e.recurrent
      ? `  ${tag(e)} — recurrent state ${bytes(e.bytes_per_request)}/request`
      : `  ${tag(e)} — ${bytes(e.bytes_per_position)}/token, stored ${e.stored}${kvAt(e, false)}`);
  }
  if (kv.reads.length) lines.push("reads:");
  for (const e of kv.reads) {
    lines.push(e.recurrent
      ? `  ${tag(e)} — recurrent state ${bytes(e.bytes_per_request)}/request${e.bytes_per_step == null ? "" : `, ${bytes(e.bytes_per_step)}/step`}`
      : `  ${tag(e)} — ${bytes(e.bytes_per_position)}/token × ${e.read}${kvAt(e, true)}`);
  }
  return lines.join("\n");
}

function KVBadges({ kv }: { kv?: KVMark }) {
  if (!kv) return null;
  const stateRW = kv.writes.some((e) => e.recurrent) && kv.reads.some((e) => e.recurrent);
  return (
    <div className="dag-kv-line" title={kvTitle(kv)}>
      {kv.write_bytes_per_token != null ? (
        <span className="dag-kv-badge kv-write" data-kv="write">KV write {bytes(kv.write_bytes_per_token)}/tok</span>
      ) : null}
      {kv.read_bytes_per_token != null ? (
        <span className="dag-kv-badge kv-read" data-kv="read">KV read {kvReadText(kv)}</span>
      ) : null}
      {kv.state_bytes_per_request != null ? (
        <span className="dag-kv-badge kv-state" data-kv="state">state {stateRW ? "R/W" : "R"} {bytes(kv.state_bytes_per_request)}/req</span>
      ) : null}
    </div>
  );
}

// A box around the node, by how it touches the cache: solid for a writer,
// dashed for a reader, doubled for a recurrent state.
function kvClass(kv?: KVMark): string {
  if (!kv) return "";
  if (kv.state_bytes_per_request != null) return " kv-s";
  if (kv.write_bytes_per_token != null) return " kv-w";
  return kv.read_bytes_per_token != null || kv.read_bytes_per_step != null ? " kv-r" : "";
}

// The legend says what the drawing uses, read off the payload: the node
// styles that need saying (fused kernels, collectives), the KV marks, the
// non-activation edge kinds.
const STYLE_LEGEND: Record<string, string> = { fused: "fused kernel", comm: "collective" };

export function DagLegend({ nodes, edges }: { nodes: BlockNode[]; edges: BlockEdge[] }) {
  const styles = new Set<string>();
  const kinds = new Set<string>();
  let write = false, read = false, state = false;
  (function walk(ns: BlockNode[], es: BlockEdge[]) {
    for (const e of es) if (e.kind && e.kind !== "activation") kinds.add(e.kind);
    for (const n of ns) {
      if (n.style && STYLE_LEGEND[n.style]) styles.add(n.style);
      for (const kv of [n.kv, ...(n.compare || []).map((r) => r.kv)]) {
        if (kv?.write_bytes_per_token != null) write = true;
        if (kv?.read_bytes_per_token != null) read = true;
        if (kv?.state_bytes_per_request != null) state = true;
      }
      walk(n.children || [], n.child_edges || []);
    }
  })(nodes, edges);
  if (!styles.size && !kinds.size && !write && !read && !state) return null;
  return (
    <div className="dag-legend" data-legend>
      {[...styles].map((s) => (
        <span key={s} className="dag-legend-item" data-legend-item={`style:${s}`}>
          <span className={`dag-legend-swatch dag-node-box s-${s}`} />{STYLE_LEGEND[s]}
        </span>
      ))}
      {write ? <span className="dag-legend-item" data-legend-item="kv:write"><span className="dag-kv-badge kv-write">KV write</span>writes the cache (bytes per token)</span> : null}
      {read ? <span className="dag-legend-item" data-legend-item="kv:read"><span className="dag-kv-badge kv-read">KV read</span>reads the cache (bytes per token × read length)</span> : null}
      {state ? <span className="dag-legend-item" data-legend-item="kv:state"><span className="dag-kv-badge kv-state">state</span>recurrent state (bytes per request)</span> : null}
      {[...kinds].map((k) => {
        const st = edgeStyle(k);
        return (
          <span key={k} className="dag-legend-item" data-legend-item={`edge:${k}`}>
            <svg width="28" height="8" aria-hidden><line x1="0" y1="4" x2="28" y2="4" stroke={st.stroke} strokeWidth={st.width} strokeDasharray={st.dash} /></svg>
            {st.legend}
          </span>
        );
      })}
    </div>
  );
}

/** One cached tensor a node writes or reads (backend/rendering/kv_marks.py).
 *  Bytes are per rank and per layer; `bytes_per_position` is one cached
 *  token's row (null for a recurrent state, which has no positions), and the
 *  per-request / per-step figures are the tensor's storage (`stored`) or read
 *  (`read`) law at the graph's cache length and batch. */
export type KVEntry = {
  ref: string;
  tensor: string;
  family?: string | null;
  display?: string | null;
  dtype: string;
  recurrent: boolean;
  bytes_per_position: number | null;
  // null on a graph drawn for the model, for a growing cache (its size is a
  // law at a length); a recurrent state keeps its fixed size.
  bytes_per_request: number | null;
  stored?: string;
  read?: string;
  bytes_per_step?: number | null;
  cache_len: number | null;
  batch: number | null;
};

/** A node's KV cache marks: what it writes and reads, with the backend's sums.
 *  A compound carries the union of its descendants'. */
export type KVMark = {
  writes: KVEntry[];
  reads: KVEntry[];
  write_bytes_per_token: number | null;
  state_bytes_per_request: number | null;
  read_bytes_per_token: number | null;
  read_terms: Array<{ law: string; bytes_per_token: number }>;
  read_bytes_per_step: number | null;
  cache_len: number | null;
  batch: number | null;
};

export type AbsorbedParam = {
  label?: string;
  tensor_out?: string;
  detail?: string;
  weight_dtype?: string;
  params?: number;
};

export type BlockNode = {
  id?: string;
  // The inner streams this node's output is published as.
  stream_writes?: string[];
  op?: string;
  label?: string;
  style?: string;
  expandable?: boolean;
  l1?: { available?: boolean; reason?: string | null; [k: string]: unknown };
  badge?: string;
  subtitle?: string;
  op_formula?: string;
  op_formula_derived?: string;
  detail?: string;
  detail_full?: string;
  tensor_in?: string;
  tensor_out?: string;
  tensor_in_full?: string;
  tensor_out_full?: string;
  tensor_in_symbolic?: string;
  tensor_out_symbolic?: string;
  has_bias?: boolean;
  weight_dtype?: string;
  weight_dtype_bytes?: number;
  weight_bytes?: number;
  flops?: number;
  params?: number;
  out_bytes?: number;
  access_bytes?: number;
  arithmetic_intensity?: number;
  flops_symbolic?: string;
  flops_values?: string;
  params_symbolic?: string;
  params_values?: string;
  absorbed_params?: AbsorbedParam[];
  children?: BlockNode[];
  inputs?: string[];
  input_ports?: string[];
  output_port?: string;
  // A block's own input node: the position of the block_stack's `inputs`
  // entry bound to it.
  formal?: number;
  child_edges?: BlockEdge[];
  is_source?: boolean;
  // The serving graph's fused kernels name the fusion they stand for.
  fused?: string;
  /** Set only on a Fused graph node carrying its runtime-table row. */
  priced_runtime?: boolean;
  kv?: KVMark;
  // A compare view's per-entry lines: one row per entry, with that entry's
  // colour swatch (`s-diff-u<entry>`) and, where it touches the cache, its
  // own KV mark.
  compare?: CompareRow[];
  // A compound drawn always open, with no collapse control (alternatives
  // placed side by side).
  group?: boolean;
  // Layout band (ELK partition): a lower band is laid out above a higher one.
  partition?: number;
  // Hover card for the node, as HTML for the shared tip. Plain text goes
  // through `textHtml`.
  hoverHtml?: string;
};

// `entry` omitted: a line true of every entry, drawn without a swatch.
export type CompareRow = { entry?: number; text: string; kv?: KVMark };

export type BlockGraphEntry = {
  nodes?: BlockNode[];
  edges?: BlockEdge[];
  template?: string;
  params?: Record<string, string>;
  l1?: { available?: boolean; reason?: string | null; [k: string]: unknown };
};

const elk = new ELK();

// Root + per-compound ELK options. NETWORK_SIMPLEX placement + connected-
// component compaction + aspectRatio keep the DAG tall/narrow and tight;
// applied at every nesting level so an expanded compound's inner graph stays
// compact instead of fanning wide and blowing up the outer layout.
const ROOT_OPTS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.spacing.nodeNodeBetweenLayers": "40",
  "elk.spacing.nodeNode": "24",
  "elk.hierarchyHandling": "INCLUDE_CHILDREN",
  "elk.aspectRatio": "0.7",
  "elk.layered.compaction.connectedComponents": "true",
  "elk.layered.compaction.postCompaction.strategy": "LEFT",
  "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
};

const COMPOUND_OPTS: Record<string, string> = {
  "elk.padding": "[top=22,left=12,bottom=12,right=12]",
  "elk.layered.spacing.nodeNodeBetweenLayers": "36",
  "elk.spacing.nodeNode": "20",
  "elk.aspectRatio": "0.8",
  "elk.layered.compaction.connectedComponents": "true",
  "elk.layered.compaction.postCompaction.strategy": "LEFT",
  "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
};

// A compound's title tab: its name and the Structure / Detailed switch.
const COMPOUND_TAB_PX = 320;
// Past this a layout no longer fits the panel it is shown in.
const WIDE_LAYOUT_PX = 960;
// The second placement bends straight chains, so it has to earn that.
const NARROWER_BY = 0.9;

function withPlacement(node: ElkNode, strategy: string): ElkNode {
  const key = "elk.layered.nodePlacement.strategy";
  const options = node.layoutOptions as Record<string, string> | undefined;
  return {
    ...node,
    layoutOptions: options && key in options ? { ...options, [key]: strategy } : options,
    children: node.children?.map((c) => withPlacement(c, strategy)),
  };
}

/** NETWORK_SIMPLEX shortens every edge, which keeps a chain straight but lets a
 *  long run of fan-out / fan-in stages drift sideways stage by stage, leaving
 *  the top of the graph far from the bottom. A layout that comes out wide is
 *  laid out again with linear segments, and the narrower one is kept. */
async function layoutCompact(graph: ElkNode): Promise<ElkNode> {
  const first = await elk.layout(structuredClone(graph));
  if ((first.width || 0) <= WIDE_LAYOUT_PX) return first;
  const second = await elk.layout(structuredClone(withPlacement(graph, "LINEAR_SEGMENTS")));
  return (second.width || 0) < (first.width || 0) * NARROWER_BY ? second : first;
}

function hasAbsorbed(n: BlockNode): boolean {
  return (n.absorbed_params || []).length > 0;
}

function isResidualAdd(n: BlockNode): boolean {
  return n.op === "add" && !hasAbsorbed(n);
}

function styleClass(n: BlockNode): string {
  // A compare view's style still reaches the input pill and the residual
  // circle, on top of their own shape.
  const diff = n.style?.startsWith("diff-") ? ` s-${n.style}` : "";
  if (isResidualAdd(n)) return `s-res${diff}`;
  if (n.op === "input") return `s-io${diff}`;
  if (n.op === "add" && hasAbsorbed(n)) return "s-op";
  return `s-${n.style || "op"}`;
}

function indexParents(nodes: BlockNode[], parentOf: Map<string, string>, acc: BlockNode[] = []): BlockNode[] {
  for (const n of nodes) {
    if (n.id) acc.push(n);
    for (const c of n.children || []) {
      if (n.id && c.id) parentOf.set(c.id, n.id);
    }
    indexParents(n.children || [], parentOf, acc);
  }
  return acc;
}

function nodeSize(n: BlockNode, open: boolean): { width: number; height: number } {
  if (isResidualAdd(n) && !open) return { width: 36, height: 36 };
  if (n.op === "input" && !open) return { width: 88, height: 32 };
  let h = 44;
  if (n.subtitle || (n.op && n.op !== n.label && n.op !== "input")) h += 16;
  if (n.op_formula) h += 16;
  if (n.detail) h += 16;
  if (n.has_bias || n.weight_dtype) h += 16;
  if (hasAbsorbed(n)) h += 18;
  if (n.kv) h += 18;
  for (const r of n.compare || []) h += r.kv ? 34 : 16;
  return { width: 200, height: Math.min(h, n.compare?.length ? 360 : 160) };
}

/** Plain text as one-line-per-`<br/>` HTML for a node's `hoverHtml`. */
export function textHtml(text: string): string {
  return escapeHtml(text).replaceAll("\n", "<br/>");
}

function NodeInner({ n, open }: { n: BlockNode; open: boolean }) {
  if (isResidualAdd(n)) {
    return <div className={`dag-node-box ${styleClass(n)}`} title={n.label && n.label !== "add" ? n.label : undefined}>+</div>;
  }
  if (n.op === "input") {
    return <div className={`dag-node-box ${styleClass(n)}`} {...tipHandlers(n.hoverHtml)}>input</div>;
  }
  const canOpen = !!(n.expandable || (n.children || []).length);
  const hasMultiInSig = !!(n.tensor_in && n.tensor_in.includes("×") && n.tensor_out);
  return (
    <div className={`dag-node-box ${styleClass(n)} ${canOpen ? "expandable" : ""}${kvClass(n.kv)}`} {...tipHandlers(n.hoverHtml)}>
      <div className="dag-node-label" title={n.label}>
        {n.label}
        {n.badge ? <span className="dag-badge">{n.badge}</span> : null}
      </div>
      {n.subtitle ? (
        <div className="dag-node-detail" style={{ opacity: 0.6, fontStyle: "italic" }}>{n.subtitle}</div>
      ) : n.op && n.op !== "input" && n.op !== n.label ? (
        <div className="dag-node-detail" style={{ opacity: 0.55, fontStyle: "italic" }}>{n.op}</div>
      ) : null}
      {n.op_formula ? <div className="dag-op-formula">{n.op_formula}</div> : null}
      {n.op_formula_derived ? (
        <div className="dag-op-formula" style={{ opacity: 0.55, fontSize: 10, whiteSpace: "pre-line" }}>{n.op_formula_derived}</div>
      ) : null}
      {n.detail && !hasMultiInSig ? (
        <div className="dag-node-detail" title={n.detail_full ? `full (unsharded): ${n.detail_full}` : undefined}>{n.detail}</div>
      ) : null}
      {n.priced_runtime && (n.flops != null || n.access_bytes != null) ? (
        <div className="dag-node-detail" data-fused-resource>
          {n.params ? <>{formatCount(n.params)} params · </> : null}
          {n.flops != null ? <>{formatSI(n.flops, "FLOPs")} · </> : null}
          {n.access_bytes != null ? formatBytes(n.access_bytes) : null}
        </div>
      ) : null}
      {n.style === "view" && (n.tensor_in || n.tensor_out) ? (
        <div className="dag-view-shape">
          <span className="dag-shape-token">{n.tensor_in || "?"}</span>
          <span className="arrow">→</span>
          <span className="dag-shape-token">{n.tensor_out || "?"}</span>
        </div>
      ) : hasMultiInSig ? (
        <div className="dag-view-shape">
          <span className="dag-shape-token">{n.tensor_in}</span>
          <span className="arrow">→</span>
          <span className="dag-shape-token">{n.tensor_out}</span>
        </div>
      ) : null}
      {n.has_bias || n.weight_dtype ? (
        <div className="dag-node-detail dag-dtype-line">
          {n.has_bias ? <span className="dag-dtype-badge">+bias</span> : null}
          {n.weight_dtype ? <span className="dag-dtype-badge">W:{n.weight_dtype}</span> : null}
        </div>
      ) : null}
      {hasAbsorbed(n) ? (
        <div className="dag-absorbed-params">
          <span className="dag-absorbed-label">weights</span>
          {(n.absorbed_params || []).map((p, i) => {
            const shape = p.tensor_out
              ? p.tensor_out
              : (p.detail ? `[${p.detail.replace(/^dim=/, "")}]` : "");
            return (
              <span key={i} className="dag-absorbed-item">
                {p.label}{shape ? <span className="dag-absorbed-shape"> {shape}</span> : null}
                {p.weight_dtype ? <span className="dag-dtype-badge">W:{p.weight_dtype}</span> : null}
              </span>
            );
          })}
        </div>
      ) : null}
      <KVBadges kv={n.kv} />
      {n.stream_writes && n.stream_writes.length ? (
        <div className="dag-stream-line" data-stream-writes={n.stream_writes.join(" ")}>▲ writes {n.stream_writes.join(", ")}</div>
      ) : null}
      {(n.compare || []).map((r, i) => (
        <div key={i} className="dag-compare-row" data-compare-entry={r.entry}>
          {r.entry != null ? <span className={`dag-compare-swatch dag-node-box s-diff-u${r.entry % ENTRY_PALETTE.length}`} /> : null}
          <span className="dag-compare-text">{r.text}</span>
          <KVBadges kv={r.kv} />
        </div>
      ))}
      {canOpen ? <span className="expand-icon">{open ? "−" : "+"}</span> : null}
    </div>
  );
}

function FallbackList({ nodes, edges }: { nodes: BlockNode[]; edges: BlockEdge[] }) {
  return (
    <div className="flex flex-col items-stretch gap-2 p-4">
      <p className="text-11 text-amber-700">ELK layout failed; falling back to a stacked list with edges listed.</p>
      {nodes.map((n, i) => (
        <div key={n.id || i} className="text-xs text-slate-700 font-mono">
          {n.label || n.op || n.id}
        </div>
      ))}
      <ul className="mt-2 text-10 font-mono text-slate-500 space-y-0.5">
        {edges.map((e, i) => <li key={e.id || i}>{e.src} → {e.dst}</li>)}
      </ul>
    </div>
  );
}

export type InnerL1Mode = Record<string, "structure" | "detailed">;

/** Inner L1/L2 for expanded compounds. Provided by StructureGraph. */
export const ScopeL1Context = createContext<{
  innerMode: InnerL1Mode;
  setInnerMode: (id: string, mode: "structure" | "detailed") => void;
  apply: (nodes: BlockNode[]) => BlockNode[];
} | null>(null);

/** vanilla `renderDag` / `renderElkDag` for one compare-block column. */
/** `legend={false}` leaves the legend to the caller, which places
 *  `DagLegend` itself (the compare view folds it with its own legend). */
export function BlockElkDag({ nodes, edges, legend = true, fit = false }: {
  nodes: BlockNode[];
  edges: BlockEdge[];
  legend?: boolean;
  /** Size the canvas to the layout; skip the 280×240 floor used on full-page graphs. */
  fit?: boolean;
}) {
  const uid = useId().replace(/:/g, "");
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const scopeL1 = useContext(ScopeL1Context);
  const displayNodes = useMemo(
    () => (scopeL1 ? scopeL1.apply(nodes) : nodes),
    [nodes, scopeL1],
  );
  const [laid, setLaid] = useState<ElkNode | null>(null);
  const [labels, setLabels] = useState<Map<string, EdgeLabel>>(() => new Map());
  const [err, setErr] = useState("");

  const byId = useMemo(() => {
    const m = new Map<string, BlockNode>();
    indexParents(displayNodes, new Map()).forEach((n) => { if (n.id) m.set(n.id, n); });
    return m;
  }, [displayNodes]);
  const allNodes = useMemo(() => Array.from(byId.values()), [byId]);

  const [sizes, setSizes] = useState<Map<string, { width: number; height: number }>>(new Map());
  const measureRef = useRef<HTMLDivElement | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const focusRef = useRef<string | null>(null);

  // Measure the real collapsed size of every node off-screen (same CSS) so ELK
  // allocates slots that match what actually renders. The static nodeSize
  // heuristic under-counts multi-line op_formula_derived and wrapped text,
  // which packed layers too tightly and let a tall box overlap the node below.
  // Mirrors vanilla dag.js's off-screen measurement pass.
  useLayoutEffect(() => {
    const el = measureRef.current;
    if (!el) return;
    const m = new Map<string, { width: number; height: number }>();
    el.querySelectorAll<HTMLElement>("[data-measure-id]").forEach((wrap) => {
      const id = wrap.getAttribute("data-measure-id");
      if (!id) return;
      const box = (wrap.firstElementChild as HTMLElement | null) || wrap;
      const r = box.getBoundingClientRect();
      m.set(id, { width: Math.max(Math.ceil(r.width), 60), height: Math.max(Math.ceil(r.height), 36) });
    });
    setSizes((prev) => {
      if (prev.size === m.size && [...m].every(([k, v]) => prev.get(k)?.width === v.width && prev.get(k)?.height === v.height)) return prev;
      return m;
    });
  }, [allNodes]);

  useEffect(() => {
    if (!displayNodes.length) {
      setLaid(null);
      setErr("");
      return;
    }
    let cancelled = false;

    // Whole-tree index so endpoint lookups resolve at any nesting depth.
    const idOf = new Map<string, BlockNode>();
    (function idx(ns: BlockNode[]) {
      for (const n of ns) { if (n.id) idOf.set(n.id, n); if (n.children) idx(n.children); }
    })(displayNodes);

    // Root edges reference top-level sibling ids only — resolve in root scope
    // (a global suffix search would jump into a compound and break ELK).
    const topLevelIds = new Set(displayNodes.map((n) => n.id));
    const resolveInp = (inp?: string) => (inp && topLevelIds.has(inp) ? inp : null);

    // `is_source` params with no in-scope consumer are orphan weight banks;
    // ELK parks them in a trailing layer (LAST_SEPARATE) out of the flow.
    const orphanIds = new Set<string>();
    (function markOrphans(scope: BlockNode[]) {
      const idSet = new Set(scope.map((n) => n.id));
      const consumed = new Set<string>();
      for (const n of scope) for (const inp of n.inputs || []) {
        if (idSet.has(inp)) consumed.add(inp);
        else { const hit = scope.find((s) => s.id && s.id.endsWith("." + inp)); if (hit?.id) consumed.add(hit.id); }
      }
      for (const n of scope) {
        if (n.is_source && n.id && !consumed.has(n.id)) orphanIds.add(n.id);
        if (n.children) markOrphans(n.children);
      }
    })(displayNodes);

    // A `group` compound is always open; any other opens when clicked.
    const isExpandedNow = (id: string) => {
      const n = idOf.get(id);
      return !!(n && (n.group || (n.expandable && expanded.has(id))) && (n.children?.length || 0) > 0);
    };

    // Entry children = no in-scope sibling producer; exits = not consumed by a
    // sibling. Derived from the compound's child_edges (backend's port-aware
    // sibling-scope truth), cached per parent.
    const entryExitCache = new Map<string, { entries: BlockNode[]; exits: BlockNode[]; resolveChild: (q: string) => string | null }>();
    const entryExitOf = (parent: BlockNode) => {
      const cached = parent.id ? entryExitCache.get(parent.id) : undefined;
      if (cached) return cached;
      const kids = parent.children || [];
      const childIds = kids.map((c) => c.id).filter((x): x is string => !!x);
      const resolveChild = (q: string) => childIds.find((k) => k === q || k.endsWith("." + q)) || null;
      const used = new Set<string>();
      const consumedAsDst = new Set<string>();
      (parent.child_edges || []).forEach((e) => {
        const rs = e.src ? resolveChild(e.src) : null;
        const rd = e.dst ? resolveChild(e.dst) : null;
        if (rs) used.add(rs);
        if (rd) consumedAsDst.add(rd);
      });
      const entries = kids.filter((c) => !c.is_source && c.id && !consumedAsDst.has(c.id));
      const exits = kids.filter((c) => !c.is_source && c.id && !used.has(c.id));
      const out = { entries, exits, resolveChild };
      if (parent.id) entryExitCache.set(parent.id, out);
      return out;
    };

    // When an outer edge endpoint is an expanded compound, fan src into its
    // exit children / dst into its entry children (recursive, port-aware).
    const expandSrc = (id: string, srcPort?: string): string[] => {
      if (!isExpandedNow(id)) return [id];
      const { exits } = entryExitOf(idOf.get(id)!);
      if (srcPort) {
        // A port names its producer, which need not be an exit: a tensor can
        // leave the compound and also feed a sibling inside it.
        const ported = (idOf.get(id)!.children || []).filter((c) => c.output_port === srcPort);
        if (ported.length) return ported.flatMap((c) => expandSrc(c.id!, srcPort));
      }
      if (!exits.length) return [id];
      return exits.flatMap((c) => expandSrc(c.id!));
    };
    const expandDst = (id: string, src?: string, srcPort?: string): string[] => {
      if (!isExpandedNow(id)) return [id];
      const wrapper = idOf.get(id)!;
      if (src) {
        // A block names which of its input nodes each outer input binds to.
        const at = (wrapper.inputs || []).flatMap((inp, i) => (inp === src ? [i] : []));
        const bound = (wrapper.children || []).filter((c) => c.formal != null && at.includes(c.formal));
        if (bound.length) return bound.flatMap((c) => expandDst(c.id!, src, srcPort));
        const targeted = (wrapper.children || []).filter((c) => {
          const inps = c.inputs || [];
          const ports = c.input_ports || [];
          return inps.some((inp, i) => {
            const port = ports[i] || "";
            const matches = inp === src || inp.endsWith("." + src) || (!!port && inp === `${src}.${port}`);
            if (!matches) return false;
            if (srcPort) return port === srcPort;
            return !port || inp === src;
          });
        });
        if (targeted.length) return targeted.flatMap((c) => expandDst(c.id!, src, srcPort));
      }
      const { entries } = entryExitOf(wrapper);
      if (!entries.length) return [id];
      return entries.flatMap((c) => expandDst(c.id!, src, srcPort));
    };

    // Global (src,dst[,port]) dedup: outer-edge fan-out and a compound's own
    // child_edges can emit the same cross-scope pair; ELK needs only one.
    const seen = new Set<string>();
    // Label per drawn edge. One drawn edge can stand for several (closed
    // compounds, deduped pairs); when they carry different tensors no single
    // label is true, so it draws none.
    const labelOf = new Map<string, EdgeLabel>();
    const pushRewired = (out: ElkExtendedEdge[], src: string, dst: string, srcPort?: string, via?: string,
                         label: EdgeLabel = null) => {
      // A named output port is the edge's own identity and also says which
      // child it leaves from; `via` steers only when no port is named. An
      // edge leaving an opened L1 box names its member (`via`) and that
      // member's port: the box opens onto the member, the port onto the
      // member's own child when it is opened too.
      const steer = srcPort ?? via;
      const member = via && isExpandedNow(src)
        ? (idOf.get(src)!.children || []).find((c) => c.id === via) : undefined;
      const sources = member ? expandSrc(member.id!, srcPort) : expandSrc(src, steer);
      // An opened destination box names the member that reads the source
      // node (`via`), as its rewired input port.
      for (const s of sources) for (const d of expandDst(dst, src, via ?? steer)) {
        if (!s || !d || s === d) continue;
        // Distinct ports stay distinct edges (DeepSeek-V4's hc_*_pre hands
        // its post and comb outputs to one consumer) unless the port has
        // already chosen which child of an opened source `s` is — or the
        // edge still leaves a closed L1 box, which draws one edge per box
        // pair.
        const opened = member ?? idOf.get(src);
        const portChosen = !!srcPort && ((via && s === src)
          || (s !== opened?.id
              && (opened?.children || []).some((c) => c.output_port === srcPort)));
        const key = `${s}->${d}${srcPort && !portChosen ? `:${srcPort}` : ""}`;
        if (seen.has(key)) {
          // One drawn edge standing for several: a label or a kind only
          // survives when all of them agree on it.
          const had = labelOf.get(key);
          if (had && (!label || had.text !== label.text || had.kind !== label.kind || had.tone !== label.tone)) {
            const kind = label && had.kind === label.kind ? had.kind : undefined;
            const tone = label && had.tone === label.tone ? had.tone : undefined;
            labelOf.set(key, kind || tone != null ? { text: "", kind, tone } : null);
          }
          continue;
        }
        labelOf.set(key, label);
        seen.add(key);
        out.push({ id: key, sources: [s], targets: [d] });
      }
    };

    // is_source param consumed by exactly one wrapper → pulled inside it.
    const computeAbsorptions = (scope: BlockNode[]) => {
      const absorbedBy = new Map<string, string>();
      const importedTo = new Map<string, string[]>();
      for (const param of scope) {
        if (!param.is_source || !param.id) continue;
        const wrappers = scope.filter((w) => w.expandable && (w.child_edges || []).some((e) => e.src === param.id));
        if (wrappers.length !== 1 || !wrappers[0].id) continue;
        const wid = wrappers[0].id;
        absorbedBy.set(param.id, wid);
        const list = importedTo.get(wid) || [];
        list.push(param.id);
        importedTo.set(wid, list);
      }
      return { absorbedBy, importedTo };
    };

    const withConstraint = (opts: Record<string, string> | undefined, n: BlockNode) => {
      let o = opts;
      if (n.id && orphanIds.has(n.id)) o = { ...(o || {}), "elk.layered.layering.layerConstraint": "LAST_SEPARATE" };
      if (n.op === "input") o = { ...(o || {}), "elk.layered.layering.layerConstraint": "FIRST_SEPARATE" };
      if (n.op === "stream_out") o = { ...(o || {}), "elk.layered.layering.layerConstraint": "LAST_SEPARATE" };
      if (n.partition != null) o = { ...(o || {}), "elk.partitioning.partition": String(n.partition) };
      return o;
    };

    const buildElkChild = (n: BlockNode, importedNodes: BlockNode[] = []): ElkNode => {
      const id = n.id || "";
      const isExpanded = isExpandedNow(id);
      const node: ElkNode = { id };
      if (isExpanded) {
        const { importedTo } = computeAbsorptions(n.children || []);
        const ownChildren = (n.children || []).map((c) =>
          buildElkChild(c, (importedTo.get(c.id || "") || [])
            .map((pid) => (n.children || []).find((s) => s.id === pid))
            .filter((x): x is BlockNode => !!x)));
        const imported = importedNodes.map((p) => buildElkChild(p));
        node.children = [...imported, ...ownChildren];
        node.edges = [];
        const { resolveChild } = entryExitOf(n);
        (n.child_edges || []).forEach((e) => {
          if (!e.src || !e.dst) return;
          pushRewired(node.edges!, resolveChild(e.src) || e.src, resolveChild(e.dst) || e.dst, e.src_port, e.via,
            edgeLabel(e, idOf.get(resolveChild(e.src) || e.src)));
        });
        node.layoutOptions = { ...COMPOUND_OPTS };
      } else {
        const { width, height } = sizes.get(id) || nodeSize(n, false);
        node.width = width;
        node.height = height;
      }
      const opts = withConstraint(node.layoutOptions as Record<string, string> | undefined, n);
      if (opts) node.layoutOptions = opts;
      return node;
    };

    const { absorbedBy: rootAbsorbed, importedTo: rootImported } = computeAbsorptions(displayNodes);
    const elkChildren = displayNodes
      .filter((n) => n.id && !rootAbsorbed.has(n.id))
      .map((n) => buildElkChild(n, (rootImported.get(n.id!) || [])
        .map((pid) => displayNodes.find((s) => s.id === pid))
        .filter((x): x is BlockNode => !!x)));
    const elkEdges: ElkExtendedEdge[] = [];
    edges.forEach((e) => {
      if (!e.src || !e.dst) return;
      const src = resolveInp(e.src) || e.src;
      const dst = resolveInp(e.dst) || e.dst;
      if (rootAbsorbed.has(src)) return;
      pushRewired(elkEdges, src, dst, e.src_port, e.via, edgeLabel(e, idOf.get(src)));
    });

    // Bands only when some node asks for one; every other page lays out as before.
    const banded = [...idOf.values()].some((n) => n.partition != null);
    const graph: ElkNode = {
      id: "root", layoutOptions: { ...ROOT_OPTS, ...(banded ? { "elk.partitioning.activate": "true" } : {}) },
      children: elkChildren, edges: elkEdges,
    };

    // ELK rejects a FIRST_SEPARATE node with an incoming edge; expanding a
    // compound can rewire an outer edge into a block's input alias. Strip the
    // constraint from exactly those nodes rather than let ELK throw.
    const incoming = new Set<string>();
    (function collectTargets(kids?: ElkNode[], es?: ElkExtendedEdge[]) {
      (es || []).forEach((e) => (e.targets || []).forEach((t) => incoming.add(t)));
      (kids || []).forEach((k) => collectTargets(k.children, (k.edges as ElkExtendedEdge[]) || []));
    })(graph.children, graph.edges as ElkExtendedEdge[]);
    (function relax(kids?: ElkNode[]) {
      (kids || []).forEach((k) => {
        const lo = k.layoutOptions as Record<string, string> | undefined;
        if (lo && lo["elk.layered.layering.layerConstraint"] === "FIRST_SEPARATE" && k.id && incoming.has(k.id)) {
          delete lo["elk.layered.layering.layerConstraint"];
        }
        relax(k.children);
      });
    })(graph.children);

    setErr("");
    layoutCompact(graph).then((result) => {
      if (!cancelled) { setLabels(labelOf); setLaid(result); }
    }).catch((e) => {
      if (!cancelled) {
        setLaid(null);
        setErr(e instanceof Error ? e.message : String(e));
      }
    });
    return () => { cancelled = true; };
  }, [edges, expanded, displayNodes, sizes]);

  // Bring what was just opened into view once its layout has landed.
  useLayoutEffect(() => {
    const id = focusRef.current;
    const panel = scrollerRef.current;
    if (!id || !panel) return;
    const opened = panel.querySelector<HTMLElement>(`[data-compound="${CSS.escape(id)}"]`);
    if (!opened) return;
    focusRef.current = null;
    revealSideways(opened);
  }, [laid]);

  const toggle = (id: string) => {
    focusRef.current = expanded.has(id) ? null : id;
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    const n = byId.get(id);
    if (n?.l1 !== undefined && n.id && scopeL1 && scopeL1.innerMode[n.id] == null) {
      scopeL1.setInnerMode(n.id, n.l1.available ? "structure" : "detailed");
    }
  };

  if (!displayNodes.length) return null;

  // Off-screen twin used only for measurement (see the useLayoutEffect above).
  // Kept in the tree on every non-empty render so sizes are known before the
  // first paint of the real layout.
  const measureEl = (
    <div
      ref={measureRef}
      aria-hidden
      className="dag-svg-container"
      style={{ position: "fixed", left: -10000, top: 0, visibility: "hidden", pointerEvents: "none" }}
    >
      {allNodes.map((n) => (
        <div key={n.id} data-measure-id={n.id} style={{ display: "inline-block", margin: 8 }}>
          <NodeInner n={n} open={false} />
        </div>
      ))}
    </div>
  );

  if (err) return <FallbackList nodes={nodes} edges={edges} />;
  if (!laid) return <>{measureEl}<p className="p-4 text-xs text-slate-500">ELK layout…</p></>;

  const w = Math.max(laid.width || 0, fit ? 1 : 280);
  const h = Math.max(laid.height || 0, fit ? 1 : 240);
  const markerId = `dag-arrow-${uid}`;

  // Expanded compound → labeled dashed container (tab at top-left) with its
  // children rendered as absolute siblings in root coords; collapsed / leaf →
  // a slot box carrying the full node card.
  const renderNode = (c: ElkNode, ox: number, oy: number): ReactNode => {
    const n = byId.get(c.id);
    if (!n) return null;
    const x = ox + (c.x || 0);
    const y = oy + (c.y || 0);
    if (c.children && c.children.length) {
      return (
        <div key={c.id}>
          <div className={`dag-compound-wrap expanded${n.group ? " dag-group" : ""}`} data-compound={c.id} style={{ position: "absolute", left: x, top: y, width: c.width, height: c.height }}>
            <div
              className={`dag-elk-collapse-tab dag-node-box ${n.group ? "" : "expandable "}${styleClass(n)}`}
              style={{ position: "absolute", left: 8, top: -12 }}
              data-group={n.group ? "1" : undefined}
              {...tipHandlers(n.hoverHtml)}
              onClick={() => !n.group && n.id && toggle(n.id)}
            >
              <span className="dag-node-label">
                {n.label}
                {n.subtitle && n.subtitle !== n.label ? <span style={{ opacity: 0.55, fontStyle: "italic" }}> {n.subtitle}</span> : null}
              </span>
              {n.l1 !== undefined && n.id && scopeL1 ? (
                <span
                  className="inline-flex shrink-0 whitespace-nowrap ml-1 rounded overflow-hidden text-10 border border-current/30"
                  title={n.l1.available ? undefined : (n.l1.reason || "no structure view")}
                  onClick={(e) => e.stopPropagation()}
                >
                  {(["structure", "detailed"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      disabled={m === "structure" && !n.l1?.available}
                      className={`px-1 ${(scopeL1.innerMode[n.id!] ?? (n.l1?.available ? "structure" : "detailed")) === m ? "bg-slate-900/10" : "opacity-60"} disabled:opacity-30`}
                      onClick={() => scopeL1.setInnerMode(n.id!, m)}
                    >
                      {m === "structure" ? "Structure" : "Detailed"}
                    </button>
                  ))}
                </span>
              ) : null}
              {n.group ? null : <span className="expand-icon" style={{ position: "static", width: 16, height: 16, fontSize: 11, flexShrink: 0 }}>−</span>}
            </div>
          </div>
          {c.children.map((cc) => renderNode(cc, x, y))}
        </div>
      );
    }
    const canOpen = !!(n.expandable || (n.children || []).length);
    return (
      <div key={c.id}>
        <div
          className="dag-node-slot absolute"
          data-node={c.id}
          data-openable={canOpen ? "1" : undefined}
          style={{ left: x, top: y, width: c.width, height: c.height }}
          onClick={() => canOpen && n.id && toggle(n.id)}
        >
          <NodeInner n={n} open={false} />
        </div>
      </div>
    );
  };

  // Edges live at the depth they're defined; ELK returns their sections in
  // container-relative coords, so translate each by its container's absolute
  // origin into root space for the single overlay SVG.
  const edgePaths: { key: string; d: string; edge: string; kind?: string; tone?: number }[] = [];
  const edgeTexts: EdgeText[] = [];
  (function walkEdges(node: ElkNode, ox: number, oy: number) {
    for (const e of (node.edges as ElkExtendedEdge[]) || []) {
      const label = labels.get(e.id);
      (e.sections || []).forEach((sec, i) => {
        const pts = [sec.startPoint, ...(sec.bendPoints || []), sec.endPoint];
        const d = pts.map((p, idx) => `${idx ? "L" : "M"} ${p.x + ox} ${p.y + oy}`).join(" ");
        edgePaths.push({ key: `${e.id}-${i}`, d, edge: `${e.sources[0]}->${e.targets[0]}`, kind: label?.kind, tone: label?.tone });
        if (i === 0 && label?.text) {
          edgeTexts.push({ key: e.id, text: label.text, points: pts.map((p) => ({ x: p.x + ox, y: p.y + oy })) });
        }
      });
    }
    for (const c of node.children || []) walkEdges(c, ox + (c.x || 0), oy + (c.y || 0));
  })(laid, 0, 0);

  // What a label has to stay off: every card, and a compound's title tab.
  const nodeRects: Rect[] = [];
  (function walkNodes(node: ElkNode, ox: number, oy: number) {
    for (const c of node.children || []) {
      const x = ox + (c.x || 0);
      const y = oy + (c.y || 0);
      if (c.children?.length) {
        nodeRects.push({ x0: x + 8, y0: y - 12, x1: x + Math.min(c.width || 0, COMPOUND_TAB_PX), y1: y + 12 });
        walkNodes(c, x, y);
      } else {
        nodeRects.push({ x0: x, y0: y, x1: x + (c.width || 0), y1: y + (c.height || 0) });
      }
    }
  })(laid, 0, 0);

  return (
    <>
      {measureEl}
      {legend && <DagLegend nodes={displayNodes} edges={edges} />}
      <div className="dag-svg-container" ref={scrollerRef}>
      <svg width={w} height={h} className="dag-edges absolute inset-0 pointer-events-none">
        <defs>
          {Object.entries(EDGE_STYLE).map(([k, st]) => (
            <marker key={k} id={`${markerId}-${k}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 z" fill={st.stroke} />
            </marker>
          ))}
          {ENTRY_PALETTE.map((c, k) => (
            <marker key={`tone${k}`} id={`${markerId}-tone${k}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 z" fill={c} />
            </marker>
          ))}
        </defs>
        {edgePaths.map((e) => {
          const st = edgeStyle(e.kind);
          // A toned edge (one entry's own) keeps a non-activation kind's
          // line style and takes the entry's colour.
          const tone = e.tone != null ? e.tone % ENTRY_PALETTE.length : null;
          return (
            <path
              key={e.key}
              data-edge={e.edge}
              data-edge-kind={e.kind || "activation"}
              data-edge-tone={tone ?? undefined}
              d={e.d}
              fill="none"
              stroke={tone != null ? ENTRY_PALETTE[tone] : st.stroke}
              strokeWidth={st.width}
              strokeDasharray={tone != null && !e.kind ? TONE_DASH : st.dash}
              markerEnd={`url(#${markerId}-${tone != null ? `tone${tone}` : EDGE_STYLE[e.kind || "activation"] ? e.kind || "activation" : "activation"})`}
            />
          );
        })}
        {placeLabels(edgeTexts, w, nodeRects).map((t) => (
          <text key={`t:${t.key}`} className="dag-edge-label" x={t.x} y={t.y} textAnchor={t.anchor}>{t.text}</text>
        ))}
      </svg>
      <div className="dag-elk-nodes relative" style={{ width: w, height: h }}>
        {(laid.children || []).map((c) => renderNode(c, 0, 0))}
      </div>
      </div>
    </>
  );
}
