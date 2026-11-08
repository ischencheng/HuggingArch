import ELK from "elkjs/lib/elk.bundled.js";
import type { ElkExtendedEdge, ElkNode } from "elkjs";
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { revealSideways } from "../revealSideways";

/** One box of the architecture overview: a model input, a card (an op, a
 *  block, a kind with its variants, a cycle) or an auxiliary module's
 *  section. `content` is what the box draws; the graph only places it. */
export type ArchGraphNode = {
  id: string;
  kind: "input" | "card" | "section";
  // A card that draws its layers' stream map, wider to fit a column a layer.
  wide?: boolean;
  content: ReactNode;
};

/** Who reads whom (`architecture_view.edges`): a read of a layer's output,
 *  or of a stream (`streams` names it) — a seed, a stack's final value, an
 *  appended tap. */
export type ArchGraphEdge = {
  from: string;
  to: string;
  // `loop`: a denoise loop hands a latent from the head that updates it to
  // whoever reads it once the loop ends (`label` says after how many steps).
  kind: "data" | "stream" | "loop";
  streams?: string[];
  label?: string;
};

type Laid = {
  width: number;
  height: number;
  at: Map<string, { x: number; y: number }>;
  edges: { key: string; from: string; to: string; kind: string; d: string;
    label?: { x: number; y: number; text: string } }[];
};

const elk = new ELK();

const OPTS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.spacing.nodeNodeBetweenLayers": "36",
  "elk.layered.spacing.edgeNodeBetweenLayers": "14",
  "elk.spacing.nodeNode": "28",
  "elk.spacing.edgeNode": "16",
  "elk.spacing.edgeEdge": "10",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
  "elk.edgeLabels.inline": "false",
};

function streamLabel(streams: string[]): string {
  return streams.length > 2 ? `${streams.slice(0, 2).join(", ")} +${streams.length - 2}` : streams.join(", ");
}

/** The overview as a graph: every box measured as it renders, laid out by
 *  ELK, its edges drawn as routed. Re-laid whenever a box changes size (a
 *  card opened onto its block's DAG). */
export function ArchGraph({ nodes, edges }: { nodes: ArchGraphNode[]; edges: ArchGraphEdge[] }) {
  const boxes = useRef(new Map<string, HTMLDivElement>());
  const [sizes, setSizes] = useState<Record<string, { w: number; h: number }>>({});
  const [laid, setLaid] = useState<Laid | null>(null);
  const [err, setErr] = useState("");
  // The parent builds `nodes` / `edges` afresh each render; the layout
  // depends only on which boxes there are and how they connect.
  const nodeKey = nodes.map((n) => `${n.id}:${n.kind}`).join("|");
  const edgeKey = JSON.stringify(edges);
  const latest = useRef({ nodes, edges });
  latest.current = { nodes, edges };
  const laidSizes = useRef<Record<string, { w: number; h: number }>>({});

  useLayoutEffect(() => {
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const next: Record<string, { w: number; h: number }> = {};
        boxes.current.forEach((el, id) => {
          next[id] = { w: Math.ceil(el.offsetWidth), h: Math.ceil(el.offsetHeight) };
        });
        setSizes((prev) => {
          const keys = Object.keys(next);
          const same = keys.length === Object.keys(prev).length
            && keys.every((k) => prev[k] && prev[k].w === next[k].w && prev[k].h === next[k].h);
          return same ? prev : next;
        });
      });
    };
    const observer = new ResizeObserver(measure);
    boxes.current.forEach((el) => observer.observe(el));
    measure();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [nodeKey]);

  useEffect(() => {
    const { nodes, edges } = latest.current;
    if (!nodes.length || nodes.some((n) => !sizes[n.id])) return;
    let cancelled = false;
    const graph: ElkNode = {
      id: "root",
      layoutOptions: OPTS,
      children: nodes.map((n) => ({
        id: n.id,
        width: sizes[n.id].w,
        height: sizes[n.id].h,
        ...(n.kind === "input"
          ? { layoutOptions: { "elk.layered.layering.layerConstraint": "FIRST_SEPARATE" } }
          : {}),
      })),
      edges: edges.map((e, i): ElkExtendedEdge => {
        const text = e.streams && e.streams.length ? streamLabel(e.streams) : (e.label || "");
        return {
          id: `e${i}`,
          sources: [e.from],
          targets: [e.to],
          labels: text ? [{ id: `e${i}-l`, text, width: text.length * 6 + 8, height: 14 }] : [],
          layoutOptions: { "elk.layered.priority.straightness": e.kind === "data" ? "10" : "1" },
        };
      }),
    };
    elk.layout(graph).then(
      (out) => {
        if (cancelled) return;
        const at = new Map<string, { x: number; y: number }>();
        for (const c of out.children || []) at.set(c.id, { x: c.x || 0, y: c.y || 0 });
        const drawn: Laid["edges"] = [];
        (out.edges || []).forEach((e, i) => {
          const s = e.sections?.[0];
          if (!s) return;
          const pts = [s.startPoint, ...(s.bendPoints || []), s.endPoint];
          const l = e.labels?.[0];
          drawn.push({
            key: e.id,
            from: edges[i].from,
            to: edges[i].to,
            kind: edges[i].kind,
            d: pts.map((p, k) => `${k ? "L" : "M"}${p.x},${p.y}`).join(" "),
            label: l && l.text ? { x: (l.x || 0) + 4, y: (l.y || 0) + 10, text: l.text } : undefined,
          });
        });
        setLaid({ width: out.width || 0, height: out.height || 0, at, edges: drawn });
        setErr("");
      },
      (e: unknown) => {
        if (!cancelled) setErr(String(e));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [sizes, nodeKey, edgeKey]);

  // A box that just widened was just opened: once the graph has re-laid
  // around it, bring it into view.
  useLayoutEffect(() => {
    if (!laid) return;
    const before = laidSizes.current;
    laidSizes.current = sizes;
    const opened = Object.keys(sizes).find((id) => before[id] && sizes[id].w > before[id].w);
    const box = opened ? boxes.current.get(opened) : undefined;
    if (box) revealSideways(box);
  }, [laid]);

  return (
    <div style={{ overflowX: "auto" }}>
      {err ? <p className="text-11 text-amber-700">Layout failed: {err}</p> : null}
      <div
        className="arch-graph"
        style={{ position: "relative", margin: "0 auto", width: laid?.width, height: laid ? laid.height : 400 }}
      >
        {laid ? (
          <svg
            width={laid.width}
            height={laid.height}
            style={{ position: "absolute", left: 0, top: 0, overflow: "visible", pointerEvents: "none" }}
          >
            <defs>
              <marker id="arch-arrow-data" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8" />
              </marker>
              <marker id="arch-arrow-stream" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="#d97706" />
              </marker>
              <marker id="arch-arrow-loop" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="#7c3aed" />
              </marker>
            </defs>
            {laid.edges.map((e) => (
              <g key={e.key} data-arch-edge={e.kind} data-from={e.from} data-to={e.to}>
                <path
                  d={e.d}
                  fill="none"
                  stroke={e.kind === "stream" ? "#d97706" : e.kind === "loop" ? "#7c3aed" : "#94a3b8"}
                  strokeWidth={e.kind === "stream" ? 1.25 : 1.75}
                  strokeDasharray={e.kind === "stream" ? "5 4" : e.kind === "loop" ? "2 3" : undefined}
                  markerEnd={`url(#arch-arrow-${e.kind})`}
                />
                {e.label ? (
                  <text x={e.label.x} y={e.label.y} className="arch-edge-label">{e.label.text}</text>
                ) : null}
              </g>
            ))}
          </svg>
        ) : null}
        {nodes.map((n) => {
          const p = laid?.at.get(n.id);
          return (
            <div
              key={n.id}
              ref={(el) => {
                if (el) boxes.current.set(n.id, el);
                else boxes.current.delete(n.id);
              }}
              className={`arch-node arch-node-${n.kind}${n.wide ? " arch-node-wide" : ""}`}
              data-arch-node={n.id}
              style={{
                position: "absolute",
                left: p ? p.x : 0,
                top: p ? p.y : 0,
                visibility: p ? "visible" : "hidden",
              }}
            >
              {n.content}
            </div>
          );
        })}
      </div>
    </div>
  );
}
