// Where a DAG's edge labels go. A label is only useful where it can be read,
// and the node layer paints over the edge layer, so a label that lands on a
// node is simply cut off. It goes to the first free spot along its own edge:
// the middle of the longest segment when that is free, then further along
// that segment, then the shorter ones.

export type Point = { x: number; y: number };
export type Rect = { x0: number; y0: number; x1: number; y1: number };
export type EdgeText = { key: string; text: string; points: Point[] };
export type PlacedLabel = {
  key: string;
  text: string;
  x: number;
  y: number;
  anchor: "start" | "middle" | "end";
};

// ~6.2px per 10px mono char; a line is 12px; the text sits 4px off its edge.
const CHAR_WIDTH = 6.2;
const LINE_HEIGHT = 12;
const EDGE_GAP = 4;
// Along one segment, from its middle outwards.
const ALONG = [0.5, 0.3, 0.7, 0.15, 0.85];

const overlaps = (a: Rect, b: Rect) =>
  a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

function spotsAlong(points: Point[]): Point[] {
  const segments = points.slice(1)
    .map((b, i) => ({ a: points[i], b, length: Math.hypot(b.x - points[i].x, b.y - points[i].y) }))
    .sort((s, t) => t.length - s.length);
  return segments.flatMap(({ a, b }) =>
    ALONG.map((f) => ({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f - EDGE_GAP })));
}

function placedAt(text: string, spot: Point, anchor: PlacedLabel["anchor"]): Rect {
  const length = text.length * CHAR_WIDTH;
  const x0 = anchor === "end" ? spot.x - length : anchor === "start" ? spot.x : spot.x - length / 2;
  return { x0, y0: spot.y - LINE_HEIGHT + 2, x1: x0 + length, y1: spot.y + 2 };
}

/** The anchor that keeps a label inside the canvas when centred on `x` is not. */
function fittingAnchor(text: string, x: number, width: number): PlacedLabel["anchor"] {
  const half = (text.length * CHAR_WIDTH) / 2;
  return x + half > width - EDGE_GAP ? "end" : x - half < EDGE_GAP ? "start" : "middle";
}

export function placeLabels(texts: EdgeText[], width: number, boxes: Rect[]): PlacedLabel[] {
  const taken: Rect[] = [];
  const free = (r: Rect) =>
    r.x0 >= 0 && r.x1 <= width && !boxes.some((b) => overlaps(r, b)) && !taken.some((t) => overlaps(r, t));
  return texts.map(({ key, text, points }) => {
    const spots = spotsAlong(points);
    for (const spot of spots) {
      // Centred on the edge, or beside it when a node crowds one side.
      const sides: [Point, PlacedLabel["anchor"]][] = [
        [spot, fittingAnchor(text, spot.x, width)],
        [{ x: spot.x + EDGE_GAP, y: spot.y }, "start"],
        [{ x: spot.x - EDGE_GAP, y: spot.y }, "end"],
      ];
      for (const [at, anchor] of sides) {
        const rect = placedAt(text, at, anchor);
        if (free(rect)) {
          taken.push(rect);
          return { key, text, x: at.x, y: at.y, anchor };
        }
      }
    }
    // Nowhere free along the whole edge: the middle of its longest segment,
    // stepped down clear of the labels already placed.
    const at = { ...spots[0] };
    const anchor = fittingAnchor(text, at.x, width);
    for (let step = 0; step < 6 && taken.some((t) => overlaps(placedAt(text, at, anchor), t)); step++) {
      at.y += LINE_HEIGHT;
    }
    taken.push(placedAt(text, at, anchor));
    return { key, text, x: at.x, y: at.y, anchor };
  });
}
