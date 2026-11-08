// An edge label has to be readable: off the nodes, off the other labels and
// inside the canvas, with the old spot kept whenever that spot was fine.
import { describe, expect, test } from "vitest";

import { placeLabels, type EdgeText, type Rect } from "./dagLabels";

const vertical: EdgeText = {
  key: "e", text: "[B,s,128]", points: [{ x: 300, y: 0 }, { x: 300, y: 200 }],
};

describe("placeLabels", () => {
  test("a free edge keeps its label at the middle of its longest segment", () => {
    expect(placeLabels([vertical], 800, [])).toEqual([
      { key: "e", text: "[B,s,128]", x: 300, y: 96, anchor: "middle" },
    ]);
  });

  test("a label moves along its edge off a node that covers the middle", () => {
    const node: Rect = { x0: 250, y0: 80, x1: 350, y1: 120 };
    const [label] = placeLabels([vertical], 800, [node]);
    expect(label.y < 70 || label.y > 130).toBe(true);
    expect(label.x).toBe(300);
  });

  test("a label goes beside its edge when a node crowds one side all along", () => {
    const left: Rect = { x0: 200, y0: 0, x1: 296, y1: 200 };
    const [label] = placeLabels([vertical], 800, [left]);
    expect(label.anchor).toBe("start");
    expect(label.x).toBeGreaterThan(300);
  });

  test("two labels on one spot do not overprint", () => {
    const [a, b] = placeLabels([vertical, { ...vertical, key: "f" }], 800, []);
    expect([a.x, a.y]).not.toEqual([b.x, b.y]);
  });

  test("a label near the canvas edge stays inside it", () => {
    const edge: EdgeText = { ...vertical, points: [{ x: 790, y: 0 }, { x: 790, y: 200 }] };
    expect(placeLabels([edge], 800, [])[0].anchor).toBe("end");
  });

  test("with no free spot the label still appears, on its longest segment", () => {
    const wall: Rect = { x0: 0, y0: -50, x1: 800, y1: 300 };
    const [label] = placeLabels([vertical], 800, [wall]);
    expect([label.x, label.y]).toEqual([300, 96]);
  });
});
