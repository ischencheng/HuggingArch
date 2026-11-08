// A graph wider than its panel scrolls sideways, and panels nest: the overview
// scrolls, and so does each opened block inside it. Whatever was just opened
// can land outside all of them, leaving the reader looking at empty canvas.

const MARGIN_PX = 16;

/** Scroll every horizontally scrolling ancestor of `el` so it is in view,
 *  centred when it fits and from its left edge when it does not. Vertical
 *  position is left alone: the reader's place on the page is theirs. */
export function revealSideways(el: HTMLElement): void {
  for (let box = el.parentElement; box; box = box.parentElement) {
    const overflow = getComputedStyle(box).overflowX;
    if ((overflow !== "auto" && overflow !== "scroll") || box.scrollWidth <= box.clientWidth) continue;
    const target = el.getBoundingClientRect();
    const view = box.getBoundingClientRect();
    box.scrollLeft += target.left - view.left - Math.max((box.clientWidth - target.width) / 2, MARGIN_PX);
  }
}
