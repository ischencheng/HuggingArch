/** A backend refusal. `message` is the response's `detail`; `body` is the
 *  rest of the JSON error body, where an endpoint that can say WHY it refused
 *  puts a stable `reason` code beside its facts. */
export class ApiError extends Error {
  readonly status: number;
  readonly body: Record<string, unknown>;

  constructor(message: string, status: number, body: Record<string, unknown>) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }

  get reason(): string | undefined {
    return typeof this.body.reason === "string" ? this.body.reason : undefined;
  }
}

/** How long a request keeps asking for a result the server is still
 *  computing (a 503 with `reason: "still_computing"`) without the work
 *  moving: an inactivity budget, not a total one. A reported
 *  `progress.done` or `jobs_done` that advances restarts it, so a cold
 *  build that keeps finishing jobs is never given up on mid-way, however
 *  long one step of its progress takes; only 15 minutes without any job
 *  finishing are. The server goes on computing while it is asked; the same
 *  request joins that work, and work no request asks for is cancelled. */
const STILL_COMPUTING_WAIT_MS = 15 * 60 * 1000;

/** How long a request keeps asking while the server says it cannot run it
 *  right now (`compute_unavailable`: the compute pool is being rebuilt;
 *  `measurements_unavailable`: the Kernel Bench service is restarting).
 *  Nothing reports progress there, so it is one total budget: long enough
 *  to ride out a restart, short enough that a service that stays down
 *  surfaces as the error it is. */
const UNAVAILABLE_WAIT_MS = 3 * 60 * 1000;

/** The 503 reasons the server answers for a request it asks to have sent
 *  again (`backend/api/_app.py`), and how long each may keep being sent. */
const RETRY_BUDGET_MS: Record<string, number> = {
  still_computing: STILL_COMPUTING_WAIT_MS,
  compute_unavailable: UNAVAILABLE_WAIT_MS,
  measurements_unavailable: UNAVAILABLE_WAIT_MS,
};

function waitMs(ms: number, signal?: AbortSignal | null): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(signal.reason);
    }, { once: true });
  });
}

/** What a still-computing request's work published so far that the client
 *  does not have yet (`partial` in the 503 body): the work's `tag`, how many
 *  items it has published in all (`count`), the frame every item is drawn
 *  under (`head`), and the items past the client's cursor. */
export type Partial = { tag: number; count: number; head: Record<string, unknown> | null; items: unknown[] };

/** How far a still-computing request's work got, when the server says
 *  (`progress` in the 503 body: steps done of steps in all), how many
 *  compute jobs it has finished (`jobs_done`, which also moves between two
 *  steps), and what it has published so far (`partial`). */
export type StillComputing = { progress?: { done: number; total: number }; jobsDone?: number; partial?: Partial };

/** The status line while the server is still computing what nobody asked
 *  for before: the request keeps waiting, so say how far it got. One text
 *  for every page that waits (`fetchJson`'s `onStillComputing`). */
export function stillComputingNote(info: StillComputing): string {
  const p = info.progress;
  const far = p ? ` (${p.done} / ${p.total})`
    : info.jobsDone ? ` (${info.jobsDone} compute job${info.jobsDone === 1 ? "" : "s"} done)` : "";
  return `Computing — first request for this view takes a few minutes${far}…`;
}

/** A 503 the server asks to have sent again: after how long, within what
 *  budget, and (still computing) how far the work got. */
async function retryable(response: Response): Promise<{ reason: string; retryMs: number; budgetMs: number; info: StillComputing } | null> {
  if (response.status !== 503) return null;
  try {
    const body: unknown = await response.clone().json();
    if (!body || typeof body !== "object") return null;
    const { reason, retry_after_s: after, progress, jobs_done: jobsDone, partial } = body as {
      reason?: unknown; retry_after_s?: unknown; progress?: unknown; jobs_done?: unknown; partial?: unknown;
    };
    const budgetMs = typeof reason === "string" ? RETRY_BUDGET_MS[reason] : undefined;
    if (typeof reason !== "string" || budgetMs === undefined) return null;
    const p = progress as { done?: unknown; total?: unknown } | undefined;
    const info: StillComputing = p && typeof p.done === "number" && typeof p.total === "number"
      ? { progress: { done: p.done, total: p.total } }
      : {};
    if (typeof jobsDone === "number") info.jobsDone = jobsDone;
    const part = partial as Partial | undefined;
    if (part && typeof part.tag === "number" && Array.isArray(part.items)) info.partial = part;
    return { retryMs: (typeof after === "number" && after > 0 ? after : 5) * 1000, budgetMs, info, reason };
  } catch {
    return null;
  }
}

/** True when `fetchJson` failed because the caller aborted it. */
export function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

export async function fetchJson<T>(
  /** The URL, or how to build it for each attempt — a request that tells
   *  the server, on every retry, what it already holds and what it wants
   *  first. */
  url: string | (() => string),
  init?: RequestInit,
  /** Called each time the server answers "still computing" and the
   *  request waits to ask again. */
  onStillComputing?: (info: StillComputing) => void,
): Promise<T> {
  let response: Response;
  let giveUpAt: number | undefined;
  let lastMoved: string | undefined;
  for (;;) {
    try {
      response = await fetch(typeof url === "string" ? url : url(), init);
    } catch (e) {
      if (isAbort(e)) throw e;
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(`Cannot reach the backend (${msg}).`);
    }
    const waiting = await retryable(response);
    if (waiting === null) break;
    giveUpAt ??= Date.now() + waiting.budgetMs;
    // Work whose reported `done` or finished jobs advance is alive: the
    // budget restarts there, so a long computation that keeps finishing
    // jobs never reads as stuck.
    const { progress, jobsDone, partial } = waiting.info;
    const moved = progress === undefined && jobsDone === undefined && partial === undefined
      ? undefined : `${progress?.done}:${jobsDone}:${partial?.count}`;
    if (moved !== undefined && moved !== lastMoved) {
      lastMoved = moved;
      giveUpAt = Date.now() + waiting.budgetMs;
    }
    if (Date.now() + waiting.retryMs > giveUpAt) break;
    if (waiting.reason === "still_computing") onStillComputing?.(waiting.info);
    await waitMs(waiting.retryMs, init?.signal);
  }
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    let body: Record<string, unknown> = {};
    try {
      const parsed: unknown = await response.json();
      if (parsed && typeof parsed === "object") body = parsed as Record<string, unknown>;
      if (typeof body.detail === "string") detail = body.detail;
      if (body.detail && typeof body.detail === "object") {
        const structured = body.detail as { message?: unknown; errors?: unknown };
        const messages = Array.isArray(structured.errors)
          ? structured.errors.map((item) => String(item))
          : [];
        if (typeof structured.message === "string") detail = structured.message;
        if (messages.length) detail = [detail, ...messages].join(" ");
      }
    } catch { /* keep status */ }
    throw new ApiError(detail, response.status, body);
  }
  if (contentType && !contentType.includes("application/json")) {
    throw new Error(`Backend unreachable — HTTP ${response.status} instead of JSON.`);
  }
  return response.json() as Promise<T>;
}

const HF_REPO_ID_PART = /^[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?$/;

/** Match huggingface_hub's local repo-id rules for the ids this app requests:
 * exactly `owner/name`, ASCII word characters plus separators, at most 96
 * characters per part, no leading/trailing separator, no `--` / `..`, and no
 * `.git` repo suffix. */
export function isHfRepoId(id: string): boolean {
  const parts = id.split("/");
  if (parts.length !== 2 || parts.some((part) => !HF_REPO_ID_PART.test(part))) {
    return false;
  }
  return !parts.some((part) => part.includes("--") || part.includes(".."))
    && !parts[1].endsWith(".git");
}

/** Model ids accepted by the backend's config-fetch boundary. Custom local
 * models do not have an HF owner/name shape; everything else must. */
export function isRequestableModelId(id: string): boolean {
  return (id.startsWith("custom:") && id.length > "custom:".length)
    || isHfRepoId(id);
}
