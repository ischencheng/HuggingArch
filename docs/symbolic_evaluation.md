# Symbolic evaluation: from one walk to the page

This document follows one quantity from the walk that produces it to the page
that shows it. The running examples are Llama-3.2-3B (dense GQA),
DeepSeek-V3 (MLA), gpt-oss-20b (sliding-window layers) and DeepSeek-V4-Flash
(CSA: a sliding window joined with a pooled, top-k selected stream). Unless
stated otherwise, the point is batch 1, a 4096-token prompt, 256 output
tokens and a verify width of 1.

## 1. One walk: the workload is symbolic, the topology is fixed

`Workload.symbolic()` (`backend/arch/eval/transforms.py:201`) replaces each
count a request varies with a declared symbol: `B`, `s_input`, `s_output`,
`s_prefix`, `s_verify`, and the batch's routing skew `rho` (a ratio of at
least one, `dim.ratio_symbol`). It returns the pair `(symbolic Workload, point)`: the
workload the walk runs over, and the counts that read the walk back as that
request. The model's dims stay numbers. `Phase.resolve`
(`transforms.py:358`) builds the step's lengths from those counts, and the
`Evaluator` walks the program once. The graph, the layouts and the inserted
collectives are the same at every point. Only the extents are laws.

*Example (Llama-3.2-3B decode).* The phase binds
`s_kv = s_verify + Max(1, s_input + floor(s_output/2 + 1/2)) - 1`. At the
point it reads `1 + Max(1, 4096 + 128) - 1 = 4224`.

## 2. The two halves of a `Dim`

Every extent is a `Dim` (`backend/arch/eval/dim.py:114`):

- **Identity half** (`expr_of`, `dim.py:933`): the quantity in the model's
  names, such as `n_kv_heads*d_head`.
- **Law half** (`value_of`, `dim.py:918`): the same quantity as a law in the
  workload's counts, with every model dim a number.

A frame binds a name to its identity (`Dim.symbol`, `dim.py:267`), so a
formula is printed in the names the spec writes.

*Example (Llama-3.2-3B decode).* `s_past` has the identity `s_kv - n_verify`
and the law `Max(1, s_input + floor(s_output/2 + 1/2)) - 1`, which reads 4223
at the point.

## 3. The walk decides nothing about the workload

A walk is every point's walk, so it takes no decision that differs between
points:

- **Cases.** A value that differs between points is written as its cases.
  `where` (`dim.py:449`) produces `Cases`; `ratio` (`dim.py:489`) produces
  `Quotient` (`backend/arch/law_text.py:51`, `:108`). `Min` and `Max` are
  their own cases.
- **Conditions.** A condition on the point is recorded with `require`
  (`dim.py:713`) and checked when a point is read.
- **Domain.** What the counts' declarations settle (a batch is at least one)
  is settled once, over the walk's `Domain` (`dim.py:335`, `settled` at
  `:402`). The domain is built from the binding before the walk runs
  (`backend/arch/eval/evaluator.py:406`).
- **Comparisons.** Comparing two laws directly raises `Undecided`.

*Example (gpt-oss-20b sliding layer).* The keys a query attends are
`Min(s_kv, 128)`. That is one law, not a branch the walk takes.

## 4. Reading a point

`Analysis.at(Point(...))` (`backend/arch/eval/analysis.py:481`; `Point` at
`:1404`) returns the analysis a walk at that point would produce:

1. The point is admitted against the recorded conditions (`admit`, `:519`).
   A refused point gets the same refusal a direct walk there raises.
2. Every layer is substituted.
3. The program and every value that does not read the workload stay the same
   objects.

`tests/backend/inference/test_sweep_parity.py:158` holds rows read at a point
equal to rows walked at that point.

*Example.* A point with `s_verify = 0` is refused when the point is formed,
because the verify width counts at least one (`Domain.refusal`). A point
that breaks a recorded condition is refused with the message a walk at that
point raises.

## 5. Cells and reuse

A cell is one model's walks under one deployment and precision:

- **Keys.** `_model_cells` (`backend/inference/capacity.py:849`) keys a
  `ModelCells` entry on the model, `result_version`, the parallel config and
  the precision. Inside it, `StepCells.cell`
  (`backend/inference/step_eval.py:386`) keys each `Cell` on the spec object,
  the parallel config, the phase, the precision, the encoder stack and the
  media. A request of the same phase, stack and media, at any batch, prompt,
  cached prefix, output or verify width, is a point read off the same
  cell.
- **Bounds.** Up to 16 model/deployment families are kept
  (`WALK_CELLS`, `backend/inference/walk_registry.py:79`) **and** their
  resident object bytes are capped at 768 MiB per process: entry count alone
  does not bound Kimi-K3 against a 3B model. Inside a family, up to 8 cells
  are kept (`CELLS_KEPT`, `backend/inference/step_eval.py:218`) and up to 32
  readings per cell (`READINGS_KEPT`, `:215`). An oversized family is
  returned to its caller but not retained in `WALK_CELLS`.
- **Locks.** A cell builds what it holds under its own lock, so threads share
  it (`CellBasis`, `capacity.py:788`).
- **`breakdown`.** With `breakdown`, a phase also carries its module table
  and formulas (`batch_sweep.phase_at`, `backend/inference/batch_sweep.py:398`);
  without it, the numbers are the same.

*Example.* A Scaling sweep of Llama-3.2-3B over several batches and context
lengths walks each phase once per cell and reads every grid point off that
walk.

## 6. From law to page

A law exists only as a backend expression. The page receives text:

- **Identity line.** The formula from the identity halves
  (`display_formulas`, `backend/arch/eval/projection/formulas.py:110`).
- **Substituted line.** The same terms, each bound with its node's frame at
  the point, kept unevaluated (`dim.bind_keep_products`, `dim.py:1185`).
  Identical terms are merged as `n*(term)`.
- **Result.** The law read at the point.
- **Where column.** The names the formula uses whose law is neither the name
  itself nor a number (`FormulaDefs`, `backend/inference/phase_formulas.py:54`).
  A name with a declared definition (§7) is shown as `name = definition =
  definition with the point's numbers in it = value`; any other as
  `name = folded law = value`.

Rows follow the same rule: a row holds laws and declared invariant facts,
and its numbers exist only at the one edge where it is read at a point. No
row field can therefore keep the walked batch's value. The dashboard is the
breakdown summed by role at every model
(`tests/backend/inference/test_dashboard_is_the_breakdown.py:68`), and every
`with_values` evaluates to its result
(`tests/backend/arch/eval/test_formula_arithmetic.py`).

*Example (Llama-3.2-3B prefill).* The attention row's formula is
`4*B*d_head*n_heads*s_pairs + 5*B*n_heads*s_pairs`, and the where column
shows `s_pairs = s_past·s_query + s_query·(s_query + 1)/2 =
0·4096 + 4096·(4096 + 1)/2 = 8,390,656`.

## 7. Where a derived quantity comes from

A derived name has exactly one declaration. There are three places:

- **The declaration table.** `derived:` rows in `backend/arch/dims.yaml`.
  When a call frame rebinds a name a row reads, the row is derived again
  (`rederivations`, `backend/arch/eval/binding.py:312`).
- **The phase binding.** `Phase.resolve` binds the step's lengths
  `s_query` and `s_kv`, and derives `s_past` and `s_pairs` from the laws it
  declares (`STEP_LAWS`, `backend/arch/eval/binding.py:288`):
  `s_past = s_kv − s_query`, and `s_pairs = s_query·s_past +
  s_query·(s_query + 1)/2`, the live (query, key) pairs of the dense causal
  matrix. An attention's pair-priced FLOPs are written in `s_pairs`
  (`ops_base._pairs`).
- **Node frames.** A node whose matrix is its own rebinds `s_kv` and
  `s_pairs` in its frame (`evaluator.py:1738`), with the definition
  beside the value (`Definition`, `binding.py:243`), recorded in
  `Analysis.node_rebinds`. The pairs are what its op declares: a selection's
  (`kv_pairs_per_step_fn`: a top-k, a `sliding_window_kv` or pooled stream
  marker), a band it streams (`cost_pairs_fn`: `swa_attention_*`), or a
  union's streams' sum. A node that narrowed `s_kv` with no such
  declaration binds no `s_pairs`, and an op that reads it there is refused.

A reader of names follows each frame through its declarations
(`declared_names`, `binding.py:338`; `frame_names`,
`backend/playground/module_bench.py:2690`): a rebound `s_pairs`
resolves to the names its definition reads, and the phase's derived
lengths to `s_kv` and `s_query`.

*Example (DeepSeek-V4-Flash CSA, prefill).* The phase's `s_pairs` is
8,390,656. The CSA score node rebinds `s_kv = Min(128, 4096) + Min(512,
1024) = 640` and `s_pairs` to the sum of the window stream's and the
selected pooled stream's pairs, 2,088,512. Collecting the call's
dependencies, that `s_pairs` resolves to `s_window`, `compress_ratio`,
`n_topk_keys` and the step's lengths; the call's other dependencies (its
widths and head counts, `s_ctx`) come from its other extents.

## 8. Counterexamples and boundaries

- **Topology fixed by declaration.** Under cp=8, the DeepSeek-V4
  compressor slices `floor(s_input/m)*m` rows before pooling. That slice is
  the whole axis only where `m` divides `s_input`. `layout.whole_interval`
  (`backend/arch/layout/layout.py:77`) asks whether the slice is the whole
  axis of the laws, not of one point's numbers. The slice is therefore always
  a proper subinterval, and the cut axis is gathered before it, as sglang's
  compressor does. The inserted collectives do not change with the point.
- **Media axes are numbers.** A request's image sides and frame counts are
  whole numbers (`ModelInput.media_values`,
  `backend/arch/eval/program.py:811`). Only the per-item encoder
  formula (`encoder_item_formulas`) binds them to symbols, and an input
  contract that cannot check a symbolic side skips it there.
- **Unnamed subexpressions.** A `Min` or `Max` inside an identity that no
  frame names is not listed in the where column. It appears only as a
  number on the substituted line.
- **Window-narrowed `s_kv` in encoders.** An encoder node's `s_kv` narrowed
  by its op's `kv_positions` law keeps the identity `s_kv`, and the names
  that law reads are not followed when dependencies are collected.

*Example (DeepSeek-V4-Flash, cp=8, prefill).* `s_input` = 1024 (a multiple
of 128) and 1032 (not a multiple) produce the same collectives for the HCA
compressor, because whether its slice is the whole axis is decided on the
laws `floor(s_input/128)*128` and `s_input`, not on their numbers.
