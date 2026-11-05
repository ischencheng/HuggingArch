"""Transfer-table contracts and small owner-map oracles."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import layout
from backend.arch.layout import transfer_table as tt
from backend.arch.layout.groups import ProcessGroupFamily
from backend.arch.eval.refusal import LayoutRefusal


def fam(name, *groups):
    return ProcessGroupFamily(name, tuple(tuple(group) for group in groups))


SCOPE8 = frozenset(range(8))
REP8 = layout.replicated(SCOPE8)

# Three independent binary coordinates over eight ranks.
X2 = fam("x", (0, 1), (2, 3), (4, 5), (6, 7))
Y2 = fam("y", (0, 2), (1, 3), (4, 6), (5, 7))
Z2 = fam("z", (0, 4), (1, 5), (2, 6), (3, 7))
BX = layout.sharded(X2)
BY = layout.sharded(Y2)
BZ = layout.sharded(Z2)
PXY = layout.product((BX, BY))
SEG_XY = layout.segmented(((0, 4, BX), (4, 4, BY)))


class TestVocabularyAndContracts(unittest.TestCase):
    def test_exactly_42_keys_with_the_documented_shape(self):
        self.assertEqual(len(tt.TRANSFER), 42)
        self.assertEqual(set(tt.TRANSFER), set(tt.EXPECTED_TRANSFER_KEYS))
        self.assertTrue(all(callable(cell) for cell in tt.TRANSFER.values()))
        self.assertEqual(set(tt.ROW_CONTRACT), set(tt.Relation))

    def test_row_contract_rejects_bad_success_shapes(self):
        with self.assertRaisesRegex(ValueError, "output arity"):
            tt.validate_cell_result(tt.PRESERVE, tt.Derived((REP8, REP8)))
        with self.assertRaisesRegex(ValueError, "expected_outputs"):
            tt.validate_cell_result(tt.SPLIT, tt.Derived((REP8,)))
        with self.assertRaisesRegex(ValueError, "output arity"):
            tt.validate_cell_result(tt.SPLIT, tt.Derived((REP8,)), expected_outputs=2)
        pending = layout.Partial(layout.SUM, X2)
        with self.assertRaisesRegex(ValueError, "only contract"):
            tt.validate_cell_result(tt.PRESERVE, tt.Derived((REP8,), pending))
        with self.assertRaisesRegex(TypeError, "foreign result"):
            tt.validate_cell_result(tt.PRESERVE, object())

    def test_result_primitives_refuse_unknown_cases(self):
        with self.assertRaisesRegex(ValueError, "useful reason"):
            tt.Rejected("")
        with self.assertRaisesRegex(ValueError, "derive Unknown"):
            tt.Derived((layout.Unknown("bottom"),))
        with self.assertRaisesRegex(TypeError, "not an AxisLayout"):
            tt.Derived((object(),))
        with self.assertRaisesRegex(TypeError, "Partial or None"):
            tt.Derived((REP8,), object())

    def test_a_rejected_geometry_is_a_layout_refusal(self):
        with self.assertRaisesRegex(LayoutRefusal, "outermost owner map"):
            tt.Rejected("the outermost owner map changed").require(
                "split", "node")

    def test_call_contract_rejects_malformed_shape_facts(self):
        with self.assertRaisesRegex(ValueError, "unknown axis relation"):
            tt.transfer("reshape-ish", (REP8,))
        with self.assertRaisesRegex(ValueError, "expects 1 input"):
            tt.transfer(tt.PRESERVE, ())
        with self.assertRaisesRegex(ValueError, "expects arguments"):
            tt.transfer(tt.PRESERVE, (REP8,), typo=1)
        with self.assertRaisesRegex(ValueError, "index map tiles"):
            tt.transfer(tt.SPLIT, (REP8,), extent=8, output_extents=(2, 3))
        with self.assertRaisesRegex(ValueError, "one extent"):
            tt.transfer(tt.CONCAT, (REP8, BX), extents=(4,))
        with self.assertRaisesRegex(ValueError, "bad interval"):
            tt.transfer(tt.SLICE, (REP8,), start=3, stop=2, extent=4)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            tt.transfer(tt.REPEAT, (REP8,), factor=True)

        # Label the subtest by index, not the callable: a lambda in the
        # subTest context can't cross pytest-xdist's execnet channel.
        for i, call in enumerate((
            lambda: tt.transfer(tt.SPLIT, (REP8,), extent=True, output_extents=(True,)),
            lambda: tt.transfer(tt.CONCAT, (REP8,), extents=(True,)),
            lambda: tt.transfer(tt.SLICE, (REP8,), start=False, stop=True, extent=True),
        )):
            with self.subTest(call=i), self.assertRaisesRegex(ValueError, "integer"):
                call()

    def test_flattened_axis_preserves_a_factored_row_major_owner_map(self):
        axis = layout.flattened((BX, BY), (4, 4))
        self.assertEqual(layout.physical_local_extent(axis, 16), 4)
        for rank in SCOPE8:
            owned = [
                index for index in range(16)
                if rank in layout.owner_ranks(axis, index, 16, SCOPE8)
            ]
            local = [
                layout.local_index(axis, rank, index, 16, SCOPE8)
                for index in owned
            ]
            self.assertEqual(local, list(range(4)))

    def test_extent_aware_rows_validate_input_layouts(self):
        for i, call in enumerate((
            lambda: tt.transfer(tt.SPLIT, (BX,), extent=3, output_extents=(3,)),
            lambda: tt.transfer(tt.CONCAT, (BX, REP8), extents=(3, 1)),
            lambda: tt.transfer(tt.SLICE, (BX,), start=0, stop=3, extent=3),
        )):
            with (
                self.subTest(call=i),
                self.assertRaisesRegex(ValueError, "not divisible"),
            ):
                call()

    def test_an_empty_range_is_replicated_whatever_the_input_cut(self):
        """No element, no owner: an empty slice or an empty axis's split
        (DSv4's complete windows in a decode step) is Replicated, where a
        proper subinterval of the same Block is rejected."""
        for axis in (REP8, BX, PXY):
            with self.subTest(axis=axis):
                self.assertEqual(
                    tt.transfer(tt.SLICE, (axis,), start=0, stop=0, extent=8),
                    tt.Derived((REP8,)))
                self.assertEqual(
                    tt.transfer(tt.SPLIT, (axis,), extent=0,
                                output_extents=(0, 4)),
                    tt.Derived((REP8, REP8)))
        self.assertIsInstance(
            tt.transfer(tt.SLICE, (BX,), start=0, stop=4, extent=8),
            tt.Rejected)

    def test_split_normalizes_one_shot_output_extents_once(self):
        outputs = (extent for extent in (2, 2))
        self.assertEqual(
            tt.transfer(tt.SPLIT, (REP8,), extent=4, output_extents=outputs),
            tt.Derived((REP8, REP8)),
        )


class TestPreserveContractAndRepeat(unittest.TestCase):
    def test_preserve_carries_every_constructor_exactly(self):
        for axis in (REP8, BX, SEG_XY, PXY):
            with self.subTest(constructor=type(axis).__name__):
                self.assertEqual(tt.transfer(tt.PRESERVE, (axis,)), tt.Derived((axis,)))

    def test_contract_replicated_and_padding_only_have_no_debt(self):
        self.assertEqual(
            tt.transfer(tt.CONTRACT, (REP8,), reducer=layout.SUM), tt.Derived(())
        )
        padding_only = layout.block(X2, ((0, 1),), pad_to=1)
        self.assertEqual(
            tt.transfer(tt.CONTRACT, (padding_only,), reducer=layout.SUM),
            tt.Derived(()),
        )
        with self.assertRaisesRegex(ValueError, "unknown reducer"):
            tt.transfer(tt.CONTRACT, (REP8,), reducer="topk")

    def test_contract_block_uses_group_or_cross_replica_family(self):
        direct = tt.transfer(tt.CONTRACT, (BX,), reducer=layout.SUM)
        self.assertEqual(direct.pending_reduction, layout.Partial(layout.SUM, X2))

        tp8 = fam("tp8", range(8))
        pairs = tuple((2 * i, 2 * i + 1) for i in range(4))
        replicated_cells = layout.block(tp8, pairs)
        result = tt.transfer(tt.CONTRACT, (replicated_cells,), reducer=layout.SUM)
        pending = result.pending_reduction
        self.assertIsNotNone(pending)
        self.assertEqual(
            set(pending.group.groups),
            {(0, 2, 4, 6), (1, 3, 5, 7)},
        )

    def test_contract_segmented_requires_one_complete_pattern(self):
        identical = layout.segmented(((0, 4, BX), (4, 8, BX)))
        result = tt.transfer(tt.CONTRACT, (identical,), reducer=layout.SUM)
        self.assertEqual(result.pending_reduction, layout.Partial(layout.SUM, X2))

        replicated_segments = layout.segmented(((0, 3, REP8), (3, 5, REP8)))
        self.assertEqual(
            tt.transfer(tt.CONTRACT, (replicated_segments,), reducer=layout.SUM),
            tt.Derived(()),
        )
        self.assertIsInstance(
            tt.transfer(tt.CONTRACT, (SEG_XY,), reducer=layout.SUM),
            tt.Rejected,
        )

    def test_contract_product_joins_orthogonal_pending_groups(self):
        result = tt.transfer(tt.CONTRACT, (PXY,), reducer=layout.SUM)
        pending = result.pending_reduction
        self.assertIsNotNone(pending)
        self.assertEqual(pending.group.ranks(), SCOPE8)
        self.assertEqual(pending.group.group_size, 4)

    def test_repeat_preserves_replication_and_block_only(self):
        for axis in (REP8, BX):
            self.assertEqual(
                tt.transfer(tt.REPEAT, (axis,), factor=3),
                tt.Derived((axis,)),
            )
        for axis in (SEG_XY, PXY):
            self.assertIsInstance(
                tt.transfer(tt.REPEAT, (axis,), factor=3), tt.Rejected
            )

    def test_repeat_scales_padding_quantum_and_preserves_owners(self):
        padded = layout.sharded(X2, pad_to=1)
        repeated = tt.transfer(tt.REPEAT, (padded,), factor=2).out_axes[0]
        self.assertEqual(repeated.pad_to, 2)
        for output_index in range(6):
            self.assertEqual(
                layout.owner_ranks(repeated, output_index, 6, SCOPE8),
                layout.owner_ranks(padded, output_index // 2, 3, SCOPE8),
            )


class TestSplit(unittest.TestCase):
    def test_replicated_split_makes_every_output_replicated(self):
        result = tt.transfer(tt.SPLIT, (REP8,), extent=8, output_extents=(2, 4))
        self.assertEqual(result, tt.Derived((REP8, REP8)))

    def test_block_cut_goes_to_outermost_factor_it_divides(self):
        result = tt.transfer(tt.SPLIT, (BX,), extent=8, output_extents=(2, 4))
        self.assertEqual(result, tt.Derived((BX, REP8)))

        tp8 = fam("tp8", range(8))
        full_cut = layout.sharded(tp8)
        self.assertIsInstance(
            tt.transfer(tt.SPLIT, (full_cut,), extent=32, output_extents=(4, 8)),
            tt.Rejected,
        )

        result = tt.transfer(tt.SPLIT, (full_cut,), extent=8, output_extents=(1, 8))
        self.assertEqual(result, tt.Derived((REP8, full_cut)))

    def test_nondivisible_and_padded_split_are_rejected(self):
        tp8 = fam("tp8", range(8))
        full_cut = layout.sharded(tp8)
        self.assertIsInstance(
            tt.transfer(tt.SPLIT, (full_cut,), extent=8, output_extents=(2, 4)),
            tt.Rejected,
        )
        padding_only = layout.block(X2, ((0, 1),), pad_to=1)
        self.assertIsInstance(
            tt.transfer(tt.SPLIT, (padding_only,), extent=8, output_extents=(2, 4)),
            tt.Rejected,
        )

    def test_segmented_split_is_rejected(self):
        self.assertIsInstance(
            tt.transfer(tt.SPLIT, (SEG_XY,), extent=8, output_extents=(2, 4)),
            tt.Rejected,
        )

    def test_product_cuts_are_assigned_in_nesting_order(self):
        separated = tt.transfer(tt.SPLIT, (PXY,), extent=4, output_extents=(2, 2))
        self.assertEqual(separated, tt.Derived((BX, BY)))

        together = tt.transfer(tt.SPLIT, (PXY,), extent=4, output_extents=(4,))
        self.assertEqual(together, tt.Derived((PXY,)))


class TestMerge(unittest.TestCase):
    def test_replicated_and_outer_cut_cases(self):
        self.assertEqual(tt.transfer(tt.MERGE, (REP8, REP8)), tt.Derived((REP8,)))
        self.assertEqual(tt.transfer(tt.MERGE, (BX, REP8)), tt.Derived((BX,)))
        self.assertEqual(tt.transfer(tt.MERGE, (PXY, REP8)), tt.Derived((PXY,)))
        self.assertIsInstance(tt.transfer(tt.MERGE, (REP8, BX)), tt.Rejected)
        self.assertIsInstance(tt.transfer(tt.MERGE, (REP8, PXY)), tt.Rejected)

        padded = layout.sharded(X2, pad_to=1)
        self.assertIsInstance(tt.transfer(tt.MERGE, (padded, REP8)), tt.Rejected)

    def test_block_pair_is_product_only_when_orthogonal(self):
        result = tt.transfer(tt.MERGE, (BX, BY))
        self.assertEqual(result, tt.Derived((layout.product((BX, BY)),)))
        self.assertIsInstance(tt.transfer(tt.MERGE, (BX, BX)), tt.Rejected)

    def test_product_combinations_keep_outer_to_inner_order(self):
        self.assertEqual(
            tt.transfer(tt.MERGE, (PXY, BZ)),
            tt.Derived((layout.product((BX, BY, BZ)),)),
        )
        self.assertEqual(
            tt.transfer(tt.MERGE, (BZ, PXY)),
            tt.Derived((layout.product((BZ, BX, BY)),)),
        )

        def bit_family(bit):
            groups = []
            for low in range(16):
                if low & (1 << bit):
                    continue
                groups.append((low, low | (1 << bit)))
            return fam(f"bit{bit}", *groups)

        cuts = tuple(layout.sharded(bit_family(bit)) for bit in range(4))
        outer = layout.product(cuts[:2])
        inner = layout.product(cuts[2:])
        self.assertEqual(
            tt.transfer(tt.MERGE, (outer, inner)),
            tt.Derived((layout.product(cuts),)),
        )

    def test_every_merge_cell_touching_segmented_rejects(self):
        examples = (REP8, BX, SEG_XY, PXY)
        count = 0
        for outer in examples:
            for inner in examples:
                if isinstance(outer, layout.Segmented) or isinstance(inner, layout.Segmented):
                    count += 1
                    self.assertIsInstance(
                        tt.transfer(tt.MERGE, (outer, inner)), tt.Rejected
                    )
        self.assertEqual(count, 7)


class TestConcatSliceBroadcast(unittest.TestCase):
    def test_concat_builds_segments_and_flattens_segmented_inputs(self):
        result = tt.transfer(tt.CONCAT, (REP8, BX), extents=(3, 4))
        axis = result.out_axes[0]
        self.assertIsInstance(axis, layout.Segmented)
        self.assertEqual(axis.segments, ((0, 3, REP8), (3, 4, BX)))

        flattened = tt.transfer(tt.CONCAT, (SEG_XY, BZ), extents=(8, 2)).out_axes[0]
        self.assertEqual(
            flattened.segments,
            ((0, 4, BX), (4, 4, BY), (8, 2, BZ)),
        )

    def test_concat_uses_only_unit_normalization(self):
        self.assertEqual(tt.transfer(tt.CONCAT, (BX,), extents=(4,)), tt.Derived((BX,)))
        same = tt.transfer(tt.CONCAT, (BX, BX), extents=(4, 4)).out_axes[0]
        self.assertIsInstance(same, layout.Segmented)

    def test_concat_rejects_different_rank_universes(self):
        small = fam("small", (0, 1), (2, 3))
        result = tt.transfer(tt.CONCAT, (REP8, layout.sharded(small)), extents=(4, 4))
        self.assertIsInstance(result, tt.Rejected)

    def test_slice_rules_are_table_cells(self):
        self.assertEqual(
            tt.transfer(tt.SLICE, (REP8,), start=1, stop=3, extent=4),
            tt.Derived((REP8,)),
        )
        self.assertEqual(
            tt.transfer(tt.SLICE, (BX,), start=0, stop=4, extent=4),
            tt.Derived((BX,)),
        )
        self.assertIsInstance(
            tt.transfer(tt.SLICE, (BX,), start=0, stop=2, extent=4),
            tt.Rejected,
        )
        self.assertEqual(
            tt.transfer(tt.SLICE, (SEG_XY,), start=0, stop=4, extent=8),
            tt.Derived((BX,)),
        )
        self.assertIsInstance(
            tt.transfer(tt.SLICE, (SEG_XY,), start=2, stop=6, extent=8),
            tt.Rejected,
        )
        with self.assertRaisesRegex(ValueError, "tiles 8 elements"):
            tt.transfer(tt.SLICE, (SEG_XY,), start=0, stop=4, extent=9)
        self.assertIsInstance(
            tt.transfer(tt.SLICE, (PXY,), start=0, stop=2, extent=4),
            tt.Rejected,
        )

    def test_broadcast_new_axis_or_inherits_counterpart(self):
        self.assertEqual(
            tt.transfer(tt.BROADCAST, (), scope=SCOPE8), tt.Derived((REP8,))
        )
        for counterpart in (REP8, BX, SEG_XY, PXY):
            self.assertEqual(
                tt.transfer(tt.BROADCAST, (counterpart,)),
                tt.Derived((counterpart,)),
            )


class TestEntryRulesAndOracles(unittest.TestCase):
    def test_unknown_is_loud_before_every_lookup_shape(self):
        unknown = layout.Unknown("test bottom")
        calls = (
            lambda: tt.transfer(tt.PRESERVE, (unknown,)),
            lambda: tt.transfer(tt.CONTRACT, (unknown,), reducer=layout.SUM),
            lambda: tt.transfer(tt.SPLIT, (unknown,), extent=4, output_extents=(2, 2)),
            lambda: tt.transfer(tt.MERGE, (unknown, REP8)),
            lambda: tt.transfer(tt.CONCAT, (unknown,), extents=(4,)),
            lambda: tt.transfer(tt.SLICE, (unknown,), start=0, stop=4, extent=4),
            lambda: tt.transfer(tt.BROADCAST, (unknown,)),
            lambda: tt.transfer(tt.REPEAT, (unknown,), factor=2),
        )
        for call in calls:
            with self.assertRaisesRegex(ValueError, "Unknown layout"):
                call()

    def test_routed_rows_only_preserve_or_broadcast(self):
        self.assertIsInstance(
            tt.transfer(tt.PRESERVE, (REP8,), routed=True), tt.Derived
        )
        self.assertIsInstance(
            tt.transfer(tt.BROADCAST, (REP8,), routed=True), tt.Derived
        )
        rejected = (
            tt.transfer(tt.CONTRACT, (REP8,), reducer=layout.SUM, routed=True),
            tt.transfer(
                tt.SPLIT,
                (REP8,),
                extent=4,
                output_extents=(2, 2),
                routed=True,
            ),
            tt.transfer(tt.MERGE, (REP8, REP8), routed=True),
            tt.transfer(tt.CONCAT, (REP8,), extents=(4,), routed=True),
            tt.transfer(
                tt.SLICE,
                (REP8,),
                start=0,
                stop=4,
                extent=4,
                routed=True,
            ),
            tt.transfer(tt.REPEAT, (REP8,), factor=2, routed=True),
        )
        self.assertTrue(all(isinstance(result, tt.Rejected) for result in rejected))

    def test_split_owner_map_matches_flat_input(self):
        out_axes = tt.transfer(
            tt.SPLIT, (BX,), extent=8, output_extents=(2, 4)
        ).out_axes
        for flat in range(8):
            outer, inner = divmod(flat, 4)
            output_owners = layout.owner_ranks(out_axes[0], outer, 2, SCOPE8)
            output_owners &= layout.owner_ranks(out_axes[1], inner, 4, SCOPE8)
            self.assertEqual(output_owners, layout.owner_ranks(BX, flat, 8, SCOPE8))

    def test_product_split_and_merge_preserve_owner_map(self):
        split_axes = tt.transfer(
            tt.SPLIT, (PXY,), extent=4, output_extents=(2, 2)
        ).out_axes
        merged = tt.transfer(tt.MERGE, split_axes).out_axes[0]
        self.assertEqual(merged, PXY)
        for flat in range(4):
            outer, inner = divmod(flat, 2)
            split_owners = layout.owner_ranks(split_axes[0], outer, 2, SCOPE8)
            split_owners &= layout.owner_ranks(split_axes[1], inner, 2, SCOPE8)
            self.assertEqual(split_owners, layout.owner_ranks(PXY, flat, 4, SCOPE8))


if __name__ == "__main__":
    unittest.main()
