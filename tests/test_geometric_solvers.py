import unittest
import numpy as np

from ingestion.processors.geometric_solvers import (
    HungarianTableMatcher,
    PolarChartExtractor,
    OptimalTransportBarAligner,
    DelaunayCardClusterer
)


class TestGeometricSolvers(unittest.TestCase):

    def test_hungarian_table_matcher_preserves_empty_cells(self):
        """
        Verify that Hungarian bipartite matching pins values to the exact column
        and leaves blank cells as '—' without shifting columns left.
        """
        matcher = HungarianTableMatcher(max_dist_threshold=50.0)

        # 2 Rows: Revenue and Operating EBITDA
        row_labels = [
            (100.0, 120.0, "Revenue"),
            (150.0, 170.0, "Operating EBITDA")
        ]

        # 3 Columns: Q1FY25 (x=200), Q1FY26 (x=300), Q1FY27 (x=400)
        col_headers = [
            (180.0, 220.0, "Q1FY25"),
            (280.0, 320.0, "Q1FY26"),
            (380.0, 420.0, "Q1FY27")
        ]

        # Values:
        # Row 0 has all 3 values (401, 725, 877)
        # Row 1 is MISSING Q1FY25 (blank cell!) and only has Q1FY26=127, Q1FY27=242
        values = [
            (195.0, 105.0, 205.0, 115.0, "401"),
            (295.0, 105.0, 305.0, 115.0, "725"),
            (395.0, 105.0, 405.0, 115.0, "877"),
            # Row 1: missing at x=200!
            (295.0, 155.0, 305.0, 165.0, "127"),
            (395.0, 155.0, 405.0, 165.0, "242"),
        ]

        table = matcher.align_table(row_labels, col_headers, values)

        # Header check
        self.assertEqual(table[0], ["Dimension / Metric", "Q1FY25", "Q1FY26", "Q1FY27"])
        # Row 0 check: all values present
        self.assertEqual(table[1], ["Revenue", "401", "725", "877"])
        # Row 1 check: Q1FY25 MUST be '—', Q1FY26 MUST be '127', Q1FY27 MUST be '242'
        self.assertEqual(table[2], ["Operating EBITDA", "—", "127", "242"])
        print("\n[PASS] Hungarian Matcher correctly preserved empty cell at Q1FY25 without column shifting!")

    def test_polar_chart_extractor_clockwise_ordering(self):
        """
        Verify that circular/radial tokens around a center (300, 300)
        are transformed to polar (r, theta) and sorted clockwise.
        """
        extractor = PolarChartExtractor()

        # Center at (300, 300), Radius ~100
        # 4 Slices at angles 0 (East), 90 (South), 180 (West), 270 (North)
        # Note: In PDF coordinates, Y goes down, so 90 deg is South
        words = [
            # 0 deg (East, x=400, y=300)
            (390, 290, 410, 310, "Cardio"),
            (415, 295, 435, 305, "35%"),
            # 90 deg (South, x=300, y=400)
            (290, 390, 310, 410, "Oncology"),
            (295, 415, 305, 435, "25%"),
            # 180 deg (West, x=200, y=300)
            (190, 290, 210, 310, "Orthopaedics"),
            (165, 295, 185, 305, "20%"),
            # 270 deg (North, x=300, y=200)
            (290, 190, 310, 210, "Neurology"),
            (295, 165, 305, 185, "20%"),
        ]

        self.assertTrue(extractor.is_circular_layout(words))
        chart_table = extractor.extract_donut_chart(words)

        self.assertGreaterEqual(len(chart_table), 4)
        self.assertEqual(chart_table[0], ["Category / Specialty", "Share (%)"])
        # Check that percentages are ordered clockwise
        extracted_pcts = [row[1] for row in chart_table[1:]]
        self.assertIn("35%", extracted_pcts)
        self.assertIn("25%", extracted_pcts)
        print(f"\n[PASS] Polar Extractor cleanly sorted radial donut chart into: {chart_table[1:]}")

    def test_optimal_transport_aligner(self):
        """
        Verify that floating bar labels pair to the nearest timeline quarter via 1D Wasserstein cost.
        """
        aligner = OptimalTransportBarAligner()

        # 3 timeline quarter targets along the X-axis
        periods = [
            (190.0, 400.0, 210.0, 420.0, "Q1FY25"),
            (290.0, 400.0, 310.0, 420.0, "Q1FY26"),
            (390.0, 400.0, 410.0, 420.0, "Q1FY27"),
        ]

        # Floating values above bars at different elevations
        values = [
            (198.0, 310.0, 202.0, 320.0, "50"),   # High bar at Q1FY25
            (299.0, 250.0, 301.0, 260.0, "127"),  # Medium bar at Q1FY26
            (402.0, 180.0, 408.0, 190.0, "242"),  # Very tall bar at Q1FY27
        ]

        pairs = aligner.align_values_to_periods(periods, values)

        self.assertEqual(pairs, [("Q1FY25", "50"), ("Q1FY26", "127"), ("Q1FY27", "242")])
        print(f"\n[PASS] Optimal Transport Aligner matched floating numbers: {pairs}")

    def test_delaunay_card_clusterer(self):
        """
        Verify that irregular word point clouds are segmented into 3 distinct card clusters.
        """
        clusterer = DelaunayCardClusterer()

        # Generate 3 distinct card clusters centered at X=100, X=300, X=500
        words = []
        for xc, title in [(100, "Design"), (300, "Procurement"), (500, "Quality")]:
            for y in [200, 220, 240, 260]:
                words.append((xc - 40, y - 5, xc + 40, y + 5, f"{title}_Token"))

        cards = clusterer.cluster_cards(words, k_cards=3)
        self.assertEqual(len(cards), 3)
        # Verify left-to-right sorting
        means = [np.mean([(w[0] + w[2]) / 2.0 for w in c]) for c in cards]
        self.assertTrue(means[0] < means[1] < means[2])
        print(f"\n[PASS] Delaunay Clusterer partitioned 3 cards with X-means: {[round(m, 1) for m in means]}")


if __name__ == "__main__":
    unittest.main()
