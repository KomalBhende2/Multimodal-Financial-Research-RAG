"""
Geometric Solvers for Complex Document Patterns
------------------------------------------------
Production-Grade Computational Geometry Engine

Implements 4 core mathematical and graph-theoretic solvers:
1. HungarianTableMatcher: Solves bipartite lattice matching (Kuhn-Munkres) for borderless tables with empty cells.
2. PolarChartExtractor: Transforms 2D Cartesian coordinates to Polar (r, theta) space for circular pie & donut charts.
3. OptimalTransportBarAligner: Pairs floating metric values to periodic horizontal timeline quarters via 1D Wasserstein optimization.
4. DelaunayCardClusterer: Partitions irregular/staggered dashboard cards via Delaunay triangulation and Graph Laplacian spectral clustering.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
import re


class HungarianTableMatcher:
    """
    Bipartite Lattice Matching using the Hungarian (Kuhn-Munkres) algorithm.
    Guarantees zero column shifting when tables contain blank / missing cells.
    """

    def __init__(self, max_dist_threshold: float = 45.0, y_penalty_multiplier: float = 2.0):
        self.max_dist_threshold = max_dist_threshold
        self.y_penalty_multiplier = y_penalty_multiplier

    def align_table(
        self,
        row_labels: List[Tuple[float, float, str]],  # [(y0, y1, "Row Name"), ...]
        col_headers: List[Tuple[float, float, str]],  # [(x0, x1, "Col Name"), ...]
        values: List[Tuple[float, float, float, float, str]],  # [(x0, y0, x1, y1, "Val"), ...]
    ) -> List[List[str]]:
        num_rows = len(row_labels)
        num_cols = len(col_headers)

        if num_rows == 0 or num_cols == 0:
            return []

        # 1. Target grid intersection points: (Row Y-center, Col X-center)
        grid_targets = []
        for r_idx, (r_y0, r_y1, _) in enumerate(row_labels):
            r_yc = (r_y0 + r_y1) / 2.0
            for c_idx, (c_x0, c_x1, _) in enumerate(col_headers):
                c_xc = (c_x0 + c_x1) / 2.0
                grid_targets.append((r_idx, c_idx, c_xc, r_yc))

        num_cells = len(grid_targets)
        num_vals = len(values)

        if num_vals == 0:
            header_row = ["Dimension / Metric"] + [c[2] for c in col_headers]
            empty_table = [header_row]
            for _, _, r_name in row_labels:
                empty_table.append([r_name] + ["—"] * num_cols)
            return empty_table

        # 2. Build Anisotropic Cost Matrix C of size (num_vals x num_cells)
        cost_matrix = np.full((num_vals, num_cells), fill_value=1e5)

        for i, (vx0, vy0, vx1, vy1, val_text) in enumerate(values):
            v_xc = (vx0 + vx1) / 2.0
            v_yc = (vy0 + vy1) / 2.0

            for j, (r_idx, c_idx, t_xc, t_yc) in enumerate(grid_targets):
                dx = abs(v_xc - t_xc)
                dy = abs(v_yc - t_yc) * self.y_penalty_multiplier
                dist = np.sqrt(dx**2 + dy**2)

                if dist < self.max_dist_threshold:
                    cost_matrix[i, j] = dist

        # 3. Solve Maximum-Weight / Minimum-Cost Bipartite Matching in O(V^3)
        val_indices, cell_indices = linear_sum_assignment(cost_matrix)

        # 4. Construct Table Matrix M[num_rows][num_cols]
        table_matrix = [["—" for _ in range(num_cols)] for _ in range(num_rows)]

        for v_idx, cell_idx in zip(val_indices, cell_indices):
            if cost_matrix[v_idx, cell_idx] < self.max_dist_threshold:
                r_idx, c_idx, _, _ = grid_targets[cell_idx]
                val_str = values[v_idx][4].strip()
                if table_matrix[r_idx][c_idx] == "—":
                    table_matrix[r_idx][c_idx] = val_str
                else:
                    table_matrix[r_idx][c_idx] += " " + val_str

        # 5. Build final Markdown table rows
        header_row = ["Dimension / Metric"] + [c[2] for c in col_headers]
        final_rows = [header_row]
        for r_idx, (_, _, r_label) in enumerate(row_labels):
            final_rows.append([r_label] + table_matrix[r_idx])

        return final_rows


class PolarChartExtractor:
    """
    Transforms 2D Cartesian coordinates (x, y) into Polar coordinates (r, theta)
    relative to a detected chart centroid (x0, y0).
    Traverses circular slices clockwise (0 to 360 deg), pairing categories to percentages.
    """

    def is_circular_layout(self, words: List[Tuple], min_radial_pct_count: int = 3) -> bool:
        """Detect if words form a circular / radial arrangement around a central point."""
        pct_words = [w for w in words if re.search(r"\d{1,3}%", w[4])]
        if len(pct_words) < min_radial_pct_count:
            return False

        pts = np.array([[(w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0] for w in pct_words])
        center = np.mean(pts, axis=0)
        radii = np.linalg.norm(pts - center, axis=1)

        # Radii to center should have relatively low variance (they lie on a circle)
        std_r = np.std(radii)
        mean_r = np.mean(radii)
        if mean_r > 30.0 and (std_r / mean_r) < 0.45:
            return True
        return False

    def extract_donut_chart(
        self, words: List[Tuple], page_w: float = 720.0, page_h: float = 540.0
    ) -> List[List[str]]:
        if not words:
            return []

        pts = np.array([[(w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0] for w in words])
        x0, y0 = np.mean(pts[:, 0]), np.mean(pts[:, 1])

        # Convert to Polar Coordinates (r, theta)
        polar_tokens = []
        for w in words:
            xc = (w[0] + w[2]) / 2.0
            yc = (w[1] + w[3]) / 2.0

            dx = xc - x0
            dy = yc - y0

            r = float(np.hypot(dx, dy))
            theta = float(np.degrees(np.arctan2(dy, dx))) % 360.0

            is_pct = bool(re.search(r"\d{1,3}%", w[4]))
            polar_tokens.append({
                "word": w,
                "text": w[4].strip(),
                "r": r,
                "theta": theta,
                "is_pct": is_pct
            })

        # Separate into slice percentages and category labels
        pct_items = [p for p in polar_tokens if p["is_pct"]]
        label_items = [p for p in polar_tokens if not p["is_pct"] and len(p["text"]) >= 2]

        if not pct_items:
            return []

        # Sort percentages clockwise by theta (0 -> 360 deg)
        pct_items.sort(key=lambda p: p["theta"])

        # Pair each percentage with its closest angular label
        pairs = []
        used_labels = set()

        for pct in pct_items:
            best_label = None
            min_ang_dist = 999.0

            for idx, lbl in enumerate(label_items):
                if idx in used_labels:
                    continue
                # Angular difference modulo 360
                diff = abs(pct["theta"] - lbl["theta"])
                ang_dist = min(diff, 360.0 - diff)
                if ang_dist < min_ang_dist and ang_dist < 40.0:
                    min_ang_dist = ang_dist
                    best_label = (idx, lbl["text"])

            if best_label:
                used_labels.add(best_label[0])
                pairs.append([best_label[1], pct["text"]])
            else:
                pairs.append(["Segment", pct["text"]])

        header = ["Category / Specialty", "Share (%)"]
        return [header] + pairs


class OptimalTransportBarAligner:
    """
    Pairs floating data values to periodic horizontal timeline quarters (Q1FY25, Q1FY26, etc.)
    with minimum 1D Wasserstein transport work along the X-axis.
    """

    def align_values_to_periods(
        self,
        period_tokens: List[Tuple],
        value_tokens: List[Tuple],
        corridor_tolerance: float = 45.0
    ) -> List[Tuple[str, str]]:
        if not period_tokens or not value_tokens:
            return []

        period_tokens_sorted = sorted(period_tokens, key=lambda p: (p[0] + p[2]) / 2.0)
        m = len(period_tokens_sorted)
        n = len(value_tokens)

        # 1D Horizontal distance matrix C[i, j] = |x_val - x_period|
        cost_matrix = np.zeros((n, m))
        for i, val in enumerate(value_tokens):
            val_xc = (val[0] + val[2]) / 2.0
            for j, per in enumerate(period_tokens_sorted):
                per_xc = (per[0] + per[2]) / 2.0
                cost_matrix[i, j] = abs(val_xc - per_xc)

        # Optimal transport matching
        val_idx, per_idx = linear_sum_assignment(cost_matrix)

        assigned_dict = {}
        for v, p in zip(val_idx, per_idx):
            if cost_matrix[v, p] <= corridor_tolerance:
                per_name = period_tokens_sorted[p][4].strip()
                val_text = value_tokens[v][4].strip()
                assigned_dict[per_name] = val_text

        result = []
        for p in period_tokens_sorted:
            per_name = p[4].strip()
            result.append((per_name, assigned_dict.get(per_name, "—")))

        return result


class DelaunayCardClusterer:
    """
    Partitions staggered, irregular cards using Delaunay triangulation
    and Normalized Graph Laplacian spectral decomposition.
    """

    def cluster_cards(self, words: List[Tuple], k_cards: int = 3) -> List[List[Tuple]]:
        if len(words) < k_cards * 3 or k_cards <= 1:
            return [words]

        points = np.array([[(w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0] for w in words])
        n = len(points)

        try:
            tri = Delaunay(points)
        except Exception:
            # Fallback to simple X-coordinate sort if triangulation fails
            words_sorted = sorted(words, key=lambda w: (w[0] + w[2]) / 2.0)
            chunk_size = len(words) // k_cards
            return [words_sorted[i * chunk_size : (i + 1) * chunk_size] for i in range(k_cards)]

        # Construct Adjacency Matrix A
        A = np.zeros((n, n))
        for simplex in tri.simplices:
            for u in simplex:
                for v in simplex:
                    if u != v:
                        dist = float(np.linalg.norm(points[u] - points[v]))
                        # Radial Basis Function (RBF) affinity with sigma = 50.0 pt
                        A[u, v] = np.exp(-(dist**2) / (2.0 * (50.0**2)))

        # Normalized Graph Laplacian: L = I - D^(-1/2) A D^(-1/2)
        d_vec = np.sum(A, axis=1)
        d_vec[d_vec == 0] = 1e-6
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d_vec))
        L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

        # Eigenvalue Decomposition
        try:
            eigenvals, eigenvecs = np.linalg.eigh(L)
            features = eigenvecs[:, 1 : min(k_cards + 1, n)]

            # K-means clustering on spectral embedding
            from scipy.cluster.vq import kmeans2
            _, labels = kmeans2(features, k_cards, minit="points")
        except Exception:
            # Fallback on X-projection
            x_vals = points[:, 0]
            labels = np.digitize(x_vals, np.linspace(np.min(x_vals), np.max(x_vals), k_cards + 1)[1:-1])

        # Group words by cluster label
        card_clusters = [[] for _ in range(k_cards)]
        for w_idx, label in enumerate(labels):
            card_clusters[min(label, k_cards - 1)].append(words[w_idx])

        # Filter empty clusters and sort left-to-right
        active_clusters = [c for c in card_clusters if c]
        active_clusters.sort(key=lambda c: np.mean([(w[0] + w[2]) / 2.0 for w in c]))

        return active_clusters
