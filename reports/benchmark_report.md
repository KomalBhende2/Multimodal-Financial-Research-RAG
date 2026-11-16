# Institutional Equity Research RAG - Benchmark Evaluation Summary

This report presents quality metrics, latency breakdowns, and dynamic grounding metrics for the newly updated **Hybrid Retrieval Engine**.

## 1. Quality & Performance Table

| Query Name | Coverage Score (%) | Total Latency (s) | Citation Score (Pages) | Theme Diversity | Duplicate Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ratnaveer Supply Chain | 100.0% | 322.2624s | 4 | 3 | 47.5% |
| Ratnaveer Business Model | 92.0% | 428.173s | 5 | 5 | 52.5% |
| Ratnaveer Moat | 92.0% | 321.2276s | 4 | 4 | 40.62% |
| Ather Margin Drivers | 0.0% | 10.2513s | 0 | 0 | 40.62% |
| Ather Sourcing & CapEx | 0.0% | 23.0346s | 0 | 0 | 0.0% |

## 2. Retrieval Latency Breakdown

| Query Name | Vector Search Latency (s) | Lexical FTS Latency (s) | Reranking Latency (s) | Total Latency (s) |
| :--- | :---: | :---: | :---: | :---: |
| Ratnaveer Supply Chain | 0.1839s | 0.4064s | 275.2317s | 322.2624s |
| Ratnaveer Business Model | 0.2311s | 0.4594s | 384.4673s | 428.173s |
| Ratnaveer Moat | 0.0519s | 0.1578s | 287.2887s | 321.2276s |
| Ather Margin Drivers | 0.1093s | 0.0261s | 287.2887s | 10.2513s |
| Ather Sourcing & CapEx | 0.0355s | 0.0069s | 0.0s | 23.0346s |

## 3. Hybrid Retrieval Contribution Breakdown

| Query Name | Vector-Only Chunks (%) | Lexical-Only Chunks (%) | Both-Search Chunks (%) | Reranker Rank Shift |
| :--- | :---: | :---: | :---: | :---: |
| Ratnaveer Supply Chain | 33.7% | 33.7% | 32.6% | 2.0 |
| Ratnaveer Business Model | 33.7% | 33.7% | 32.6% | 2.4 |
| Ratnaveer Moat | 39.09% | 39.09% | 21.83% | 2.0 |
| Ather Margin Drivers | 0.0% | 0.0% | 0.0% | 0 |
| Ather Sourcing & CapEx | 0.0% | 0.0% | 0.0% | 0 |

## 4. Missing Concept Analysis

| Query Name | Missing Concepts detected |
| :--- | :--- |
| Ratnaveer Supply Chain | *None* |
| Ratnaveer Business Model | *asset-light/asset-heavy, acquisitions* |
| Ratnaveer Moat | *differentiation, patents* |
| Ather Margin Drivers | *EBITDA margins, revenue growth, cost optimization, cash flow margins, operating leverage, margin drivers, gross margin, profitability milestones* |
| Ather Sourcing & CapEx | *In-house manufacturing, battery assembly, cells sourcing, vertical integration, technology collaborations, assembly lines capacity, domestic supply chain, imported parts dependency, vendor ecosystem* |
