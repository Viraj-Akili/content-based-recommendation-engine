# Recommendation Engine Evaluation Framework

This directory contains the reproducible offline evaluation benchmark and error analysis suite for the **Content-Based Recommendation Engine**.

---

## 1. Evaluation Methodology

The recommendation engine is evaluated against a curated benchmark dataset of **60 standardized user skill profiles** ([test_cases.json](file:///c:/Users/Viraj%20Akili/OneDrive/Desktop/project-3/content-based-recommendation-engine/evaluation/test_cases.json)) spanning 15 technology categories.

Each test case defines:
- A set of user skills (including common aliases and abbreviations, e.g. `k8s`, `ml`, `js`, `postgres`).
- Ground-truth relevant target career roles.

The evaluation executes the exact TF-IDF vectorization, Cosine Similarity matching, and Weighted Skill Coverage ranking pipeline used by the production engine.

---

## 2. Metric Definitions

### 1. Precision@3 (P@3)
Measures the proportion of recommended roles in the top-3 that are relevant:

$$\text{Precision@3} = \frac{|\text{Top-3 Recommendations} \cap \text{Relevant Roles}|}{3}$$

### 2. Recall@3 (R@3)
Measures the proportion of all relevant target roles successfully retrieved in the top-3:

$$\text{Recall@3} = \frac{|\text{Top-3 Recommendations} \cap \text{Relevant Roles}|}{|\text{Relevant Roles}|}$$

### 3. Mean Reciprocal Rank (MRR)
Measures the quality of ranking by assessing the position of the **first** relevant recommendation:

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

*(where $\text{rank}_i$ is the 1-indexed position of the first relevant role; $0$ if none are in the top-K)*

### 4. Catalog Coverage
Measures catalog discovery diversity by calculating the percentage of unique roles in the 118-role corpus recommended at least once in the top-3 across all queries:

$$\text{Catalog Coverage} = \frac{\left|\bigcup_{i=1}^N \text{Top-3}(q_i)\right|}{|V_{\text{roles}}|}$$

### 5. Average Inference Latency
Measures the pure mathematical execution time per recommendation query (vectorization, similarity computation, and multi-tier sorting) isolated from any frontend timers or rendering operations:

$$\text{Average Latency} = \frac{1}{M \cdot N} \sum_{m=1}^M \sum_{i=1}^N t_{\text{inference}}(q_i)$$

*(Measured across $M=20$ warm benchmark iterations using high-precision timers)*

---

## 3. Measured Experimental Results & Mathematical Proofs

Generated via `python evaluation/evaluate.py`:

| Metric | Measured Value | Description & Mathematical Grounding |
| :--- | :---: | :--- |
| **Total Test Profiles** | `60` | Standardized test profiles spanning 15 tech domains |
| **Corpus Role Count** | `118` | Total career paths in catalog |
| **Precision@3** | **`0.6667`** (66.7%) | $\frac{\text{Total Hits}}{60 \times 3} = \frac{120}{180} = \frac{2}{3}$. Exactly 2 out of 3 top recommendations on average are relevant. |
| **Recall@3** | **`0.6667`** (66.7%) | $\frac{\text{Total Hits}}{60 \times |\text{Relevant}|} = \frac{120}{180} = \frac{2}{3}$. Because each test profile specifies exactly 3 relevant roles, P@3 and R@3 are mathematically identical. |
| **Mean Reciprocal Rank (MRR)** | **`1.0000`** | $\frac{1}{60} \sum_{i=1}^{60} \frac{1}{1} = 1.0000$. **In 60 out of 60 profiles (100%)**, the recommendation placed at **Rank #1** is an exact match to a ground-truth relevant role. |
| **Catalog Coverage** | **`88.98%`** (105 / 118) | $\frac{105\text{ unique recommended roles}}{118\text{ total roles}} = 0.88983...$. Surfaces 105 distinct career paths in the top-3 across the 60 test queries. |
| **Average Inference Latency** | **`0.297 ms`** | Pure vector transformation, cosine similarity, weighted coverage, and sorting time per query (strictly excluding network I/O, DOM parsing, and UI paint). |

### Hits Distribution Across 60 Profiles
- **3 Hits in Top 3**: `13 profiles` (100% precision)
- **2 Hits in Top 3**: `34 profiles` (66.7% precision)
- **1 Hit in Top 3**: `13 profiles` (33.3% precision)
- **0 Hits in Top 3**: `0 profiles` (0% - zero complete misses)
- **Total Hits**: $(13 \times 3) + (34 \times 2) + (13 \times 1) = 39 + 68 + 13 = \mathbf{120\text{ hits}}$ out of $180$ recommendation slots.

---

## 4. Per-Category Performance Breakdown

| Category | Profiles Tested | Precision@3 | Recall@3 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Emerging Technology** | 1 | 1.000 | 1.000 | 1.000 |
| **Systems / Business Analysis** | 2 | 0.833 | 0.833 | 1.000 |
| **Blockchain / Web3** | 2 | 0.833 | 0.833 | 1.000 |
| **Mobile Development** | 4 | 0.750 | 0.750 | 1.000 |
| **Cybersecurity** | 5 | 0.733 | 0.733 | 1.000 |
| **AI / Machine Learning** | 6 | 0.722 | 0.722 | 1.000 |
| **Web Development** | 7 | 0.714 | 0.714 | 1.000 |
| **Data Engineering** | 4 | 0.667 | 0.667 | 1.000 |
| **Embedded / IoT** | 3 | 0.667 | 0.667 | 1.000 |
| **Game Development** | 3 | 0.667 | 0.667 | 1.000 |
| **Data Science** | 6 | 0.611 | 0.611 | 1.000 |
| **Software Engineering** | 7 | 0.571 | 0.571 | 1.000 |
| **Database Engineering** | 3 | 0.556 | 0.556 | 1.000 |
| **Cloud / DevOps** | 5 | 0.533 | 0.533 | 1.000 |
| **Research** | 2 | 0.500 | 0.500 | 1.000 |

---

## 5. Diagnostic Error Analysis & Limitations

### Findings & Observations:
1. **Rank #1 Accuracy (100%)**: In all 60 test profiles, the top recommended role was an exact match to a ground-truth relevant role.
2. **Sub-Domain Overlap**: In broad categories like *Cloud / DevOps* and *Software Engineering*, foundational skills like *Docker*, *Linux*, and *SQL* are shared across adjacent roles (*DevOps Engineer*, *Platform Engineer*, *Site Reliability Engineer*). As a result, the 2nd and 3rd recommendations occasionally feature highly compatible sibling roles not explicitly in the ground-truth target set, slightly moderating Precision@3.
3. **Cold Start & Sparse Inputs**: Profiles with fewer than 4 input skills produce flatter TF-IDF representations. The addition of domain-specific core skills (e.g. *PyTorch*, *Qiskit*, *Solidity*) sharply sharpens the top-3 discriminative accuracy.

---

## 6. How to Reproduce

```bash
# Execute evaluation suite and regenerate results.json
python evaluation/evaluate.py
```
