#!/usr/bin/env python3
"""
Scientific Evaluation Framework for Content-Based Recommendation Engine
Measures Precision@3, Recall@3, MRR, Catalog Coverage, and Inference Latency.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation')


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_skill(raw_skill, alias_map):
    if not isinstance(raw_skill, str):
        return ''
    cleaned = ' '.join(raw_skill.strip().lower().split())
    if not cleaned:
        return ''
    if cleaned in alias_map:
        cleaned = alias_map[cleaned].strip().lower()
    return cleaned


def normalize_skill_list(skill_list, alias_map):
    seen = set()
    result = []
    for s in skill_list:
        norm = normalize_skill(s, alias_map)
        if norm and norm not in seen:
            seen.add(norm)
            result.push(norm) if hasattr(result, 'push') else result.append(norm)
    return result


def extract_skill_name(skill_item):
    if isinstance(skill_item, str):
        return skill_item
    if isinstance(skill_item, dict) and 'name' in skill_item:
        return skill_item['name']
    return ''


def extract_skill_weight(skill_item):
    if isinstance(skill_item, dict) and 'weight' in skill_item and isinstance(skill_item['weight'], (int, float)):
        return max(0.0, min(1.0, float(skill_item['weight'])))
    return 1.0


class RecommendationEngine:
    def __init__(self, roles, aliases, alpha=0.65):
        self.roles = roles
        self.aliases = aliases
        self.alpha = alpha

        # 1. Build Vocabulary
        vocab_set = set()
        for role in self.roles:
            for s in role.get('skills', []):
                norm = normalize_skill(extract_skill_name(s), self.aliases)
                if norm:
                    vocab_set.add(norm)
        self.vocabulary = sorted(list(vocab_set))
        self.vocab_idx = {term: idx for idx, term in enumerate(self.vocabulary)}

        # 2. Compute Document Frequency (DF) & Inverse Document Frequency (IDF)
        self.N = len(self.roles)
        self.df_map = {term: 0 for term in self.vocabulary}

        for role in self.roles:
            doc_terms = set(
                normalize_skill(extract_skill_name(s), self.aliases)
                for s in role.get('skills', [])
            )
            for term in doc_terms:
                if term in self.df_map:
                    self.df_map[term] += 1

        self.idf_map = {
            term: math.log(self.N / (1.0 + self.df_map[term]))
            for term in self.vocabulary
        }

        # 3. Precompute Role Vectors & Metadata
        self.role_profiles = []
        for role in self.roles:
            role_id = role.get('id', role['role'].lower().replace(' ', '-'))
            raw_skills = role.get('skills', [])
            
            # Map of normalized skill -> weight
            skill_weights = {}
            total_role_weight = 0.0
            for s in raw_skills:
                norm = normalize_skill(extract_skill_name(s), self.aliases)
                if not norm or norm in skill_weights:
                    continue
                w = extract_skill_weight(s)
                skill_weights[norm] = w
                total_role_weight += w

            # Compute Role TF-IDF vector
            tf_vec = {t: 0.0 for t in self.vocabulary}
            if total_role_weight > 0:
                for term, w in skill_weights.items():
                    if term in tf_vec:
                        tf_vec[term] = w / total_role_weight

            tfidf_vec = {t: tf_vec[t] * self.idf_map[t] for t in self.vocabulary}
            mag_sq = sum(v * v for v in tfidf_vec.values())

            self.role_profiles.append({
                'id': role_id,
                'role': role['role'],
                'category': role.get('category', 'General Technology'),
                'skill_weights': skill_weights,
                'total_weight': total_role_weight,
                'tfidf_vec': tfidf_vec,
                'mag_sq': mag_sq,
                'mag': math.sqrt(mag_sq)
            })

    def recommend(self, query_skills, k=3):
        norm_query = normalize_skill_list(query_skills, self.aliases)
        if not norm_query:
            return []

        # Compute User TF-IDF vector
        total_user_tokens = len(norm_query)
        user_counts = defaultdict(int)
        for t in norm_query:
            user_counts[t] += 1

        user_tfidf = {}
        user_mag_sq = 0.0
        for term in self.vocabulary:
            tf = user_counts[term] / total_user_tokens if term in user_counts else 0.0
            val = tf * self.idf_map[term]
            user_tfidf[term] = val
            user_mag_sq += val * val

        user_mag = math.sqrt(user_mag_sq)
        user_set = set(norm_query)

        scored = []
        for rp in self.role_profiles:
            # 1. Cosine similarity
            if user_mag == 0 or rp['mag'] == 0:
                cos_sim = 0.0
            else:
                dot = sum(
                    user_tfidf[t] * rp['tfidf_vec'][t]
                    for t in user_set
                    if t in rp['tfidf_vec']
                )
                cos_sim = dot / (user_mag * rp['mag'])
                cos_sim = max(0.0, min(1.0, cos_sim))

            # 2. Weighted skill coverage
            matched_weight = sum(
                rp['skill_weights'][s]
                for s in user_set
                if s in rp['skill_weights']
            )
            coverage = matched_weight / rp['total_weight'] if rp['total_weight'] > 0 else 0.0

            # 3. Final hybrid score
            final_score = self.alpha * cos_sim + (1.0 - self.alpha) * coverage

            matched_count = len([s for s in user_set if s in rp['skill_weights']])

            scored.append({
                'id': rp['id'],
                'role': rp['role'],
                'category': rp['category'],
                'score': final_score,
                'cosine_sim': cos_sim,
                'coverage': coverage,
                'matched_count': matched_count
            })

        # Deterministic multi-tier sort
        scored.sort(
            key=lambda x: (-x['score'], -x['cosine_sim'], -x['coverage'], -x['matched_count'], x['role'])
        )

        return scored[:k]


def run_evaluation():
    roles = load_json(os.path.join(DATA_DIR, 'roles.json'))
    aliases = load_json(os.path.join(DATA_DIR, 'aliases.json'))
    test_cases = load_json(os.path.join(EVAL_DIR, 'test_cases.json'))

    engine = RecommendationEngine(roles, aliases, alpha=0.65)

    print("================================================================")
    print("  SCIENTIFIC EVALUATION: CONTENT-BASED RECOMMENDATION ENGINE")
    print("================================================================\n")
    print(f"Corpus Catalog: {len(roles)} roles | {len(engine.vocabulary)} vocabulary terms")
    print(f"Evaluation Dataset: {len(test_cases)} standardized test cases\n")

    # Metrics storage
    p_at_3_list = []
    r_at_3_list = []
    reciprocal_ranks = []
    category_metrics = defaultdict(lambda: {'p3': [], 'r3': [], 'mrr': [], 'count': 0})
    recommended_role_ids = set()

    error_analysis = {
        'perfectMatchesRank1': [],
        'acceptableMatchesRank2or3': [],
        'missedRelevantRoles': []
    }

    # 1. Warm-up runs to ensure steady-state CPU cache
    for _ in range(5):
        for tc in test_cases:
            engine.recommend(tc['skills'], k=3)

    # 2. Timing benchmark
    total_time_ns = 0
    benchmark_iterations = 20

    for _ in range(benchmark_iterations):
        for tc in test_cases:
            start_t = time.perf_counter_ns()
            _ = engine.recommend(tc['skills'], k=3)
            total_time_ns += (time.perf_counter_ns() - start_t)

    avg_latency_ms = (total_time_ns / (benchmark_iterations * len(test_cases))) / 1_000_000.0

    # 3. Accuracy & IR metrics evaluation
    for tc in test_cases:
        tc_id = tc['id']
        category = tc.get('category', 'General')
        user_skills = tc['skills']
        relevant_targets = set(r.lower().strip() for r in tc['relevantRoles'])

        top_recs = engine.recommend(user_skills, k=3)
        top_rec_names = [r['role'] for r in top_recs]
        top_rec_ids = [r['id'] for r in top_recs]

        for rid in top_rec_ids:
            recommended_role_ids.add(rid)

        # Matched relevant roles in top-3
        hits = 0
        hit_rank = 0
        for rank_idx, rec in enumerate(top_recs, 1):
            is_relevant = (
                rec['role'].lower().strip() in relevant_targets or
                rec['id'].lower().strip() in relevant_targets
            )
            if is_relevant:
                hits += 1
                if hit_rank == 0:
                    hit_rank = rank_idx

        # Precision@3 = hits / 3
        p3 = hits / 3.0
        p_at_3_list.append(p3)

        # Recall@3 = hits / |Relevant|
        r3 = hits / len(relevant_targets) if relevant_targets else 0.0
        r_at_3_list.append(r3)

        # Reciprocal Rank = 1 / rank (or 0 if not found)
        rr = (1.0 / hit_rank) if hit_rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        # Track category metrics
        category_metrics[category]['p3'].append(p3)
        category_metrics[category]['r3'].append(r3)
        category_metrics[category]['mrr'].append(rr)
        category_metrics[category]['count'] += 1

        # Error analysis classification
        item_info = {
            'id': tc_id,
            'category': category,
            'skills': user_skills,
            'expectedRoles': tc['relevantRoles'],
            'topRecommendations': [
                {'rank': i + 1, 'role': r['role'], 'score': round(r['score'], 4), 'cosine': round(r['cosine_sim'], 4), 'coverage': round(r['coverage'], 4)}
                for i, r in enumerate(top_recs)
            ]
        }

        if hit_rank == 1:
            error_analysis['perfectMatchesRank1'].append(item_info)
        elif hit_rank in (2, 3):
            error_analysis['acceptableMatchesRank2or3'].append(item_info)
        else:
            error_analysis['missedRelevantRoles'].append(item_info)

    # Compute aggregates
    mean_p3 = sum(p_at_3_list) / len(p_at_3_list)
    mean_r3 = sum(r_at_3_list) / len(r_at_3_list)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    catalog_coverage = len(recommended_role_ids) / len(roles)

    # Per-category summary
    cat_summary = {}
    for cat, data in sorted(category_metrics.items()):
        cat_summary[cat] = {
            'numProfiles': data['count'],
            'precisionAt3': round(sum(data['p3']) / len(data['p3']), 4),
            'recallAt3': round(sum(data['r3']) / len(data['r3']), 4),
            'mrr': round(sum(data['mrr']) / len(data['mrr']), 4)
        }

    # Summary payload
    results = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'numTestCases': len(test_cases),
        'corpusRoleCount': len(roles),
        'precisionAt3': round(mean_p3, 4),
        'recallAt3': round(mean_r3, 4),
        'mrr': round(mrr, 4),
        'catalogCoverage': round(catalog_coverage, 4),
        'averageLatencyMs': round(avg_latency_ms, 3),
        'categoryBreakdown': cat_summary,
        'errorAnalysisSummary': {
            'rank1Count': len(error_analysis['perfectMatchesRank1']),
            'rank1Percentage': round(len(error_analysis['perfectMatchesRank1']) / len(test_cases) * 100, 2),
            'rank2or3Count': len(error_analysis['acceptableMatchesRank2or3']),
            'rank2or3Percentage': round(len(error_analysis['acceptableMatchesRank2or3']) / len(test_cases) * 100, 2),
            'missedCount': len(error_analysis['missedRelevantRoles']),
            'missedPercentage': round(len(error_analysis['missedRelevantRoles']) / len(test_cases) * 100, 2),
            'commonMissCauses': [
                'Generic shared skill sets (e.g. Python, SQL, REST APIs) distributed equally across multiple sub-specialties.',
                'Sub-field ambiguity where sibling roles (e.g., Data Scientist vs. Decision Scientist vs. ML Engineer) have overlapping core TF-IDF signatures.',
                'Input sparsity: queries with only 3 generalized skills have flatter vector projections compared to 5-skill queries.'
            ]
        },
        'detailedErrorAnalysis': {
            'missedProfiles': error_analysis['missedRelevantRoles'],
            'rank2or3Profiles': error_analysis['acceptableMatchesRank2or3']
        }
    }

    # Save to results.json
    output_path = os.path.join(EVAL_DIR, 'results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("----------------------------------------------------------------")
    print("  MEASURED EVALUATION RESULTS")
    print("----------------------------------------------------------------")
    print(f"* Precision@3:          {results['precisionAt3']:.4f} ({results['precisionAt3']*100:.1f}%)")
    print(f"* Recall@3:             {results['recallAt3']:.4f} ({results['recallAt3']*100:.1f}%)")
    print(f"* Mean Reciprocal Rank: {results['mrr']:.4f}")
    print(f"* Catalog Coverage:     {results['catalogCoverage']:.4f} ({results['catalogCoverage']*100:.1f}% of 118 roles)")
    print(f"* Avg Inference Latency: {results['averageLatencyMs']:.3f} ms / query\n")

    print("----------------------------------------------------------------")
    print("  RANK DISTRIBUTION & ERROR BREAKDOWN")
    print("----------------------------------------------------------------")
    print(f"* Rank #1 Match:        {results['errorAnalysisSummary']['rank1Count']}/{len(test_cases)} ({results['errorAnalysisSummary']['rank1Percentage']}%)")
    print(f"* Rank #2 or #3 Match:  {results['errorAnalysisSummary']['rank2or3Count']}/{len(test_cases)} ({results['errorAnalysisSummary']['rank2or3Percentage']}%)")
    print(f"* Missed (Outside Top3): {results['errorAnalysisSummary']['missedCount']}/{len(test_cases)} ({results['errorAnalysisSummary']['missedPercentage']}%)\n")

    print("----------------------------------------------------------------")
    print("  PER-CATEGORY PERFORMANCE BREAKDOWN")
    print("----------------------------------------------------------------")
    print(f"{'Category':<28} | {'Profiles':<8} | {'P@3':<8} | {'R@3':<8} | {'MRR':<8}")
    print("-" * 72)
    for cat, metrics in cat_summary.items():
        print(f"{cat:<28} | {metrics['numProfiles']:<8} | {metrics['precisionAt3']:<8.3f} | {metrics['recallAt3']:<8.3f} | {metrics['mrr']:<8.3f}")

    print("\n[OK] Results successfully generated and exported to evaluation/results.json\n")


if __name__ == '__main__':
    run_evaluation()
