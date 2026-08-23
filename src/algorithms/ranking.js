/**
 * Ranking, Hybrid Scoring, and Explainability Recommendation Module.
 */
import { normalizeSkill, normalizeSkillList, getCanonicalSkill, buildCanonicalSkillMap } from '../utils/normalization.js';
import { buildVocabulary, computeIDF, computeTFIDFVector, extractSkillName, extractSkillWeight } from './tfidf.js';
import { cosineSimilarity } from './similarity.js';

/**
 * @typedef {Object} SkillDetail
 * @property {string} name - Canonical skill name
 * @property {number} weight - Role-specific skill weight [0.0, 1.0]
 * @property {boolean} isCore - True if weight >= 0.8
 * @property {number} [userWeight] - Normalized user TF-IDF weight
 * @property {number} [contribution] - Numerical contribution to cosine dot product
 */

/**
 * @typedef {Object} RecommendationExplanation
 * @property {SkillDetail[]} matchedSkills - Detailed breakdown of matched skills
 * @property {SkillDetail[]} missingSkills - Detailed breakdown of missing skills
 * @property {SkillDetail[]} missingCoreSkills - High-importance missing skills (weight >= 0.8)
 * @property {Array<{ skill: string, contribution: number }>} topContributors - Strongest matching skills
 * @property {number} coverage - Weighted skill coverage ratio [0.0, 1.0]
 * @property {string} summary - Human-interpretable explanation summary
 */

/**
 * @typedef {Object} RecommendationResult
 * @property {string} id - Unique role identifier
 * @property {string} role - Role title
 * @property {string} category - Career category
 * @property {string} description - Role description
 * @property {number} score - Final combined hybrid score [0.0, 1.0]
 * @property {number} finalRankingScore - Explicit alias for final hybrid score
 * @property {number} cosineSimilarity - Pure TF-IDF cosine similarity score [0.0, 1.0]
 * @property {number} rawCosineSimilarity - Backwards-compatible alias for cosine similarity
 * @property {number} weightedSkillCoverage - Ratio of matched skill weight to total role skill weight [0.0, 1.0]
 * @property {number} skillCoveragePercentage - Integer percentage of weighted coverage (0-100)
 * @property {number} matchedSkillCount - Count of overlapping user skills
 * @property {number} totalRoleSkillCount - Total skills in role
 * @property {number} weightedMatchedScore - Sum of weights of matched skills
 * @property {number} totalRoleSkillWeight - Sum of all skill weights in role
 * @property {string[]} matchedSkills - Canonical display names of matched skills
 * @property {string[]} missingSkills - Canonical display names of missing skills
 * @property {string[]} missingCoreSkills - Canonical display names of missing core skills (weight >= 0.8)
 * @property {string[]} relatedRoles - Associated role IDs
 * @property {RecommendationExplanation} explanation - Structured explainability payload
 * @property {Array<{ term: string, displayTerm: string, weight: number }>} featureWeights - User TF-IDF feature weights
 */

/**
 * Computes recommendations and ranks candidate career roles against a user's skill set.
 *
 * @param {string[]} rawUserSkills - List of input skills provided by the user
 * @param {Array<{
 *   id?: string,
 *   role: string,
 *   category?: string,
 *   description: string,
 *   skills: Array<string | { name: string, weight: number }>,
 *   relatedRoles?: string[]
 * }>} roles - List of candidate career roles
 * @param {Object} [options={}] - Configuration options
 * @param {Record<string, string>} [options.aliasMap={}] - Skill alias mappings
 * @param {string[]} [options.vocabulary] - Optional precomputed vocabulary
 * @param {Record<string, number>} [options.idfMap] - Optional precomputed IDF map
 * @param {Map<string, string>} [options.canonicalMap] - Optional canonical name lookup map
 * @param {number} [options.alpha=0.65] - Hybrid weight: alpha * cosineSimilarity + (1 - alpha) * weightedCoverage
 * @returns {{
 *   recommendations: RecommendationResult[],
 *   userNormalizedSkills: string[],
 *   userFeatureWeights: Array<{ term: string, displayTerm: string, weight: number }>,
 *   hasValidQuery: boolean
 * }}
 */
export function rankRoles(rawUserSkills, roles, options = {}) {
  const {
    aliasMap = {},
    vocabulary: preVocab = null,
    idfMap: preIdf = null,
    canonicalMap: preCanonical = null,
    alpha = 0.65
  } = options;

  if (!Array.isArray(roles) || roles.length === 0) {
    return {
      recommendations: [],
      userNormalizedSkills: [],
      userFeatureWeights: [],
      hasValidQuery: false
    };
  }

  // 1. Build or use cached vocabulary & canonical maps
  const vocabulary = preVocab || buildVocabulary(roles, aliasMap);
  const idfMap = preIdf || computeIDF(roles, vocabulary, { aliasMap });
  const canonicalMap = preCanonical || buildCanonicalSkillMap(
    roles.flatMap(r => (r.skills || []).map(s => extractSkillName(s)))
  );

  // 2. Normalize and deduplicate user skills
  const userNormalizedSkills = normalizeSkillList(rawUserSkills, aliasMap);
  const userSkillSet = new Set(userNormalizedSkills);

  // 3. Compute user TF-IDF vector
  const userVec = computeTFIDFVector(userNormalizedSkills, idfMap, vocabulary, { aliasMap });

  // 4. Extract user non-zero feature weights sorted descending
  const userFeatureWeights = vocabulary
    .map(term => ({
      term,
      displayTerm: getCanonicalSkill(term, canonicalMap),
      weight: userVec[term] || 0
    }))
    .filter(item => item.weight > 0)
    .sort((a, b) => b.weight - a.weight);

  const hasValidQuery = userNormalizedSkills.length > 0 && userFeatureWeights.length > 0;

  // 5. Score each role
  const scoredRoles = roles.map(roleItem => {
    const roleId = roleItem.id || roleItem.role.toLowerCase().replace(/\s+/g, '-');
    const rawRoleSkills = Array.isArray(roleItem.skills) ? roleItem.skills : [];

    // Parse role skills with weights and normalized keys
    const roleSkillEntries = [];
    const seenRoleSkills = new Set();
    let totalRoleSkillWeight = 0;

    for (const item of rawRoleSkills) {
      const name = extractSkillName(item);
      const normalized = normalizeSkill(name, aliasMap);
      if (!normalized || seenRoleSkills.has(normalized)) continue;

      seenRoleSkills.add(normalized);
      const weight = extractSkillWeight(item);
      const canonicalName = getCanonicalSkill(normalized, canonicalMap) || name;

      roleSkillEntries.push({
        normalized,
        canonicalName,
        weight,
        isCore: weight >= 0.8
      });
      totalRoleSkillWeight += weight;
    }

    // Build role vector & compute cosine similarity
    const roleVec = computeTFIDFVector(rawRoleSkills, idfMap, vocabulary, { aliasMap });
    const cosSim = hasValidQuery ? cosineSimilarity(userVec, roleVec, vocabulary) : 0;

    // Calculate matched vs missing skills and individual contributions
    let weightedMatchedScore = 0;
    const matchedSkillDetails = [];
    const missingSkillDetails = [];
    const contributions = [];

    for (const skillEntry of roleSkillEntries) {
      const isMatched = userSkillSet.has(skillEntry.normalized);
      const userTermWeight = userVec[skillEntry.normalized] || 0;
      const roleTermWeight = roleVec[skillEntry.normalized] || 0;
      const dotContribution = userTermWeight * roleTermWeight;

      if (isMatched) {
        weightedMatchedScore += skillEntry.weight;
        matchedSkillDetails.push({
          name: skillEntry.canonicalName,
          weight: skillEntry.weight,
          isCore: skillEntry.isCore,
          userWeight: Number(userTermWeight.toFixed(4)),
          contribution: Number(dotContribution.toFixed(4))
        });

        contributions.push({
          skill: skillEntry.canonicalName,
          contribution: dotContribution
        });
      } else {
        missingSkillDetails.push({
          name: skillEntry.canonicalName,
          weight: skillEntry.weight,
          isCore: skillEntry.isCore
        });
      }
    }

    // Sort missing skills by weight descending (highest priority missing first)
    missingSkillDetails.sort((a, b) => b.weight - a.weight);

    // Filter missing core skills (weight >= 0.8)
    const missingCoreSkills = missingSkillDetails.filter(s => s.isCore);

    // Sort contributors by contribution descending
    contributions.sort((a, b) => b.contribution - a.contribution);

    // Compute weighted coverage [0.0, 1.0]
    const weightedSkillCoverage = totalRoleSkillWeight > 0
      ? (weightedMatchedScore / totalRoleSkillWeight)
      : 0;

    // Hybrid final ranking score: alpha * cosineSimilarity + (1 - alpha) * weightedSkillCoverage
    const finalScore = hasValidQuery
      ? (alpha * cosSim + (1 - alpha) * weightedSkillCoverage)
      : 0;

    // Construct human-readable explainability summary
    let explanationSummary = '';
    if (matchedSkillDetails.length > 0) {
      const topMatchedNames = matchedSkillDetails.slice(0, 3).map(s => s.name).join(', ');
      explanationSummary = `Matched on key skills: ${topMatchedNames}.`;
      if (missingCoreSkills.length > 0) {
        explanationSummary += ` Recommended next skill to acquire: ${missingCoreSkills[0].name}.`;
      } else {
        explanationSummary += ` Strong overall coverage of required role capabilities.`;
      }
    } else {
      explanationSummary = 'No overlapping technical skills detected for this role.';
    }

    const explanation = {
      matchedSkills: matchedSkillDetails,
      missingSkills: missingSkillDetails,
      missingCoreSkills,
      topContributors: contributions.slice(0, 5),
      coverage: Number(weightedSkillCoverage.toFixed(4)),
      summary: explanationSummary
    };

    return {
      id: roleId,
      role: roleItem.role,
      category: roleItem.category || 'General Technology',
      description: roleItem.description || '',
      score: Number(finalScore.toFixed(4)),
      finalRankingScore: Number(finalScore.toFixed(4)),
      cosineSimilarity: Number(cosSim.toFixed(4)),
      rawCosineSimilarity: cosSim,
      weightedSkillCoverage: Number(weightedSkillCoverage.toFixed(4)),
      skillCoveragePercentage: Math.round(weightedSkillCoverage * 100),
      matchedSkillCount: matchedSkillDetails.length,
      totalRoleSkillCount: roleSkillEntries.length,
      weightedMatchedScore: Number(weightedMatchedScore.toFixed(2)),
      totalRoleSkillWeight: Number(totalRoleSkillWeight.toFixed(2)),
      matchedSkills: matchedSkillDetails.map(s => s.name),
      missingSkills: missingSkillDetails.map(s => s.name),
      missingCoreSkills: missingCoreSkills.map(s => s.name),
      relatedRoles: Array.isArray(roleItem.relatedRoles) ? roleItem.relatedRoles : [],
      explanation,
      featureWeights: userFeatureWeights
    };
  });

  // 6. Deterministic multi-tier sort:
  // 1st: Final Ranking Score descending
  // 2nd: Cosine Similarity descending
  // 3rd: Weighted Skill Coverage descending
  // 4th: Matched Skill Count descending
  // 5th: Role Title alphabetical ascending
  scoredRoles.sort((a, b) => {
    if (b.finalRankingScore !== a.finalRankingScore) return b.finalRankingScore - a.finalRankingScore;
    if (b.cosineSimilarity !== a.cosineSimilarity) return b.cosineSimilarity - a.cosineSimilarity;
    if (b.weightedSkillCoverage !== a.weightedSkillCoverage) return b.weightedSkillCoverage - a.weightedSkillCoverage;
    if (b.matchedSkillCount !== a.matchedSkillCount) return b.matchedSkillCount - a.matchedSkillCount;
    return a.role.localeCompare(b.role);
  });

  return {
    recommendations: scoredRoles,
    userNormalizedSkills,
    userFeatureWeights,
    hasValidQuery
  };
}

/**
 * Extracts the top K recommendations from a ranked recommendation list.
 *
 * @param {RecommendationResult[]} recommendations - Array of ranked recommendations
 * @param {number} [k=3] - Number of top results to return
 * @returns {RecommendationResult[]} Top K recommendations
 */
export function getTopKRecommendations(recommendations, k = 3) {
  if (!Array.isArray(recommendations)) return [];
  const limit = Math.max(1, Math.min(k, recommendations.length));
  return recommendations.slice(0, limit);
}
