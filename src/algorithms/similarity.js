/**
 * Similarity metric calculations and vector algebra utilities.
 */

/**
 * Computes the dot product between two numerical vectors over a vocabulary.
 *
 * @param {Record<string, number>} vecA - First vector
 * @param {Record<string, number>} vecB - Second vector
 * @param {string[]} [vocabulary] - Optional vocabulary keys list; defaults to object keys
 * @returns {number} Dot product scalar
 */
export function dotProduct(vecA, vecB, vocabulary) {
  if (!vecA || !vecB) return 0;

  const keys = vocabulary || Array.from(new Set([...Object.keys(vecA), ...Object.keys(vecB)]));
  let dot = 0;

  for (const key of keys) {
    const a = Number(vecA[key]) || 0;
    const b = Number(vecB[key]) || 0;
    dot += a * b;
  }

  return dot;
}

/**
 * Computes the Euclidean norm (L2 magnitude) of a vector.
 *
 * @param {Record<string, number>} vec - Vector
 * @param {string[]} [vocabulary] - Optional vocabulary keys list
 * @returns {number} Vector magnitude (L2 norm)
 */
export function vectorMagnitude(vec, vocabulary) {
  if (!vec) return 0;

  const keys = vocabulary || Object.keys(vec);
  let sumSq = 0;

  for (const key of keys) {
    const val = Number(vec[key]) || 0;
    sumSq += val * val;
  }

  return Math.sqrt(sumSq);
}

/**
 * Computes the Cosine Similarity between two vectors with comprehensive safeguards:
 * Cosine Similarity = (A . B) / (||A|| * ||B||)
 *
 * @param {Record<string, number>} vecA - User profile vector
 * @param {Record<string, number>} vecB - Candidate role vector
 * @param {string[]} [vocabulary] - Vocabulary key list
 * @returns {number} Cosine similarity in range [0.0, 1.0] (for non-negative TF-IDF)
 */
export function cosineSimilarity(vecA, vecB, vocabulary) {
  if (!vecA || !vecB) return 0;

  const keys = vocabulary || Array.from(new Set([...Object.keys(vecA), ...Object.keys(vecB)]));
  let dot = 0;
  let magASq = 0;
  let magBSq = 0;

  for (const term of keys) {
    const a = Number(vecA[term]) || 0;
    const b = Number(vecB[term]) || 0;

    if (a !== 0 && b !== 0) {
      dot += a * b;
    }
    if (a !== 0) magASq += a * a;
    if (b !== 0) magBSq += b * b;
  }

  // Safeguards for empty or zero vectors
  if (magASq === 0 || magBSq === 0 || dot === 0) {
    return 0;
  }

  const denominator = Math.sqrt(magASq) * Math.sqrt(magBSq);
  if (denominator === 0 || !Number.isFinite(denominator)) {
    return 0;
  }

  const similarity = dot / denominator;

  // Numerical precision clamping
  if (Number.isNaN(similarity)) return 0;
  if (similarity < 0) return 0;
  if (similarity > 1) return 1;

  return similarity;
}
