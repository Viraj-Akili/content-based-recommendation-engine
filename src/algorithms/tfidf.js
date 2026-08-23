/**
 * TF-IDF (Term Frequency - Inverse Document Frequency) Vectorization Module.
 */
import { normalizeSkill } from '../utils/normalization.js';

/**
 * Extracts a normalized skill name from either a string or a skill object { name, weight }.
 *
 * @param {string | { name: string, weight?: number }} skillItem
 * @returns {string} Raw skill name string
 */
export function extractSkillName(skillItem) {
  if (!skillItem) return '';
  if (typeof skillItem === 'string') return skillItem;
  if (typeof skillItem === 'object' && typeof skillItem.name === 'string') {
    return skillItem.name;
  }
  return '';
}

/**
 * Extracts the explicit weight of a skill item (default 1.0).
 *
 * @param {string | { name: string, weight?: number }} skillItem
 * @returns {number} Weight in range [0, 1]
 */
export function extractSkillWeight(skillItem) {
  if (typeof skillItem === 'object' && typeof skillItem.weight === 'number') {
    return Math.max(0, Math.min(1, skillItem.weight));
  }
  return 1.0;
}

/**
 * Extracts a deduplicated, normalized vocabulary from a list of documents/roles.
 *
 * @param {Array<{ skills: Array<string | { name: string, weight?: number }> } | string[]>} documents - Role objects or skill arrays
 * @param {Record<string, string>} [aliasMap={}] - Optional alias dictionary
 * @returns {string[]} Sorted array of unique normalized vocabulary terms
 */
export function buildVocabulary(documents, aliasMap = {}) {
  if (!Array.isArray(documents)) return [];

  const vocabSet = new Set();

  for (const doc of documents) {
    const rawSkills = Array.isArray(doc) ? doc : (doc && Array.isArray(doc.skills) ? doc.skills : []);
    for (const skillItem of rawSkills) {
      const name = extractSkillName(skillItem);
      const normalized = normalizeSkill(name, aliasMap);
      if (normalized) {
        vocabSet.add(normalized);
      }
    }
  }

  return Array.from(vocabSet).sort();
}

/**
 * Computes the Document Frequency (DF) for each term in the vocabulary across all documents.
 * DF(t) = count of documents that contain term t.
 *
 * @param {Array<{ skills: Array<string | { name: string, weight?: number }> } | string[]>} documents - Corpus of documents/roles
 * @param {string[]} vocabulary - List of vocabulary terms
 * @param {Record<string, string>} [aliasMap={}] - Optional alias dictionary
 * @returns {Record<string, number>} Map of term -> document frequency count
 */
export function computeDocumentFrequency(documents, vocabulary, aliasMap = {}) {
  const dfMap = {};
  for (const term of vocabulary) {
    dfMap[term] = 0;
  }

  if (!Array.isArray(documents) || documents.length === 0) {
    return dfMap;
  }

  for (const doc of documents) {
    const rawSkills = Array.isArray(doc) ? doc : (doc && Array.isArray(doc.skills) ? doc.skills : []);
    const docSkillSet = new Set(
      rawSkills.map(s => normalizeSkill(extractSkillName(s), aliasMap)).filter(Boolean)
    );

    for (const term of vocabulary) {
      if (docSkillSet.has(term)) {
        dfMap[term] += 1;
      }
    }
  }

  return dfMap;
}

/**
 * Computes the Inverse Document Frequency (IDF) for all terms in the vocabulary.
 * Uses standard smoothed formula: IDF(t) = ln(N / (1 + DF(t)))
 *
 * @param {Array<{ skills: Array<string | { name: string, weight?: number }> } | string[]>} documents - Corpus of documents/roles
 * @param {string[]} vocabulary - List of vocabulary terms
 * @param {Object} [options={}] - Calculation options
 * @param {Record<string, string>} [options.aliasMap={}] - Alias dictionary
 * @param {'standard' | 'smooth' | 'probabilistic'} [options.formula='standard'] - IDF formula variant
 * @returns {Record<string, number>} Map of term -> IDF value
 */
export function computeIDF(documents, vocabulary, options = {}) {
  const { aliasMap = {}, formula = 'standard' } = options;
  const N = Array.isArray(documents) ? documents.length : 0;
  const dfMap = computeDocumentFrequency(documents, vocabulary, aliasMap);
  const idfMap = {};

  if (N === 0) {
    for (const term of vocabulary) {
      idfMap[term] = 0;
    }
    return idfMap;
  }

  for (const term of vocabulary) {
    const df = dfMap[term] || 0;
    if (formula === 'smooth') {
      // Smooth IDF: ln((1 + N) / (1 + df)) + 1
      idfMap[term] = Math.log((1 + N) / (1 + df)) + 1;
    } else if (formula === 'probabilistic') {
      // Probabilistic IDF: max(0, ln((N - df) / df))
      idfMap[term] = df > 0 ? Math.max(0, Math.log((N - df) / df)) : Math.log(N);
    } else {
      // Standard formula: ln(N / (1 + df))
      idfMap[term] = Math.log(N / (1 + df));
    }
  }

  return idfMap;
}

/**
 * Computes the Term Frequency (TF) vector for a list of tokens/weighted skills.
 *
 * @param {Array<string | { name: string, weight?: number }>} tokens - List of input skills or weighted objects
 * @param {string[]} vocabulary - Vocabulary terms
 * @param {Object} [options={}] - Options
 * @param {'relative' | 'raw' | 'log' | 'weighted'} [options.mode='relative'] - TF computation mode
 * @param {Record<string, string>} [options.aliasMap={}] - Alias dictionary
 * @returns {Record<string, number>} Map of term -> TF score
 */
export function computeTF(tokens, vocabulary, options = {}) {
  const { mode = 'relative', aliasMap = {} } = options;
  const tfVector = {};
  for (const term of vocabulary) {
    tfVector[term] = 0;
  }

  if (!Array.isArray(tokens) || tokens.length === 0) {
    return tfVector;
  }

  const tokenWeights = {};
  let totalWeight = 0;

  for (const item of tokens) {
    const rawName = extractSkillName(item);
    const normalized = normalizeSkill(rawName, aliasMap);
    if (!normalized) continue;

    const weight = extractSkillWeight(item);
    tokenWeights[normalized] = (tokenWeights[normalized] || 0) + weight;
    totalWeight += weight;
  }

  if (totalWeight === 0) {
    return tfVector;
  }

  for (const term of vocabulary) {
    const weight = tokenWeights[term] || 0;
    if (mode === 'raw') {
      tfVector[term] = weight;
    } else if (mode === 'log') {
      tfVector[term] = weight > 0 ? 1 + Math.log(weight) : 0;
    } else {
      // 'relative' / 'weighted' normalized by sum of weights
      tfVector[term] = weight / totalWeight;
    }
  }

  return tfVector;
}

/**
 * Constructs a TF-IDF vector for a given skill list.
 *
 * @param {Array<string | { name: string, weight?: number }>} tokens - Input skills or weighted skill objects
 * @param {Record<string, number>} idfMap - Computed IDF lookup map
 * @param {string[]} vocabulary - Vocabulary list
 * @param {Object} [options={}] - Options
 * @param {boolean} [options.normalizeL2=false] - Whether to apply L2 normalization
 * @param {Record<string, string>} [options.aliasMap={}] - Alias dictionary
 * @returns {Record<string, number>} TF-IDF weight vector
 */
export function computeTFIDFVector(tokens, idfMap, vocabulary, options = {}) {
  const { normalizeL2 = false, aliasMap = {} } = options;
  const tfVector = computeTF(tokens, vocabulary, { mode: 'relative', aliasMap });
  const tfidfVector = {};

  let sumSquares = 0;
  for (const term of vocabulary) {
    const tf = tfVector[term] || 0;
    const idf = idfMap[term] || 0;
    const val = tf * idf;
    tfidfVector[term] = val;
    sumSquares += val * val;
  }

  if (normalizeL2 && sumSquares > 0) {
    const norm = Math.sqrt(sumSquares);
    for (const term of vocabulary) {
      tfidfVector[term] = tfidfVector[term] / norm;
    }
  }

  return tfidfVector;
}
