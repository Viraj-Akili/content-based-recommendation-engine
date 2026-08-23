/**
 * Input validation utilities for skill vectors and user queries.
 */
import { normalizeSkill, normalizeSkillList } from './normalization.js';

/**
 * Validates a user's skill input list against constraints and known vocabulary.
 *
 * @param {string[]} rawSkills - Array of input skill strings from UI or API
 * @param {Object} options - Validation options
 * @param {number} [options.minSkills=3] - Minimum required skills
 * @param {number} [options.maxSkills=10] - Maximum allowable skills
 * @param {Set<string>|string[]} [options.knownVocabulary] - Optional set of known normalized vocabulary terms
 * @param {Record<string, string>} [options.aliasMap={}] - Alias dictionary for normalization
 * @returns {{
 *   isValid: boolean,
 *   normalizedSkills: string[],
 *   rawCount: number,
 *   uniqueCount: number,
 *   unknownSkills: string[],
 *   duplicateSkills: string[],
 *   error: string | null,
 *   warnings: string[]
 * }}
 */
export function validateSkillInputs(rawSkills, options = {}) {
  const {
    minSkills = 3,
    maxSkills = 10,
    knownVocabulary = null,
    aliasMap = {}
  } = options;

  const rawList = Array.isArray(rawSkills) ? rawSkills.filter(s => typeof s === 'string' && s.trim()) : [];
  const normalizedList = [];
  const duplicates = new Set();
  const seen = new Set();

  for (const raw of rawList) {
    const norm = normalizeSkill(raw, aliasMap);
    if (!norm) continue;

    if (seen.has(norm)) {
      duplicates.add(norm);
    } else {
      seen.add(norm);
      normalizedList.push(norm);
    }
  }

  const vocabSet = knownVocabulary
    ? (knownVocabulary instanceof Set ? knownVocabulary : new Set(knownVocabulary.map(v => v.toLowerCase().trim())))
    : null;

  const unknownSkills = [];
  if (vocabSet) {
    for (const skill of normalizedList) {
      if (!vocabSet.has(skill)) {
        unknownSkills.push(skill);
      }
    }
  }

  const warnings = [];
  if (duplicates.size > 0) {
    warnings.push(`Duplicate skills ignored: ${Array.from(duplicates).join(', ')}`);
  }
  if (unknownSkills.length > 0) {
    warnings.push(`Unknown skills not in dictionary: ${unknownSkills.join(', ')} (will have zero corpus weight)`);
  }

  let error = null;
  if (normalizedList.length < minSkills) {
    error = `Please enter at least ${minSkills} distinct skills (currently provided: ${normalizedList.length}).`;
  } else if (normalizedList.length > maxSkills) {
    error = `Please enter no more than ${maxSkills} skills (currently provided: ${normalizedList.length}).`;
  }

  return {
    isValid: error === null,
    normalizedSkills: normalizedList,
    rawCount: rawList.length,
    uniqueCount: normalizedList.length,
    unknownSkills,
    duplicateSkills: Array.from(duplicates),
    error,
    warnings
  };
}
