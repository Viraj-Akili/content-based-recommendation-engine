/**
 * Normalization utilities for skill tokens and text processing.
 */

/**
 * Normalizes a single skill string:
 * - Trims leading/trailing whitespace
 * - Converts to lowercase
 * - Collapses consecutive whitespace
 * - Resolves aliases if an alias dictionary is provided
 * - Preserves technical symbols (e.g. C++, C#, Node.js, CI/CD, UI/UX)
 *
 * @param {string} rawSkill - The raw skill string input by user or from dataset
 * @param {Record<string, string>} [aliasMap={}] - Mapping of alias -> canonical normalized name
 * @returns {string} The normalized skill string
 */
export function normalizeSkill(rawSkill, aliasMap = {}) {
  if (typeof rawSkill !== 'string') return '';

  let cleaned = rawSkill
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ');

  if (!cleaned) return '';

  // Check alias map
  if (aliasMap && Object.prototype.hasOwnProperty.call(aliasMap, cleaned)) {
    cleaned = aliasMap[cleaned].trim().toLowerCase();
  }

  return cleaned;
}

/**
 * Normalizes an array of skills, filtering empty values and removing duplicates.
 *
 * @param {string[]} skillList - Array of skill strings
 * @param {Record<string, string>} [aliasMap={}] - Mapping of alias -> canonical normalized name
 * @returns {string[]} Array of unique, normalized skills
 */
export function normalizeSkillList(skillList, aliasMap = {}) {
  if (!Array.isArray(skillList)) return [];

  const seen = new Set();
  const result = [];

  for (const raw of skillList) {
    const normalized = normalizeSkill(raw, aliasMap);
    if (normalized && !seen.has(normalized)) {
      seen.add(normalized);
      result.push(normalized);
    }
  }

  return result;
}

/**
 * Builds a lookup map from normalized skill string to its canonical display name.
 *
 * @param {string[]} canonicalSkills - List of display-formatted skill strings (e.g. "Machine Learning", "Node.js")
 * @returns {Map<string, string>} Map from lowercase/normalized key to canonical display string
 */
export function buildCanonicalSkillMap(canonicalSkills = []) {
  const map = new Map();
  for (const skill of canonicalSkills) {
    if (typeof skill === 'string' && skill.trim()) {
      map.set(skill.trim().toLowerCase(), skill.trim());
    }
  }
  return map;
}

/**
 * Retrieves the display/canonical name for a normalized skill, falling back to capitalized form.
 *
 * @param {string} normalizedSkill - Normalized skill string
 * @param {Map<string, string>|Record<string, string>} [canonicalMap] - Lookup map of canonical names
 * @returns {string} Formatted display name
 */
export function getCanonicalSkill(normalizedSkill, canonicalMap) {
  if (!normalizedSkill) return '';
  const key = normalizedSkill.toLowerCase().trim();

  if (canonicalMap instanceof Map && canonicalMap.has(key)) {
    return canonicalMap.get(key);
  }
  if (canonicalMap && typeof canonicalMap === 'object' && Object.prototype.hasOwnProperty.call(canonicalMap, key)) {
    return canonicalMap[key];
  }

  // Fallback: Title Case capitalization
  return key
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Escapes special HTML characters to prevent XSS injection.
 *
 * @param {string} text - Raw input text
 * @returns {string} Sanitized string safe for HTML interpolation
 */
export function escapeHtml(text) {
  if (typeof text !== 'string') return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

