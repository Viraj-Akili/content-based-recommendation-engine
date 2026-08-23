/**
 * Automated Verification & Testing Suite for Content-Based Recommendation Engine.
 * Run with: node tests/run-tests.js
 */

import { normalizeSkill, normalizeSkillList, getCanonicalSkill, buildCanonicalSkillMap, escapeHtml } from '../src/utils/normalization.js';
import { validateSkillInputs } from '../src/utils/validation.js';
import { buildVocabulary, computeDocumentFrequency, computeIDF, computeTF, computeTFIDFVector } from '../src/algorithms/tfidf.js';
import { cosineSimilarity, dotProduct, vectorMagnitude } from '../src/algorithms/similarity.js';
import { rankRoles, getTopKRecommendations } from '../src/algorithms/ranking.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load datasets from data/ directory
const roles = JSON.parse(fs.readFileSync(path.join(__dirname, '../data/roles.json'), 'utf8'));
const skills = JSON.parse(fs.readFileSync(path.join(__dirname, '../data/skills.json'), 'utf8'));
const aliases = JSON.parse(fs.readFileSync(path.join(__dirname, '../data/aliases.json'), 'utf8'));
const canonicalMap = buildCanonicalSkillMap(skills);

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;

function assert(condition, message) {
  totalTests++;
  if (condition) {
    passedTests++;
    console.log(`  ✓ ${message}`);
  } else {
    failedTests++;
    console.error(`  ✗ FAIL: ${message}`);
  }
}

function assertEquals(actual, expected, message) {
  const isMatch = actual === expected;
  assert(isMatch, `${message} (Expected: ${expected}, Actual: ${actual})`);
}

function assertClose(actual, expected, tolerance = 0.001, message) {
  const diff = Math.abs(actual - expected);
  assert(diff <= tolerance, `${message} (Expected ~${expected}, Actual: ${actual}, Diff: ${diff.toFixed(4)})`);
}

console.log('================================================================');
console.log('  RUNNING CONTENT-BASED RECOMMENDATION ENGINE TEST SUITE');
console.log('================================================================\n');

// -------------------------------------------------------------
// 1. Normalization & Alias Resolution Tests
// -------------------------------------------------------------
console.log('--- 1. Normalization & Alias Resolution Tests ---');

assertEquals(normalizeSkill('  Python  '), 'python', 'Trims and lowercases skill');
assertEquals(normalizeSkill('k8s', aliases), 'kubernetes', 'Resolves alias k8s -> kubernetes');
assertEquals(normalizeSkill('ML', aliases), 'machine learning', 'Resolves alias ML -> machine learning');
assertEquals(normalizeSkill('AI', aliases), 'deep learning', 'Resolves alias AI -> deep learning');
assertEquals(normalizeSkill('JS', aliases), 'javascript', 'Resolves alias JS -> javascript');
assertEquals(normalizeSkill('TS', aliases), 'typescript', 'Resolves alias TS -> typescript');
assertEquals(normalizeSkill('postgres', aliases), 'postgresql', 'Resolves alias postgres -> postgresql');
assertEquals(normalizeSkill('mongo', aliases), 'mongodb', 'Resolves alias mongo -> mongodb');
assertEquals(normalizeSkill('TF', aliases), 'tensorflow', 'Resolves alias TF -> tensorflow');
assertEquals(normalizeSkill('C++', aliases), 'c++', 'Preserves C++ symbol');
assertEquals(normalizeSkill('C#', aliases), 'c#', 'Preserves C# symbol');
assertEquals(normalizeSkill('CI/CD', aliases), 'ci/cd', 'Preserves CI/CD symbol');

const rawList = ['  Python', 'python', 'py', 'k8s', 'Kubernetes', '   '];
const normalizedList = normalizeSkillList(rawList, aliases);
assertEquals(normalizedList.length, 2, 'Deduplicates normalized/alias skills (Python & Kubernetes)');
assert(normalizedList.includes('python') && normalizedList.includes('kubernetes'), 'Contains canonical normalized skills');

assertEquals(getCanonicalSkill('kubernetes', canonicalMap), 'Kubernetes', 'Canonical display name for kubernetes');
assertEquals(getCanonicalSkill('machine learning', canonicalMap), 'Machine Learning', 'Canonical display name for machine learning');

// -------------------------------------------------------------
// 2. Input Validation Tests
// -------------------------------------------------------------
console.log('\n--- 2. Input Validation Tests ---');

const vocab = buildVocabulary(roles, aliases);

const valFew = validateSkillInputs(['Python', 'SQL'], { minSkills: 3, knownVocabulary: vocab, aliasMap: aliases });
assertEquals(valFew.isValid, false, 'Rejects fewer than 3 skills');
assert(valFew.error.includes('at least 3'), 'Returns descriptive error for <3 skills');

const valDupes = validateSkillInputs(['Python', 'python', 'py', 'SQL', 'Machine Learning'], { minSkills: 3, knownVocabulary: vocab, aliasMap: aliases });
assertEquals(valDupes.isValid, true, 'Accepts 3 unique skills despite alias/casing duplicates');
assertEquals(valDupes.uniqueCount, 3, 'Counts exactly 3 unique skills');
assert(valDupes.duplicateSkills.length > 0, 'Records duplicate warnings');

const valUnknown = validateSkillInputs(['Python', 'SQL', 'FlurbGlock9000'], { minSkills: 3, knownVocabulary: vocab, aliasMap: aliases });
assertEquals(valUnknown.isValid, true, 'Allows unknown skills to proceed with warnings');
assertEquals(valUnknown.unknownSkills.length, 1, 'Detects 1 unknown skill');
assertEquals(valUnknown.unknownSkills[0], 'flurbglock9000', 'Identifies exact unknown skill token');

// -------------------------------------------------------------
// 3. TF-IDF Algorithmic Tests with Weighted Skills
// -------------------------------------------------------------
console.log('\n--- 3. TF-IDF Vectorization Tests ---');

assert(vocab.length >= 100, `Vocabulary extracted (${vocab.length} terms)`);

const dfMap = computeDocumentFrequency(roles, vocab, aliases);
assert(dfMap['python'] > 1, `Python appears in multiple roles (DF = ${dfMap['python']})`);
assert(dfMap['qiskit'] >= 1, `Qiskit is a domain-specific skill (DF = ${dfMap['qiskit']})`);

const idfMap = computeIDF(roles, vocab, { aliasMap: aliases });
assert(idfMap['qiskit'] > idfMap['python'], 'Rare skill (Qiskit) has higher IDF than frequent skill (Python)');

const tfVec = computeTF(['python', 'python', 'sql'], vocab, { aliasMap: aliases });
assertClose(tfVec['python'], 2 / 3, 0.001, 'TF for Python with 2 occurrences out of 3 tokens is 2/3');
assertClose(tfVec['sql'], 1 / 3, 0.001, 'TF for SQL with 1 occurrence out of 3 tokens is 1/3');
assertEquals(tfVec['docker'], 0, 'TF for unmentioned skill is 0');

const tfidfVec = computeTFIDFVector(['python', 'sql'], idfMap, vocab, { aliasMap: aliases });
assert(tfidfVec['python'] > 0, 'TF-IDF weight for Python is positive');
assert(tfidfVec['sql'] > 0, 'TF-IDF weight for SQL is positive');
assertEquals(tfidfVec['docker'], 0, 'TF-IDF weight for Docker is 0');

// -------------------------------------------------------------
// 4. Similarity Metric & Vector Safeguard Tests
// -------------------------------------------------------------
console.log('\n--- 4. Cosine Similarity & Vector Safeguard Tests ---');

const vecA = { 'python': 0.8, 'sql': 0.6 };
const vecB = { 'python': 0.8, 'sql': 0.6 };
const vecC = { 'docker': 1.0, 'kubernetes': 1.0 };
const vecEmpty = {};
const vecZero = { 'python': 0, 'sql': 0 };

assertClose(cosineSimilarity(vecA, vecB), 1.0, 0.0001, 'Cosine similarity of identical vectors is 1.0');
assertEquals(cosineSimilarity(vecA, vecC), 0, 'Cosine similarity of disjoint orthogonal vectors is 0.0');
assertEquals(cosineSimilarity(vecA, vecEmpty), 0, 'Safeguard: Empty vector yields similarity 0.0');
assertEquals(cosineSimilarity(vecA, vecZero), 0, 'Safeguard: Zero magnitude vector yields similarity 0.0');
assertEquals(cosineSimilarity(null, vecA), 0, 'Safeguard: Null vector yields 0.0');

// -------------------------------------------------------------
// 5. 5+ Career Profile Tests & Explainability Verification
// -------------------------------------------------------------
console.log('\n--- 5. Career Profile Recommendation Tests ---');

// Profile 1: AI / ML Engineer Profile
console.log('Test Profile 1: Machine Learning & MLOps');
const mlOutput = rankRoles(['Python', 'Machine Learning', 'PyTorch', 'MLOps'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const topML = getTopKRecommendations(mlOutput.recommendations, 3);
assertEquals(topML.length, 3, 'Returns top 3 recommendations');
assert(topML[0].role.includes('Machine Learning') || topML[0].role.includes('Deep Learning') || topML[0].role.includes('MLOps'), `Top ML match is appropriate (Got: ${topML[0].role})`);
assert(topML[0].matchedSkillCount >= 3, `Matched skill count is at least 3 (Got: ${topML[0].matchedSkillCount})`);
assert(topML[0].weightedSkillCoverage > 0.3, `Weighted coverage is strong (Got: ${topML[0].weightedSkillCoverage})`);
assert(topML[0].explanation.summary.length > 10, 'Generates explanatory summary');

// Profile 2: Cloud / DevOps Profile
console.log('Test Profile 2: DevOps & Cloud');
const devopsOutput = rankRoles(['Docker', 'k8s', 'AWS', 'Terraform'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const topDevOps = getTopKRecommendations(devopsOutput.recommendations, 3);
assert(topDevOps[0].category === 'Cloud / DevOps', `Top match is in Cloud / DevOps category (Got: ${topDevOps[0].category} - ${topDevOps[0].role})`);
assert(topDevOps[0].matchedSkills.includes('Kubernetes'), 'Resolves k8s alias to Kubernetes in matched skills');

// Profile 3: Modern Web / Frontend Profile
console.log('Test Profile 3: Frontend & UI');
const feOutput = rankRoles(['js', 'React', 'HTML', 'CSS', 'TypeScript'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const topFE = getTopKRecommendations(feOutput.recommendations, 3);
assert(topFE[0].category === 'Web Development', `Top match is in Web Development (Got: ${topFE[0].category} - ${topFE[0].role})`);
assert(topFE[0].matchedSkills.includes('JavaScript'), 'Resolves js alias to JavaScript');

// Profile 4: Cybersecurity Profile
console.log('Test Profile 4: Cybersecurity');
const secOutput = rankRoles(['Security', 'Penetration Testing', 'Linux', 'Python'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const topSec = getTopKRecommendations(secOutput.recommendations, 3);
assert(topSec[0].category === 'Cybersecurity', `Top match is in Cybersecurity category (Got: ${topSec[0].category} - ${topSec[0].role})`);
assert(topSec[0].explanation.matchedSkills.some(s => s.name === 'Penetration Testing' && s.isCore), 'Identifies Penetration Testing as core matched skill');

// Profile 5: Embedded / Robotics Profile
console.log('Test Profile 5: Embedded & Robotics');
const robOutput = rankRoles(['C++', 'ROS', 'Computer Vision', 'Linux'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const topRob = getTopKRecommendations(robOutput.recommendations, 3);
assert(topRob[0].role === 'Robotics Software Engineer' || topRob[0].role === 'Autonomous Vehicles Systems Engineer', `Top robotics match (Got: ${topRob[0].role})`);

// Profile 6: Quantum Computing Profile
console.log('Test Profile 6: Quantum Computing');
const quantumOutput = rankRoles(['Python', 'Qiskit', 'Quantum', 'Physics'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const topQuantum = getTopKRecommendations(quantumOutput.recommendations, 1);
assertEquals(topQuantum[0].role, 'Quantum Computing Engineer', 'High-IDF domain terms reliably match Quantum Computing Engineer');

// Profile 7: Blockchain Profile
console.log('Test Profile 7: Blockchain / Smart Contracts');
const web3Output = rankRoles(['Solidity', 'Smart Contracts', 'Ethereum', 'Security'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const topWeb3 = getTopKRecommendations(web3Output.recommendations, 1);
assert(topWeb3[0].category === 'Blockchain / Web3', `Top match is in Blockchain / Web3 (Got: ${topWeb3[0].role})`);

// -------------------------------------------------------------
// 6. Ranking Sensitivity & Dynamic Shift Tests
// -------------------------------------------------------------
console.log('\n--- 6. Ranking Sensitivity & Dynamic Shift Tests ---');

// Baseline: Data analysis skills
const baseQuery = rankRoles(['Python', 'SQL', 'Data Analysis'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const baseTop = baseQuery.recommendations[0].role;

// Add Deep Learning & PyTorch -> Should shift towards ML / Deep Learning
const shiftedQuery = rankRoles(['Python', 'SQL', 'Data Analysis', 'Deep Learning', 'PyTorch'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const shiftedTop = shiftedQuery.recommendations[0].role;

assert(baseTop !== shiftedTop || baseQuery.recommendations[0].finalRankingScore !== shiftedQuery.recommendations[0].finalRankingScore, `Adding specialized skills shifted top role or score (${baseTop} -> ${shiftedTop})`);
assert(shiftedQuery.recommendations.some(r => r.role === 'Deep Learning Engineer' || r.role === 'Machine Learning Engineer' || r.role === 'AI Research Scientist'), 'ML/Deep Learning roles rise to top after adding DL skills');

// -------------------------------------------------------------
// 7. Explainability & Missing Core Skills Tests
// -------------------------------------------------------------
console.log('\n--- 7. Explainability & Missing Skills Tests ---');

const mlExp = topML[0].explanation;
assert(Array.isArray(mlExp.matchedSkills) && mlExp.matchedSkills.length > 0, 'Matched skills array populated');
assert(Array.isArray(mlExp.missingSkills), 'Missing skills array populated');
assert(mlExp.missingSkills.length > 0, 'Missing skills contains unentered skills');
assert(Array.isArray(mlExp.topContributors) && mlExp.topContributors.length > 0, 'Top contributors identified');
assert(mlExp.coverage > 0 && mlExp.coverage <= 1, 'Coverage ratio is bounded between 0 and 1');

// -------------------------------------------------------------
// 8. Determinism & Security Tests
// -------------------------------------------------------------
console.log('\n--- 8. Determinism & Security Tests ---');

const run1 = rankRoles(['Python', 'Docker', 'Linux', 'Bash'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
const run2 = rankRoles(['Python', 'Docker', 'Linux', 'Bash'], roles, { aliasMap: aliases, vocabulary: vocab, idfMap, canonicalMap });
assertEquals(run1.recommendations[0].id, run2.recommendations[0].id, 'Deterministic role ID ordering across runs');
assertEquals(run1.recommendations[0].finalRankingScore, run2.recommendations[0].finalRankingScore, 'Deterministic float score across runs');

// HTML sanitization test
const sanitized = escapeHtml('<script>alert("test")</script>');
assert(sanitized.includes('&lt;script&gt;'), 'HTML script tags are sanitized to &lt;script&gt;');
assert(!sanitized.includes('<script>'), 'Raw HTML is removed');

// Edge case: Exactly 3 skills
const exact3 = validateSkillInputs(['python', 'sql', 'docker'], { minSkills: 3, maxSkills: 10 });
assertEquals(exact3.isValid, true, 'Validates exactly 3 skills');

// Edge case: 5 skills
const valid5 = validateSkillInputs(['python', 'sql', 'docker', 'aws', 'linux'], { minSkills: 3, maxSkills: 10 });
assertEquals(valid5.isValid, true, 'Validates 5 skills profile');

// Edge case: Empty input
const valEmpty = validateSkillInputs([], { minSkills: 3 });
assertEquals(valEmpty.isValid, false, 'Rejects empty skill list');

// Edge case: Whitespace only
const valWhitespace = validateSkillInputs(['   ', '  ', ''], { minSkills: 3 });
assertEquals(valWhitespace.isValid, false, 'Rejects whitespace-only entries');

console.log('\n================================================================');
console.log(`TEST RESULTS: ${passedTests}/${totalTests} Passed (${failedTests} Failed)`);
console.log('================================================================\n');

if (failedTests > 0) {
  process.exit(1);
}
