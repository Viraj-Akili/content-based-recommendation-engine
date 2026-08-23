/**
 * Dataset Integrity & Quality Verification Script
 * Run with: node tests/validate-dataset.js
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rolesPath = path.join(__dirname, '../data/roles.json');
const skillsPath = path.join(__dirname, '../data/skills.json');
const aliasesPath = path.join(__dirname, '../data/aliases.json');

const roles = JSON.parse(fs.readFileSync(rolesPath, 'utf8'));
const skills = JSON.parse(fs.readFileSync(skillsPath, 'utf8'));
const aliases = JSON.parse(fs.readFileSync(aliasesPath, 'utf8'));

let errors = [];
let warnings = [];

console.log('================================================================');
console.log('  VALIDATING CAREER RECOMMENDATION DATASET INTEGRITY');
console.log('================================================================\n');

// 1. Role ID uniqueness & structure
const roleIds = new Set();
const allCorpusSkills = new Set();

roles.forEach((r, idx) => {
  const roleLabel = r.id || `Index ${idx}`;

  if (!r.id || typeof r.id !== 'string' || !r.id.trim()) {
    errors.push(`[${roleLabel}] Missing or invalid 'id'`);
  } else if (roleIds.has(r.id)) {
    errors.push(`[${roleLabel}] Duplicate role id: '${r.id}'`);
  } else {
    roleIds.add(r.id);
  }

  if (!r.role || typeof r.role !== 'string' || !r.role.trim()) {
    errors.push(`[${roleLabel}] Missing or empty 'role' title`);
  }

  if (!r.category || typeof r.category !== 'string' || !r.category.trim()) {
    errors.push(`[${roleLabel}] Missing or empty 'category'`);
  }

  if (!r.description || typeof r.description !== 'string' || !r.description.trim()) {
    errors.push(`[${roleLabel}] Missing or empty 'description'`);
  }

  if (!Array.isArray(r.skills) || r.skills.length < 3) {
    errors.push(`[${roleLabel}] Must have at least 3 skills (found ${r.skills ? r.skills.length : 0})`);
  } else {
    const roleSkillNames = new Set();
    let hasCore = false;

    r.skills.forEach(s => {
      if (!s || typeof s !== 'object' || typeof s.name !== 'string' || !s.name.trim()) {
        errors.push(`[${roleLabel}] Contains malformed skill item: ${JSON.stringify(s)}`);
        return;
      }

      if (typeof s.weight !== 'number' || s.weight < 0 || s.weight > 1) {
        errors.push(`[${roleLabel}] Skill '${s.name}' has invalid weight: ${s.weight} (must be between 0.0 and 1.0)`);
      }

      if (s.weight >= 0.8) {
        hasCore = true;
      }

      const lowerName = s.name.trim().toLowerCase();
      if (roleSkillNames.has(lowerName)) {
        errors.push(`[${roleLabel}] Duplicate skill inside role: '${s.name}'`);
      }
      roleSkillNames.add(lowerName);
      allCorpusSkills.add(s.name.trim());
    });

    if (!hasCore) {
      warnings.push(`[${roleLabel}] Role has no core skill with weight >= 0.8`);
    }
  }
});

// 2. Relational Integrity: relatedRoles must resolve to valid role IDs
roles.forEach(r => {
  if (Array.isArray(r.relatedRoles)) {
    r.relatedRoles.forEach(relId => {
      if (!roleIds.has(relId)) {
        errors.push(`[${r.id}] References non-existent relatedRole id: '${relId}'`);
      }
      if (relId === r.id) {
        warnings.push(`[${r.id}] References itself in relatedRoles`);
      }
    });
  }
});

// 3. Skills Dictionary Verification
const skillsDictionarySet = new Set(skills.map(s => s.toLowerCase()));
for (const skill of allCorpusSkills) {
  if (!skillsDictionarySet.has(skill.toLowerCase())) {
    errors.push(`[skills.json] Missing canonical skill: '${skill}'`);
  }
}

// 4. Aliases Dictionary Verification
for (const [alias, target] of Object.entries(aliases)) {
  if (!alias || typeof alias !== 'string' || !alias.trim()) {
    errors.push(`[aliases.json] Empty or invalid alias key`);
  }
  if (!target || typeof target !== 'string' || !target.trim()) {
    errors.push(`[aliases.json] Empty target for alias: '${alias}'`);
  }
}

console.log(`Summary Statistics:`);
console.log(`- Total Career Roles: ${roles.length}`);
console.log(`- Unique Categories: ${new Set(roles.map(r => r.category)).size}`);
console.log(`- Unique Skills: ${allCorpusSkills.size}`);
console.log(`- Total Aliases: ${Object.keys(aliases).length}`);
console.log(`- Validation Errors: ${errors.length}`);
console.log(`- Validation Warnings: ${warnings.length}\n`);

if (errors.length > 0) {
  console.error('Validation Errors:');
  errors.forEach(e => console.error(`  ✗ ${e}`));
  process.exit(1);
} else {
  console.log('✓ All dataset schema and relational integrity checks PASSED successfully!\n');
}
