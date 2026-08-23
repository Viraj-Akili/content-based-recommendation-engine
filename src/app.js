/**
 * CareerMatch - Explainable Career Recommendation Engine
 * Main Application Controller & UI Orchestrator
 */

import { loadDataset } from './data/data-loader.js';
import { normalizeSkill, getCanonicalSkill, escapeHtml } from './utils/normalization.js';
import { validateSkillInputs } from './utils/validation.js';
import { buildVocabulary, computeIDF } from './algorithms/tfidf.js';
import { rankRoles, getTopKRecommendations } from './algorithms/ranking.js';

// Application State
const state = {
  roles: [],
  skills: [],
  aliases: {},
  canonicalMap: new Map(),
  vocabulary: [],
  idfMap: {},
  selectedSkills: new Set(),
  isInitialized: false
};

// Preset Example Profiles
const PRESET_PROFILES = {
  aiml: ['Python', 'Machine Learning', 'PyTorch', 'MLOps', 'Docker'],
  fullstack: ['JavaScript', 'React', 'Node.js', 'SQL', 'TypeScript'],
  devops: ['Docker', 'Kubernetes', 'AWS', 'Terraform', 'Linux'],
  data: ['Python', 'SQL', 'Spark', 'Airflow', 'Pandas']
};

// DOM References
const elements = {
  skillSearchInput: document.getElementById('skillSearchInput'),
  suggestionsMenu: document.getElementById('suggestionsMenu'),
  tagsCloud: document.getElementById('tagsCloud'),
  skillCounter: document.getElementById('skillCounter'),
  validationAlert: document.getElementById('validationAlert'),
  runButton: document.getElementById('runButton'),
  resetButton: document.getElementById('resetButton'),
  resultsContainer: document.getElementById('resultsContainer'),
  emptyStateCard: document.getElementById('emptyStateCard'),
  telemetrySection: document.getElementById('telemetrySection'),
  exampleButtons: document.querySelectorAll('.example-btn')
};

/**
 * Initializes dataset and precomputes corpus statistics.
 */
async function initializeApp() {
  try {
    const data = await loadDataset();
    state.roles = data.roles;
    state.skills = data.skills;
    state.aliases = data.aliases;
    state.canonicalMap = data.canonicalMap;

    // Precompute vocabulary and smoothed IDF corpus statistics
    state.vocabulary = buildVocabulary(state.roles, state.aliases);
    state.idfMap = computeIDF(state.roles, state.vocabulary, { aliasMap: state.aliases });
    state.isInitialized = true;

    setupEventListeners();
    renderTags();
  } catch (error) {
    console.error('Initialization error:', error);
    showAlert('Failed to load recommendation dataset. Please refresh.', 'error');
  }
}

/**
 * Sets up all UI event listeners.
 */
function setupEventListeners() {
  // Autocomplete search input
  if (elements.skillSearchInput) {
    elements.skillSearchInput.addEventListener('input', handleSearchInput);
    elements.skillSearchInput.addEventListener('focus', handleSearchInput);
    elements.skillSearchInput.addEventListener('keydown', handleSearchKeydown);
  }

  // Close suggestions on outside click
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-wrapper')) {
      closeSuggestions();
    }
  });

  // Example profile preset buttons
  if (elements.exampleButtons) {
    elements.exampleButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const profileKey = btn.dataset.profile;
        if (PRESET_PROFILES[profileKey]) {
          loadPresetProfile(PRESET_PROFILES[profileKey]);
        }
      });
    });
  }

  // Primary Action Buttons
  if (elements.runButton) {
    elements.runButton.addEventListener('click', () => executeRecommendation());
  }

  if (elements.resetButton) {
    elements.resetButton.addEventListener('click', resetAll);
  }
}

/**
 * Loads a preset skill profile and triggers recommendations immediately.
 */
function loadPresetProfile(skillArray) {
  state.selectedSkills.clear();
  for (const raw of skillArray) {
    const norm = normalizeSkill(raw, state.aliases);
    const canonical = getCanonicalSkill(norm, state.canonicalMap) || raw;
    state.selectedSkills.add(canonical);
  }
  renderTags();
  executeRecommendation();
}

/**
 * Handles live typing in the skill search input.
 */
function handleSearchInput() {
  const query = elements.skillSearchInput.value.trim().toLowerCase();
  elements.suggestionsMenu.innerHTML = '';

  if (!query) {
    closeSuggestions();
    return;
  }

  const results = [];
  const seen = new Set();

  // 1. Direct matches in canonical skills
  for (const skill of state.skills) {
    const norm = skill.toLowerCase();
    if (norm.startsWith(query) || norm.includes(query)) {
      if (!seen.has(norm) && !state.selectedSkills.has(skill)) {
        seen.add(norm);
        results.push({ display: skill, value: skill, isAlias: false });
      }
    }
  }

  // 2. Alias lookups (e.g. k8s -> Kubernetes, ml -> Machine Learning)
  for (const [aliasKey, targetCanonical] of Object.entries(state.aliases)) {
    if (aliasKey.startsWith(query)) {
      const canonicalName = getCanonicalSkill(targetCanonical, state.canonicalMap);
      const uniqueKey = canonicalName.toLowerCase();
      if (!seen.has(uniqueKey) && !state.selectedSkills.has(canonicalName)) {
        seen.add(uniqueKey);
        results.push({
          display: canonicalName,
          aliasLabel: `alias: ${aliasKey}`,
          value: canonicalName,
          isAlias: true
        });
      }
    }
  }

  if (results.length === 0) {
    closeSuggestions();
    return;
  }

  // Render suggestion items
  results.slice(0, 8).forEach((item, index) => {
    const div = document.createElement('div');
    div.className = 'suggestion-item';
    div.setAttribute('role', 'option');
    div.setAttribute('tabindex', '-1');
    div.dataset.index = index;

    const nameSpan = document.createElement('span');
    nameSpan.textContent = item.display;
    div.appendChild(nameSpan);

    if (item.aliasLabel) {
      const aliasSpan = document.createElement('span');
      aliasSpan.className = 'suggestion-alias';
      aliasSpan.textContent = item.aliasLabel;
      div.appendChild(aliasSpan);
    }

    div.addEventListener('mousedown', (e) => {
      e.preventDefault();
      addSkill(item.value);
    });

    elements.suggestionsMenu.appendChild(div);
  });

  elements.suggestionsMenu.classList.add('open');
}

/**
 * Handles keyboard navigation (Arrow Down/Up, Enter, Escape) in search input.
 */
function handleSearchKeydown(e) {
  const items = elements.suggestionsMenu.querySelectorAll('.suggestion-item');
  const activeItem = elements.suggestionsMenu.querySelector('.suggestion-item.highlighted');
  let currentIndex = activeItem ? parseInt(activeItem.dataset.index, 10) : -1;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (items.length === 0) return;
    currentIndex = (currentIndex + 1) % items.length;
    highlightSuggestion(items, currentIndex);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (items.length === 0) return;
    currentIndex = (currentIndex - 1 + items.length) % items.length;
    highlightSuggestion(items, currentIndex);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (activeItem) {
      const selectedSkill = activeItem.querySelector('span').textContent;
      addSkill(selectedSkill);
    } else {
      const text = elements.skillSearchInput.value.trim();
      if (text) {
        const norm = normalizeSkill(text, state.aliases);
        const canonical = getCanonicalSkill(norm, state.canonicalMap) || text;
        addSkill(canonical);
      } else if (!elements.runButton.disabled) {
        executeRecommendation();
      }
    }
  } else if (e.key === 'Escape') {
    closeSuggestions();
  }
}

function highlightSuggestion(items, index) {
  items.forEach(item => item.classList.remove('highlighted'));
  if (items[index]) {
    items[index].classList.add('highlighted');
    items[index].scrollIntoView({ block: 'nearest' });
  }
}

function closeSuggestions() {
  if (elements.suggestionsMenu) {
    elements.suggestionsMenu.classList.remove('open');
    elements.suggestionsMenu.innerHTML = '';
  }
}

/**
 * Adds a skill to the selected skills collection.
 */
function addSkill(skillName) {
  if (!skillName || !skillName.trim()) return;

  if (state.selectedSkills.size >= 10) {
    showAlert('Maximum of 10 skills allowed per profile.', 'warn');
    return;
  }

  const norm = normalizeSkill(skillName, state.aliases);
  const canonical = getCanonicalSkill(norm, state.canonicalMap) || skillName.trim();

  state.selectedSkills.add(canonical);
  elements.skillSearchInput.value = '';
  closeSuggestions();
  renderTags();
  elements.skillSearchInput.focus();
}

/**
 * Removes a skill from the selected collection.
 */
function removeSkill(skillName) {
  state.selectedSkills.delete(skillName);
  renderTags();
}

/**
 * Renders interactive skill tag chips and updates counters & validation.
 */
function renderTags() {
  const skillsArray = Array.from(state.selectedSkills);
  elements.tagsCloud.innerHTML = '';

  const validation = validateSkillInputs(skillsArray, {
    minSkills: 3,
    maxSkills: 10,
    knownVocabulary: state.vocabulary,
    aliasMap: state.aliases
  });

  if (skillsArray.length === 0) {
    elements.tagsCloud.innerHTML = '<span class="tags-placeholder">Type skills or aliases above (e.g., Python, ml, k8s, React, SQL) or click an example profile ↑</span>';
    elements.skillCounter.textContent = '0 / 10 skills (minimum 3 required)';
    elements.skillCounter.classList.remove('ready');
    elements.runButton.disabled = true;
    hideAlert();
    return;
  }

  // Render tag chips
  skillsArray.forEach(skill => {
    const tag = document.createElement('span');
    tag.className = 'skill-tag';
    const escaped = escapeHtml(skill);
    tag.innerHTML = `${escaped} <span class="skill-tag-remove" role="button" aria-label="Remove ${escaped}" title="Remove skill">×</span>`;

    tag.querySelector('.skill-tag-remove').addEventListener('click', () => {
      removeSkill(skill);
    });

    elements.tagsCloud.appendChild(tag);
  });

  // Update counter
  const isReady = validation.isValid;
  elements.skillCounter.textContent = `${skillsArray.length} / 10 skills ${isReady ? '✓ (Ready)' : '(minimum 3 required)'}`;
  elements.skillCounter.classList.toggle('ready', isReady);

  // Validation alerts
  if (validation.warnings.length > 0) {
    showAlert(validation.warnings.join(' • '), 'warn');
  } else {
    hideAlert();
  }

  elements.runButton.disabled = !isReady;
}

function showAlert(message, type = 'warn') {
  if (!elements.validationAlert) return;
  elements.validationAlert.className = `alert-box ${type}`;
  elements.validationAlert.textContent = `${type === 'error' ? '❌' : '⚠️'} ${message}`;
}

function hideAlert() {
  if (!elements.validationAlert) return;
  elements.validationAlert.className = 'alert-box';
  elements.validationAlert.textContent = '';
}

/**
 * Executes the recommendation pipeline with high-precision latency measurement.
 */
function executeRecommendation() {
  const skillsArray = Array.from(state.selectedSkills);
  if (skillsArray.length < 3) return;

  // High-precision execution timer
  const startTime = performance.now();

  const rankOutput = rankRoles(skillsArray, state.roles, {
    aliasMap: state.aliases,
    vocabulary: state.vocabulary,
    idfMap: state.idfMap,
    canonicalMap: state.canonicalMap,
    alpha: 0.65
  });

  const durationMs = performance.now() - startTime;
  const top3 = getTopKRecommendations(rankOutput.recommendations, 3);

  renderResults(top3, rankOutput.userFeatureWeights, {
    skillsCount: skillsArray.length,
    totalRoles: state.roles.length,
    latencyMs: durationMs
  });
}

/**
 * Renders the results dashboard: telemetry bar, hero card, secondary cards, feature weights, and comparison matrix.
 */
function renderResults(topRecommendations, featureWeights, telemetry) {
  if (!elements.resultsContainer) return;
  elements.emptyStateCard.style.display = 'none';

  if (!topRecommendations || topRecommendations.length === 0) {
    elements.resultsContainer.innerHTML = `
      <div class="empty-state-card">
        <div class="empty-icon">🎯</div>
        <h3>No matching roles found</h3>
        <p>Try adding more varied technical skills.</p>
      </div>
    `;
    return;
  }

  const top1 = topRecommendations[0];
  const secondary = topRecommendations.slice(1, 3);

  // Render Telemetry & Result Cards
  elements.resultsContainer.innerHTML = `
    <!-- Real-time Telemetry Stats -->
    <div class="telemetry-bar" aria-label="Recommendation System Telemetry">
      <div class="telemetry-card">
        <span class="telemetry-label">Skills Analyzed</span>
        <span class="telemetry-value">${telemetry.skillsCount}</span>
        <span class="telemetry-sub">Normalized Tokens</span>
      </div>
      <div class="telemetry-card">
        <span class="telemetry-label">Candidate Roles</span>
        <span class="telemetry-value">${telemetry.totalRoles}</span>
        <span class="telemetry-sub">Corpus Catalog</span>
      </div>
      <div class="telemetry-card">
        <span class="telemetry-label">Inference Latency</span>
        <span class="telemetry-value" style="color:var(--success);">${telemetry.latencyMs < 1 ? telemetry.latencyMs.toFixed(2) + ' ms' : telemetry.latencyMs.toFixed(1) + ' ms'}</span>
        <span class="telemetry-sub">Vector Scoring Time</span>
      </div>
      <div class="telemetry-card">
        <span class="telemetry-label">Peak Match Strength</span>
        <span class="telemetry-value" style="color:var(--accent-gold);">${top1.score.toFixed(3)}</span>
        <span class="telemetry-sub">Hybrid Score</span>
      </div>
    </div>

    <!-- Recommendations Grid -->
    <div class="recs-container">
      <!-- #1 Priority Hero Card -->
      ${renderHeroCard(top1)}

      <!-- Rank #2 & #3 Secondary Cards Grid -->
      ${secondary.length > 0 ? `
        <div class="secondary-grid">
          ${secondary.map((role, idx) => renderSecondaryCard(role, idx + 2)).join('')}
        </div>
      ` : ''}
    </div>

    <!-- Analytics & Comparison Grid -->
    <div class="analytics-grid">
      <!-- TF-IDF Feature Importance Breakdown -->
      ${renderFeatureImportance(featureWeights)}

      <!-- Multi-Role Comparison Table -->
      ${renderComparisonTable(topRecommendations)}
    </div>
  `;
}

/**
 * Generates HTML for the #1 Hero Recommendation Card with visual priority.
 */
function renderHeroCard(item) {
  const matchPct = Math.round(item.score * 100);
  const circumference = 2 * Math.PI * 34;
  const offset = circumference - (item.score * circumference);

  const roleTitle = escapeHtml(item.role);
  const category = escapeHtml(item.category);
  const description = escapeHtml(item.description);
  const summary = escapeHtml(item.explanation.summary);

  return `
    <article class="hero-rec-card" aria-label="Rank 1: ${roleTitle}">
      <div class="rec-badge-row">
        <span class="rank-pill rank-1">🥇 TOP CAREER MATCH • RANK 1</span>
        <span class="category-tag">${category}</span>
      </div>

      <div class="rec-main-grid">
        <div class="rec-info">
          <h2>${roleTitle}</h2>
          <p class="role-desc">${description}</p>

          <div class="metrics-row">
            <span class="metric-badge" title="Combined: 65% Cosine Similarity + 35% Weighted Coverage">
              Match Strength: <strong>${item.score.toFixed(3)}</strong>
            </span>
            <span class="metric-badge" title="Pure TF-IDF Vector Cosine Proximity">
              Cosine Sim: <strong>${item.cosineSimilarity.toFixed(3)}</strong>
            </span>
            <span class="metric-badge" title="Weighted skill points satisfied">
              Skill Coverage: <strong>${item.skillCoveragePercentage}%</strong> (${item.weightedMatchedScore}/${item.totalRoleSkillWeight} pts)
            </span>
          </div>

          <!-- Why this role callout -->
          <div class="why-role-box">
            <span class="why-icon">💡</span>
            <div>
              <strong>Why this role?</strong>
              <div style="margin-top:2px;">${summary}</div>
            </div>
          </div>
        </div>

        <!-- Animated Radial Score Ring -->
        <div class="score-widget">
          <div class="ring-outer">
            <svg width="80" height="80" viewBox="0 0 80 80" aria-hidden="true">
              <circle class="ring-bg-track" cx="40" cy="40" r="34"/>
              <circle class="ring-progress" cx="40" cy="40" r="34"
                stroke-dasharray="${circumference}"
                stroke-dashoffset="${offset}"
              />
            </svg>
            <div class="ring-center-text">
              ${matchPct}%
              <small>SCORE</small>
            </div>
          </div>
          <span class="score-legend">cos θ = ${item.cosineSimilarity.toFixed(3)}</span>
        </div>
      </div>

      <!-- Skills Breakdown -->
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:8px;">
        <div class="skills-block">
          <span class="skills-heading">Matched Profile Skills (${item.matchedSkillCount})</span>
          <div class="pill-cloud">
            ${item.matchedSkills.map(s => `<span class="badge-matched" title="Matched skill">✓ ${escapeHtml(s)}</span>`).join('')}
          </div>
        </div>

        <div class="skills-block">
          <span class="skills-heading">High-Priority Skills to Develop</span>
          <div class="pill-cloud">
            ${item.missingCoreSkills.length > 0 
              ? item.missingCoreSkills.map(s => `<span class="badge-missing-core" title="Missing core requirement">+ ${escapeHtml(s)} (Core)</span>`).join('')
              : '<span style="font-size:0.75rem;color:var(--success);font-family:var(--mono);">✓ All core skills satisfied!</span>'
            }
            ${item.missingSkills.filter(s => !item.missingCoreSkills.includes(s)).slice(0, 3).map(s => `<span class="badge-missing-aux" title="Supporting skill">${escapeHtml(s)}</span>`).join('')}
          </div>
        </div>
      </div>
    </article>
  `;
}

/**
 * Generates HTML for secondary recommendation cards (Rank 2 and 3).
 */
function renderSecondaryCard(item, rankNum) {
  const matchPct = Math.round(item.score * 100);
  const rankClass = rankNum === 2 ? 'rank-2' : 'rank-3';
  const medal = rankNum === 2 ? '🥈' : '🥉';
  const roleTitle = escapeHtml(item.role);
  const category = escapeHtml(item.category);
  const description = escapeHtml(item.description);
  const summary = escapeHtml(item.explanation.summary);

  return `
    <article class="secondary-card" aria-label="Rank ${rankNum}: ${roleTitle}">
      <div class="rec-badge-row">
        <span class="rank-pill ${rankClass}">${medal} RANK ${rankNum}</span>
        <span class="category-tag">${category}</span>
      </div>

      <div>
        <h3>${roleTitle}</h3>
        <p class="role-desc">${description}</p>
        
        <div class="metrics-row">
          <span class="metric-badge">Score: <strong>${item.score.toFixed(3)}</strong></span>
          <span class="metric-badge">Cosine: <strong>${item.cosineSimilarity.toFixed(3)}</strong></span>
          <span class="metric-badge">Coverage: <strong>${item.skillCoveragePercentage}%</strong></span>
        </div>

        <div style="font-size:0.76rem;color:var(--text-secondary);font-family:var(--mono);margin-bottom:12px;">
          💡 <em>${summary}</em>
        </div>

        <div class="skills-block">
          <span class="skills-heading">Matched Capabilities</span>
          <div class="pill-cloud">
            ${item.matchedSkills.map(s => `<span class="badge-matched">✓ ${escapeHtml(s)}</span>`).join('')}
          </div>
        </div>

        <div class="skills-block" style="margin-top:10px;">
          <span class="skills-heading">Skills to Develop</span>
          <div class="pill-cloud">
            ${item.missingCoreSkills.slice(0, 2).map(s => `<span class="badge-missing-core">+ ${escapeHtml(s)} (Core)</span>`).join('')}
            ${item.missingSkills.filter(s => !item.missingCoreSkills.includes(s)).slice(0, 2).map(s => `<span class="badge-missing-aux">${escapeHtml(s)}</span>`).join('')}
          </div>
        </div>
      </div>
    </article>
  `;
}

/**
 * Generates the TF-IDF Profile Feature Importance bar chart.
 */
function renderFeatureImportance(featureWeights) {
  if (!featureWeights || featureWeights.length === 0) return '';
  const topTerms = featureWeights.slice(0, 8);
  const maxWeight = topTerms[0]?.weight || 1.0;

  return `
    <section class="panel-card" aria-label="Profile TF-IDF Feature Importance">
      <div class="panel-header">
        <div>
          <h3>Profile Feature Importance</h3>
          <span style="color:var(--text-muted);">Term Frequency × Corpus Inverse Document Frequency</span>
        </div>
        <span>TF-IDF Weight</span>
      </div>

      <div class="vector-barchart">
        ${topTerms.map(fw => {
          const widthPct = Math.max(10, Math.round((fw.weight / maxWeight) * 100));
          const label = escapeHtml(fw.displayTerm);
          return `
            <div class="barchart-row">
              <div class="barchart-label" title="${label}">${label}</div>
              <div class="barchart-track">
                <div class="barchart-fill" style="width: ${widthPct}%;"></div>
              </div>
              <div class="barchart-value">${fw.weight.toFixed(3)}</div>
            </div>
          `;
        }).join('')}
      </div>
    </section>
  `;
}

/**
 * Generates the Multi-Role Comparison Matrix table.
 */
function renderComparisonTable(topRecommendations) {
  if (!topRecommendations || topRecommendations.length === 0) return '';

  return `
    <section class="panel-card" aria-label="Top 3 Roles Comparison Matrix">
      <div class="panel-header">
        <h3>Top Roles Comparison Matrix</h3>
        <span>Side-by-Side Dimensional Analysis</span>
      </div>

      <div class="comparison-table-wrapper">
        <table class="comparison-table">
          <thead>
            <tr>
              <th>Rank & Role</th>
              <th>Category</th>
              <th>Match Score</th>
              <th>Cosine Sim</th>
              <th>Coverage</th>
              <th>Matched Core Skills</th>
              <th>Missing Core Skills</th>
            </tr>
          </thead>
          <tbody>
            ${topRecommendations.map((role, idx) => `
              <tr>
                <td>
                  <div class="table-role-title">${idx === 0 ? '🥇' : idx === 1 ? '🥈' : '🥉'} ${escapeHtml(role.role)}</div>
                </td>
                <td>
                  <span class="category-tag">${escapeHtml(role.category)}</span>
                </td>
                <td>
                  <strong style="color:var(--accent-gold);">${role.score.toFixed(3)}</strong>
                </td>
                <td>
                  <span>${role.cosineSimilarity.toFixed(3)}</span>
                </td>
                <td>
                  <span>${role.skillCoveragePercentage}% (${role.weightedMatchedScore}/${role.totalRoleSkillWeight})</span>
                </td>
                <td>
                  <div style="font-family:var(--mono);font-size:0.72rem;color:var(--success);">
                    ${escapeHtml(role.matchedSkills.slice(0, 3).join(', ')) || 'None'}
                  </div>
                </td>
                <td>
                  <div style="font-family:var(--mono);font-size:0.72rem;color:#fbbf24;">
                    ${escapeHtml(role.missingCoreSkills.slice(0, 3).join(', ')) || 'All satisfied'}
                  </div>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

/**
 * Resets profile and restores empty state.
 */
function resetAll() {
  state.selectedSkills.clear();
  if (elements.skillSearchInput) {
    elements.skillSearchInput.value = '';
  }
  closeSuggestions();
  renderTags();
  hideAlert();

  if (elements.resultsContainer && elements.emptyStateCard) {
    elements.emptyStateCard.style.display = 'flex';
    elements.resultsContainer.innerHTML = '';
    elements.resultsContainer.appendChild(elements.emptyStateCard);
  }
}

// Expose reset globally
window.resetAll = resetAll;

// Boot application on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}
