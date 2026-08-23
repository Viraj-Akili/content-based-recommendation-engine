/**
 * Data loader module with asynchronous fetching and fallback defaults.
 */
import { buildCanonicalSkillMap } from '../utils/normalization.js';
import { extractSkillName } from '../algorithms/tfidf.js';

// Fallback embedded subset to guarantee functionality across restricted environments (including file://)
const FALLBACK_ROLES = [
  {
    id: "data-scientist",
    role: "Data Scientist",
    category: "Data Science",
    description: "Analyze complex datasets, build predictive models, and extract actionable insights using statistical and ML techniques.",
    skills: [
      { name: "Python", weight: 1.0 },
      { name: "Machine Learning", weight: 1.0 },
      { name: "SQL", weight: 0.9 },
      { name: "Statistics", weight: 0.9 },
      { name: "Pandas", weight: 0.8 },
      { name: "Scikit-learn", weight: 0.8 },
      { name: "Data Analysis", weight: 0.8 },
      { name: "NumPy", weight: 0.7 }
    ],
    relatedRoles: ["ml-engineer", "statistician", "decision-scientist"]
  },
  {
    id: "ml-engineer",
    role: "Machine Learning Engineer",
    category: "AI / Machine Learning",
    description: "Design, train, and deploy scalable machine learning models and inference systems in production environments.",
    skills: [
      { name: "Python", weight: 1.0 },
      { name: "Machine Learning", weight: 1.0 },
      { name: "PyTorch", weight: 0.9 },
      { name: "TensorFlow", weight: 0.8 },
      { name: "MLOps", weight: 0.9 },
      { name: "Docker", weight: 0.8 },
      { name: "Kubernetes", weight: 0.7 }
    ],
    relatedRoles: ["ai-research-scientist", "data-scientist", "mlops-engineer"]
  },
  {
    id: "backend-developer",
    role: "Backend Developer",
    category: "Software Engineering",
    description: "Build robust server-side business logic, microservices, REST/GraphQL APIs, and database persistence layers.",
    skills: [
      { name: "Python", weight: 0.9 },
      { name: "Java", weight: 0.9 },
      { name: "SQL", weight: 0.9 },
      { name: "REST APIs", weight: 0.9 },
      { name: "PostgreSQL", weight: 0.8 },
      { name: "Docker", weight: 0.7 }
    ],
    relatedRoles: ["full-stack-developer", "api-engineer", "python-backend-developer"]
  },
  {
    id: "devops-engineer",
    role: "DevOps Engineer",
    category: "Cloud / DevOps",
    description: "Automate build and deployment pipelines, manage container fleets, and automate cloud infrastructure.",
    skills: [
      { name: "AWS", weight: 0.9 },
      { name: "Docker", weight: 0.9 },
      { name: "Kubernetes", weight: 0.9 },
      { name: "CI/CD", weight: 0.9 },
      { name: "Linux", weight: 0.8 },
      { name: "Terraform", weight: 0.8 }
    ],
    relatedRoles: ["site-reliability-engineer", "cloud-architect", "platform-engineer"]
  },
  {
    id: "frontend-developer",
    role: "Frontend Developer",
    category: "Web Development",
    description: "Craft responsive, accessible user interfaces and web clients using modern JavaScript frameworks and design systems.",
    skills: [
      { name: "JavaScript", weight: 1.0 },
      { name: "React", weight: 1.0 },
      { name: "HTML", weight: 0.9 },
      { name: "CSS", weight: 0.9 },
      { name: "TypeScript", weight: 0.8 }
    ],
    relatedRoles: ["full-stack-developer", "react-specialist", "ui-engineer"]
  },
  {
    id: "quantum-computing-engineer",
    role: "Quantum Computing Engineer",
    category: "Emerging Technology",
    description: "Implement quantum algorithms, circuit compilers, gate calibrations, and quantum error correction routines.",
    skills: [
      { name: "Python", weight: 1.0 },
      { name: "Qiskit", weight: 1.0 },
      { name: "Quantum", weight: 1.0 },
      { name: "Mathematics", weight: 1.0 },
      { name: "Linear Algebra", weight: 1.0 },
      { name: "Physics", weight: 0.9 }
    ],
    relatedRoles: ["quantum-algorithm-developer", "computational-scientist", "ai-research-scientist"]
  }
];

const FALLBACK_ALIASES = {
  "ml": "machine learning",
  "ai": "deep learning",
  "js": "javascript",
  "ts": "typescript",
  "py": "python",
  "postgres": "postgresql",
  "mongo": "mongodb",
  "k8s": "kubernetes",
  "tf": "tensorflow",
  "torch": "pytorch",
  "dl": "deep learning",
  "cv": "computer vision",
  "nlp": "nlp",
  "aws": "aws",
  "gcp": "gcp",
  "azure": "azure",
  "rest": "rest apis",
  "sol": "solidity",
  "quant": "quantum"
};

/**
 * Loads dataset from JSON files with graceful fallback.
 *
 * @param {string} [basePath='./data/'] - Base directory path for data files
 * @returns {Promise<{
 *   roles: Array<{ id: string, role: string, category?: string, description: string, skills: any[], relatedRoles?: string[] }>,
 *   skills: string[],
 *   aliases: Record<string, string>,
 *   canonicalMap: Map<string, string>
 * }>}
 */
export async function loadDataset(basePath = './data/') {
  let roles = FALLBACK_ROLES;
  let aliases = FALLBACK_ALIASES;
  let skills = [];

  try {
    const rolesRes = await fetch(`${basePath}roles.json`);
    if (rolesRes.ok) {
      roles = await rolesRes.json();
    }
  } catch (err) {
    console.info('Using fallback embedded roles dataset (file:// mode).');
  }

  try {
    const aliasesRes = await fetch(`${basePath}aliases.json`);
    if (aliasesRes.ok) {
      aliases = await aliasesRes.json();
    }
  } catch (err) {
    console.info('Using fallback embedded aliases dictionary.');
  }

  try {
    const skillsRes = await fetch(`${basePath}skills.json`);
    if (skillsRes.ok) {
      skills = await skillsRes.json();
    }
  } catch (err) {
    // Generate canonical skills from roles
    const skillSet = new Set(roles.flatMap(r => (r.skills || []).map(s => extractSkillName(s))));
    skills = Array.from(skillSet).sort();
  }

  if (!skills || skills.length === 0) {
    const skillSet = new Set(roles.flatMap(r => (r.skills || []).map(s => extractSkillName(s))));
    skills = Array.from(skillSet).sort();
  }

  const canonicalMap = buildCanonicalSkillMap(skills);

  return {
    roles,
    skills,
    aliases,
    canonicalMap
  };
}
