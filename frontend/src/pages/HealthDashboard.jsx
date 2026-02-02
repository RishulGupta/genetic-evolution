import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shield, AlertTriangle, TrendingUp, Github, Bug, Heart, Zap, Flame, Activity, Clock } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function HealthDashboard() {
  const [rateLimit, setRateLimit] = useState(null);

  useEffect(() => {
    fetchRateLimit();
  }, []);

  const fetchRateLimit = async () => {
    try {
      const response = await axios.get(`${API}/rate-limit`);
      setRateLimit(response.data);
    } catch (error) {
      console.error('Error fetching rate limit:', error);
    }
  };

  const getRateLimitPercentage = () => {
    if (!rateLimit) return 0;
    return (rateLimit.remaining / rateLimit.limit) * 100;
  };

  const getRateLimitColor = () => {
    const percentage = getRateLimitPercentage();
    if (percentage > 50) return 'text-emerald-500';
    if (percentage > 20) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="container mx-auto px-6 md:px-12 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl md:text-6xl font-bold tracking-tighter text-white mb-6">
          Health Dashboard
        </h1>
        <p className="text-lg text-slate-400 mb-12">
          Monitor system health, API usage, and understand the bug evolution analysis methodology.
        </p>

        {/* API Status Grid */}
        <div className="grid md:grid-cols-3 gap-6 grid-borders mb-8">
          <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8 h-full" data-testid="api-status">
            <Shield className="w-8 h-8 text-emerald-500 mb-4" strokeWidth={1.5} />
            <h3 className="text-xl font-medium text-white mb-2">API Status</h3>
            <div className="text-3xl font-bold text-emerald-400 font-mono mb-2">Operational</div>
            <p className="text-sm text-slate-400">All systems running smoothly</p>
          </div>

          <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8 h-full" data-testid="rate-limit-status">
            <Github className="w-8 h-8 text-blue-500 mb-4" strokeWidth={1.5} />
            <h3 className="text-xl font-medium text-white mb-2">GitHub API Rate Limit</h3>
            {rateLimit ? (
              <>
                <div className={`text-3xl font-bold font-mono mb-2 ${getRateLimitColor()}`}>
                  {rateLimit.remaining}/{rateLimit.limit}
                </div>
                <div className="w-full bg-slate-800 h-2 rounded-none mb-2">
                  <div 
                    className="bg-emerald-500 h-2 transition-all"
                    style={{ width: `${getRateLimitPercentage()}%` }}
                  />
                </div>
                <p className="text-sm text-slate-400">
                  Resets at {new Date(rateLimit.reset * 1000).toLocaleTimeString()}
                </p>
              </>
            ) : (
              <div className="text-slate-400">Loading...</div>
            )}
          </div>

          <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8 h-full" data-testid="analysis-count">
            <TrendingUp className="w-8 h-8 text-purple-500 mb-4" strokeWidth={1.5} />
            <h3 className="text-xl font-medium text-white mb-2">Analysis Engine</h3>
            <div className="text-3xl font-bold text-purple-400 font-mono mb-2">v2.0</div>
            <p className="text-sm text-slate-400">Bug Evolution Analysis enabled</p>
          </div>
        </div>

        {/* Bug Evolution Methodology */}
        <div className="space-y-6">
          {/* Main Health Scoring */}
          <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8">
            <div className="flex items-center gap-3 mb-6">
              <Activity className="w-8 h-8 text-emerald-500" strokeWidth={1.5} />
              <h3 className="text-2xl font-medium text-white">Health Score Methodology</h3>
            </div>
            <div className="text-slate-400 space-y-3">
              <p>
                The health score (0-100) is calculated using six key metrics, including bug evolution factors:
              </p>
              <div className="grid md:grid-cols-2 gap-4 mt-4">
                <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-emerald-500 font-bold text-lg">25</span>
                    <span className="text-white font-medium">Bug Frequency Score</span>
                  </div>
                  <p className="text-sm text-slate-400">Lower bug ratios indicate healthier codebases. Calculated from the percentage of bug-fix commits.</p>
                </div>
                <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-blue-500 font-bold text-lg">25</span>
                    <span className="text-white font-medium">Commit Stability Score</span>
                  </div>
                  <p className="text-sm text-slate-400">Consistent commit sizes show predictable development. Uses coefficient of variation.</p>
                </div>
                <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-purple-500 font-bold text-lg">25</span>
                    <span className="text-white font-medium">Contributor Diversity Score</span>
                  </div>
                  <p className="text-sm text-slate-400">Better distribution prevents single-point failures. Uses Gini coefficient.</p>
                </div>
                <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-yellow-500 font-bold text-lg">25</span>
                    <span className="text-white font-medium">Change Volatility Score</span>
                  </div>
                  <p className="text-sm text-slate-400">Recent bug patterns affect long-term risk. Analyzes last 10 commits.</p>
                </div>
                <div className="bg-black/30 border border-red-500/30 rounded-none p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-red-500 font-bold text-lg">-15</span>
                    <span className="text-white font-medium">Contagion Penalty</span>
                  </div>
                  <p className="text-sm text-slate-400">Deduction based on bug spread patterns. Higher contagion = lower score.</p>
                </div>
                <div className="bg-black/30 border border-emerald-500/30 rounded-none p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-emerald-500 font-bold text-lg">+10</span>
                    <span className="text-white font-medium">Recovery Bonus</span>
                  </div>
                  <p className="text-sm text-slate-400">Bonus for good recovery metrics. Fast fixes and low regression rates.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Bug Evolution Analysis */}
          <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8">
            <div className="flex items-center gap-3 mb-6">
              <Bug className="w-8 h-8 text-red-500" strokeWidth={1.5} />
              <h3 className="text-2xl font-medium text-white">Bug Evolution Analysis</h3>
            </div>
            <p className="text-slate-400 mb-6">
              Treats bugs as genetic mutations that propagate through code. This advanced analysis tracks how bugs spread, mutate, and survive across the codebase.
            </p>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <Zap className="w-6 h-6 text-yellow-500 mb-3" strokeWidth={1.5} />
                <h4 className="text-white font-medium mb-2">Bug Contagion Score</h4>
                <p className="text-sm text-slate-400">
                  Measures how infectious bugs are (0-1 scale). Considers propagation depth, lifespan, reinfection rate, and contributor spread.
                </p>
                <div className="mt-3 space-y-1 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-500" />
                    <span className="text-slate-400">0.0-0.3: Contained</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-yellow-500" />
                    <span className="text-slate-400">0.3-0.6: Moderate</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-red-500" />
                    <span className="text-slate-400">0.6-1.0: Highly Infectious</span>
                  </div>
                </div>
              </div>

              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <Heart className="w-6 h-6 text-pink-500 mb-3" strokeWidth={1.5} />
                <h4 className="text-white font-medium mb-2">Recovery & Immunity</h4>
                <p className="text-sm text-slate-400">
                  Measures how well a project recovers from bugs. Tracks time-to-fix, regression probability, and clean commit streaks.
                </p>
                <div className="mt-3 space-y-1 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-500" />
                    <span className="text-slate-400">75-100: Antifragile</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-500" />
                    <span className="text-slate-400">50-74: Resilient</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-yellow-500" />
                    <span className="text-slate-400">25-49: Moderate</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-red-500" />
                    <span className="text-slate-400">0-24: Fragile</span>
                  </div>
                </div>
              </div>

              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <Flame className="w-6 h-6 text-orange-500 mb-3" strokeWidth={1.5} />
                <h4 className="text-white font-medium mb-2">File Hotspots</h4>
                <p className="text-sm text-slate-400">
                  Identifies bug-prone files and modules. Calculates hotspot intensity based on bug density and code churn.
                </p>
                <div className="mt-3 text-xs text-slate-400">
                  Files with high bug-fix frequency and high churn are flagged as hotspots requiring attention.
                </div>
              </div>

              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <AlertTriangle className="w-6 h-6 text-red-500 mb-3" strokeWidth={1.5} />
                <h4 className="text-white font-medium mb-2">Super-Spreader Commits</h4>
                <p className="text-sm text-slate-400">
                  Identifies commits that introduced bugs affecting many files. Calculates amplification score based on spread.
                </p>
              </div>

              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <Clock className="w-6 h-6 text-purple-500 mb-3" strokeWidth={1.5} />
                <h4 className="text-white font-medium mb-2">Temporal Bug Waves</h4>
                <p className="text-sm text-slate-400">
                  Detects bug surges after major events like releases, refactors, or contributor churn. Measures recovery duration.
                </p>
              </div>

              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <TrendingUp className="w-6 h-6 text-blue-500 mb-3" strokeWidth={1.5} />
                <h4 className="text-white font-medium mb-2">Bug Lineage Tracking</h4>
                <p className="text-sm text-slate-400">
                  Tracks bug lifecycles from introduction to fix. Monitors reintroductions and calculates resolution confidence.
                </p>
              </div>
            </div>
          </div>

          {/* Risk Levels */}
          <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8">
            <h3 className="text-2xl font-medium text-white mb-4">Risk Levels</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-none p-4">
                <div className="text-emerald-500 font-bold mb-2">Low Risk (70-100)</div>
                <p className="text-sm text-slate-400">
                  Healthy project with stable patterns, low bug rates, good contributor diversity, and strong recovery metrics.
                </p>
              </div>
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-none p-4">
                <div className="text-yellow-500 font-bold mb-2">Medium Risk (40-69)</div>
                <p className="text-sm text-slate-400">
                  Moderate health with some concerns. May have inconsistent patterns, elevated bug rates, or moderate contagion.
                </p>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded-none p-4">
                <div className="text-red-500 font-bold mb-2">High Risk (0-39)</div>
                <p className="text-sm text-slate-400">
                  Significant issues detected. High bug rates, volatile changes, poor contributor balance, or high bug contagion.
                </p>
              </div>
            </div>
          </div>

          {/* ML Features */}
          <div className="bg-emerald-950/10 border border-emerald-500/30 rounded-none p-8">
            <AlertTriangle className="w-8 h-8 text-emerald-500 mb-4" strokeWidth={1.5} />
            <h3 className="text-2xl font-medium text-white mb-3">Machine Learning Features</h3>
            <p className="text-slate-400 leading-relaxed mb-4">
              The bug prediction model trains automatically on analyzed repositories using Random Forest classification. 
              It learns from commit patterns (size, files touched, time gaps, code churn) to predict whether future 
              commits are likely to be bug-prone.
            </p>
            <div className="grid md:grid-cols-4 gap-4 text-sm">
              <div className="bg-black/30 border border-slate-800 rounded-none p-3">
                <div className="text-emerald-400 font-medium mb-1">Commit Size</div>
                <div className="text-slate-400">Lines added + deleted</div>
              </div>
              <div className="bg-black/30 border border-slate-800 rounded-none p-3">
                <div className="text-emerald-400 font-medium mb-1">Files Touched</div>
                <div className="text-slate-400">Number of files changed</div>
              </div>
              <div className="bg-black/30 border border-slate-800 rounded-none p-3">
                <div className="text-emerald-400 font-medium mb-1">Time Gap</div>
                <div className="text-slate-400">Hours since last commit</div>
              </div>
              <div className="bg-black/30 border border-slate-800 rounded-none p-3">
                <div className="text-emerald-400 font-medium mb-1">Code Churn</div>
                <div className="text-slate-400">Deletion/addition ratio</div>
              </div>
            </div>
          </div>

          {/* API Endpoints */}
          <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8">
            <h3 className="text-2xl font-medium text-white mb-4">Available API Endpoints</h3>
            <div className="space-y-3 font-mono text-sm">
              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <span className="text-emerald-400">POST</span>
                <span className="text-white ml-2">/api/analyze-repo</span>
                <span className="text-slate-500 ml-4">- Analyze a GitHub repository</span>
              </div>
              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <span className="text-blue-400">POST</span>
                <span className="text-white ml-2">/api/compare-repos</span>
                <span className="text-slate-500 ml-4">- Compare two repositories</span>
              </div>
              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <span className="text-purple-400">GET</span>
                <span className="text-white ml-2">/api/health-score/{'{owner}'}/{'{repo}'}</span>
                <span className="text-slate-500 ml-4">- Get cached health score</span>
              </div>
              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <span className="text-purple-400">GET</span>
                <span className="text-white ml-2">/api/bug-evolution/{'{owner}'}/{'{repo}'}</span>
                <span className="text-slate-500 ml-4">- Get bug evolution analysis</span>
              </div>
              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <span className="text-purple-400">GET</span>
                <span className="text-white ml-2">/api/hotspots/{'{owner}'}/{'{repo}'}</span>
                <span className="text-slate-500 ml-4">- Get file hotspots</span>
              </div>
              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <span className="text-yellow-400">POST</span>
                <span className="text-white ml-2">/api/predict-bug</span>
                <span className="text-slate-500 ml-4">- Predict if commit is bug-prone</span>
              </div>
              <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                <span className="text-slate-400">GET</span>
                <span className="text-white ml-2">/api/rate-limit</span>
                <span className="text-slate-500 ml-4">- Check GitHub API rate limit</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default HealthDashboard;