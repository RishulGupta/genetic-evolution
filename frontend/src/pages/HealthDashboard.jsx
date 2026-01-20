import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shield, AlertTriangle, TrendingUp, Github } from 'lucide-react';
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
          Monitor system health, API usage, and application metrics.
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
            <h3 className="text-xl font-medium text-white mb-2">Analyses Run</h3>
            <div className="text-3xl font-bold text-purple-400 font-mono mb-2">-</div>
            <p className="text-sm text-slate-400">Session statistics</p>
          </div>
        </div>

        {/* Information Cards */}
        <div className="space-y-6">
          <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8">
            <h3 className="text-2xl font-medium text-white mb-4">About Health Scoring</h3>
            <div className="text-slate-400 space-y-3">
              <p>
                The health score (0-100) is calculated using four key metrics:
              </p>
              <ul className="list-none space-y-2 ml-4">
                <li className="flex items-start gap-3">
                  <span className="text-emerald-500 font-bold">•</span>
                  <span><strong className="text-white">Bug Frequency (25pts):</strong> Lower bug ratios indicate healthier codebases</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 font-bold">•</span>
                  <span><strong className="text-white">Commit Stability (25pts):</strong> Consistent commit sizes show predictable development</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-purple-500 font-bold">•</span>
                  <span><strong className="text-white">Contributor Diversity (25pts):</strong> Better distribution prevents single-point failures</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-yellow-500 font-bold">•</span>
                  <span><strong className="text-white">Change Volatility (25pts):</strong> Recent bug patterns affect long-term risk</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8">
            <h3 className="text-2xl font-medium text-white mb-4">Risk Levels</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-none p-4">
                <div className="text-emerald-500 font-bold mb-2">Low Risk (70-100)</div>
                <p className="text-sm text-slate-400">
                  Healthy project with stable patterns, low bug rates, and good contributor diversity.
                </p>
              </div>
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-none p-4">
                <div className="text-yellow-500 font-bold mb-2">Medium Risk (40-69)</div>
                <p className="text-sm text-slate-400">
                  Moderate health with some concerns. May have inconsistent patterns or elevated bug rates.
                </p>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded-none p-4">
                <div className="text-red-500 font-bold mb-2">High Risk (0-39)</div>
                <p className="text-sm text-slate-400">
                  Significant issues detected. High bug rates, volatile changes, or poor contributor balance.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-emerald-950/10 border border-emerald-500/30 rounded-none p-8">
            <AlertTriangle className="w-8 h-8 text-emerald-500 mb-4" strokeWidth={1.5} />
            <h3 className="text-2xl font-medium text-white mb-3">Machine Learning Features</h3>
            <p className="text-slate-400 leading-relaxed">
              The bug prediction model trains automatically on analyzed repositories using Random Forest classification. 
              It learns from commit patterns (size, files touched, time gaps, code churn) to predict whether future 
              commits are likely to be bug-prone. The model improves as more repositories are analyzed.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default HealthDashboard;
