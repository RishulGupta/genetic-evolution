import { useState } from 'react';
import { motion } from 'framer-motion';
import { GitCompare, Loader2, Trophy, Shield } from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid } from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function ProjectComparison() {
  const [repoAUrl, setRepoAUrl] = useState('');
  const [repoBUrl, setRepoBUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);

  const compareRepositories = async () => {
    if (!repoAUrl.trim() || !repoBUrl.trim()) {
      toast.error('Please enter both repository URLs');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/compare-repos`, {
        repo_a_url: repoAUrl,
        repo_b_url: repoBUrl
      });
      setComparisonData(response.data);
      toast.success('Comparison complete!');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to compare repositories');
      console.error('Comparison error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'Low': return 'text-emerald-500';
      case 'Medium': return 'text-yellow-500';
      case 'High': return 'text-red-500';
      default: return 'text-slate-500';
    }
  };

  // Prepare comparison chart data
  const comparisonChartData = comparisonData ? [
    {
      metric: 'Health Score',
      repoA: comparisonData.comparison.metrics.repo_a.health_score,
      repoB: comparisonData.comparison.metrics.repo_b.health_score
    },
    {
      metric: 'Bug Ratio',
      repoA: (1 - comparisonData.comparison.metrics.repo_a.bug_ratio) * 100,
      repoB: (1 - comparisonData.comparison.metrics.repo_b.bug_ratio) * 100
    }
  ] : [];

  return (
    <div className="container mx-auto px-6 md:px-12 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl md:text-6xl font-bold tracking-tighter text-white mb-6">
          Project Comparison
        </h1>
        <p className="text-lg text-slate-400 mb-12">
          Compare two GitHub repositories to determine which is healthier, more stable, and less risky.
        </p>

        {/* Input Section */}
        <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8 mb-8">
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm uppercase tracking-widest text-slate-500 mb-2">Project A URL</label>
              <input
                type="text"
                value={repoAUrl}
                onChange={(e) => setRepoAUrl(e.target.value)}
                placeholder="https://github.com/owner/repo-a"
                className="w-full bg-black border border-slate-800 focus:border-emerald-500 rounded-none h-12 px-4 text-white font-mono placeholder:text-slate-600 outline-none transition-colors"
                disabled={loading}
                data-testid="repo-a-input"
              />
            </div>
            <div>
              <label className="block text-sm uppercase tracking-widest text-slate-500 mb-2">Project B URL</label>
              <input
                type="text"
                value={repoBUrl}
                onChange={(e) => setRepoBUrl(e.target.value)}
                placeholder="https://github.com/owner/repo-b"
                className="w-full bg-black border border-slate-800 focus:border-emerald-500 rounded-none h-12 px-4 text-white font-mono placeholder:text-slate-600 outline-none transition-colors"
                disabled={loading}
                data-testid="repo-b-input"
              />
            </div>
          </div>
          <button
            onClick={compareRepositories}
            disabled={loading}
            className="w-full bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-none px-8 py-3 hover:shadow-[0_0_20px_rgba(16,185,129,0.4)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            data-testid="compare-button"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Comparing...
              </>
            ) : (
              <>
                <GitCompare className="w-5 h-5" />
                Compare Projects
              </>
            )}
          </button>
        </div>

        {/* Results */}
        {comparisonData && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            {/* Winner Announcement */}
            <div className="bg-emerald-950/10 border border-emerald-500/30 rounded-none p-8 text-center relative overflow-hidden" data-testid="winner-announcement">
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/5 to-blue-500/5" />
              <div className="relative">
                <Trophy className="w-12 h-12 text-emerald-500 mx-auto mb-4" strokeWidth={1.5} />
                <h2 className="text-3xl font-bold text-white mb-4">Winner</h2>
                <div className="text-5xl font-bold text-emerald-400 font-mono mb-4">
                  {comparisonData.comparison.winner}
                </div>
                <p className="text-lg text-slate-300 max-w-3xl mx-auto leading-relaxed">
                  {comparisonData.comparison.explanation}
                </p>
              </div>
            </div>

            {/* Side-by-Side Comparison */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Project A */}
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6" data-testid="project-a-details">
                <h3 className="text-xl font-medium text-white mb-4">{comparisonData.comparison.repo_a}</h3>
                <div className="space-y-4">
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Health Score</div>
                    <div className="text-4xl font-bold text-emerald-400 font-mono">
                      {comparisonData.comparison.metrics.repo_a.health_score.toFixed(0)}
                    </div>
                  </div>
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Risk Level</div>
                    <div className={`text-2xl font-bold ${getRiskColor(comparisonData.comparison.metrics.repo_a.risk_level)}`}>
                      {comparisonData.comparison.metrics.repo_a.risk_level}
                    </div>
                  </div>
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Total Commits</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {comparisonData.comparison.metrics.repo_a.total_commits}
                    </div>
                  </div>
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Bug Ratio</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {(comparisonData.comparison.metrics.repo_a.bug_ratio * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Project B */}
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6" data-testid="project-b-details">
                <h3 className="text-xl font-medium text-white mb-4">{comparisonData.comparison.repo_b}</h3>
                <div className="space-y-4">
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Health Score</div>
                    <div className="text-4xl font-bold text-emerald-400 font-mono">
                      {comparisonData.comparison.metrics.repo_b.health_score.toFixed(0)}
                    </div>
                  </div>
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Risk Level</div>
                    <div className={`text-2xl font-bold ${getRiskColor(comparisonData.comparison.metrics.repo_b.risk_level)}`}>
                      {comparisonData.comparison.metrics.repo_b.risk_level}
                    </div>
                  </div>
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Total Commits</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {comparisonData.comparison.metrics.repo_b.total_commits}
                    </div>
                  </div>
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Bug Ratio</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {(comparisonData.comparison.metrics.repo_b.bug_ratio * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Comparison Chart */}
            <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6">
              <h3 className="text-xl font-medium text-white mb-4">Visual Comparison</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={comparisonChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="metric" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#020617', 
                      border: '1px solid #1e293b',
                      borderRadius: 0
                    }}
                  />
                  <Legend />
                  <Bar dataKey="repoA" fill="#10b981" name="Project A" />
                  <Bar dataKey="repoB" fill="#3b82f6" name="Project B" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}

export default ProjectComparison;
