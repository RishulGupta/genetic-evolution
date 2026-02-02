import { useState } from 'react';
import { motion } from 'framer-motion';
import { GitCompare, Loader2, Trophy, Shield, Bug, Heart, Zap, Flame } from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

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

  const getContagionColor = (score) => {
    if (score < 0.3) return 'text-emerald-500';
    if (score < 0.6) return 'text-yellow-500';
    return 'text-red-500';
  };

  // Prepare comparison chart data
  const comparisonChartData = comparisonData ? [
    {
      metric: 'Health Score',
      repoA: comparisonData.comparison.metrics.repo_a.health_score,
      repoB: comparisonData.comparison.metrics.repo_b.health_score
    },
    {
      metric: 'Bug Safety',
      repoA: (1 - comparisonData.comparison.metrics.repo_a.bug_ratio) * 100,
      repoB: (1 - comparisonData.comparison.metrics.repo_b.bug_ratio) * 100
    },
    {
      metric: 'Immunity',
      repoA: comparisonData.comparison.metrics.repo_a.immunity_score || 50,
      repoB: comparisonData.comparison.metrics.repo_b.immunity_score || 50
    },
    {
      metric: 'Containment',
      repoA: (1 - (comparisonData.comparison.metrics.repo_a.contagion_score || 0)) * 100,
      repoB: (1 - (comparisonData.comparison.metrics.repo_b.contagion_score || 0)) * 100
    }
  ] : [];

  // Radar comparison data
  const radarDataA = comparisonData?.repo_a_analysis ? [
    { metric: 'Health', A: comparisonData.repo_a_analysis.health_score.overall_score },
    { metric: 'Stability', A: comparisonData.repo_a_analysis.health_score.commit_stability_score * 4 },
    { metric: 'Diversity', A: comparisonData.repo_a_analysis.health_score.contributor_diversity_score * 4 },
    { metric: 'Freshness', A: comparisonData.repo_a_analysis.health_score.change_volatility_score * 4 },
    { metric: 'Recovery', A: comparisonData.repo_a_analysis.bug_evolution?.recovery_metrics?.immunity_score || 50 }
  ] : [];

  const radarDataB = comparisonData?.repo_b_analysis ? [
    { metric: 'Health', B: comparisonData.repo_b_analysis.health_score.overall_score },
    { metric: 'Stability', B: comparisonData.repo_b_analysis.health_score.commit_stability_score * 4 },
    { metric: 'Diversity', B: comparisonData.repo_b_analysis.health_score.contributor_diversity_score * 4 },
    { metric: 'Freshness', B: comparisonData.repo_b_analysis.health_score.change_volatility_score * 4 },
    { metric: 'Recovery', B: comparisonData.repo_b_analysis.bug_evolution?.recovery_metrics?.immunity_score || 50 }
  ] : [];

  // Merge radar data
  const radarData = radarDataA.map((item, idx) => ({
    ...item,
    ...(radarDataB[idx] || {})
  }));

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

            {/* Bug Evolution Comparison */}
            {comparisonData.comparison.bug_evolution_comparison && Object.keys(comparisonData.comparison.bug_evolution_comparison).length > 0 && (
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8" data-testid="bug-evolution-comparison">
                <div className="flex items-center gap-3 mb-6">
                  <Bug className="w-8 h-8 text-red-500" strokeWidth={1.5} />
                  <h3 className="text-2xl font-bold text-white">Bug Evolution Comparison</h3>
                </div>
                
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="bg-black/30 border border-slate-800 rounded-none p-6 text-center">
                    <Zap className="w-6 h-6 text-yellow-500 mx-auto mb-3" strokeWidth={1.5} />
                    <div className="text-sm uppercase tracking-widest text-slate-500 mb-2">Lower Contagion</div>
                    <div className="text-xl font-bold text-emerald-400">
                      {comparisonData.comparison.bug_evolution_comparison.contagion_winner}
                    </div>
                  </div>
                  
                  <div className="bg-black/30 border border-slate-800 rounded-none p-6 text-center">
                    <Heart className="w-6 h-6 text-pink-500 mx-auto mb-3" strokeWidth={1.5} />
                    <div className="text-sm uppercase tracking-widest text-slate-500 mb-2">Better Recovery</div>
                    <div className="text-xl font-bold text-emerald-400">
                      {comparisonData.comparison.bug_evolution_comparison.recovery_winner}
                    </div>
                  </div>
                  
                  <div className="bg-black/30 border border-slate-800 rounded-none p-6 text-center">
                    <Flame className="w-6 h-6 text-orange-500 mx-auto mb-3" strokeWidth={1.5} />
                    <div className="text-sm uppercase tracking-widest text-slate-500 mb-2">Hotspots</div>
                    <div className="text-lg font-mono">
                      <span className="text-blue-400">{comparisonData.comparison.bug_evolution_comparison.hotspot_count_a}</span>
                      <span className="text-slate-500 mx-2">vs</span>
                      <span className="text-purple-400">{comparisonData.comparison.bug_evolution_comparison.hotspot_count_b}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Radar Comparison */}
            <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6">
              <h3 className="text-xl font-medium text-white mb-4">Multi-Dimensional Comparison</h3>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#1e293b" />
                  <PolarAngleAxis dataKey="metric" stroke="#94a3b8" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#1e293b" />
                  <Radar name="Project A" dataKey="A" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                  <Radar name="Project B" dataKey="B" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Side-by-Side Comparison */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Project A */}
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6" data-testid="project-a-details">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-4 h-4 bg-emerald-500" />
                  <h3 className="text-xl font-medium text-white">{comparisonData.comparison.repo_a}</h3>
                </div>
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
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Bug Contagion</div>
                    <div className={`text-2xl font-bold font-mono ${getContagionColor(comparisonData.comparison.metrics.repo_a.contagion_score || 0)}`}>
                      {((comparisonData.comparison.metrics.repo_a.contagion_score || 0) * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Immunity Score</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {(comparisonData.comparison.metrics.repo_a.immunity_score || 0).toFixed(0)}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                      <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Commits</div>
                      <div className="text-xl font-bold text-white font-mono">
                        {comparisonData.comparison.metrics.repo_a.total_commits}
                      </div>
                    </div>
                    <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                      <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Bug Ratio</div>
                      <div className="text-xl font-bold text-white font-mono">
                        {(comparisonData.comparison.metrics.repo_a.bug_ratio * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Project B */}
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6" data-testid="project-b-details">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-4 h-4 bg-blue-500" />
                  <h3 className="text-xl font-medium text-white">{comparisonData.comparison.repo_b}</h3>
                </div>
                <div className="space-y-4">
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Health Score</div>
                    <div className="text-4xl font-bold text-blue-400 font-mono">
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
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Bug Contagion</div>
                    <div className={`text-2xl font-bold font-mono ${getContagionColor(comparisonData.comparison.metrics.repo_b.contagion_score || 0)}`}>
                      {((comparisonData.comparison.metrics.repo_b.contagion_score || 0) * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Immunity Score</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {(comparisonData.comparison.metrics.repo_b.immunity_score || 0).toFixed(0)}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                      <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Commits</div>
                      <div className="text-xl font-bold text-white font-mono">
                        {comparisonData.comparison.metrics.repo_b.total_commits}
                      </div>
                    </div>
                    <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                      <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Bug Ratio</div>
                      <div className="text-xl font-bold text-white font-mono">
                        {(comparisonData.comparison.metrics.repo_b.bug_ratio * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Comparison Chart */}
            <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6">
              <h3 className="text-xl font-medium text-white mb-4">Metric Comparison</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={comparisonChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="metric" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[0, 100]} />
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