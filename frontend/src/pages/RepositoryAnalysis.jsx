import { useState } from 'react';
import { motion } from 'framer-motion';
import { Github, Loader2, TrendingUp, GitBranch, Users, AlertTriangle, Activity } from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function RepositoryAnalysis() {
  const [repoUrl, setRepoUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);

  const analyzeRepository = async () => {
    if (!repoUrl.trim()) {
      toast.error('Please enter a repository URL');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/analyze-repo`, {
        url: repoUrl,
        force_refresh: false
      });
      setAnalysisData(response.data);
      toast.success('Repository analyzed successfully!');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to analyze repository');
      console.error('Analysis error:', error);
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

  const getRiskBg = (risk) => {
    switch (risk) {
      case 'Low': return 'bg-emerald-500/10 border-emerald-500/30';
      case 'Medium': return 'bg-yellow-500/10 border-yellow-500/30';
      case 'High': return 'bg-red-500/10 border-red-500/30';
      default: return 'bg-slate-500/10 border-slate-500/30';
    }
  };

  // Prepare chart data
  const commitChartData = analysisData?.commits?.slice(0, 20).reverse().map((commit, idx) => ({
    name: `C${idx + 1}`,
    size: commit.additions + commit.deletions,
    additions: commit.additions,
    deletions: commit.deletions,
    isBug: commit.is_bug_fix ? 1 : 0
  })) || [];

  const bugDistribution = analysisData ? [
    { name: 'Features', value: analysisData.commits.filter(c => c.commit_type === 'feature').length },
    { name: 'Bug Fixes', value: analysisData.commits.filter(c => c.commit_type === 'bug').length },
    { name: 'Refactors', value: analysisData.commits.filter(c => c.commit_type === 'refactor').length }
  ] : [];

  return (
    <div className="container mx-auto px-6 md:px-12 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl md:text-6xl font-bold tracking-tighter text-white mb-6">
          Repository Analysis
        </h1>
        <p className="text-lg text-slate-400 mb-12">
          Analyze the DNA of any GitHub repository—commits, bugs, contributors, and health metrics.
        </p>

        {/* Input Section */}
        <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8 mb-8">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <label className="block text-sm uppercase tracking-widest text-slate-500 mb-2">GitHub Repository URL</label>
              <input
                type="text"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/owner/repo or owner/repo"
                className="w-full bg-black border border-slate-800 focus:border-emerald-500 rounded-none h-12 px-4 text-white font-mono placeholder:text-slate-600 outline-none transition-colors"
                disabled={loading}
                data-testid="repo-url-input"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={analyzeRepository}
                disabled={loading}
                className="bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-none px-8 py-3 h-12 hover:shadow-[0_0_20px_rgba(16,185,129,0.4)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                data-testid="analyze-button"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Github className="w-5 h-5" />
                    Analyze
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Results */}
        {analysisData && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            {/* Health Score Card */}
            <div className={`border rounded-none p-8 ${getRiskBg(analysisData.health_score.risk_level)}`} data-testid="health-score-display">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-white mb-2">{analysisData.repository.full_name}</h2>
                  <p className="text-slate-400">{analysisData.repository.description || 'No description'}</p>
                </div>
                <div className={`text-right`}>
                  <div className="text-xs uppercase tracking-widest text-slate-500 mb-1">Health Score</div>
                  <div className="text-5xl font-bold text-emerald-400 font-mono">
                    {analysisData.health_score.overall_score.toFixed(0)}
                  </div>
                  <div className={`text-lg font-medium ${getRiskColor(analysisData.health_score.risk_level)}`}>
                    {analysisData.health_score.risk_level} Risk
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                  <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Bug Frequency</div>
                  <div className="text-2xl font-bold text-white font-mono">{analysisData.health_score.bug_frequency_score.toFixed(1)}</div>
                </div>
                <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                  <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Commit Stability</div>
                  <div className="text-2xl font-bold text-white font-mono">{analysisData.health_score.commit_stability_score.toFixed(1)}</div>
                </div>
                <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                  <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Contributor Diversity</div>
                  <div className="text-2xl font-bold text-white font-mono">{analysisData.health_score.contributor_diversity_score.toFixed(1)}</div>
                </div>
                <div className="bg-black/30 border border-slate-800 rounded-none p-4">
                  <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Change Volatility</div>
                  <div className="text-2xl font-bold text-white font-mono">{analysisData.health_score.change_volatility_score.toFixed(1)}</div>
                </div>
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid md:grid-cols-4 gap-6 grid-borders">
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6 h-full" data-testid="total-commits-stat">
                <GitBranch className="w-6 h-6 text-emerald-500 mb-3" strokeWidth={1.5} />
                <div className="text-4xl font-bold text-white font-mono mb-1">{analysisData.analysis_summary.total_commits}</div>
                <div className="text-sm text-slate-400">Total Commits</div>
              </div>
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6 h-full" data-testid="bug-fixes-stat">
                <AlertTriangle className="w-6 h-6 text-red-500 mb-3" strokeWidth={1.5} />
                <div className="text-4xl font-bold text-white font-mono mb-1">{analysisData.analysis_summary.bug_fixes}</div>
                <div className="text-sm text-slate-400">Bug Fixes</div>
              </div>
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6 h-full" data-testid="contributors-stat">
                <Users className="w-6 h-6 text-blue-500 mb-3" strokeWidth={1.5} />
                <div className="text-4xl font-bold text-white font-mono mb-1">{analysisData.analysis_summary.total_contributors}</div>
                <div className="text-sm text-slate-400">Contributors</div>
              </div>
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6 h-full" data-testid="forks-stat">
                <TrendingUp className="w-6 h-6 text-purple-500 mb-3" strokeWidth={1.5} />
                <div className="text-4xl font-bold text-white font-mono mb-1">{analysisData.analysis_summary.total_forks}</div>
                <div className="text-sm text-slate-400">Forks</div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6">
                <h3 className="text-xl font-medium text-white mb-4">Commit Size Evolution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={commitChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#020617', 
                        border: '1px solid #1e293b',
                        borderRadius: 0
                      }}
                    />
                    <Line type="monotone" dataKey="size" stroke="#10b981" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6">
                <h3 className="text-xl font-medium text-white mb-4">Commit Type Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={bugDistribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#020617', 
                        border: '1px solid #1e293b',
                        borderRadius: 0
                      }}
                    />
                    <Bar dataKey="value" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Recent Commits */}
            <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6">
              <h3 className="text-xl font-medium text-white mb-4">Recent Commits</h3>
              <div className="space-y-2 max-h-96 overflow-y-auto" data-testid="commits-list">
                {analysisData.commits.slice(0, 15).map((commit) => (
                  <div 
                    key={commit.sha} 
                    className="bg-black/30 border border-slate-800 rounded-none p-4 hover:border-emerald-500/50 transition-colors"
                    data-testid={`commit-item-${commit.sha.substring(0, 7)}`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-sm font-mono text-slate-500">{commit.sha.substring(0, 7)}</span>
                      {commit.is_bug_fix && (
                        <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded-none border border-red-500/30">BUG FIX</span>
                      )}
                    </div>
                    <p className="text-white mb-2">{commit.message.split('\n')[0]}</p>
                    <div className="flex items-center gap-4 text-sm text-slate-400">
                      <span>{commit.author_name}</span>
                      <span>•</span>
                      <span>{new Date(commit.author_date).toLocaleDateString()}</span>
                      <span>•</span>
                      <span className="text-emerald-400">+{commit.additions}</span>
                      <span className="text-red-400">-{commit.deletions}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Contributors */}
            <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-6">
              <h3 className="text-xl font-medium text-white mb-4">Top Contributors</h3>
              <div className="grid md:grid-cols-2 gap-4">
                {analysisData.contributors.slice(0, 6).map((contributor) => (
                  <div 
                    key={contributor.login}
                    className="flex items-center gap-4 bg-black/30 border border-slate-800 rounded-none p-4 hover:border-emerald-500/50 transition-colors"
                    data-testid={`contributor-${contributor.login}`}
                  >
                    <img 
                      src={contributor.avatar_url} 
                      alt={contributor.login}
                      className="w-12 h-12 rounded-none border border-slate-700"
                    />
                    <div className="flex-1">
                      <div className="text-white font-medium">{contributor.login}</div>
                      <div className="text-sm text-slate-400">{contributor.contributions} contributions</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}

export default RepositoryAnalysis;
