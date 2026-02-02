import { useState } from 'react';
import { motion } from 'framer-motion';
import { Github, Loader2, TrendingUp, GitBranch, Users, AlertTriangle, Activity, Bug, Shield, Flame, Zap, Heart, Clock } from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

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

  const getContagionColor = (interpretation) => {
    switch (interpretation) {
      case 'Contained': return 'text-emerald-500';
      case 'Moderate': return 'text-yellow-500';
      case 'Highly Infectious': return 'text-red-500';
      default: return 'text-slate-500';
    }
  };

  const getResilienceColor = (resilience) => {
    switch (resilience) {
      case 'Antifragile': return 'text-emerald-500';
      case 'Resilient': return 'text-blue-500';
      case 'Moderate': return 'text-yellow-500';
      case 'Fragile': return 'text-red-500';
      default: return 'text-slate-500';
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
    { name: 'Features', value: analysisData.commits.filter(c => c.commit_type === 'feature').length, color: '#10b981' },
    { name: 'Bug Fixes', value: analysisData.commits.filter(c => c.commit_type === 'bug').length, color: '#ef4444' },
    { name: 'Refactors', value: analysisData.commits.filter(c => c.commit_type === 'refactor').length, color: '#3b82f6' },
    { name: 'Chores', value: analysisData.commits.filter(c => c.commit_type === 'chore').length, color: '#8b5cf6' }
  ] : [];

  // Bug evolution radar data
  const bugEvolutionRadar = analysisData?.bug_evolution ? [
    { metric: 'Contagion', value: (1 - analysisData.bug_evolution.contagion_score.score) * 100, fullMark: 100 },
    { metric: 'Recovery', value: analysisData.bug_evolution.recovery_metrics.immunity_score, fullMark: 100 },
    { metric: 'Stability', value: analysisData.health_score.commit_stability_score * 4, fullMark: 100 },
    { metric: 'Diversity', value: analysisData.health_score.contributor_diversity_score * 4, fullMark: 100 },
    { metric: 'Freshness', value: analysisData.health_score.change_volatility_score * 4, fullMark: 100 }
  ] : [];

  // Hotspot intensity data
  const hotspotData = analysisData?.bug_evolution?.file_hotspots?.slice(0, 10).map(h => ({
    name: h.filename.split('/').pop(),
    intensity: h.hotspot_intensity * 100,
    bugs: h.bug_count + h.bug_fix_count
  })) || [];

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

            {/* Bug Evolution Section */}
            {analysisData.bug_evolution && (
              <div className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8" data-testid="bug-evolution-section">
                <div className="flex items-center gap-3 mb-6">
                  <Bug className="w-8 h-8 text-red-500" strokeWidth={1.5} />
                  <h3 className="text-2xl font-bold text-white">Bug Evolution Analysis</h3>
                </div>

                <div className="grid md:grid-cols-3 gap-6 mb-8">
                  {/* Contagion Score */}
                  <div className="bg-black/30 border border-slate-800 rounded-none p-6" data-testid="contagion-score">
                    <div className="flex items-center gap-2 mb-4">
                      <Zap className="w-5 h-5 text-yellow-500" strokeWidth={1.5} />
                      <span className="text-sm uppercase tracking-widest text-slate-500">Bug Contagion</span>
                    </div>
                    <div className="text-4xl font-bold text-white font-mono mb-2">
                      {(analysisData.bug_evolution.contagion_score.score * 100).toFixed(0)}%
                    </div>
                    <div className={`text-lg font-medium ${getContagionColor(analysisData.bug_evolution.contagion_score.interpretation)}`}>
                      {analysisData.bug_evolution.contagion_score.interpretation}
                    </div>
                    <div className="mt-4 space-y-2 text-sm text-slate-400">
                      <div className="flex justify-between">
                        <span>Avg Lifespan</span>
                        <span className="font-mono">{analysisData.bug_evolution.contagion_score.avg_lifespan_days.toFixed(1)} days</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Reinfection Rate</span>
                        <span className="font-mono">{(analysisData.bug_evolution.contagion_score.reinfection_rate * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Recovery Metrics */}
                  <div className="bg-black/30 border border-slate-800 rounded-none p-6" data-testid="recovery-metrics">
                    <div className="flex items-center gap-2 mb-4">
                      <Heart className="w-5 h-5 text-pink-500" strokeWidth={1.5} />
                      <span className="text-sm uppercase tracking-widest text-slate-500">Recovery & Immunity</span>
                    </div>
                    <div className="text-4xl font-bold text-white font-mono mb-2">
                      {analysisData.bug_evolution.recovery_metrics.immunity_score.toFixed(0)}
                    </div>
                    <div className={`text-lg font-medium ${getResilienceColor(analysisData.bug_evolution.recovery_metrics.resilience_class)}`}>
                      {analysisData.bug_evolution.recovery_metrics.resilience_class}
                    </div>
                    <div className="mt-4 space-y-2 text-sm text-slate-400">
                      <div className="flex justify-between">
                        <span>Avg Fix Time</span>
                        <span className="font-mono">{analysisData.bug_evolution.recovery_metrics.avg_time_to_fix_hours.toFixed(1)}h</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Clean Streak</span>
                        <span className="font-mono">{analysisData.bug_evolution.recovery_metrics.clean_commit_streak} commits</span>
                      </div>
                    </div>
                  </div>

                  {/* Bug Summary */}
                  <div className="bg-black/30 border border-slate-800 rounded-none p-6" data-testid="bug-summary">
                    <div className="flex items-center gap-2 mb-4">
                      <Activity className="w-5 h-5 text-emerald-500" strokeWidth={1.5} />
                      <span className="text-sm uppercase tracking-widest text-slate-500">Bug Status</span>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-3xl font-bold text-emerald-400 font-mono">{analysisData.bug_evolution.resolved_bugs}</div>
                        <div className="text-sm text-slate-400">Resolved</div>
                      </div>
                      <div>
                        <div className="text-3xl font-bold text-red-400 font-mono">{analysisData.bug_evolution.active_bugs}</div>
                        <div className="text-sm text-slate-400">Active</div>
                      </div>
                      <div>
                        <div className="text-3xl font-bold text-white font-mono">{analysisData.bug_evolution.total_bugs_detected}</div>
                        <div className="text-sm text-slate-400">Total Detected</div>
                      </div>
                      <div>
                        <div className="text-3xl font-bold text-yellow-400 font-mono">{analysisData.bug_evolution.file_hotspots?.length || 0}</div>
                        <div className="text-sm text-slate-400">Hotspots</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Bug Evolution Radar & Hotspots */}
                <div className="grid md:grid-cols-2 gap-6">
                  {/* Radar Chart */}
                  <div className="bg-black/30 border border-slate-800 rounded-none p-6">
                    <h4 className="text-lg font-medium text-white mb-4">Health Radar</h4>
                    <ResponsiveContainer width="100%" height={250}>
                      <RadarChart data={bugEvolutionRadar}>
                        <PolarGrid stroke="#1e293b" />
                        <PolarAngleAxis dataKey="metric" stroke="#94a3b8" />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#1e293b" />
                        <Radar name="Score" dataKey="value" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Hotspots Bar Chart */}
                  <div className="bg-black/30 border border-slate-800 rounded-none p-6">
                    <h4 className="text-lg font-medium text-white mb-4">File Hotspots</h4>
                    {hotspotData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={hotspotData} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis type="number" stroke="#94a3b8" domain={[0, 100]} />
                          <YAxis type="category" dataKey="name" stroke="#94a3b8" width={100} />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: '#020617', 
                              border: '1px solid #1e293b',
                              borderRadius: 0
                            }}
                          />
                          <Bar dataKey="intensity" fill="#ef4444" name="Intensity %" />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-[250px] text-slate-500">
                        No hotspots detected - Great job!
                      </div>
                    )}
                  </div>
                </div>

                {/* Developer Influence */}
                {analysisData.bug_evolution.developer_influence?.length > 0 && (
                  <div className="mt-6 bg-black/30 border border-slate-800 rounded-none p-6">
                    <h4 className="text-lg font-medium text-white mb-4">Developer Impact Analysis</h4>
                    <p className="text-sm text-slate-500 mb-4">Systemic analysis of contribution patterns (not personal blame)</p>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {analysisData.bug_evolution.developer_influence.slice(0, 6).map((dev, idx) => (
                        <div key={idx} className="bg-black/20 border border-slate-800 rounded-none p-4">
                          <div className="text-white font-medium mb-2">{dev.contributor}</div>
                          <div className="grid grid-cols-3 gap-2 text-sm">
                            <div>
                              <div className="text-slate-500">Commits</div>
                              <div className="font-mono text-white">{dev.commits_total}</div>
                            </div>
                            <div>
                              <div className="text-slate-500">Fixes</div>
                              <div className="font-mono text-emerald-400">{dev.bug_fixes}</div>
                            </div>
                            <div>
                              <div className="text-slate-500">Impact</div>
                              <div className={`font-mono ${dev.net_impact_score >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {dev.net_impact_score >= 0 ? '+' : ''}{(dev.net_impact_score * 100).toFixed(0)}%
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Temporal Bug Waves */}
                {analysisData.bug_evolution.temporal_waves?.length > 0 && (
                  <div className="mt-6 bg-black/30 border border-slate-800 rounded-none p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <Clock className="w-5 h-5 text-purple-500" strokeWidth={1.5} />
                      <h4 className="text-lg font-medium text-white">Bug Waves Detected</h4>
                    </div>
                    <div className="space-y-3">
                      {analysisData.bug_evolution.temporal_waves.map((wave, idx) => (
                        <div key={idx} className="bg-black/20 border border-slate-800 rounded-none p-4 flex items-center justify-between">
                          <div>
                            <span className="text-purple-400 font-medium capitalize">{wave.trigger_event}</span>
                            <span className="text-slate-500 ml-2">triggered {wave.bug_count} bugs</span>
                          </div>
                          <div className="text-sm text-slate-400">
                            Recovery: <span className="font-mono text-white">{wave.recovery_duration_days.toFixed(1)} days</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

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
                  <PieChart>
                    <Pie
                      data={bugDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {bugDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#020617', 
                        border: '1px solid #1e293b',
                        borderRadius: 0
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-4 mt-4">
                  {bugDistribution.map((item, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <div className="w-3 h-3" style={{ backgroundColor: item.color }} />
                      <span className="text-sm text-slate-400">{item.name}</span>
                    </div>
                  ))}
                </div>
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
                      <div className="flex gap-2">
                        {commit.is_bug_fix && (
                          <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded-none border border-red-500/30">BUG FIX</span>
                        )}
                        {commit.is_bug_introducing && (
                          <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded-none border border-yellow-500/30">RISKY</span>
                        )}
                        <span className={`text-xs px-2 py-1 rounded-none border ${
                          commit.commit_type === 'feature' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' :
                          commit.commit_type === 'bug' ? 'bg-red-500/20 text-red-400 border-red-500/30' :
                          commit.commit_type === 'refactor' ? 'bg-blue-500/20 text-blue-400 border-blue-500/30' :
                          'bg-slate-500/20 text-slate-400 border-slate-500/30'
                        }`}>
                          {commit.commit_type.toUpperCase()}
                        </span>
                      </div>
                    </div>
                    <p className="text-white mb-2">{commit.message.split('\n')[0]}</p>
                    <div className="flex items-center gap-4 text-sm text-slate-400">
                      <span>{commit.author_name}</span>
                      <span>•</span>
                      <span>{new Date(commit.author_date).toLocaleDateString()}</span>
                      <span>•</span>
                      <span className="text-emerald-400">+{commit.additions}</span>
                      <span className="text-red-400">-{commit.deletions}</span>
                      {commit.files_changed > 0 && (
                        <>
                          <span>•</span>
                          <span>{commit.files_changed} files</span>
                        </>
                      )}
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