import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Github, Activity, GitCompare, TrendingUp, Shield, Dna } from 'lucide-react';

function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: 'url(https://images.unsplash.com/photo-1610519911439-e4eec73e03b9?crop=entropy&cs=srgb&fm=jpg&q=85)',
            backgroundSize: 'cover',
            backgroundPosition: 'center'
          }}
        />
        <div className="relative container mx-auto px-6 md:px-12 py-24 md:py-32">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-4xl"
          >
            <div className="flex items-center gap-3 mb-6">
              <Dna className="w-10 h-10 text-emerald-500" strokeWidth={1.5} />
              <h1 className="text-5xl md:text-7xl font-bold tracking-tighter text-white">
                Codebase Evolution Genome
              </h1>
            </div>
            <p className="text-xl md:text-2xl text-slate-400 leading-relaxed mb-8">
              Analyze how software projects evolve over time. Treat code like DNA—detect patterns, 
              measure health, and compare projects to determine which is more stable and less risky.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link to="/analyze" data-testid="analyze-cta-button">
                <button className="bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-none px-8 py-3 hover:shadow-[0_0_20px_rgba(16,185,129,0.4)] transition-all">
                  Analyze Repository
                </button>
              </Link>
              <Link to="/compare" data-testid="compare-cta-button">
                <button className="bg-transparent border border-slate-700 text-white hover:border-emerald-500 hover:text-emerald-500 rounded-none px-8 py-3 transition-all">
                  Compare Projects
                </button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="container mx-auto px-6 md:px-12 py-24">
        <div className="grid md:grid-cols-3 gap-6 grid-borders">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8 hover:border-emerald-500/50 transition-colors h-full"
            data-testid="feature-dna-analysis"
          >
            <Activity className="w-8 h-8 text-emerald-500 mb-4" strokeWidth={1.5} />
            <h3 className="text-xl md:text-2xl font-medium text-white mb-3">DNA Feature Analysis</h3>
            <p className="text-slate-400 leading-relaxed">
              Extract commit signatures: size, churn, time gaps, and bug patterns. 
              Understand the biological rhythm of your codebase.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8 hover:border-emerald-500/50 transition-colors h-full"
            data-testid="feature-health-scoring"
          >
            <Shield className="w-8 h-8 text-blue-500 mb-4" strokeWidth={1.5} />
            <h3 className="text-xl md:text-2xl font-medium text-white mb-3">Health Scoring</h3>
            <p className="text-slate-400 leading-relaxed">
              Get a 0-100 health score based on bug frequency, commit stability, 
              contributor diversity, and change volatility.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="bg-black/40 border border-slate-800 backdrop-blur-sm rounded-none p-8 hover:border-emerald-500/50 transition-colors h-full"
            data-testid="feature-project-comparison"
          >
            <GitCompare className="w-8 h-8 text-purple-500 mb-4" strokeWidth={1.5} />
            <h3 className="text-xl md:text-2xl font-medium text-white mb-3">Project Comparison</h3>
            <p className="text-slate-400 leading-relaxed">
              Compare two repositories side-by-side. Get a clear winner with 
              plain-English explanations of risks and stability.
            </p>
          </motion.div>
        </div>
      </section>

      {/* How It Works */}
      <section className="container mx-auto px-6 md:px-12 py-24">
        <h2 className="text-3xl md:text-5xl font-semibold tracking-tight text-white mb-12">How It Works</h2>
        <div className="grid md:grid-cols-2 gap-12">
          <div>
            <div className="flex items-start gap-4 mb-8">
              <div className="bg-emerald-500 text-black font-bold w-10 h-10 flex items-center justify-center rounded-none font-mono">1</div>
              <div>
                <h3 className="text-xl font-medium text-white mb-2">Data Ingestion</h3>
                <p className="text-slate-400">Fetch commits, contributors, forks, and file changes from GitHub repositories using the REST API.</p>
              </div>
            </div>
            <div className="flex items-start gap-4 mb-8">
              <div className="bg-blue-500 text-white font-bold w-10 h-10 flex items-center justify-center rounded-none font-mono">2</div>
              <div>
                <h3 className="text-xl font-medium text-white mb-2">DNA Feature Engineering</h3>
                <p className="text-slate-400">Calculate commit size, code churn, time gaps, and classify commits as bugs, features, or refactors.</p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="bg-purple-500 text-white font-bold w-10 h-10 flex items-center justify-center rounded-none font-mono">3</div>
              <div>
                <h3 className="text-xl font-medium text-white mb-2">Health Analysis</h3>
                <p className="text-slate-400">Generate comprehensive health scores and risk assessments using machine learning.</p>
              </div>
            </div>
          </div>
          <div className="relative">
            <img 
              src="https://images.unsplash.com/photo-1738082956220-a1f20a8632ce?crop=entropy&cs=srgb&fm=jpg&q=85"
              alt="Network visualization"
              className="w-full h-full object-cover border border-slate-800 rounded-none"
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-6 md:px-12 py-24">
        <div className="bg-emerald-950/10 border border-emerald-500/30 rounded-none p-12 text-center relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/5 to-blue-500/5" />
          <div className="relative">
            <TrendingUp className="w-12 h-12 text-emerald-500 mx-auto mb-6" strokeWidth={1.5} />
            <h2 className="text-3xl md:text-5xl font-semibold tracking-tight text-white mb-6">
              Ready to analyze your codebase?
            </h2>
            <p className="text-slate-400 text-lg mb-8 max-w-2xl mx-auto">
              Discover the health patterns, bug trends, and evolution trajectory of any GitHub repository.
            </p>
            <Link to="/analyze" data-testid="get-started-button">
              <button className="bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-none px-12 py-4 hover:shadow-[0_0_20px_rgba(16,185,129,0.4)] transition-all">
                Get Started
              </button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;
