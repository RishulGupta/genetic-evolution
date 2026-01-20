import { Link, useLocation } from 'react-router-dom';
import { Dna } from 'lucide-react';

function Navigation() {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="fixed top-0 w-full z-50 bg-black/70 backdrop-blur-xl border-b border-white/10 h-16 flex items-center px-6 md:px-12" data-testid="main-navigation">
      <div className="flex items-center gap-8 flex-1">
        <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity" data-testid="nav-home">
          <Dna className="w-6 h-6 text-emerald-500" strokeWidth={1.5} />
          <span className="text-white font-bold text-lg hidden md:inline">CEG</span>
        </Link>
        
        <div className="flex items-center gap-1">
          <Link to="/" data-testid="nav-home-link">
            <button className={`text-slate-400 hover:text-white hover:bg-white/5 rounded-none px-4 py-2 transition-colors ${
              isActive('/') ? 'text-emerald-500 bg-emerald-500/10' : ''
            }`}>
              Home
            </button>
          </Link>
          <Link to="/analyze" data-testid="nav-analyze-link">
            <button className={`text-slate-400 hover:text-white hover:bg-white/5 rounded-none px-4 py-2 transition-colors ${
              isActive('/analyze') ? 'text-emerald-500 bg-emerald-500/10' : ''
            }`}>
              Analyze
            </button>
          </Link>
          <Link to="/compare" data-testid="nav-compare-link">
            <button className={`text-slate-400 hover:text-white hover:bg-white/5 rounded-none px-4 py-2 transition-colors ${
              isActive('/compare') ? 'text-emerald-500 bg-emerald-500/10' : ''
            }`}>
              Compare
            </button>
          </Link>
          <Link to="/dashboard" data-testid="nav-dashboard-link">
            <button className={`text-slate-400 hover:text-white hover:bg-white/5 rounded-none px-4 py-2 transition-colors ${
              isActive('/dashboard') ? 'text-emerald-500 bg-emerald-500/10' : ''
            }`}>
              Dashboard
            </button>
          </Link>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <a 
          href="https://github.com" 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-slate-400 hover:text-emerald-500 transition-colors"
          data-testid="github-link"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
          </svg>
        </a>
      </div>
    </nav>
  );
}

export default Navigation;
