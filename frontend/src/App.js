import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from '@/components/ui/sonner';
import '@/App.css';

import Home from '@/pages/Home';
import RepositoryAnalysis from '@/pages/RepositoryAnalysis';
import ProjectComparison from '@/pages/ProjectComparison';
import HealthDashboard from '@/pages/HealthDashboard';
import Navigation from '@/components/Navigation';

function App() {
  return (
    <div className="App min-h-screen bg-[#09090B]">
      <BrowserRouter>
        <Navigation />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analyze" element={<RepositoryAnalysis />} />
          <Route path="/compare" element={<ProjectComparison />} />
          <Route path="/dashboard" element={<HealthDashboard />} />
        </Routes>
        <Toaster position="top-right" theme="dark" />
      </BrowserRouter>
    </div>
  );
}

export default App;
