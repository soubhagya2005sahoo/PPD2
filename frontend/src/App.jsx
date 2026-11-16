import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import ImageSearch from './pages/ImageSearch';
import Results from './pages/Results';
import ModelInfo from './pages/ModelInfo';
import Evaluation from './pages/Evaluation';
import About from './pages/About';
import api from './services/api';

export default function App() {
  const [status, setStatus] = useState(null);
  const [lastSearchResult, setLastSearchResult] = useState(null);

  useEffect(() => {
    async function checkHealth() {
      try {
        const data = await api.getHealth();
        setStatus(data);
      } catch (err) {
        console.error('Failed to connect to backend API:', err);
      }
    }
    checkHealth();
  }, []);

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      {/* Fixed Left Navigation Sidebar */}
      <Sidebar status={status} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Navbar status={status} />

        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <Routes>
            <Route path="/" element={<Home status={status} />} />
            <Route
              path="/search"
              element={
                <ImageSearch
                  onSearchComplete={(result) => setLastSearchResult(result)}
                />
              }
            />
            <Route
              path="/results"
              element={<Results lastSearchResult={lastSearchResult} />}
            />
            <Route path="/model" element={<ModelInfo />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/about" element={<About />} />
            {/* Fallback to home */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

