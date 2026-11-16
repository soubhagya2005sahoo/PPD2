import React from 'react';
import { ShieldCheck, Cpu, HardDrive } from 'lucide-react';

export default function Navbar({ status }) {
  return (
    <header className="h-16 bg-white border-b border-slate-200 px-6 flex items-center justify-between shadow-xs sticky top-0 z-30">
      <div className="flex items-center space-x-3">
        <span className="hidden sm:inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
        <h2 className="text-sm sm:text-base font-semibold text-slate-800 truncate">
          View-Aware Multi-Scale Image Enhancement & Feature Fusion
        </h2>
        <span className="hidden md:inline-flex text-xs px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 font-medium border border-blue-200">
          Proposed Architecture
        </span>
      </div>

      <div className="flex items-center space-x-3 text-xs">
        <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 text-slate-700">
          <HardDrive className="w-3.5 h-3.5 text-blue-600" />
          <span className="font-medium">Gallery:</span>
          <span className="font-mono font-semibold">
            {status?.gallery_count ? status.gallery_count.toLocaleString() : '40,473'}
          </span>
        </div>

        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
          <span className="font-medium">Model:</span>
          <span className="font-mono font-semibold">26.17M Active</span>
        </div>

        <div className="hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 text-slate-700">
          <Cpu className="w-3.5 h-3.5 text-slate-500" />
          <span className="uppercase font-mono font-medium">{status?.device || 'MPS/CPU'}</span>
        </div>
      </div>
    </header>
  );
}

