import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Home,
  Search,
  Grid,
  Cpu,
  BarChart3,
  Info,
  ShieldCheck,
  Database,
  Layers,
} from 'lucide-react';

const navigation = [
  { name: 'Home', href: '/', icon: Home },
  { name: 'Image Search', href: '/search', icon: Search, primary: true },
  { name: 'Retrieval Results', href: '/results', icon: Grid },
  { name: 'Model Information', href: '/model', icon: Cpu },
  { name: 'Evaluation', href: '/evaluation', icon: BarChart3 },
  { name: 'About Project', href: '/about', icon: Info },
];

export default function Sidebar({ status }) {
  return (
    <aside className="w-64 bg-slate-900 text-slate-200 flex flex-col flex-shrink-0 border-r border-slate-800 shadow-xl min-h-screen">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800 flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-lg shadow-blue-500/20">
          <Layers className="w-6 h-6" />
        </div>
        <div>
          <h1 className="text-base font-bold text-white leading-tight">Aerial–Ground</h1>
          <p className="text-xs text-blue-400 font-medium">Person Re-ID System</p>
        </div>
      </div>

      {/* Main Navigation */}
      <div className="px-3 py-6 flex-1 space-y-1">
        <p className="px-3 text-[11px] font-semibold tracking-wider text-slate-400 uppercase mb-2">
          Research Console
        </p>
        {navigation.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.name}
              to={item.href}
              end={item.href === '/'}
              className={({ isActive }) =>
                `flex items-center px-3 py-2.5 rounded-lg text-sm font-medium transition-all group ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                    : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                } ${item.primary && !isActive ? 'border border-blue-500/30 text-blue-300 hover:bg-blue-950/40' : ''}`
              }
            >
              {({ isActive }) => (
                <>
                  <Icon
                    className={`w-5 h-5 mr-3 transition-colors ${
                      isActive ? 'text-white' : item.primary ? 'text-blue-400' : 'text-slate-400 group-hover:text-slate-200'
                    }`}
                  />
                  <span>{item.name}</span>
                  {item.primary && (
                    <span className="ml-auto text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-400/30">
                      Primary
                    </span>
                  )}
                </>
              )}
            </NavLink>
          );
        })}
      </div>

      {/* System Status Panel */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/50">
        <div className="space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center">
              <ShieldCheck className="w-3.5 h-3.5 mr-1 text-emerald-400" />
              AI Checkpoint
            </span>
            <span className="font-semibold text-emerald-400">26.17M Ready</span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center">
              <Database className="w-3.5 h-3.5 mr-1 text-blue-400" />
              Gallery
            </span>
            <span className="font-mono text-slate-200">
              {status?.gallery_count ? status.gallery_count.toLocaleString() : '40,473'} imgs
            </span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-400">Compute Device</span>
            <span className="uppercase font-mono text-[11px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
              {status?.device || 'Active'}
            </span>
          </div>
        </div>

        <div className="mt-3 pt-3 border-t border-slate-800/80 text-[11px] text-slate-500 text-center">
          AG-ReID.v2 Research Platform
        </div>
      </div>
    </aside>
  );
}

