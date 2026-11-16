import React, { useEffect, useState } from 'react';
import api from '../services/api';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
} from 'recharts';
import { BarChart3, TrendingUp, Zap, CheckCircle2, ShieldCheck } from 'lucide-react';

export default function Evaluation() {
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadEval() {
      try {
        const data = await api.getEvaluation();
        setEvalData(data);
      } catch (err) {
        console.error('Failed to load evaluation data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadEval();
  }, []);

  const cmcData = evalData?.cmc_curve || [
    { rank: 'Rank-1', Baseline: 22.94, Proposed: 26.89 },
    { rank: 'Rank-5', Baseline: 35.60, Proposed: 42.96 },
    { rank: 'Rank-10', Baseline: 42.63, Proposed: 50.94 },
  ];

  const directionalData = [
    {
      direction: 'Aerial UAV → CCTV',
      'Baseline Rank-1': 47.13,
      'Proposed Rank-1': 53.45,
      'Baseline mAP': 23.72,
      'Proposed mAP': 31.81,
    },
    {
      direction: 'CCTV → Aerial UAV',
      'Baseline Rank-1': 39.80,
      'Proposed Rank-1': 49.27,
      'Baseline mAP': 24.36,
      'Proposed mAP': 32.25,
    },
  ];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2.5 py-0.5 rounded-md border border-blue-200">
            Empirical Benchmark Results
          </span>
          <h1 className="text-xl font-bold text-slate-900 mt-2">
            AG-ReID.v2 Evaluation & Performance Gains
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Strict Market-1501 evaluation protocol adapted for Aerial-Ground multi-platform surveillance.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <div className="px-3.5 py-2 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center font-bold">
            <TrendingUp className="w-4 h-4 mr-1.5 text-emerald-600" />
            +46.6% mAP Relative Gain
          </div>
        </div>
      </div>

      {/* Top Level Metric Gains */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <span className="text-xs font-semibold text-slate-400 uppercase">Overall Rank-1 Accuracy</span>
          <div className="flex items-baseline space-x-3 mt-2">
            <span className="text-2xl font-extrabold text-blue-700">26.89%</span>
            <span className="text-xs font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
              +3.94% absolute
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">vs Baseline ResNet-50 (22.94%)</p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <span className="text-xs font-semibold text-slate-400 uppercase">Mean Average Precision (mAP)</span>
          <div className="flex items-baseline space-x-3 mt-2">
            <span className="text-2xl font-extrabold text-indigo-700">18.49%</span>
            <span className="text-xs font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
              +5.87% absolute
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">vs Baseline ResNet-50 (12.61%)</p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <span className="text-xs font-semibold text-slate-400 uppercase">Aerial → Ground Rank-1</span>
          <div className="flex items-baseline space-x-3 mt-2">
            <span className="text-2xl font-extrabold text-emerald-700">53.45%</span>
            <span className="text-xs font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
              +6.32% absolute
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">vs Baseline ResNet-50 (47.13%)</p>
        </div>
      </div>

      {/* Visual Charts: CMC Curve & Directional Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* CMC Curve Chart */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
          <h3 className="text-sm font-bold text-slate-800">
            Cumulative Matching Characteristics (CMC)
          </h3>
          <p className="text-xs text-slate-500">
            Recognition accuracy comparison across Rank-1, Rank-5, and Rank-10:
          </p>

          <div className="h-64 pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cmcData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="rank" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} unit="%" />
                <Tooltip
                  formatter={(val, name) => [`${val}%`, name]}
                  contentStyle={{ fontSize: 12, borderRadius: 8, padding: '8px 12px' }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="Baseline" fill="#94a3b8" radius={[4, 4, 0, 0]} name="Baseline ResNet-50" />
                <Bar dataKey="Proposed" fill="#2563eb" radius={[4, 4, 0, 0]} name="Proposed Framework" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Directional Retrieval Chart */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
          <h3 className="text-sm font-bold text-slate-800">
            Cross-View Directional Retrieval Performance
          </h3>
          <p className="text-xs text-slate-500">
            Rank-1 accuracy across aerial-to-ground and ground-to-aerial search directions:
          </p>

          <div className="h-64 pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={directionalData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="direction" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} unit="%" />
                <Tooltip
                  formatter={(val, name) => [`${val}%`, name]}
                  contentStyle={{ fontSize: 12, borderRadius: 8, padding: '8px 12px' }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="Baseline Rank-1" fill="#94a3b8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Proposed Rank-1" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Summary Table */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
        <h3 className="text-base font-bold text-slate-900">Official Experimental Results Summary</h3>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-700 uppercase font-semibold border-b border-slate-200">
              <tr>
                <th className="px-4 py-3">Evaluation Metric</th>
                <th className="px-4 py-3">Baseline ResNet-50</th>
                <th className="px-4 py-3">Proposed Framework</th>
                <th className="px-4 py-3 text-right">Absolute Gain</th>
                <th className="px-4 py-3 text-right">Relative Gain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-normal">
              {evalData?.comparison_summary?.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50/50">
                  <td className="px-4 py-3 font-semibold text-slate-800">{row.metric}</td>
                  <td className="px-4 py-3 font-mono text-slate-600">{row.baseline}</td>
                  <td className="px-4 py-3 font-mono font-bold text-blue-700">{row.proposed}</td>
                  <td className="px-4 py-3 font-mono text-right text-emerald-600 font-semibold">
                    {row.absolute_gain}
                  </td>
                  <td className="px-4 py-3 font-mono text-right text-emerald-700 font-bold">
                    {row.relative_gain}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

