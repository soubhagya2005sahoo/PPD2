import React, { useEffect, useState } from 'react';
import api from '../services/api';
import { Cpu, Layers, GitMerge, Sliders, ShieldCheck, Zap } from 'lucide-react';

export default function ModelInfo() {
  const [modelData, setModelData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadModel() {
      try {
        const data = await api.getModelInfo();
        setModelData(data);
      } catch (err) {
        console.error('Failed to load model info:', err);
      } finally {
        setLoading(false);
      }
    }
    loadModel();
  }, []);

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2.5 py-0.5 rounded-md border border-blue-200">
            Model Specifications
          </span>
          <h1 className="text-xl font-bold text-slate-900 mt-2">
            View-Aware Multi-Scale Re-ID Architecture
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Full PyTorch implementation integrating multi-scale hierarchical features, view-conditioned scale attention, and latent diffusion.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <div className="px-3.5 py-2 rounded-xl bg-slate-900 text-white text-xs flex items-center shadow-xs">
            <Cpu className="w-4 h-4 mr-2 text-blue-400" />
            <span>Total Params:</span>
            <span className="font-mono font-bold ml-1.5 text-blue-300">26.17M</span>
          </div>
        </div>
      </div>

      {/* Model Parameter Breakdown Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs">
          <span className="text-xs text-slate-400 uppercase font-semibold">Backbone</span>
          <p className="text-base font-bold text-slate-800 mt-1">ResNet-50 Multi-Scale</p>
          <p className="text-xs text-slate-500 mt-0.5">layer1, layer2, layer4 (stride=1)</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs">
          <span className="text-xs text-slate-400 uppercase font-semibold">Embedding Dim</span>
          <p className="text-base font-bold text-blue-700 mt-1">512 Dimensions</p>
          <p className="text-xs text-slate-500 mt-0.5">L2-Normalized Unit Vector</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs">
          <span className="text-xs text-slate-400 uppercase font-semibold">View-Aware Learning</span>
          <p className="text-base font-bold text-emerald-700 mt-1">Dynamic Scale Attention</p>
          <p className="text-xs text-slate-500 mt-0.5">UAV (C0), CCTV (C3), Glasses (C2)</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs">
          <span className="text-xs text-slate-400 uppercase font-semibold">Diffusion Denoising</span>
          <p className="text-base font-bold text-purple-700 mt-1">2-Step Latent Diffusion</p>
          <p className="text-xs text-slate-500 mt-0.5">1D compact latent space</p>
        </div>
      </div>

      {/* Architectural Components List */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900">Framework Components & Parameter Allocation</h2>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-700 uppercase font-semibold border-b border-slate-200">
              <tr>
                <th className="px-4 py-3">Component / Module</th>
                <th className="px-4 py-3">Technical Description & Functionality</th>
                <th className="px-4 py-3 text-right">Parameter Count</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-normal">
              {modelData?.components ? (
                modelData.components.map((comp, idx) => (
                  <tr key={idx} className="hover:bg-slate-50/50">
                    <td className="px-4 py-3 font-bold text-slate-800 whitespace-nowrap">{comp.name}</td>
                    <td className="px-4 py-3 text-slate-600 leading-relaxed">{comp.detail}</td>
                    <td className="px-4 py-3 text-right font-mono font-semibold text-blue-700 whitespace-nowrap">
                      {comp.params}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={3} className="px-4 py-4 text-center text-slate-400">Loading components...</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Learned View-Conditioned Scale Attention Weights */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
        <div>
          <h2 className="text-base font-bold text-slate-900">Learned Dynamic Scale Attention Weights</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Empirical attention weights (Fine, Medium, Coarse) dynamically allocated per viewpoint:
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800 text-xs">Aerial UAV Drone (C0)</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-blue-100 text-blue-800 font-semibold">Camera 0</span>
            </div>
            <div className="space-y-1.5 text-xs pt-1">
              <div className="flex justify-between">
                <span className="text-slate-500">Fine (layer1):</span>
                <span className="font-mono font-bold text-slate-800">17.45%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Medium (layer2):</span>
                <span className="font-mono font-bold text-slate-800">19.56%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Coarse (layer4):</span>
                <span className="font-mono font-bold text-blue-700">62.98%</span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800 text-xs">Wearable Glasses (C2)</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-purple-100 text-purple-800 font-semibold">Camera 2</span>
            </div>
            <div className="space-y-1.5 text-xs pt-1">
              <div className="flex justify-between">
                <span className="text-slate-500">Fine (layer1):</span>
                <span className="font-mono font-bold text-slate-800">17.99%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Medium (layer2):</span>
                <span className="font-mono font-bold text-slate-800">19.89%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Coarse (layer4):</span>
                <span className="font-mono font-bold text-blue-700">62.12%</span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800 text-xs">Ground CCTV Camera (C3)</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-semibold">Camera 3</span>
            </div>
            <div className="space-y-1.5 text-xs pt-1">
              <div className="flex justify-between">
                <span className="text-slate-500">Fine (layer1):</span>
                <span className="font-mono font-bold text-slate-800">16.55%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Medium (layer2):</span>
                <span className="font-mono font-bold text-slate-800">19.28%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Coarse (layer4):</span>
                <span className="font-mono font-bold text-blue-700">64.17%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
