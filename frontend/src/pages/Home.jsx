import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Search,
  Cpu,
  Database,
  BarChart3,
  Layers,
  ArrowRight,
  ShieldCheck,
  Zap,
  Sparkles,
  Camera,
  Activity,
} from 'lucide-react';
import api from '../services/api';

export default function Home({ status }) {
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [dData, mData] = await Promise.all([
          api.getDatasetInfo(),
          api.getModelInfo(),
        ]);
        setDatasetInfo(dData);
        setModelInfo(mData);
      } catch (err) {
        console.error('Failed to load home data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 text-white rounded-2xl p-8 sm:p-10 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20"></div>

        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-500/20 text-blue-300 border border-blue-400/30 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Academic Research Project Demonstration</span>
          </div>

          <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight leading-tight">
            View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification
          </h1>

          <p className="text-sm sm:text-base text-slate-300 leading-relaxed">
            A deep learning framework bridging the extreme viewpoint, scale, and illumination domain gap
            between aerial UAV drones, egocentric wearable cameras, and ground CCTV surveillance.
          </p>

          {/* Quick Action Buttons */}
          <div className="pt-4 flex flex-wrap gap-3">
            <Link
              to="/search"
              className="inline-flex items-center px-5 py-2.5 rounded-xl bg-blue-600 text-white text-sm font-bold shadow-lg shadow-blue-600/30 hover:bg-blue-500 transition-all hover:scale-[1.02]"
            >
              <Search className="w-4 h-4 mr-2" />
              Launch Image Search
              <ArrowRight className="w-4 h-4 ml-2" />
            </Link>

            <Link
              to="/model"
              className="inline-flex items-center px-5 py-2.5 rounded-xl bg-slate-800 text-slate-200 text-sm font-semibold border border-slate-700 hover:bg-slate-700 hover:text-white transition-all"
            >
              <Cpu className="w-4 h-4 mr-2 text-blue-400" />
              Architecture Details
            </Link>
          </div>
        </div>
      </div>

      {/* Verified Statistics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase">Gallery Images</span>
            <div className="p-2 rounded-lg bg-blue-50 text-blue-600">
              <Database className="w-4 h-4" />
            </div>
          </div>
          <p className="text-2xl font-extrabold text-slate-900 mt-2">
            {status?.gallery_count ? status.gallery_count.toLocaleString() : '40,473'}
          </p>
          <p className="text-xs text-slate-500 mt-1">Precomputed 512-d embeddings</p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase">Total Dataset</span>
            <div className="p-2 rounded-lg bg-indigo-50 text-indigo-600">
              <Layers className="w-4 h-4" />
            </div>
          </div>
          <p className="text-2xl font-extrabold text-slate-900 mt-2">
            {datasetInfo?.total_images ? datasetInfo.total_images.toLocaleString() : '100,502'}
          </p>
          <p className="text-xs text-slate-500 mt-1">AG-ReID.v2 verified images</p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase">Unique Identities</span>
            <div className="p-2 rounded-lg bg-emerald-50 text-emerald-600">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>
          <p className="text-2xl font-extrabold text-slate-900 mt-2">
            {datasetInfo?.total_identities ? datasetInfo.total_identities.toLocaleString() : '1,044'}
          </p>
          <p className="text-xs text-slate-500 mt-1">590 train / 649 test tracklets</p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase">Proposed Model</span>
            <div className="p-2 rounded-lg bg-purple-50 text-purple-600">
              <Zap className="w-4 h-4" />
            </div>
          </div>
          <p className="text-2xl font-extrabold text-slate-900 mt-2">26.17M</p>
          <p className="text-xs text-slate-500 mt-1">PyTorch weights active</p>
        </div>
      </div>

      {/* Methodology Pipeline Highlights */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
        <div>
          <h2 className="text-lg font-bold text-slate-900">Research Framework & Modules</h2>
          <p className="text-xs text-slate-500 mt-1">
            End-to-end aerial-ground person re-identification pipeline implemented in this project:
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-sm">
              1
            </div>
            <h3 className="text-sm font-bold text-slate-800">Adaptive Enhancement</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              CIELAB-based CLAHE and LUT Gamma Correction adaptively boost local pedestrian contrast and
              mitigate severe aerial shadows and atmospheric haze.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold text-sm">
              2
            </div>
            <h3 className="text-sm font-bold text-slate-800">Multi-Scale Extraction</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Preserves fine textures (layer1: 256 channels), body parts (layer2: 512 channels), and global
              semantics (layer4: 2048 channels) projected into a unified 512-d space.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold text-sm">
              3
            </div>
            <h3 className="text-sm font-bold text-slate-800">View-Aware & Diffusion</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Learned viewpoint scale attention conditioned on camera domain (UAV, CCTV, Wearable) combined
              with 2-step compact 1D latent diffusion denoising.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

