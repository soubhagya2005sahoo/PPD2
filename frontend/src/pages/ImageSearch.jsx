import React, { useState, useEffect } from 'react';
import ImageUploader from '../components/ImageUploader';
import ResultsGrid from '../components/ResultsGrid';
import api from '../services/api';
import {
  Search,
  Camera,
  Layers,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Shuffle,
  ShieldCheck,
  Scale,
  Zap,
} from 'lucide-react';

export default function ImageSearch({ onSearchComplete }) {
  // Mode selection: 1-to-N Gallery Retrieval vs 1-to-1 Pairwise Compare
  const [activeTab, setActiveTab] = useState('gallery'); // 'gallery' | 'compare'

  // Gallery Search State
  const [queryFile, setQueryFile] = useState(null);
  const [queryPreview, setQueryPreview] = useState(null);
  const [queryMeta, setQueryMeta] = useState(null);
  const [selectedView, setSelectedView] = useState('auto');
  const [topK, setTopK] = useState(5);
  const [isSearching, setIsSearching] = useState(false);
  const [searchStep, setSearchStep] = useState(0);
  const [searchResult, setSearchResult] = useState(null);
  const [searchError, setSearchError] = useState(null);

  // Quick Test Samples
  const [samples, setSamples] = useState([]);
  const [loadingSamples, setLoadingSamples] = useState(true);

  // Pairwise Compare State
  const [file1, setFile1] = useState(null);
  const [prev1, setPrev1] = useState(null);
  const [meta1, setMeta1] = useState(null);
  const [cam1, setCam1] = useState('uav');

  const [file2, setFile2] = useState(null);
  const [prev2, setPrev2] = useState(null);
  const [meta2, setMeta2] = useState(null);
  const [cam2, setCam2] = useState('cctv');

  const [compareResult, setCompareResult] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const [compareError, setCompareError] = useState(null);

  useEffect(() => {
    async function loadSamples() {
      try {
        const data = await api.getDatasetSamples();
        setSamples(data);
      } catch (err) {
        console.error('Failed to load dataset samples:', err);
      } finally {
        setLoadingSamples(false);
      }
    }
    loadSamples();
  }, []);

  // Helper to load sample image as a query
  const loadSampleAsQuery = async (sample) => {
    if (!sample.thumbnail_base64) return;
    try {
      const res = await fetch(sample.thumbnail_base64);
      const blob = await res.blob();
      const file = new File([blob], `${sample.id}.jpg`, { type: 'image/jpeg' });

      setQueryFile(file);
      setQueryPreview(sample.thumbnail_base64);
      setQueryMeta({
        width: 128,
        height: 256,
        aspectRatio: '0.50',
        sizeKb: (file.size / 1024).toFixed(1),
        name: `${sample.title} (${sample.pid})`,
      });

      // Auto set view based on camera
      if (sample.camera.includes('C0')) setSelectedView('uav');
      else if (sample.camera.includes('C3')) setSelectedView('cctv');
      else if (sample.camera.includes('C2')) setSelectedView('wearable');
      else setSelectedView('auto');

      // Clear previous result
      setSearchResult(null);
      setSearchError(null);
    } catch (err) {
      console.error('Error loading sample:', err);
    }
  };

  // Run the primary search pipeline
  const handleFindMatchingImages = async () => {
    if (!queryFile) {
      setSearchError('Please upload a person query image first.');
      return;
    }

    setIsSearching(true);
    setSearchError(null);
    setSearchStep(1);

    try {
      // Step simulation for visual research clarity
      setTimeout(() => setSearchStep(2), 250);
      setTimeout(() => setSearchStep(3), 500);

      const result = await api.searchQuery(queryFile, selectedView, topK);
      setSearchStep(4);
      setSearchResult(result);
      if (onSearchComplete) onSearchComplete(result);
    } catch (err) {
      console.error('Search failed:', err);
      const msg = err.response?.data?.detail || err.message || 'Feature extraction and search failed.';
      setSearchError(`Inference Error: ${msg}`);
    } finally {
      setIsSearching(false);
      setSearchStep(0);
    }
  };

  // Run Pairwise Comparison
  const handleCompareImages = async () => {
    if (!file1 || !file2) {
      setCompareError('Please select both person images to compare.');
      return;
    }

    setIsComparing(true);
    setCompareError(null);
    try {
      const res = await api.compareImages(file1, file2, cam1, cam2, 0.40);
      setCompareResult(res);
    } catch (err) {
      console.error('Comparison error:', err);
      const msg = err.response?.data?.detail || err.message || 'Comparison failed.';
      setCompareError(`Verification Error: ${msg}`);
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Page Title & Mode Switcher */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Aerial–Ground Person Re-Identification</h1>
          <p className="text-xs text-slate-500 mt-1">
            Upload a person query image to automatically extract view-aware multi-scale embeddings and retrieve matching gallery identities.
          </p>
        </div>

        {/* Tab Controls */}
        <div className="inline-flex rounded-lg border border-slate-200 p-1 bg-slate-50 text-xs font-semibold self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setActiveTab('gallery')}
            className={`px-3.5 py-1.5 rounded-md transition-all ${
              activeTab === 'gallery'
                ? 'bg-blue-600 text-white shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            1-to-N Gallery Retrieval
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('compare')}
            className={`px-3.5 py-1.5 rounded-md transition-all ${
              activeTab === 'compare'
                ? 'bg-blue-600 text-white shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            1-to-1 Cross-View Verification
          </button>
        </div>
      </div>

      {/* -------------------- TAB 1: GALLERY RETRIEVAL (MAIN) -------------------- */}
      {activeTab === 'gallery' && (
        <div className="space-y-6">
          {/* If result is ready, show ResultsGrid */}
          {searchResult ? (
            <ResultsGrid
              searchData={searchResult}
              onNewSearch={() => {
                setSearchResult(null);
                setSearchError(null);
              }}
            />
          ) : (
            <div className="space-y-6">
              {/* Uploader Card */}
              <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
                <div>
                  <h3 className="text-base font-bold text-slate-900">Step 1: Select Query Person Image</h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Upload an aerial UAV, CCTV, or wearable camera capture.
                  </p>
                </div>

                {/* Drag and Drop Uploader */}
                <ImageUploader
                  imageFile={queryFile}
                  imagePreview={queryPreview}
                  imageMeta={queryMeta}
                  onImageSelected={(file, preview, meta) => {
                    setQueryFile(file);
                    setQueryPreview(preview);
                    setQueryMeta(meta);
                    setSearchError(null);
                  }}
                  onImageRemoved={() => {
                    setQueryFile(null);
                    setQueryPreview(null);
                    setQueryMeta(null);
                  }}
                />

                {/* Viewpoint & Top-K Configuration */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-slate-100">
                  {/* Viewpoint Selector */}
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1.5 flex items-center">
                      <Camera className="w-3.5 h-3.5 mr-1 text-slate-400" />
                      Camera Platform / Viewpoint (Optional)
                    </label>
                    <select
                      value={selectedView}
                      onChange={(e) => setSelectedView(e.target.value)}
                      className="w-full text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-hidden"
                    >
                      <option value="auto">Unknown / Automatic (View-Aware Predictor)</option>
                      <option value="uav">Aerial UAV Drone (Camera C0)</option>
                      <option value="cctv">Ground CCTV Surveillance (Camera C3)</option>
                      <option value="wearable">Wearable Smart Glasses (Camera C2)</option>
                    </select>
                    <p className="text-[11px] text-slate-400 mt-1">
                      Our View-Aware module automatically predicts viewpoint if set to Automatic.
                    </p>
                  </div>

                  {/* Top-K Selector */}
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1.5 flex items-center">
                      <Layers className="w-3.5 h-3.5 mr-1 text-slate-400" />
                      Results to Retrieve
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        onClick={() => setTopK(5)}
                        className={`py-2 px-3 text-xs font-semibold rounded-lg border transition-all ${
                          topK === 5
                            ? 'bg-blue-50 border-blue-500 text-blue-700 shadow-2xs'
                            : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        Top-5 (Default)
                      </button>
                      <button
                        type="button"
                        onClick={() => setTopK(10)}
                        className={`py-2 px-3 text-xs font-semibold rounded-lg border transition-all ${
                          topK === 10
                            ? 'bg-blue-50 border-blue-500 text-blue-700 shadow-2xs'
                            : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        Top-10 Extended
                      </button>
                    </div>
                  </div>
                </div>

                {/* THE PRIMARY MAIN BUTTON: "FIND MATCHING IMAGES" */}
                <div className="pt-2">
                  <button
                    type="button"
                    onClick={handleFindMatchingImages}
                    disabled={!queryFile || isSearching}
                    className={`w-full py-3.5 px-6 rounded-xl text-sm font-bold text-white shadow-lg transition-all flex items-center justify-center space-x-2 ${
                      !queryFile || isSearching
                        ? 'bg-slate-300 text-slate-500 cursor-not-allowed shadow-none'
                        : 'bg-blue-600 hover:bg-blue-500 shadow-blue-600/30 hover:shadow-xl hover:scale-[1.008]'
                    }`}
                  >
                    {isSearching ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Executing Research Pipeline...</span>
                      </>
                    ) : (
                      <>
                        <Search className="w-5 h-5" />
                        <span>FIND MATCHING IMAGES</span>
                      </>
                    )}
                  </button>
                </div>

                {/* Live Step Progress Indicator */}
                {isSearching && (
                  <div className="p-4 rounded-xl bg-blue-50/80 border border-blue-200 space-y-2.5 animate-fadeIn">
                    <div className="flex items-center justify-between text-xs font-semibold text-blue-900">
                      <span className="flex items-center">
                        <Zap className="w-4 h-4 mr-1.5 text-blue-600" />
                        Research Inference Pipeline in Progress:
                      </span>
                      <span className="font-mono">Stage {searchStep} of 4</span>
                    </div>

                    <div className="w-full bg-blue-200/60 h-2 rounded-full overflow-hidden">
                      <div
                        className="bg-blue-600 h-full rounded-full transition-all duration-300"
                        style={{ width: `${(searchStep / 4) * 100}%` }}
                      />
                    </div>

                    <div className="grid grid-cols-4 text-[10px] text-blue-800 text-center font-medium gap-1 pt-1">
                      <span className={searchStep >= 1 ? 'font-bold text-blue-950' : 'text-blue-400'}>
                        1. CLAHE/Gamma
                      </span>
                      <span className={searchStep >= 2 ? 'font-bold text-blue-950' : 'text-blue-400'}>
                        2. Multi-Scale
                      </span>
                      <span className={searchStep >= 3 ? 'font-bold text-blue-950' : 'text-blue-400'}>
                        3. View Diffusion
                      </span>
                      <span className={searchStep >= 4 ? 'font-bold text-blue-950' : 'text-blue-400'}>
                        4. Gallery Match
                      </span>
                    </div>
                  </div>
                )}

                {searchError && (
                  <div className="p-3.5 rounded-lg bg-red-50 border border-red-200 text-red-700 text-xs flex items-center space-x-2">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    <span>{searchError}</span>
                  </div>
                )}
              </div>

              {/* 1-Click Curated Test Probes from AG-ReID.v2 */}
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                      1-Click Research Test Probes (AG-ReID.v2)
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      Click any sample below to instantly load it as query:
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 pt-1">
                  {samples.map((s) => (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => loadSampleAsQuery(s)}
                      className="group p-2 rounded-lg border border-slate-200 bg-slate-50 hover:bg-white hover:border-blue-400 hover:shadow-xs transition-all text-left flex flex-col items-center"
                    >
                      <div className="w-16 h-28 bg-slate-200 rounded overflow-hidden flex items-center justify-center mb-1.5 shadow-inner">
                        {s.thumbnail_base64 ? (
                          <img
                            src={s.thumbnail_base64}
                            alt={s.title}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                          />
                        ) : (
                          <span className="text-[10px] text-slate-400">Loading...</span>
                        )}
                      </div>
                      <span className="text-[11px] font-bold text-slate-800 truncate w-full text-center">
                        Person #{s.pid}
                      </span>
                      <span className="text-[10px] text-blue-600 font-medium truncate w-full text-center">
                        {s.camera.split(' ')[0]}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* -------------------- TAB 2: PAIRWISE CROSS-VIEW VERIFICATION -------------------- */}
      {activeTab === 'compare' && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
          <div>
            <h3 className="text-base font-bold text-slate-900">1-to-1 Cross-View Person Verification</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Upload two person images (e.g. UAV Drone vs. CCTV Camera) to compute cosine similarity and determine identity match.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Image 1 Box */}
            <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-3">
              <span className="text-xs font-bold text-blue-700 uppercase">Image 1 (e.g. Aerial UAV)</span>
              <ImageUploader
                imageFile={file1}
                imagePreview={prev1}
                imageMeta={meta1}
                label="Select Image 1"
                onImageSelected={(f, p, m) => {
                  setFile1(f);
                  setPrev1(p);
                  setMeta1(m);
                }}
                onImageRemoved={() => {
                  setFile1(null);
                  setPrev1(null);
                  setMeta1(null);
                }}
              />
              <select
                value={cam1}
                onChange={(e) => setCam1(e.target.value)}
                className="w-full text-xs bg-white border border-slate-300 rounded-lg p-2"
              >
                <option value="uav">Aerial UAV Drone (C0)</option>
                <option value="cctv">Ground CCTV Surveillance (C3)</option>
                <option value="wearable">Wearable Glasses (C2)</option>
              </select>
            </div>

            {/* Image 2 Box */}
            <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-3">
              <span className="text-xs font-bold text-blue-700 uppercase">Image 2 (e.g. Ground CCTV)</span>
              <ImageUploader
                imageFile={file2}
                imagePreview={prev2}
                imageMeta={meta2}
                label="Select Image 2"
                onImageSelected={(f, p, m) => {
                  setFile2(f);
                  setPrev2(p);
                  setMeta2(m);
                }}
                onImageRemoved={() => {
                  setFile2(null);
                  setPrev2(null);
                  setMeta2(null);
                }}
              />
              <select
                value={cam2}
                onChange={(e) => setCam2(e.target.value)}
                className="w-full text-xs bg-white border border-slate-300 rounded-lg p-2"
              >
                <option value="cctv">Ground CCTV Surveillance (C3)</option>
                <option value="uav">Aerial UAV Drone (C0)</option>
                <option value="wearable">Wearable Glasses (C2)</option>
              </select>
            </div>
          </div>

          <button
            type="button"
            onClick={handleCompareImages}
            disabled={!file1 || !file2 || isComparing}
            className={`w-full py-3 rounded-xl text-sm font-bold text-white shadow-md transition-all flex items-center justify-center space-x-2 ${
              !file1 || !file2 || isComparing
                ? 'bg-slate-300 cursor-not-allowed shadow-none'
                : 'bg-blue-600 hover:bg-blue-500 shadow-blue-600/30'
            }`}
          >
            {isComparing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Comparing Representations...</span>
              </>
            ) : (
              <>
                <Shuffle className="w-4 h-4" />
                <span>Compare & Verify Identity</span>
              </>
            )}
          </button>

          {compareError && (
            <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-xs">
              {compareError}
            </div>
          )}

          {/* Comparison Result Verdict */}
          {compareResult && (
            <div
              className={`p-6 rounded-xl border text-center space-y-3 ${
                compareResult.is_same
                  ? 'bg-emerald-50/60 border-emerald-300 text-emerald-950'
                  : 'bg-red-50/60 border-red-300 text-red-950'
              }`}
            >
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-white shadow-xs mx-auto">
                {compareResult.is_same ? (
                  <CheckCircle2 className="w-7 h-7 text-emerald-600" />
                ) : (
                  <AlertCircle className="w-7 h-7 text-red-600" />
                )}
              </div>

              <h4 className="text-xl font-extrabold tracking-tight">
                {compareResult.verdict}
              </h4>

              <p className="text-sm font-medium">
                Cosine Similarity: <span className="font-mono font-bold text-base">{compareResult.similarity_pct}</span> (Threshold: {(compareResult.threshold * 100).toFixed(0)}%)
              </p>

              <div className="flex justify-center items-center space-x-4 pt-2 text-xs text-slate-600">
                <span className="px-2.5 py-1 rounded bg-white border border-slate-200">
                  Confidence: <b>{compareResult.confidence}</b>
                </span>
                <span className="px-2.5 py-1 rounded bg-white border border-slate-200">
                  Latency: <b>{compareResult.time_ms} ms</b>
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

