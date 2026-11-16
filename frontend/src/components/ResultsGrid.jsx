import React, { useState } from 'react';
import ResultCard from './ResultCard';
import { Clock, Database, Eye, X, Filter, Shuffle, ArrowRight } from 'lucide-react';

export default function ResultsGrid({ searchData, onNewSearch }) {
  const [topK, setTopK] = useState(5);
  const [crossViewOnly, setCrossViewOnly] = useState(false);
  const [activeModalItem, setActiveModalItem] = useState(null);

  if (!searchData) return null;

  const matches = searchData.matches || [];
  const filteredMatches = crossViewOnly
    ? matches.filter((m) => m.is_cross_view).slice(0, topK)
    : matches.slice(0, topK);

  return (
    <div className="space-y-6">
      {/* Search Header Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="inline-flex items-center space-x-2 text-xs font-semibold text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-200">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>Re-Identification Successful</span>
            </div>
            <h3 className="text-lg font-bold text-slate-900">Retrieval Results</h3>
            <p className="text-xs text-slate-500">
              Query matched against all {searchData.total_gallery_count?.toLocaleString() || '40,473'} gallery identities.
            </p>
          </div>

          {/* Performance Chips */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <div className="px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-700 flex items-center">
              <Clock className="w-3.5 h-3.5 mr-1.5 text-blue-600" />
              <span>Inference Time:</span>
              <span className="font-mono font-bold ml-1 text-slate-900">{searchData.search_time_ms} ms</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-700 flex items-center">
              <Database className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
              <span>Gallery Vectors:</span>
              <span className="font-mono font-bold ml-1 text-slate-900">
                {searchData.total_gallery_count?.toLocaleString() || '40,473'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Query & Retrieved Gallery Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Column: Query Image Panel */}
        <div className="lg:col-span-1">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs sticky top-20 space-y-4">
            <div className="border-b border-slate-100 pb-3">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2.5 py-0.5 rounded-md border border-blue-200">
                Input Query
              </span>
              <h4 className="text-sm font-bold text-slate-800 mt-2">Uploaded Person Image</h4>
            </div>

            {/* Query Preview with Raw vs Enhanced Tabs */}
            <div className="flex justify-center bg-slate-900/5 rounded-lg p-2 border border-slate-200 overflow-hidden">
              <img
                src={searchData.query_enhanced_base64 || searchData.query_raw_base64}
                alt="Query person"
                className="max-h-72 object-contain rounded"
              />
            </div>

            {/* Query Details */}
            <div className="space-y-2 text-xs">
              <div className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-100">
                <span className="text-slate-500">Query Viewpoint:</span>
                <span className="font-semibold text-slate-800 text-right">{searchData.query_view}</span>
              </div>
              <div className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-100">
                <span className="text-slate-500">Feature Embedding:</span>
                <span className="font-mono font-semibold text-blue-700">512-d (L2-norm)</span>
              </div>
              <div className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-100">
                <span className="text-slate-500">Enhancement:</span>
                <span className="font-medium text-slate-700">CLAHE + Gamma (1.2)</span>
              </div>
            </div>

            {onNewSearch && (
              <button
                type="button"
                onClick={onNewSearch}
                className="w-full py-2 px-3 text-xs font-semibold rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors"
              >
                Search Another Image
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Ranked Matches */}
        <div className="lg:col-span-3 space-y-4">
          {/* Controls Bar: Top-5 / Top-10 Toggle & Filter */}
          <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-xs flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-semibold text-slate-600">Display:</span>
              <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-slate-50 text-xs">
                <button
                  type="button"
                  onClick={() => setTopK(5)}
                  className={`px-3 py-1 rounded-md font-medium transition-all ${
                    topK === 5
                      ? 'bg-blue-600 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Top 5
                </button>
                <button
                  type="button"
                  onClick={() => setTopK(10)}
                  className={`px-3 py-1 rounded-md font-medium transition-all ${
                    topK === 10
                      ? 'bg-blue-600 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Top 10
                </button>
              </div>
            </div>

            {/* Cross View Filter */}
            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={() => setCrossViewOnly(!crossViewOnly)}
                className={`inline-flex items-center text-xs font-semibold px-3 py-1.5 rounded-lg border transition-colors ${
                  crossViewOnly
                    ? 'bg-purple-50 text-purple-700 border-purple-300'
                    : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
                }`}
              >
                <Shuffle className="w-3.5 h-3.5 mr-1.5" />
                Cross-View Matches Only
              </button>
            </div>
          </div>

          {/* Results Grid */}
          {filteredMatches.length === 0 ? (
            <div className="bg-white border border-slate-200 rounded-xl p-12 text-center text-slate-500">
              <Filter className="w-8 h-8 mx-auto text-slate-400 mb-2" />
              <p className="font-semibold text-slate-700">No matching results for current filter.</p>
              <p className="text-xs mt-1">Try switching to all matches.</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3.5">
              {filteredMatches.map((item) => (
                <ResultCard
                  key={item.rank}
                  item={item}
                  onClickPreview={(it) => setActiveModalItem(it)}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Modal Inspector for Selected Gallery Match */}
      {activeModalItem && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full p-6 shadow-2xl relative space-y-4">
            <button
              onClick={() => setActiveModalItem(null)}
              className="absolute top-4 right-4 p-1.5 rounded-full text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center space-x-2">
              <span className="text-xs font-bold px-2.5 py-1 rounded-md bg-blue-600 text-white">
                Rank #{activeModalItem.rank}
              </span>
              <h3 className="text-base font-bold text-slate-800">
                Identity Inspection (Person #{activeModalItem.pid})
              </h3>
            </div>

            <div className="grid grid-cols-2 gap-4 items-center bg-slate-50 p-4 rounded-xl border border-slate-200">
              <div className="text-center">
                <span className="text-[11px] font-bold text-slate-500 uppercase block mb-1">
                  Query Probe
                </span>
                <div className="h-64 bg-slate-900/5 rounded-lg flex items-center justify-center border border-slate-200 p-2">
                  <img
                    src={searchData.query_enhanced_base64 || searchData.query_raw_base64}
                    alt="Query"
                    className="max-h-full object-contain"
                  />
                </div>
                <span className="text-xs text-slate-600 mt-2 block">{searchData.query_view}</span>
              </div>

              <div className="text-center">
                <span className="text-[11px] font-bold text-slate-500 uppercase block mb-1">
                  Retrieved Match
                </span>
                <div className="h-64 bg-slate-900/5 rounded-lg flex items-center justify-center border border-slate-200 p-2">
                  <img
                    src={activeModalItem.image_url}
                    alt="Match"
                    className="max-h-full object-contain"
                  />
                </div>
                <span className="text-xs text-slate-600 mt-2 block">{activeModalItem.cam_name}</span>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 text-xs">
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[10px] uppercase">Similarity Metric</span>
                <span className="font-mono text-base font-bold text-blue-700">
                  {activeModalItem.similarity_pct}
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[10px] uppercase">Cosine Value</span>
                <span className="font-mono text-base font-bold text-slate-800">
                  {activeModalItem.similarity.toFixed(4)}
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[10px] uppercase">Domain Relation</span>
                <span className="font-semibold text-purple-700 text-sm">
                  {activeModalItem.is_cross_view ? 'Cross-View Match' : 'Same-View Match'}
                </span>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                type="button"
                onClick={() => setActiveModalItem(null)}
                className="px-4 py-2 bg-slate-800 text-white rounded-lg text-xs font-semibold hover:bg-slate-900 transition-colors"
              >
                Close Inspection
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

