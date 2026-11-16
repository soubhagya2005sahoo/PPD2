import React from 'react';
import { Camera, CheckCircle2, Shuffle, Eye } from 'lucide-react';

export default function ResultCard({ item, onClickPreview }) {
  // Similarity color indicator
  const simVal = item.similarity;
  let badgeColor = 'bg-blue-50 text-blue-700 border-blue-200';
  let simBg = 'bg-blue-600';

  if (simVal >= 0.60) {
    badgeColor = 'bg-emerald-50 text-emerald-800 border-emerald-300';
    simBg = 'bg-emerald-600';
  } else if (simVal >= 0.40) {
    badgeColor = 'bg-blue-50 text-blue-800 border-blue-300';
    simBg = 'bg-blue-600';
  } else {
    badgeColor = 'bg-slate-100 text-slate-700 border-slate-200';
    simBg = 'bg-slate-500';
  }

  // Camera badge color
  const isAerial = item.cam_id === 0;
  const isWearable = item.cam_id === 2;
  const isCCTV = item.cam_id === 3;

  return (
    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs hover:shadow-md transition-all duration-200 flex flex-col group">
      {/* Card Header with Rank & Similarity */}
      <div className="px-3.5 py-2.5 bg-slate-50/80 border-b border-slate-100 flex items-center justify-between">
        <span className="inline-flex items-center text-xs font-bold px-2 py-0.5 rounded-md bg-slate-900 text-white shadow-2xs">
          Rank #{item.rank}
        </span>
        <span className={`inline-flex items-center text-xs font-bold px-2.5 py-0.5 rounded-full border ${badgeColor}`}>
          {item.similarity_pct} Match
        </span>
      </div>

      {/* Image Container with Zoom Preview */}
      <div
        className="relative bg-slate-100 flex items-center justify-center p-2 cursor-pointer overflow-hidden aspect-[1/2] max-h-64 sm:max-h-72"
        onClick={() => onClickPreview && onClickPreview(item)}
      >
        <img
          src={item.image_url}
          alt={`Rank ${item.rank} match PID ${item.pid}`}
          className="w-full h-full object-contain rounded transition-transform duration-300 group-hover:scale-105"
          loading="lazy"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="200" viewBox="0 0 100 200"><rect width="100" height="200" fill="%23e2e8f0"/><text x="50" y="100" font-family="sans-serif" font-size="12" fill="%2394a3b8" text-anchor="middle">No Image</text></svg>';
          }}
        />

        {/* Hover Action Overlay */}
        <div className="absolute inset-0 bg-slate-950/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
          <span className="inline-flex items-center px-3 py-1.5 rounded-full bg-white text-slate-900 text-xs font-semibold shadow-lg">
            <Eye className="w-3.5 h-3.5 mr-1 text-blue-600" />
            Inspect Match
          </span>
        </div>

        {/* Cross-View Tag Badge */}
        {item.is_cross_view && (
          <span className="absolute top-2 left-2 inline-flex items-center text-[10px] font-bold px-2 py-0.5 rounded-md bg-purple-600 text-white shadow-md">
            <Shuffle className="w-3 h-3 mr-1" />
            Cross-View
          </span>
        )}
      </div>

      {/* Details Footer */}
      <div className="p-3.5 space-y-2 flex-1 flex flex-col justify-between">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 font-medium">Identity (PID):</span>
            <span className="font-mono font-bold text-slate-800">Person #{item.pid}</span>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 font-medium">Platform:</span>
            <span className="font-medium text-slate-700 flex items-center truncate max-w-[130px]" title={item.cam_name}>
              <Camera className="w-3 h-3 mr-1 text-slate-400 flex-shrink-0" />
              {item.cam_name}
            </span>
          </div>
        </div>

        {/* Cosine Metric Bar */}
        <div className="pt-2 border-t border-slate-100">
          <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
            <span>Cosine Metric</span>
            <span className="font-mono font-semibold text-slate-700">{item.similarity.toFixed(4)}</span>
          </div>
          <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${simBg}`}
              style={{ width: `${Math.max(0, Math.min(100, item.similarity * 100))}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

