import React from 'react';
import { Link } from 'react-router-dom';
import ResultsGrid from '../components/ResultsGrid';
import { Search, ArrowRight, Grid, AlertCircle } from 'lucide-react';

export default function Results({ lastSearchResult }) {
  if (!lastSearchResult) {
    return (
      <div className="bg-white border border-slate-200 rounded-2xl p-12 text-center max-w-2xl mx-auto shadow-xs space-y-4">
        <div className="w-16 h-16 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center mx-auto">
          <Grid className="w-8 h-8" />
        </div>
        <h2 className="text-xl font-bold text-slate-800">No Active Retrieval Results</h2>
        <p className="text-xs text-slate-500 max-w-md mx-auto">
          You haven't executed a Re-ID search yet. Head over to Image Search to upload a query image
          and match against the 40,473 AG-ReID.v2 gallery database.
        </p>
        <div className="pt-2">
          <Link
            to="/search"
            className="inline-flex items-center px-5 py-2.5 rounded-xl bg-blue-600 text-white text-xs font-bold hover:bg-blue-500 shadow-md shadow-blue-600/20 transition-all"
          >
            <Search className="w-4 h-4 mr-2" />
            Go to Image Search
            <ArrowRight className="w-4 h-4 ml-1.5" />
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <ResultsGrid searchData={lastSearchResult} />
    </div>
  );
}

