import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, X, RefreshCw, AlertCircle } from 'lucide-react';

export default function ImageUploader({
  imageFile,
  imagePreview,
  imageMeta,
  onImageSelected,
  onImageRemoved,
  label = "Upload Person Query Image",
  sublabel = "Drag & drop your query image here, or click to browse files"
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const validateAndProcessFile = (file) => {
    setError(null);
    if (!file) return;

    // Check mime type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
      setError('Please select a valid image file (JPG, JPEG, or PNG).');
      return;
    }

    // Check size limit (max 15MB)
    if (file.size > 15 * 1024 * 1024) {
      setError('Image file is too large (maximum size is 15MB).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      const img = new Image();
      img.onload = () => {
        const sizeKb = file.size / 1024;
        const formattedSize = sizeKb >= 1024
          ? `${(sizeKb / 1024).toFixed(1)} MB`
          : `${sizeKb.toFixed(1)} KB`;

        onImageSelected(file, dataUrl, {
          width: img.width,
          height: img.height,
          aspectRatio: (img.width / img.height).toFixed(2),
          sizeKb: sizeKb.toFixed(1),
          formattedSize: formattedSize,
          name: file.name,
        });
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const openFileDialog = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="w-full">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        className="hidden"
      />

      {!imagePreview ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={openFileDialog}
          className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
            isDragging
              ? 'border-blue-500 bg-blue-50/50 scale-[0.99]'
              : 'border-slate-300 hover:border-blue-400 bg-white hover:bg-slate-50/50'
          }`}
        >
          <div className="flex flex-col items-center justify-center space-y-3">
            <div className="w-14 h-14 rounded-full bg-blue-50 flex items-center justify-center text-blue-600 shadow-xs">
              <UploadCloud className="w-7 h-7" />
            </div>
            <div>
              <p className="text-base font-semibold text-slate-800">{label}</p>
              <p className="text-xs text-slate-500 mt-1 max-w-sm">{sublabel}</p>
            </div>
            <div className="inline-flex items-center px-4 py-2 text-xs font-semibold rounded-lg bg-blue-600 text-white shadow-xs hover:bg-blue-700 transition-colors">
              Browse Files
            </div>
            <p className="text-[11px] text-slate-400">Supported: JPG, JPEG, PNG (Auto-resized to 256×128)</p>
          </div>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl p-3.5 sm:p-4 shadow-xs overflow-hidden">
          <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4 min-w-0">
            {/* Image Preview Box */}
            <div className="relative group w-28 sm:w-32 h-36 sm:h-44 bg-slate-900/5 rounded-lg overflow-hidden flex items-center justify-center border border-slate-200 shadow-inner flex-shrink-0 p-1">
              <img
                src={imagePreview}
                alt="Selected query preview"
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-slate-950/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center space-x-2">
                <button
                  type="button"
                  onClick={openFileDialog}
                  className="p-1.5 bg-white text-slate-800 rounded-full shadow-md hover:bg-slate-100"
                  title="Replace Image"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={onImageRemoved}
                  className="p-1.5 bg-red-600 text-white rounded-full shadow-md hover:bg-red-700"
                  title="Remove Image"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Metadata and Controls */}
            <div className="flex-1 min-w-0 w-full space-y-2.5">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <h4 className="text-xs sm:text-sm font-bold text-slate-800 truncate" title={imageMeta?.name}>
                    {imageMeta?.name || 'Selected Query Image'}
                  </h4>
                  <p className="text-[11px] text-slate-500 mt-0.5">Ready for feature extraction</p>
                </div>
                <button
                  type="button"
                  onClick={onImageRemoved}
                  className="text-slate-400 hover:text-red-600 p-1 rounded-md hover:bg-red-50 transition-colors flex-shrink-0"
                  title="Remove"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Dimension Badges - Responsive 2-column layout that never overflows */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2 rounded-lg bg-slate-50 border border-slate-100 col-span-2">
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Dimensions</span>
                  <span className="font-mono font-semibold text-slate-700 truncate block text-xs">
                    {imageMeta?.width} × {imageMeta?.height} px
                  </span>
                </div>
                <div className="p-2 rounded-lg bg-slate-50 border border-slate-100 col-span-1 min-w-0">
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">File Size</span>
                  <span className="font-mono font-semibold text-slate-700 truncate block text-xs">
                    {imageMeta?.formattedSize || (imageMeta?.sizeKb ? `${imageMeta.sizeKb} KB` : 'N/A')}
                  </span>
                </div>
                <div className="p-2 rounded-lg bg-slate-50 border border-slate-100 col-span-1 min-w-0">
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Aspect Ratio</span>
                  <span className="font-mono font-semibold text-slate-700 truncate block text-xs">
                    {imageMeta?.aspectRatio}:1
                  </span>
                </div>
              </div>

              <div className="pt-0.5">
                <button
                  type="button"
                  onClick={openFileDialog}
                  className="inline-flex items-center text-xs font-semibold px-2.5 py-1 rounded-lg border border-slate-200 text-slate-700 bg-white hover:bg-slate-50 hover:text-slate-900 transition-colors shadow-2xs"
                >
                  <RefreshCw className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
                  Replace Image
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-2.5 p-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
