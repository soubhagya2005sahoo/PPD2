import React from 'react';
import { User, Award, BookOpen, Layers, ShieldCheck, Mail, GraduationCap } from 'lucide-react';

export default function About() {
  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Title Card */}
      <div className="bg-white border border-slate-200 rounded-xl p-8 shadow-xs space-y-4">
        <span className="text-xs font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2.5 py-0.5 rounded-md border border-blue-200">
          Academic Research Project
        </span>

        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 leading-tight">
          View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification
        </h1>

        <p className="text-sm text-slate-600 leading-relaxed">
          Submitted in partial fulfillment of the academic requirements for the Bachelor of Technology /
          Master's degree curriculum in Computer Science and Engineering.
        </p>
      </div>

      {/* Authors and Guide Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Author 1 */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-3">
          <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center">
            <User className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[11px] font-bold text-slate-400 uppercase">Student Researcher</span>
            <h3 className="text-base font-bold text-slate-900 mt-0.5">Soubhagya Ranjan Sahoo</h3>
            <p className="text-xs font-mono text-blue-600 font-semibold mt-1">Roll No: 23110238</p>
          </div>
        </div>

        {/* Author 2 */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-3">
          <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center">
            <User className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[11px] font-bold text-slate-400 uppercase">Student Researcher</span>
            <h3 className="text-base font-bold text-slate-900 mt-0.5">Aryan Pradhan</h3>
            <p className="text-xs font-mono text-blue-600 font-semibold mt-1">Roll No: 23110197</p>
          </div>
        </div>

        {/* Project Guide */}
        <div className="bg-white border border-blue-200 rounded-xl p-6 shadow-xs space-y-3 bg-blue-50/20">
          <div className="w-12 h-12 rounded-xl bg-blue-600 text-white flex items-center justify-center shadow-md shadow-blue-500/20">
            <GraduationCap className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[11px] font-bold text-blue-700 uppercase">Project Guide / Supervisor</span>
            <h3 className="text-base font-bold text-slate-900 mt-0.5">Dr. Rajalaxmi Padhy</h3>
            <p className="text-xs text-slate-500 mt-1">Department of Computer Science & Engineering</p>
          </div>
        </div>
      </div>

      {/* Abstract and Research Contributions */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center">
          <BookOpen className="w-4 h-4 mr-2 text-blue-600" />
          Research Abstract & Methodology
        </h2>

        <div className="text-xs text-slate-600 space-y-3 leading-relaxed">
          <p>
            Person re-identification (Re-ID) across heterogeneous aerial and ground sensor platforms
            presents critical computer vision challenges, including severe perspective distortions,
            disparate camera resolutions, atmospheric shadow artifacts, and dramatic scale shifts.
          </p>
          <p>
            To resolve these challenges, this research implements a multi-stage framework:
          </p>
          <ul className="list-disc pl-5 space-y-1.5 text-slate-700">
            <li>
              <b>Adaptive Image Enhancement:</b> Applies CIELAB-space Contrast Limited Adaptive Histogram
              Equalization (CLAHE) and Look-Up Table (LUT) Gamma Correction to restore fine pedestrian textures.
            </li>
            <li>
              <b>Multi-Scale Hierarchical Feature Extraction:</b> Utilizes a modified ResNet-50 backbone
              to simultaneously capture low-level local patterns (layer1), mid-level structural features
              (layer2), and high-level semantics (layer4).
            </li>
            <li>
              <b>View-Aware Conditioning & Scale Attention:</b> Dynamically modulates multi-scale attention
              weights conditioned on the camera viewpoint domain (UAV, CCTV, Wearable).
            </li>
            <li>
              <b>Latent Feature Diffusion Refinement:</b> Applies an efficient 2-step compact 1D latent
              diffusion module to resolve residual cross-view domain discrepancy.
            </li>
          </ul>
        </div>
      </div>

      {/* Benchmark Citation */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-2">
        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Benchmark Citation</h3>
        <p className="text-xs text-slate-600">
          This system is evaluated on the official <b>AG-ReID.v2 (Aerial–Ground Person Re-Identification v2)</b> benchmark
          comprising 100,502 images across 1,044 unique identities captured by aerial UAV drones, egocentric wearable cameras,
          and ground surveillance CCTV systems.
        </p>
      </div>
    </div>
  );
}

