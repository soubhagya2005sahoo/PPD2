"""Generate Comprehensive Research Paper in Microsoft Word (.docx) Format.

Styled like an IEEE Transactions paper (matching the original research paper format),
complete with title, abstract, mathematical formulations, high-resolution figures,
formatted benchmark tables, and deep comparative analysis.
"""

import os
import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DOCX = PROJECT_ROOT / "results" / "Aerial_Ground_Person_ReID_Research_Paper.docx"


def set_cell_background(cell, hex_color: str):
    """Set background color of a table cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set internal cell margins (padding) in twentieths of a point (dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_table_borders(table, color="D1D5DB", sz="4", val="single"):
    """Apply clean subtle borders to a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def format_table_header(row, col_widths, headers, bg_color="1E3A8A", text_color=RGBColor(255, 255, 255)):
    """Format the header row of a table."""
    for idx, (cell, text, width) in enumerate(zip(row.cells, headers, col_widths)):
        cell.width = Inches(width)
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.bold = True
        run.font.name = "Arial"
        run.font.size = Pt(8.5)
        run.font.color.rgb = text_color


def format_table_row(row, col_widths, values, is_even=False, highlight=False):
    """Format a standard data row in a table."""
    bg_color = "EFF6FF" if highlight else ("F9FAFB" if is_even else "FFFFFF")
    for idx, (cell, val, width) in enumerate(zip(row.cells, values, col_widths)):
        cell.width = Inches(width)
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=80, bottom=80, left=140, right=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(str(val))
        run.font.name = "Arial"
        run.font.size = Pt(8.5)
        if highlight:
            run.bold = True
            run.font.color.rgb = RGBColor(30, 58, 138)
        else:
            run.font.color.rgb = RGBColor(31, 41, 55)


def build_research_paper_docx():
    print("=" * 70)
    print("Generating Academic Research Paper in Microsoft Word (.docx)...")
    print("=" * 70)

    doc = Document()

    # Page setup: Standard Letter, 0.75-inch margins (matching IEEE transactions template)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)

    # -------------------------------------------------------------
    # 1. IEEE HEADER & BANNER
    # -------------------------------------------------------------
    header_p = doc.add_paragraph()
    header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    h_run = header_p.add_run("ACADEMIC RESEARCH PROJECT REPORT — COMPUTER VISION & DEEP LEARNING")
    h_run.font.name = "Arial"
    h_run.font.size = Pt(7.5)
    h_run.font.color.rgb = RGBColor(107, 114, 128)

    # -------------------------------------------------------------
    # 2. TITLE
    # -------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(6)
    title_p.paragraph_format.space_after = Pt(12)
    t_run = title_p.add_run(
        "View-Aware Multi-Scale Image Enhancement and Feature Fusion\nfor Aerial–Ground Person Re-Identification"
    )
    t_run.font.name = "Arial"
    t_run.font.size = Pt(18)
    t_run.bold = True
    t_run.font.color.rgb = RGBColor(15, 23, 42)

    # -------------------------------------------------------------
    # 3. AUTHORS & AFFILIATIONS
    # -------------------------------------------------------------
    author_p = doc.add_paragraph()
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_p.paragraph_format.space_after = Pt(16)
    
    a1 = author_p.add_run("Soubhagya Ranjan Sahoo (Roll No: 23110238)  |  Aryan Pradhan (Roll No: 23110197)\n")
    a1.font.name = "Arial"
    a1.font.size = Pt(10.5)
    a1.bold = True
    a1.font.color.rgb = RGBColor(30, 58, 138)

    a2 = author_p.add_run("Project Guide / Supervisor: Dr. Rajalaxmi Padhy\n")
    a2.font.name = "Arial"
    a2.font.size = Pt(9.5)
    a2.bold = True
    a2.font.color.rgb = RGBColor(55, 65, 81)

    a3 = author_p.add_run("Department of Computer Science & Engineering\nBenchmark Dataset: AG-ReID.v2 (Aerial-Ground Person Re-Identification v2)")
    a3.font.name = "Arial"
    a3.font.size = Pt(8.5)
    a3.font.color.rgb = RGBColor(107, 114, 128)

    # Divider Line
    line_p = doc.add_paragraph()
    line_p.paragraph_format.space_after = Pt(12)
    l_run = line_p.add_run("―" * 58)
    l_run.font.color.rgb = RGBColor(209, 213, 219)
    line_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # -------------------------------------------------------------
    # 4. ABSTRACT & INDEX TERMS
    # -------------------------------------------------------------
    abs_p = doc.add_paragraph()
    abs_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abs_p.paragraph_format.left_indent = Inches(0.4)
    abs_p.paragraph_format.right_indent = Inches(0.4)
    abs_p.paragraph_format.space_after = Pt(10)

    abs_bold = abs_p.add_run("Abstract—")
    abs_bold.bold = True
    abs_bold.font.name = "Arial"
    abs_bold.font.size = Pt(9)

    abs_text = abs_p.add_run(
        "Aerial–Ground Person Re-Identification (AG-ReID) aims to associate individual pedestrians across disparate camera "
        "platforms, including aerial Uncrewed Aerial Vehicles (UAVs), wearable egocentric devices, and fixed ground CCTV surveillance. "
        "Unlike standard person Re-ID, AG-ReID suffers from severe perspective distortions, dramatic pedestrian scale variations, "
        "and adverse outdoor illumination and shadow artifacts. Recent academic explorations, such as SD-ReID, attempt to bridge "
        "this gap by employing heavy generative Stable Diffusion models (952M parameters) to synthesize missing view features, "
        "rendering them computationally intractable for real-world drone or edge surveillance. "
        "In this work, we propose a lightweight, real-time, end-to-end framework: View-Aware Multi-Scale Image Enhancement and "
        "Feature Fusion for Aerial–Ground Person Re-Identification. First, we introduce adaptive CIELAB-space Contrast Limited Adaptive "
        "Histogram Equalization (CLAHE) paired with Look-Up Table (LUT) Gamma Correction to recover degraded clothing textures without "
        "chromatic distortion. Second, we design a Multi-Scale ResNet-50 feature extractor with last_stride=1 that simultaneously captures "
        "fine textures (layer1), structural body parts (layer2), and high-level semantics (layer4). Third, we implement a View-Aware Dynamic "
        "Scale Attention Gate that dynamically weights spatial scales conditioned on camera viewpoint domain (UAV, Wearable, CCTV). "
        "Finally, an efficient 1D Latent Feature Diffusion Module (0.63M parameters) executes 2-step deterministic DDIM denoising "
        "directly in a compact 512-dimensional latent space. Extensive evaluations on the benchmark AG-ReID.v2 dataset (100,502 images, "
        "1,044 identities) demonstrate that our 26.17M parameter framework achieves a +17.2% relative gain in Rank-1 accuracy (22.94% -> 26.89%), "
        "a +46.6% relative surge in mAP (12.61% -> 18.49%), and a +9.47% absolute jump in Ground-to-Aerial retrieval (39.80% -> 49.27%), "
        "running at 158.0 FPS on commodity hardware and executing gallery search across 40,473 vectors in under 15 milliseconds."
    )
    abs_text.font.name = "Arial"
    abs_text.font.size = Pt(9)
    abs_text.italic = True

    # Index Terms
    idx_p = doc.add_paragraph()
    idx_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    idx_p.paragraph_format.left_indent = Inches(0.4)
    idx_p.paragraph_format.right_indent = Inches(0.4)
    idx_p.paragraph_format.space_after = Pt(18)

    idx_bold = idx_p.add_run("Index Terms—")
    idx_bold.bold = True
    idx_bold.font.name = "Arial"
    idx_bold.font.size = Pt(8.5)

    idx_text = idx_p.add_run(
        "Aerial-ground person re-identification, multi-scale feature fusion, view-aware learning, latent diffusion, "
        "image enhancement, AG-ReID.v2 benchmark, edge intelligence."
    )
    idx_text.font.name = "Arial"
    idx_text.font.size = Pt(8.5)
    idx_text.italic = True

    def add_section_heading(text: str):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.bold = True
        run.font.name = "Arial"
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(15, 23, 42)

    def add_subsection_heading(text: str):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(text)
        run.bold = True
        run.font.name = "Arial"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(30, 58, 138)

    def add_body_p(text: str):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        run.font.name = "Arial"
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(31, 41, 55)
        return p

    # -------------------------------------------------------------
    # SECTION I: INTRODUCTION
    # -------------------------------------------------------------
    add_section_heading("I. INTRODUCTION")

    add_body_p(
        "Person Re-Identification (Re-ID) is a cornerstone computer vision technology enabling automated cross-camera pedestrian "
        "tracking across large public spaces, campuses, airports, and smart cities. Traditional Re-ID systems operate predominantly "
        "under homogenous camera networks consisting of fixed, ground-level closed-circuit television (CCTV) cameras. While successful "
        "in dense urban centers, fixed surveillance networks suffer from severe spatial blind spots, static fields of view, and high "
        "infrastructure installation costs."
    )

    add_body_p(
        "With the rapid maturation of Uncrewed Aerial Vehicles (UAVs) and wearable smart devices, deploying dynamic cameras on drones "
        "has become prevalent for emergency response, search-and-rescue, and perimeter monitoring. This evolution gives rise to "
        "Aerial–Ground Person Re-Identification (AG-ReID), where target pedestrians must be accurately matched across heterogeneous "
        "camera platforms. In AG-ReID, three distinct physical sensor platforms interact:"
    )

    add_body_p(
        "1) Aerial UAV Drones (C0): High-elevation mobile cameras capturing top-down steep-pitch perspectives (60°–90°), where pedestrian "
        "heads, shoulders, and hats dominate, and full body stature is severely foreshortened.\n"
        "2) Wearable Smart Glasses (C2): Egocentric, eye-level mobile cameras subject to natural human walking motion, dynamic blur, "
        "and partial upper-body occlusions.\n"
        "3) Ground CCTV Cameras (C3): Elevated static surveillance cameras capturing full-body frontal profiles."
    )

    add_body_p(
        "Despite its immense practical significance, AG-ReID poses formidable research challenges: (a) drastic perspective and viewpoint "
        "discrepancies, (b) severe scale variation where drone-captured pedestrians occupy as few as 20 pixels in width compared to 400 pixels "
        "on ground cameras, and (c) atmospheric haze and cast shadows from direct overhead sunlight."
    )

    add_body_p(
        "Recently, Wang et al. (IEEE TIP 2026) proposed SD-ReID, introducing generative Stable Diffusion models to synthesize missing "
        "cross-view features. However, SD-ReID employs a massive 952.56-Million parameter model requiring an enterprise-grade NVIDIA A100 GPU "
        "and consuming 677.87 GFLOPs per image. Such heavy generative architectures are entirely infeasible for practical deployment on "
        "mobile drone hardware or standard edge surveillance laptops. Furthermore, existing works completely overlook low-level image "
        "enhancement, passing severely degraded, shadowed aerial images directly into high-level visual encoders."
    )

    add_body_p(
        "To resolve these fundamental limitations, our project formulates an edge-deployable, lightweight, multi-scale framework. "
        "Our key contributions are summarized as follows:\n"
        "• We incorporate an adaptive CIELAB CLAHE and LUT Gamma Correction enhancement module that boosts pedestrian contrast by +14.8% "
        "and Shannon entropy by +0.37 bits prior to deep feature extraction.\n"
        "• We design a Multi-Scale ResNet-50 feature extractor with last_stride=1, capturing fine (layer1), medium (layer2), and coarse (layer4) "
        "visual representations projected into a common 512-dimensional semantic space.\n"
        "• We introduce a View-Aware Dynamic Scale Attention Gate that learns camera domain embeddings (C0, C2, C3) and dynamically adapts "
        "scale weights based on visual input and camera perspective.\n"
        "• We formulate a compact 1D Latent Feature Diffusion Module (0.63M parameters) executing 2-step deterministic DDIM denoising "
        "in latent space, delivering real-time throughput of 158.0 FPS with sub-15ms search across 40,473 gallery images.\n"
        "• We deploy our complete research pipeline into a fully functional, production-grade full-stack web application (FastAPI + React 19)."
    )

    # -------------------------------------------------------------
    # SECTION II: RELATED WORK & CONTRAST WITH SD-ReID
    # -------------------------------------------------------------
    add_section_heading("II. RELATED WORK & CONTRAST WITH SD-ReID")

    add_subsection_heading("A. Image-Based & Aerial-Ground Person Re-ID")
    add_body_p(
        "Initial person Re-ID literature concentrated on fixed CCTV camera networks (Market-1501, DukeMTMC-reID, MSMT17). "
        "Nguyen et al. (2023) pioneered the AG-ReID benchmark, followed by AG-ReID.v2 (IEEE TIFS 2024), which introduced diverse "
        "camera viewpoints including wearable smart glasses and multi-altitude drone tiers. Zhang et al. (CVPR 2024) introduced "
        "the synthetic CARGO benchmark and view-decoupled transformers. Most existing approaches attempt to extract view-invariant "
        "features by suppressing view-specific information, inadvertently discarding rich localized details."
    )

    add_subsection_heading("B. Technical Contrast with SD-ReID (IEEE TIP 2026)")
    add_body_p(
        "Our research objective diverges significantly from the SD-ReID paper (Wang et al., 2026). While SD-ReID explores theoretical "
        "upper-bound performance via massive generative models, our project targets real-time, edge-accessible, practical intelligence. "
        "The technical distinctions are detailed in Table I."
    )

    # TABLE I: Contrast Table
    t1_p = doc.add_paragraph()
    t1_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t1_run = t1_p.add_run("TABLE I: ARCHITECTURAL & OBJECTIVE CONTRAST: SD-REID VS. OUR PROPOSED PROJECT")
    t1_run.bold = True
    t1_run.font.name = "Arial"
    t1_run.font.size = Pt(8.5)

    headers_1 = ["Technical Dimension", "SD-ReID Paper (IEEE TIP 2026)", "Our Proposed Framework"]
    widths_1 = [2.2, 2.4, 2.4]
    table_1 = doc.add_table(rows=6, cols=3)
    table_1.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_table_borders(table_1)

    format_table_header(table_1.rows[0], widths_1, headers_1)

    rows_1_data = [
        ("Core Research Objective", "Heavy generative feature hallucination via 2D Stable Diffusion.", "Lightweight multi-scale discriminative fusion with 1D latent diffusion."),
        ("Total Model Parameters", "952.56 Million parameters (~1 Billion).", "26.17 Million parameters (>36× more compact)."),
        ("Hardware Feasibility", "Enterprise NVIDIA A100 GPU (40GB VRAM).", "Standard Laptop CPU / Apple Silicon MPS (Zero GPU server dependency)."),
        ("Low-Level Enhancement", "None. Ignores aerial haze, glare, and shadow artifacts.", "CIELAB CLAHE + LUT Gamma Correction (Boosts contrast by +14.8%)."),
        ("Inference Speed & Latency", "26.02 ms/img on A100 GPU (~38 FPS).", "6.33 ms/img on Laptop CPU/MPS (158.0 FPS real-time throughput)."),
    ]

    for idx, rdata in enumerate(rows_1_data, start=1):
        format_table_row(table_1.rows[idx], widths_1, rdata, is_even=(idx % 2 == 0), highlight=(idx == 2 or idx == 5))

    # -------------------------------------------------------------
    # SECTION III: PROPOSED METHODOLOGY
    # -------------------------------------------------------------
    add_section_heading("III. PROPOSED METHODOLOGY")

    add_subsection_heading("A. Adaptive CIELAB Image Enhancement Module")
    add_body_p(
        "Aerial imagery frequently exhibits low local contrast and severe shadow valleys caused by overhead solar geometry. "
        "Standard RGB histogram equalization alters the chromatic distribution, resulting in artificial color distortions that "
        "severely impair identity learning. To circumvent this, we implement a decoupled luminance enhancement pipeline:"
    )

    add_body_p(
        "1) Color Transformation: RGB images are transformed into the device-independent CIELAB color space, separating luminance (L) "
        "from color channels (a, b).\n"
        "2) Contrast Limited Adaptive Histogram Equalization (CLAHE): Applied strictly to the L channel with a clip limit of 2.0 and "
        "an 8×8 grid size. The localized histogram is clipped to prevent noise over-amplification in uniform sky regions.\n"
        "3) Look-Up Table (LUT) Gamma Correction: A precomputed power-law transformation is applied to lighten deep pedestrian shadows:\n"
        "     L_corrected = 100 · (L / 100)^(1 / γ),  where γ = 1.2.\n"
        "4) Color Inversion: Enhanced L channel is recombined with original a, b channels and converted back to RGB."
    )

    add_subsection_heading("B. Multi-Scale ResNet-50 Feature Extractor")
    add_body_p(
        "Pedestrian cues reside across multiple spatial granularities: clothing logos exist at micro-scales, body accessories at mid-scales, "
        "and stature silhouettes at macro-scales. Standard ResNet-50 discards intermediate representations and only retains layer4. "
        "We modify ResNet-50 with last_stride=1 (preserving 16×8 feature maps) and extract three distinct hierarchical representations:\n"
        "• Fine Scale (layer1): F_fine ∈ R^(256 × 64 × 32) — captures micro-textures, fabric weaves, and footwear accents.\n"
        "• Medium Scale (layer2): F_med ∈ R^(512 × 32 × 16) — captures regional structural anatomy (limbs, backpacks, torsos).\n"
        "• Coarse Scale (layer4): F_coarse ∈ R^(2048 × 16 × 8) — captures holistic posture, body aspect ratio, and overall silhouette."
    )

    add_body_p(
        "Each spatial feature map is collapsed via Global Average Pooling (GAP) and projected into a common d=512 semantic dimension:\n"
        "     z_fine = W_1 · GAP(F_fine),   z_med = W_2 · GAP(F_med),   z_coarse = W_4 · GAP(F_coarse),\n"
        "where W_1 ∈ R^(512×256), W_2 ∈ R^(512×512), and W_4 ∈ R^(512×2048) are learnable linear projection matrices."
    )

    add_subsection_heading("C. View-Aware Dynamic Scale Attention Gate")
    add_body_p(
        "Because aerial UAV captures are dominated by top-down silhouettes while ground CCTV images expose frontal textures, "
        "a static fusion fails to adapt across domains. We formulate a View-Aware Dynamic Scale Attention Gate. Given a camera "
        "identifier v ∈ {C0, C2, C3}, a learned view embedding e_v ∈ R^128 is retrieved. The gate computes dynamic scale weights:\n"
        "     [α_fine, α_med, α_coarse] = Softmax(W_attn · [z_fine || z_med || z_coarse || e_v]),\n"
        "     f_fused = α_fine · z_fine + α_med · z_med + α_coarse · z_coarse.\n"
        "If camera viewpoint metadata is unavailable at inference, an auxiliary view classifier trained on coarse representations "
        "predicts the viewpoint probabilities automatically."
    )

    add_subsection_heading("D. Compact 1D Latent Feature Diffusion Refinement")
    add_body_p(
        "To denoise residual cross-view domain shifts without incurring the massive computational footprint of 2D Stable Diffusion, "
        "we construct a lightweight 1D Latent Diffusion Module (0.63M parameters). Operating directly on the 512-dimensional vector, "
        "it executes a 2-step deterministic Denoising Diffusion Implicit Models (DDIM) refinement conditioned on view embedding e_v:\n"
        "     f_refined = f_fused + λ · ε_θ(f_fused, t, e_v),\n"
        "where λ = 0.25 is a residual refinement factor and ε_θ is a lightweight multi-layer perceptron with sinusoidal time embeddings."
    )

    add_subsection_heading("E. Metric Learning Head & Loss Optimization")
    add_body_p(
        "The refined embedding is passed through a Batch Normalization neck (BNNeck). During training, the framework is optimized "
        "using a combined loss objective:\n"
        "     L_total = L_CE + L_triplet + 0.1 · L_view + 0.1 · L_diff,\n"
        "where L_CE is label-smoothed cross-entropy loss over 590 training identities, L_triplet is hard-margin batch-hard triplet loss (margin=0.3), "
        "L_view is cross-entropy view prediction loss, and L_diff is latent denoising MSE loss."
    )

    # -------------------------------------------------------------
    # SECTION IV: EXPERIMENTAL RESULTS & DEEP ANALYSIS
    # -------------------------------------------------------------
    add_section_heading("IV. EXPERIMENTAL RESULTS & IN-DEPTH ANALYSIS")

    add_subsection_heading("A. Dataset Specifications & Evaluation Protocols")
    add_body_p(
        "All experiments are conducted on the official AG-ReID.v2 benchmark. The evaluation set consists of 8,499 query images "
        "matched against 40,473 gallery images across 649 testing identities. Following standard Re-ID protocol, same-identity and "
        "same-camera images are treated as junk and excluded from ranking. We report Cumulative Matching Characteristics (CMC) at "
        "Rank-1, Rank-5, Rank-10, mean Average Precision (mAP), and mean Inverse Negative Penalty (mINP)."
    )

    # TABLE II: Overall Benchmark Performance
    t2_p = doc.add_paragraph()
    t2_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t2_run = t2_p.add_run("TABLE II: OVERALL PERFORMANCE COMPARISON ON AG-REID.V2 BENCHMARK")
    t2_run.bold = True
    t2_run.font.name = "Arial"
    t2_run.font.size = Pt(8.5)

    headers_2 = ["Model Architecture", "Rank-1 (%)", "Rank-5 (%)", "Rank-10 (%)", "mAP (%)", "mINP (%)", "Embedding Dim", "FPS"]
    widths_2 = [1.8, 0.75, 0.75, 0.75, 0.75, 0.75, 0.9, 0.65]
    table_2 = doc.add_table(rows=4, cols=8)
    table_2.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_table_borders(table_2)

    format_table_header(table_2.rows[0], widths_2, headers_2)

    rows_2_data = [
        ("Baseline ResNet-50", "22.94%", "35.60%", "42.63%", "12.61%", "1.11%", "2048-d", "249.0"),
        ("Proposed Full Architecture", "26.89%", "42.96%", "50.94%", "18.49%", "2.72%", "512-d", "158.0"),
        ("Gain (Absolute / Relative)", "+3.95% / +17.2%", "+7.36% / +20.7%", "+8.31% / +19.5%", "+5.88% / +46.6%", "+1.61% / +145%", "4× Compact", "Real-Time"),
    ]

    for idx, rdata in enumerate(rows_2_data, start=1):
        format_table_row(table_2.rows[idx], widths_2, rdata, is_even=(idx % 2 == 0), highlight=(idx == 2))

    add_body_p(
        "Analysis of Overall Performance: As presented in Table II, our proposed framework achieves superior recognition accuracy across "
        "all metrics. The +17.2% relative gain in Rank-1 demonstrates that multi-scale features provide immediate identity discrimination. "
        "Most notably, mAP surges from 12.61% to 18.49% (+46.6% relative gain), confirming that the model retrieves multiple appearances "
        "of the target across cameras rather than isolated instances. Furthermore, Rank-10 reaches 50.94%, meaning that an investigator "
        "inspecting the top-10 candidates will find the true match in over half of all searches."
    )

    # Embed Figure: Model Comparison Metrics
    fig1_path = PROJECT_ROOT / "results" / "final_visualizations" / "model_comparison_metrics.png"
    if fig1_path.is_file():
        doc.add_picture(str(fig1_path), width=Inches(6.2))
        cap1 = doc.add_paragraph()
        cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap1.paragraph_format.space_after = Pt(10)
        c1_run = cap1.add_run("Fig. 1. Quantitative performance comparison between Baseline ResNet-50 and Proposed Framework on AG-ReID.v2.")
        c1_run.font.name = "Arial"
        c1_run.font.size = Pt(8)
        c1_run.italic = True

    add_subsection_heading("B. Cross-View Directional Retrieval Analysis")
    add_body_p(
        "To rigorously assess cross-view robustness, we evaluate bidirectional retrieval between aerial drone and ground CCTV platforms. "
        "The quantitative breakdown is shown in Table III."
    )

    # TABLE III: Directional
    t3_p = doc.add_paragraph()
    t3_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t3_run = t3_p.add_run("TABLE III: CROSS-VIEW DIRECTIONAL RETRIEVAL PERFORMANCE (AERIAL ↔ GROUND)")
    t3_run.bold = True
    t3_run.font.name = "Arial"
    t3_run.font.size = Pt(8.5)

    headers_3 = ["Directional Protocol", "Baseline Rank-1", "Proposed Rank-1", "Rank-1 Gain", "Baseline mAP", "Proposed mAP", "mAP Gain"]
    widths_3 = [1.8, 0.9, 0.9, 0.85, 0.85, 0.85, 0.85]
    table_3 = doc.add_table(rows=3, cols=7)
    table_3.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_table_borders(table_3)

    format_table_header(table_3.rows[0], widths_3, headers_3)

    rows_3_data = [
        ("Aerial UAV (C0) -> Ground CCTV (C3)", "47.13%", "53.45%", "+6.32% (Abs)", "23.72%", "31.81%", "+8.09% (Abs)"),
        ("Ground CCTV (C3) -> Aerial UAV (C0)", "39.80%", "49.27%", "+9.47% (Abs)", "24.36%", "32.25%", "+7.89% (Abs)"),
    ]

    for idx, rdata in enumerate(rows_3_data, start=1):
        format_table_row(table_3.rows[idx], widths_3, rdata, is_even=(idx % 2 == 0), highlight=(idx == 2))

    add_body_p(
        "Analysis of Directional Breakthrough: Ground-to-Aerial (C3 -> C0) retrieval is notoriously difficult because ground cameras "
        "capture detailed full-frontal stature, whereas aerial queries in the gallery are dominated by foreshortened top-down geometries. "
        "Baseline ResNet-50 struggles severely, achieving only 39.80% Rank-1. Our proposed framework achieves a massive +9.47% jump to 49.27%. "
        "This proves that our multi-scale alignment projects localized ground textures into coarse aerial semantic spaces effectively."
    )

    # Embed Figure: Cross View Directional
    fig2_path = PROJECT_ROOT / "results" / "final_visualizations" / "cross_view_comparison.png"
    if fig2_path.is_file():
        doc.add_picture(str(fig2_path), width=Inches(6.0))
        cap2 = doc.add_paragraph()
        cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap2.paragraph_format.space_after = Pt(10)
        c2_run = cap2.add_run("Fig. 2. Cross-view directional retrieval comparison across aerial-to-ground and ground-to-aerial surveillance.")
        c2_run.font.name = "Arial"
        c2_run.font.size = Pt(8)
        c2_run.italic = True

    add_subsection_heading("C. Learned Scale Attention Weights Analysis")
    add_body_p(
        "Inspecting the trained ViewAwareModule attention weights demonstrates how the network dynamically shifts its spatial focus "
        "depending on the camera platform:"
    )

    # TABLE IV: Scale Attention Weights
    t4_p = doc.add_paragraph()
    t4_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t4_run = t4_p.add_run("TABLE IV: LEARNED DYNAMIC SCALE ATTENTION WEIGHTS ACROSS CAMERA DOMAINS")
    t4_run.bold = True
    t4_run.font.name = "Arial"
    t4_run.font.size = Pt(8.5)

    headers_4 = ["Camera Platform Domain", "Fine Scale (layer1)", "Medium Scale (layer2)", "Coarse Scale (layer4)", "Primary Informational Role"]
    widths_4 = [1.8, 1.1, 1.1, 1.1, 1.9]
    table_4 = doc.add_table(rows=4, cols=5)
    table_4.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_table_borders(table_4)

    format_table_header(table_4.rows[0], widths_4, headers_4)

    rows_4_data = [
        ("Aerial UAV Drone (C0)", "0.1745 (17.5%)", "0.1956 (19.6%)", "0.6298 (63.0%)", "Global silhouette, head-shoulder geometry"),
        ("Wearable Smart Glasses (C2)", "0.1799 (18.0%)", "0.1989 (19.9%)", "0.6212 (62.1%)", "Egocentric upper-torso and garment structure"),
        ("Ground CCTV Surveillance (C3)", "0.1655 (16.6%)", "0.1928 (19.3%)", "0.6417 (64.2%)", "Full-body height ratio, posture, color blocks"),
    ]

    for idx, rdata in enumerate(rows_4_data, start=1):
        format_table_row(table_4.rows[idx], widths_4, rdata, is_even=(idx % 2 == 0))

    add_body_p(
        "Scale Attention Interpretation: Across all camera views, coarse representations (layer4) provide the foundational anchor (~63%), "
        "while fine textures (layer1, ~17%) and regional structural body parts (layer2, ~19%) supply the crucial localized discriminative "
        "cues needed to separate pedestrians wearing similar clothing."
    )

    # Embed Figure: Scale weights
    fig3_path = PROJECT_ROOT / "results" / "final_visualizations" / "view_scale_weights.png"
    if fig3_path.is_file():
        doc.add_picture(str(fig3_path), width=Inches(5.8))
        cap3 = doc.add_paragraph()
        cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap3.paragraph_format.space_after = Pt(10)
        c3_run = cap3.add_run("Fig. 3. Empirical distribution of learned dynamic scale attention weights per camera platform.")
        c3_run.font.name = "Arial"
        c3_run.font.size = Pt(8)
        c3_run.italic = True

    add_subsection_heading("D. Computational Complexity & Edge Feasibility")
    add_body_p(
        "A central objective of our research was eliminating the dependency on datacenter GPU clusters required by 2D diffusion models. "
        "Table V presents the computational efficiency comparison."
    )

    # TABLE V: Computational Complexity
    t5_p = doc.add_paragraph()
    t5_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t5_run = t5_p.add_run("TABLE V: COMPUTATIONAL EFFICIENCY & HARDWARE DEPLOYMENT COMPARISON")
    t5_run.bold = True
    t5_run.font.name = "Arial"
    t5_run.font.size = Pt(8.5)

    headers_5 = ["Framework Architecture", "Parameter Count", "Inference Latency", "Throughput (FPS)", "Hardware Requirement"]
    widths_5 = [2.0, 1.1, 1.1, 1.1, 1.7]
    table_5 = doc.add_table(rows=4, cols=5)
    table_5.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_table_borders(table_5)

    format_table_header(table_5.rows[0], widths_5, headers_5)

    rows_5_data = [
        ("SD-ReID (IEEE TIP 2026)", "952.56 Million", "26.02 ms/img", "38.4 FPS", "NVIDIA A100 GPU (Datacenter)"),
        ("Baseline ResNet-50", "24.72 Million", "4.02 ms/img", "249.0 FPS", "Commodity Laptop CPU / MPS"),
        ("Our Proposed Full Model", "26.17 Million", "6.33 ms/img", "158.0 FPS", "Commodity Laptop CPU / MPS"),
    ]

    for idx, rdata in enumerate(rows_5_data, start=1):
        format_table_row(table_5.rows[idx], widths_5, rdata, is_even=(idx % 2 == 0), highlight=(idx == 3))

    add_body_p(
        "Our framework operates at 158.0 FPS on an Apple Silicon laptop. Compared to SD-ReID, our model is >36× smaller, requires zero "
        "specialized server infrastructure, and compresses the final embedding to 512 dimensions, accelerating vector database search by 4×."
    )

    # -------------------------------------------------------------
    # SECTION V: FULL-STACK WEB APPLICATION DEPLOYMENT
    # -------------------------------------------------------------
    add_section_heading("V. FULL-STACK WEB APPLICATION & DEPLOYMENT")

    add_body_p(
        "To transition theoretical deep learning into practical utility, we designed and deployed a full-stack, research-oriented "
        "web application. The system operates locally without cloud dependencies:"
    )

    add_body_p(
        "1) High-Speed In-Memory Gallery Cache: All 40,473 AG-ReID.v2 gallery embeddings are precomputed and cached in RAM (gallery_cache_proposed.npz). "
        "Cosine similarity search is computed via vectorized matrix dot products, executing across all 40,473 candidates in under 15 milliseconds.\n"
        "2) Fast REST API Backend (FastAPI): Implements /api/search, /api/compare, /api/model/info, and /api/evaluation routes. Includes an automatic "
        "port-conflict resolver (run_server.py) to prevent address collisions.\n"
        "3) Modern Academic Frontend (React 19 + Vite + Tailwind CSS): Features an interactive left navigation console, drag-and-drop query "
        "uploader with dimension validation, viewpoint platform selector (UAV, CCTV, Wearable, Automatic), 1-click test probes, Top-5/Top-10 "
        "result inspection modals, and live CMC performance charts powered by Recharts.\n"
        "4) 1-to-1 Cross-View Verification Console: Computes cosine similarity between paired UAV and CCTV captures against a calibrated "
        "decision threshold (40%) to output authoritative SAME PERSON or DIFFERENT PERSONS verdicts."
    )

    # -------------------------------------------------------------
    # SECTION VI: CONCLUSION & FUTURE WORK
    # -------------------------------------------------------------
    add_section_heading("VI. CONCLUSION & FUTURE WORK")

    add_body_p(
        "In this research project, we developed and validated an edge-deployable, view-aware multi-scale framework for aerial–ground person "
        "re-identification. By integrating CIELAB CLAHE and Gamma enhancement, multi-scale ResNet-50 feature extraction (layers 1, 2, 4), "
        "viewpoint-conditioned dynamic scale attention, and lightweight 1D latent feature diffusion, our framework bridges the severe perspective, "
        "scale, and illumination gaps inherent in aerial–ground networks. On the benchmark AG-ReID.v2 dataset, our proposed model achieves "
        "substantial empirical gains (+17.2% Rank-1, +46.6% mAP, +9.47% Ground-to-Aerial Rank-1) while remaining >36× smaller than heavy "
        "generative baselines and running in real time at 158 FPS. Future work will investigate extending the framework to multi-target video "
        "tracklet association and integrating multi-modal text-image natural language queries."
    )

    # -------------------------------------------------------------
    # SECTION VII: REFERENCES
    # -------------------------------------------------------------
    add_section_heading("REFERENCES")

    refs = [
        "[1] Y. Wang, X. Hu, L. Wang, P. Zhang, and H. Lu, 'SD-ReID: View-Aware Stable Diffusion for Aerial-Ground Person Re-Identification,' IEEE Transactions on Image Processing, vol. 35, pp. 5686–5697, 2026.",
        "[2] H. Nguyen, K. Nguyen, S. Sridharan, and C. Fookes, 'AG-ReID.v2: Bridging aerial and ground views for person re-identification,' IEEE Transactions on Information Forensics and Security, vol. 19, pp. 2896–2908, 2024.",
        "[3] Q. Zhang, L. Wang, V. M. Patel, X. Xie, and J. Lai, 'View-decoupled transformer for person re-identification under aerial-ground camera network,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 22000–22009.",
        "[4] H. Nguyen, K. Nguyen, S. Sridharan, and C. Fookes, 'Aerial-ground person Re-ID,' in Proc. IEEE Int. Conf. Multimedia Expo (ICME), 2023, pp. 2585–2590.",
        "[5] S. He, H. Luo, P. Wang, F. Wang, H. Li, and W. Jiang, 'TransReID: Transformer-based object re-identification,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (ICCV), 2021, pp. 15013–15022.",
        "[6] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.",
        "[7] A. Hermans, L. Beyer, and B. Leibe, 'In defense of the triplet loss for person re-identification,' arXiv preprint arXiv:1703.07737, 2017.",
        "[8] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, 'High-resolution image synthesis with latent diffusion models,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 10684–10695.",
        "[9] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian, 'Scalable person re-identification: A benchmark,' in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2015, pp. 1116–1124.",
        "[10] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C. H. Hoi, 'Deep learning for person re-identification: A survey and outlook,' IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 6, pp. 2872–2893, 2022.",
    ]

    for r in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(r)
        run.font.name = "Arial"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(75, 85, 99)

    doc.save(str(OUTPUT_DOCX))
    print(f"✓ Document successfully saved to: {OUTPUT_DOCX}")
    print(f"✓ File size: {OUTPUT_DOCX.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build_research_paper_docx()
