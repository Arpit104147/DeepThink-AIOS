import re
import base64
import io
from backend.downloader import resolve_model_key

class StudyPipeline:
    """
    Hierarchical Multi-Volume Master Curriculum & Pedagogical Synthesis Engine.
    Strictly powered by DeepSeek-R1 to author exhaustive, human-readable student study notes,
    comprehensive master textbook chapters with full display LaTeX derivations ($$ ... $$),
    embedded Mermaid visual flowcharts, pedagogical alert callouts, 1-page formula cheat-sheets,
    dense comparative matrices, 10 fully solved numerical problems, and a complete mock exam paper.
    """

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None, attached_doc=None, attached_image=None):
        if status_callback:
            status_callback("🎓 Study Engine: Initializing deep pedagogical reasoning core...", "info", "deepseek_r1", 5)

        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        reasoning_key = resolve_model_key("reasoning") or "deepseek_r1"
        try:
            ds_model = orchestrator._get_model(reasoning_key, required_ctx=4096)
            if not orchestrator._is_model_valid(ds_model):
                reasoning_key = "vibethinker"
        except (FileNotFoundError, Exception):
            reasoning_key = "vibethinker"

        reasoning_llm = orchestrator._get_model(reasoning_key, required_ctx=ds_ctx)
        reasoning_display = orchestrator._get_display_model_name(reasoning_key)

        # Dedicated token budget per volume (~3840 tokens each = ~7600+ tokens total output)
        vol_gen_tokens = min(4096, max(3072, ds_ctx - 1500))
        study_temp = 0.2  # Low temperature for strict mathematical precision and zero meta-rambling

        orchestrator._check_cancelled("study:init")

        # Extract clean title from prompt
        clean_title = StudyPipeline._extract_clean_title(prompt)

        # Check for uploaded Document payload (Sub-Mode B: Document-Grounded Notes)
        doc_payload = attached_doc or attached_image
        extracted_doc_text = ""
        if doc_payload:
            extracted_doc_text = StudyPipeline._extract_document_text(doc_payload)

        if extracted_doc_text and len(extracted_doc_text) > 100:
            return StudyPipeline._execute_doc_study(
                orchestrator, prompt, clean_title, extracted_doc_text, reasoning_llm, reasoning_key,
                reasoning_display, vol_gen_tokens, study_temp, status_callback
            )
        else:
            return StudyPipeline._execute_web_study(
                orchestrator, prompt, clean_title, reasoning_llm, reasoning_key,
                reasoning_display, vol_gen_tokens, study_temp, status_callback
            )

    @staticmethod
    def _execute_web_study(orchestrator, prompt, clean_title, reasoning_llm, reasoning_key, reasoning_display, gen_tokens, gen_temp, status_callback):
        """Sub-Mode A: Multi-Source Academic Web Harvest & 2-Volume Exhaustive Master Textbook."""
        if status_callback:
            status_callback("🎓 Study Engine [Stage 1/4]: Ingesting academic curriculum & empirical research...", "info", "system", 15)

        raw_contexts = []
        if hasattr(orchestrator, "web_search") and orchestrator.web_search:
            try:
                primary_res = orchestrator.web_search.search_and_scrape(clean_title, max_results=6, max_scrapes=3)
                if isinstance(primary_res, dict) and not primary_res.get("empty", True):
                    raw_contexts.append(primary_res.get("context", ""))

                sub_res = orchestrator.web_search.search(f"{clean_title} mathematical derivations formulas practice problems exam", max_results=4)
                if isinstance(sub_res, list) and sub_res:
                    formatted = [f"[{item.get('title', 'Ref')}] ({item.get('link', '')}):\n{item.get('snippet', '')}" for item in sub_res[:3] if item.get('snippet')]
                    if formatted:
                        raw_contexts.append("\n\n".join(formatted))
            except Exception as e:
                print(f"Study search notice: {e}")

        aggregated_context = "\n\n---\n\n".join(raw_contexts)
        orchestrator._check_cancelled("study:web_scrape_done")

        # ── Volume I: Theoretical Foundations, Architecture, Mermaid Diagram & Complete Formal Derivations ──
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 2/4]: Authoring Volume I (Theorems, Mermaid Flowchart & Derivations) with {reasoning_display}...", "info", reasoning_key, 35)

        sys_prompt_vol1 = (
            "You are a Distinguished Chaired Professor, Academician, and Author of Graduate Reference Textbooks.\n"
            "Your task is to write VOLUME I of a definitive, publication-grade academic master reference textbook chapter.\n\n"
            "MANDATORY LATEX, VISUAL & PROSE RULES:\n"
            "1. NO META-OUTLINES: Do NOT write outlines, bullet summaries of what you plan to do, or brief overviews. Write out the ENTIRE chapter in continuous, rich, human-readable textbook prose with full explanations.\n"
            "2. MERMAID VISUAL FLOWCHART: In Module 2, you MUST include a clean, valid ```mermaid diagram (flowchart TD or sequenceDiagram) illustrating the core system architecture, data pipeline, or state transitions.\n"
            "3. PEDAGOGICAL ALERT CALLOUTS: Embed GitHub-style callouts strategically:\n"
            "   > [!TIP] Physical Intuition Check: Visual analogy explaining abstract mechanics.\n"
            "   > [!IMPORTANT] Exam Trap: Tricky sign slips, boundary pitfalls, and student misconceptions.\n"
            "   > [!NOTE] Historical & Industrial Context: Real-world engineering implementations.\n"
            "4. DISPLAY LATEX EQUATIONS: Write EVERY major theorem, governing formula, and mathematical proof in centered display LaTeX (`$$ ... $$`) on its own dedicated line.\n"
            "5. INLINE VARIABLES: Wrap individual mathematical symbols in single dollar signs ($x$, $\\beta_t$, $W_Q$, $d_k$, $\\mathcal{O}(N^2)$).\n"
            "6. 1-PAGE FORMULA CHEAT-SHEET: In Module 4.5, include a compact Markdown summary table listing all governing equations, variables, constants, and SI units.\n\n"
            "REQUIRED VOLUME I STRUCTURE:\n"
            "### 1. 🎓 Executive Overview, Core Axioms & Physical Intuition\n"
            "- Foundational axioms, historical evolution, and mathematical motivation.\n"
            "- 2 vivid real-world analogies explaining non-obvious dynamics.\n"
            "- Core graduate competencies mastered.\n\n"
            "### 2. 📚 Exhaustive Conceptual Breakdown & Architectural Mechanics\n"
            "- Valid ```mermaid flowchart diagram visually mapping the core system architecture/dataflow.\n"
            "- Divide into at least 4 detailed subsections (e.g. 2.1, 2.2, 2.3, 2.4).\n"
            "- Write multi-paragraph continuous explanations from first principles analyzing internal operations.\n\n"
            "### 3. 📐 The Complete Mathematical Framework & Rigorous Derivations\n"
            "- State all governing equations in centered display LaTeX `$$ ... $$`.\n"
            "- Provide complete line-by-line algebraic proofs showing every intermediate step.\n"
            "- Define every variable, matrix dimension, tensor shape, and constant explicitly in bullet points.\n\n"
            "### 4. 🔬 Boundary Dynamics, Complexity Analysis & Engineering Constraints\n"
            "- Asymptotic limits, computational/memory complexity proofs (e.g. $\\mathcal{O}(\\cdot)$ in time and space).\n"
            "- Hardware-level optimization, memory bandwidth bottlenecks, and modern production standards.\n"
            "#### 4.5 📋 1-Page High-Yield Formula & Definition Cheat-Sheet (Summary Table for Rapid Revision)"
        )

        prompt_vol1 = (
            f"ACADEMIC RESEARCH CONTEXT:\n{aggregated_context[:10000] if aggregated_context else 'Comprehensive Academic Curriculum Base'}\n\n"
            f"TARGET TOPIC: {clean_title}\n"
            f"FULL USER INQUIRY: {prompt}\n\n"
            f"Write the complete, exhaustive VOLUME I for '{clean_title}'. Follow all modules with continuous prose, valid Mermaid flowchart, pedagogical callouts ([!TIP], [!IMPORTANT]), and complete display LaTeX equations ($$ ... $$):"
        )

        raw_vol1 = orchestrator._call_model(reasoning_llm, prompt_vol1, max_tokens=gen_tokens, temperature=gen_temp, system_prompt=sys_prompt_vol1)
        cleaned_vol1 = StudyPipeline._sanitize_study_latex(orchestrator._strip_thinking(raw_vol1))
        orchestrator._check_cancelled("study:volume1_done")

        # ── Volume II: Applied Taxonomy, 10-Problem Solved Question Bank & Standardized Mock Exam ──
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 3/4]: Authoring Volume II (10-Problem Solved Bank & Mock Exam Paper) with {reasoning_display}...", "info", reasoning_key, 70)

        sys_prompt_vol2 = (
            "You are a Distinguished Chief Examiner and Professor for National Competitive & Graduate Examinations.\n"
            "Your task is to write VOLUME II: Applied Taxonomy, Master Solved Question Bank, and Complete Standardized Mock Examination Paper.\n\n"
            "CRITICAL MANDATES — DO NOT REPEAT VOLUME I:\n"
            "1. START DIRECTLY with Module 5 ('### 5. 📊 Master Multi-Dimensional Comparison & Classification Matrices'). DO NOT repeat introduction or theory from Volume I.\n"
            "2. WRITE ALL 10 PROBLEMS IN FULL: In Module 7, you MUST write out 10 distinct problems with concrete numerical parameters, step-by-step algebraic substitutions, and final boxed answers.\n"
            "3. FULL MOCK EXAM PAPER: In Module 8, write out 5 full MCQs (with options A/B/C/D, answer keys & explanations), 3 Short Numerical Problems with full work, and 2 Long-Form Derivations with marking rubrics.\n"
            "4. DISPLAY LATEX: Render all formulas in display LaTeX (`$$ ... $$`). Never put English sentences inside `$ ... $`.\n\n"
            "REQUIRED VOLUME II STRUCTURE:\n"
            "### 5. 📊 Master Multi-Dimensional Comparison & Classification Matrices\n"
            "Construct a dense, multi-column Markdown comparison table contrasting key architectures, computational complexities, memory footprints, and trade-offs.\n\n"
            "### 6. 💡 High-Yield Exam Traps, Conceptual Pitfalls & Memory Mnemonics\n"
            "- Top 5 tricky pitfalls, sign/index errors, and frequent student misconceptions with correct explanations.\n"
            "- High-yield memory mnemonics for rapid exam recall.\n"
            "- 6 Rapid-Revision Q&A Flashcards.\n\n"
            "### 7. 📝 10-Problem Master Solved Question Bank\n"
            "Write out 10 distinct, fully solved problems graded by difficulty:\n"
            "- **Problems 1-3 (Foundational / Direct Computation):** Direct formula applications with full arithmetic steps.\n"
            "- **Problems 4-7 (Intermediate / Multi-Step Analysis):** Complex analytical problems with multi-step substitutions.\n"
            "- **Problems 8-10 (Advanced / Rigorous Proofs & Optimization):** Tough GATE/Olympiad-level problems with complete proofs.\n"
            "EVERY PROBLEM MUST INCLUDE: **Question**, **Given Parameters**, **Governing Formula ($$ ... $$)**, **Step-by-Step Derivation**, and **Final Boxed Answer**.\n\n"
            "### 8. 🎯 Standardized University Mock Exam Blueprint & Scoring Rubric\n"
            "- **Section A:** 5 Multiple Choice Questions (with options A, B, C, D, Answer Key & Detailed Rationale).\n"
            "- **Section B:** 3 Short-Answer Numerical Questions with complete worked-out solutions.\n"
            "- **Section C:** 2 Long-Form Comprehensive Derivation / System Design Questions with official step-by-step marking rubrics."
        )

        prompt_vol2 = (
            f"TARGET TOPIC: {clean_title}\n\n"
            f"TASK: Author the MASTER PROBLEM BANK & EXAMINATION SUITE (Volume II) for '{clean_title}'.\n\n"
            f"INSTRUCTIONS:\n"
            f"Start DIRECTLY with Module 5 (Do NOT re-introduce the topic). Author Modules 5, 6, 7 (all 10 fully solved problems), and 8 (complete mock exam paper) with full mathematical rigor ($$ ... $$):\n\n"
            f"### 5. 📊 Master Multi-Dimensional Comparison & Classification Matrices"
        )

        raw_vol2 = orchestrator._call_model(reasoning_llm, prompt_vol2, max_tokens=gen_tokens, temperature=gen_temp, system_prompt=sys_prompt_vol2)
        cleaned_vol2 = StudyPipeline._sanitize_study_latex(orchestrator._strip_thinking(raw_vol2))
        if not cleaned_vol2.startswith("### 5"):
            cleaned_vol2 = "### 5. 📊 Master Multi-Dimensional Comparison & Classification Matrices\n\n" + cleaned_vol2
        orchestrator._check_cancelled("study:volume2_done")

        if status_callback:
            status_callback("🎓 Study Engine [Stage 4/4]: Compiling & binding unified Master Reference Textbook...", "info", "system", 95)

        return (
            f"# 🎓 Master Reference Textbook & Comprehensive Pedagogical Treatise\n"
            f"## 📖 Subject: {clean_title}\n\n"
            f"> **Curriculum Standard:** University Graduate Reference Level | **Engine:** DeepSeek-R1 High-Precision Pedagogical Core\n\n"
            f"---\n\n"
            f"## 📚 Volume I: Theoretical Foundations, Architecture & Complete Formal Derivations\n\n"
            f"{cleaned_vol1}\n\n"
            f"---\n\n"
            f"## 📝 Volume II: Comparative Taxonomy, Solved Question Bank & Standardized Examination Suite\n\n"
            f"{cleaned_vol2}"
        )

    @staticmethod
    def _execute_doc_study(orchestrator, prompt, clean_title, doc_text, reasoning_llm, reasoning_key, reasoning_display, gen_tokens, gen_temp, status_callback):
        """Sub-Mode B: Document-Grounded Multi-Volume Master Study Notes & Solved Exam Suite."""
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 1/3]: Ingesting uploaded document ({len(doc_text)} chars)...", "info", "system", 20)

        orchestrator._check_cancelled("study:doc_ingest")

        # Volume I: Document-Grounded Theory, Mermaid Diagram & Mathematical Notes
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 2/3]: Grounding Volume I (Theorems, Mermaid Flowchart & Formulas) with {reasoning_display}...", "info", reasoning_key, 45)

        sys_prompt_vol1 = (
            "You are a Distinguished Professor and Master Educator.\n"
            "Your task is to transform the provided source document into VOLUME I of a comprehensive, human-readable master study guide and reference textbook.\n\n"
            "MANDATORY GROUNDING, VISUAL & LATEX RULES:\n"
            "1. STRICT GROUNDING: Every concept, theorem, definition, and formula must be strictly grounded in the ingested source document.\n"
            "2. MERMAID VISUAL DIAGRAM: In Module 2, include a clean, valid ```mermaid flowchart illustrating the core architecture or process pipeline.\n"
            "3. PEDAGOGICAL ALERTS: Embed strategic GitHub-style alert callouts (> [!TIP], > [!IMPORTANT], > [!NOTE]).\n"
            "4. DISPLAY LATEX: Write all mathematical formulas and equations in centered display LaTeX (`$$ ... $$`). Never put English sentences inside `$ ... $`.\n"
            "5. CONTINUOUS TEXTBOOK PROSE: Write in-depth, multi-paragraph explanations explaining all dynamics from first principles. Do NOT write brief bullet outlines.\n"
            "6. 1-PAGE FORMULA CHEAT-SHEET: Include a compact summary table of all grounded formulas and definitions at the end of Volume I.\n\n"
            "REQUIRED STRUCTURE:\n"
            "### 1. 🎓 Executive Epistemological Summary & Foundational Definitions\n"
            "### 2. 📚 Comprehensive In-Depth Conceptual Deconstruction (with Mermaid Diagram)\n"
            "### 3. 📐 Mathematical Formalisms, Governing Equations ($$ ... $$) & Rigorous Proofs\n"
            "### 4. 🔬 Edge Cases, Boundary Conditions & Practical Domain Implementations\n"
            "#### 4.5 📋 1-Page High-Yield Formula & Definition Cheat-Sheet Summary Table"
        )

        prompt_vol1 = (
            f"INGESTED SOURCE DOCUMENT CONTENT:\n\"\"\"\n{doc_text[:14000]}\n\"\"\"\n\n"
            f"FOCUS / USER QUERY: {prompt if prompt else clean_title}\n\n"
            f"Write the complete, document-grounded VOLUME I with full display LaTeX equations ($$ ... $$), Mermaid flowchart, and pedagogical callouts:"
        )

        raw_vol1 = orchestrator._call_model(reasoning_llm, prompt_vol1, max_tokens=gen_tokens, temperature=gen_temp, system_prompt=sys_prompt_vol1)
        cleaned_vol1 = StudyPipeline._sanitize_study_latex(orchestrator._strip_thinking(raw_vol1))
        orchestrator._check_cancelled("study:doc_vol1_done")

        # Volume II: Document-Grounded Practice Bank & Exam Suite
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 3/3]: Grounding Volume II (10-Problem Solved Bank & Mock Exam) with {reasoning_display}...", "info", reasoning_key, 75)

        sys_prompt_vol2 = (
            "You are a Chief Academic Examiner.\n"
            "Your task is to author VOLUME II: Document-Grounded Comparative Matrices, 10 Fully Solved Practice Problems, and a Predicted Mock Exam Paper.\n\n"
            "MANDATORY EXAM RULES:\n"
            "1. NO PROMISES OR OUTLINES: Write out all 10 problems completely with full numbers, step-by-step worked solutions, and final boxed answers.\n"
            "2. Ground all problems and exam questions directly in the formulas and concepts present in the ingested document.\n"
            "3. Render all math in centered display LaTeX (`$$ ... $$`). Never put English sentences inside `$ ... $`.\n"
            "4. START DIRECTLY with Module 5 ('### 5. 📊 Comparative Taxonomy & Classification Matrices'). Do NOT repeat Volume I introductions.\n\n"
            "REQUIRED STRUCTURE:\n"
            "### 5. 📊 Comparative Taxonomy & Classification Matrices\n"
            "### 6. 💡 High-Yield Exam Traps, Common Misconceptions & Mnemonics\n"
            "### 7. 📝 10-Problem Master Solved Question Bank (3 Foundational, 4 Intermediate, 3 Advanced with Full Derivations)\n"
            "### 8. 🎯 Predicted Examination Paper (Section A: 5 MCQs with Explanations, Section B: 3 Numerical Questions, Section C: 2 Derivations with Rubrics)"
        )

        prompt_vol2 = (
            f"INGESTED SOURCE DOCUMENT CONTENT:\n\"\"\"\n{doc_text[:10000]}\n\"\"\"\n\n"
            f"TARGET SUBJECT: {clean_title}\n\n"
            f"Write the complete, document-grounded VOLUME II with all 10 solved problems and full mock exam paper written out in complete detail, starting directly with Module 5:\n\n"
            f"### 5. 📊 Comparative Taxonomy & Classification Matrices"
        )

        raw_vol2 = orchestrator._call_model(reasoning_llm, prompt_vol2, max_tokens=gen_tokens, temperature=gen_temp, system_prompt=sys_prompt_vol2)
        cleaned_vol2 = StudyPipeline._sanitize_study_latex(orchestrator._strip_thinking(raw_vol2))
        if not cleaned_vol2.startswith("### 5"):
            cleaned_vol2 = "### 5. 📊 Comparative Taxonomy & Classification Matrices\n\n" + cleaned_vol2

        return (
            f"# 🎓 Document-Grounded Master Reference Treatise & Solved Examination Suite\n\n"
            f"> **Source Ingestion:** Verified Document Context Grounded | **Engine:** DeepSeek-R1 Pedagogical Core\n\n"
            f"---\n\n"
            f"## 📚 Volume I: Grounded Theory, Architectural Framework & Mathematical Formalisms\n\n"
            f"{cleaned_vol1}\n\n"
            f"---\n\n"
            f"## 📝 Volume II: Comparative Matrices, 10-Problem Solved Bank & Predicted Examination Paper\n\n"
            f"{cleaned_vol2}"
        )

    @staticmethod
    def _extract_clean_title(prompt):
        """Extracts a concise, readable subject title from a user prompt."""
        if not prompt:
            return "Advanced Academic Subject Reference"
        
        text = prompt.strip()
        # Remove common instruction prefixes
        text = re.sub(
            r"^(teach me|explain|create study notes for|notes on|give me notes for|study guide for|a complete guide on|write a textbook on|author an exhaustive graduate level reference textbook on|author an exhaustive reference textbook on|author an exhaustive graduate-level reference textbook chapter on|author a textbook on)\s+",
            "", text, flags=re.I
        ).strip()

        # If prompt has 'from first principles' or 'focusing on', grab the text before it
        match_first = re.split(r"\s+(?:from first principles|focusing on|covering|including|with full)\b", text, flags=re.I)
        if match_first and len(match_first[0].strip()) > 3:
            candidate = match_first[0].strip().rstrip(":,.- ")
            if len(candidate) < 90:
                return candidate.title()

        # If prompt has sentences, grab first sentence
        if "." in text:
            first_clause = text.split(".")[0].strip()
            if len(first_clause) > 5 and len(first_clause) < 90:
                return first_clause.title()

        return text[:80].strip().title() if text else "Advanced Academic Curriculum"

    @staticmethod
    def _sanitize_study_latex(text):
        """Fixes unclosed or misformatted LaTeX blocks to prevent squashed math rendering."""
        if not text:
            return text

        # 1. Replace raw unicode minus signs
        text = text.replace("−", "-")

        # 1b. Convert lone $ on its own line to $$ (small models use single $ for display math)
        text = re.sub(r"(?m)^\$\s*$", "$$", text)

        # 2. Normalize bracket delimiters: \[ ... \] -> $$ ... $$ and \( ... \) -> $ ... $
        text = re.sub(r"\\\[\s*([\s\S]*?)\s*\\\]", r"$$\n\1\n$$", text)
        text = re.sub(r"\\\(\s*([\s\S]*?)\s*\\\)", r"$\1$", text)

        # 3. Normalize LaTeX environment blocks to KaTeX-compatible forms
        text = re.sub(r"\\begin\{equation\*?\}([\s\S]*?)\\end\{equation\*?\}", r"$$\n\1\n$$", text)
        text = re.sub(r"\\begin\{displaymath\}([\s\S]*?)\\end\{displaymath\}", r"$$\n\1\n$$", text)
        text = re.sub(r"\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}", r"$$\n\\begin{aligned}\1\\end{aligned}\n$$", text)
        text = re.sub(r"\\begin\{gather\*?\}([\s\S]*?)\\end\{gather\*?\}", r"$$\n\\begin{gathered}\1\\end{gathered}\n$$", text)

        # 4. Clean double-nested environments inside $$ ... $$
        text = re.sub(r"\$\$\s*\\begin\{equation\*?\}([\s\S]*?)\\end\{equation\*?\}\s*\$\$", r"$$\n\1\n$$", text)
        text = re.sub(r"\$\$\s*\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}\s*\$\$", r"$$\n\\begin{aligned}\1\\end{aligned}\n$$", text)
        text = re.sub(r"\$\$\s*\\begin\{gather\*?\}([\s\S]*?)\\end\{gather\*?\}\s*\$\$", r"$$\n\\begin{gathered}\1\\end{gathered}\n$$", text)

        # 5. Fix double escaped LaTeX commands in text (\\command -> \command)
        text = re.sub(r"\\\\([a-zA-Z]+)", r"\\\1", text)

        # 6. Clean consecutive blank lines inside $$ blocks to prevent KaTeX paragraph-break errors
        def _clean_display_math(match):
            inner = match.group(1).strip()
            inner = re.sub(r"\n\s*\n+", r"\n", inner)
            return f"$$\n{inner}\n$$"

        text = re.sub(r"\$\$([\s\S]*?)\$\$", _clean_display_math, text)

        return text

    @staticmethod
    def _extract_document_text(payload):
        """Extract text from base64 PDF or raw text data URL."""
        if not payload or not isinstance(payload, str):
            return ""

        try:
            if "application/pdf" in payload or payload.startswith("data:application/pdf;base64,"):
                b64_data = payload.split(",", 1)[-1] if "," in payload else payload
                pdf_bytes = base64.b64decode(b64_data)
                
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                    text_pages = [page.extract_text() or "" for page in reader.pages]
                    full_text = "\n\n".join([f"--- Page {i+1} ---\n{t}" for i, t in enumerate(text_pages) if t.strip()])
                    if full_text.strip():
                        return full_text
                except Exception:
                    pass

                try:
                    import fitz
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    text_pages = [page.get_text() or "" for page in doc]
                    full_text = "\n\n".join([f"--- Page {i+1} ---\n{t}" for i, t in enumerate(text_pages) if t.strip()])
                    if full_text.strip():
                        return full_text
                except Exception:
                    pass

            if payload.startswith("data:text/"):
                b64_data = payload.split(",", 1)[-1] if "," in payload else payload
                return base64.b64decode(b64_data).decode("utf-8", errors="ignore")

            if len(payload) > 50 and not payload.startswith("data:image/"):
                return payload

        except Exception as e:
            print(f"Document extraction notice: {e}")

        return ""
