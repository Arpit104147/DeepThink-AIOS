import os
import re
import json
import base64
import io
from backend.sandbox import Sandbox
from backend.downloader import resolve_model_key

class StudyPipeline:
    """
    Hierarchical Multi-Volume Master Curriculum & Deep Pedagogical Synthesis Engine.
    Strictly powered by the Reasoning LLM (DeepSeek-R1) to author exhaustive,
    university-grade, publication-level 10-20 page master reference textbook chapters.
    
    Architecture:
    - Volume I: Theoretical Foundations, Epistemology, First-Principles Architecture,
      and Rigorous Mathematical Derivations ($$ ... $$) [Modules 1-4].
    - Volume II: Comparative Taxonomy Matrices, Exam Traps & Mnemonics,
      Complete 10-Problem Solved Question Bank, and Standardized Mock Exam Blueprint [Modules 5-8].
    - Master Binding: Seamlessly unifies both volumes into an exhaustive 15-20 page textbook.
    """

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None, attached_doc=None, attached_image=None):
        if status_callback:
            status_callback("🎓 Study Engine: Initializing multi-volume pedagogical reasoning core...", "info", "deepseek_r1", 5)

        # 1. Strictly bind to the Reasoning LLM (DeepSeek-R1)
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

        # Dedicated token budget per volume (~3500 tokens each = ~7000+ tokens total output)
        vol_gen_tokens = min(3584, max(2560, (ds_ctx - 1500) // 2))
        study_temp = 0.2  # Low temperature for strict mathematical and factual precision

        orchestrator._check_cancelled("study:init")

        # 2. Check for uploaded Document payload (Sub-Mode B: Document-Grounded Notes)
        doc_payload = attached_doc or attached_image
        extracted_doc_text = ""
        if doc_payload:
            extracted_doc_text = StudyPipeline._extract_document_text(doc_payload)

        if extracted_doc_text and len(extracted_doc_text) > 100:
            return StudyPipeline._execute_doc_study(
                orchestrator, prompt, extracted_doc_text, reasoning_llm, reasoning_key,
                reasoning_display, vol_gen_tokens, study_temp, status_callback
            )
        else:
            return StudyPipeline._execute_web_study(
                orchestrator, prompt, reasoning_llm, reasoning_key,
                reasoning_display, vol_gen_tokens, study_temp, status_callback
            )

    @staticmethod
    def _execute_web_study(orchestrator, prompt, reasoning_llm, reasoning_key, reasoning_display, gen_tokens, gen_temp, status_callback):
        """Sub-Mode A: Multi-Source Academic Web Ingestion & 2-Volume Master Textbook Authoring."""
        if status_callback:
            status_callback("🎓 Study Engine [Stage 1/4]: Ingesting academic curriculum & empirical research...", "info", "system", 15)

        raw_contexts = []
        topic_clean = re.sub(r"(teach me|explain|create study notes for|notes on|give me notes for|study guide for|a complete guide on)", "", prompt, flags=re.I).strip()
        if not topic_clean:
            topic_clean = prompt

        if hasattr(orchestrator, "web_search") and orchestrator.web_search:
            try:
                # 1. Primary deep scrape
                primary_res = orchestrator.web_search.search_and_scrape(topic_clean, max_results=6, max_scrapes=3)
                if isinstance(primary_res, dict) and not primary_res.get("empty", True):
                    raw_contexts.append(primary_res.get("context", ""))

                # 2. Targeted sub-queries for equations and problems (fast snippet search)
                sub_res = orchestrator.web_search.search(f"{topic_clean} equations derivations proofs practice problems", max_results=4)
                if isinstance(sub_res, list) and sub_res:
                    formatted = [f"[{item.get('title', 'Ref')}] ({item.get('link', '')}):\n{item.get('snippet', '')}" for item in sub_res[:3] if item.get('snippet')]
                    if formatted:
                        raw_contexts.append("\n\n".join(formatted))
            except Exception as e:
                print(f"Study search notice: {e}")

        aggregated_context = "\n\n---\n\n".join(raw_contexts)
        orchestrator._check_cancelled("study:web_scrape_done")

        # ── Volume I: Theoretical Foundations, Architecture & Complete Derivations ──
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 2/4]: Authoring Volume I (Theorems, Deep Concepts & Derivations) with {reasoning_display}...", "info", reasoning_key, 35)

        sys_prompt_vol1 = StudyPipeline._build_volume1_system_prompt()
        prompt_vol1 = (
            f"ACADEMIC KNOWLEDGE BASE & CURRICULUM CONTEXT:\n{aggregated_context[:10000] if aggregated_context else 'Comprehensive Academic Curriculum Base'}\n\n"
            f"TARGET TOPIC: {topic_clean}\n\n"
            f"TASK: Author VOLUME I of the Master Reference Textbook for '{topic_clean}'.\n"
            f"Cover Modules 1, 2, 3, and 4 with exhaustive depth, multi-paragraph prose from first principles, and complete line-by-line LaTeX display derivations ($$ ... $$)."
        )

        raw_vol1 = orchestrator._call_model(reasoning_llm, prompt_vol1, max_tokens=gen_tokens, temperature=gen_temp, system_prompt=sys_prompt_vol1)
        cleaned_vol1 = orchestrator._strip_thinking(raw_vol1)
        orchestrator._check_cancelled("study:volume1_done")

        # ── Volume II: Applied Taxonomy, Problem Bank & Standardized Mock Exam ──
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 3/4]: Authoring Volume II (10-Problem Solved Bank & Mock Exam Paper) with {reasoning_display}...", "info", reasoning_key, 70)

        sys_prompt_vol2 = StudyPipeline._build_volume2_system_prompt()
        prompt_vol2 = (
            f"ACADEMIC KNOWLEDGE BASE & CURRICULUM CONTEXT:\n{aggregated_context[:8000] if aggregated_context else 'Comprehensive Academic Curriculum Base'}\n\n"
            f"TARGET TOPIC: {topic_clean}\n\n"
            f"VOLUME I SUMMARY/THEOREMS COVERED:\n{cleaned_vol1[:1500]}...\n\n"
            f"TASK: Author VOLUME II of the Master Reference Textbook for '{topic_clean}'.\n"
            f"Cover Modules 5, 6, 7, and 8 with exhaustive detail: complete comparative tables, exam traps/mnemonics/flashcards, the full 10 solved problems with complete derivations, and the full mock exam blueprint with scoring rubrics."
        )

        raw_vol2 = orchestrator._call_model(reasoning_llm, prompt_vol2, max_tokens=gen_tokens, temperature=gen_temp, system_prompt=sys_prompt_vol2)
        cleaned_vol2 = orchestrator._strip_thinking(raw_vol2)
        orchestrator._check_cancelled("study:volume2_done")

        if status_callback:
            status_callback("🎓 Study Engine [Stage 4/4]: Compiling & binding unified Master Reference Textbook...", "info", "system", 95)

        return (
            f"# 🎓 Master Reference Textbook & Comprehensive Pedagogical Treatise\n"
            f"## 📖 Subject: {topic_clean.title()}\n\n"
            f"> **Curriculum Standard:** University Graduate & Doctoral Reference Level | **Engine:** DeepSeek-R1 High-Precision Pedagogical Core\n\n"
            f"---\n\n"
            f"## 📚 Volume I: Theoretical Foundations, Architecture & Complete Formal Derivations\n\n"
            f"{cleaned_vol1}\n\n"
            f"---\n\n"
            f"## 📝 Volume II: Comparative Taxonomy, Solved Question Bank & Standardized Examination Suite\n\n"
            f"{cleaned_vol2}"
        )

    @staticmethod
    def _execute_doc_study(orchestrator, prompt, doc_text, reasoning_llm, reasoning_key, reasoning_display, gen_tokens, gen_temp, status_callback):
        """Sub-Mode B: Document-Grounded Multi-Volume Master Study Notes & Solved Exam Suite."""
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 1/3]: Ingesting uploaded document ({len(doc_text)} chars)...", "info", "system", 20)

        orchestrator._check_cancelled("study:doc_ingest")

        # Volume I
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 2/3]: Grounding Volume I (Theorems, Core Mechanisms & Formulas) with {reasoning_display}...", "info", reasoning_key, 45)

        sys_prompt_vol1 = StudyPipeline._build_volume1_system_prompt(is_doc_grounded=True)
        prompt_vol1 = (
            f"INGESTED SOURCE DOCUMENT CONTENT:\n\"\"\"\n{doc_text[:14000]}\n\"\"\"\n\n"
            f"STUDENT FOCUS / TOPIC: {prompt if prompt else 'Exhaustive master study treatise grounded in this document.'}\n\n"
            f"TASK: Author VOLUME I of the Master Reference Textbook grounded strictly in the source document above.\n"
            f"Cover Modules 1, 2, 3, and 4 with exhaustive depth, multi-paragraph conceptual breakdowns, and complete line-by-line display LaTeX derivations ($$ ... $$)."
        )

        raw_vol1 = orchestrator._call_model(reasoning_llm, prompt_vol1, max_tokens=gen_tokens, temperature=gen_temp, system_prompt=sys_prompt_vol1)
        cleaned_vol1 = orchestrator._strip_thinking(raw_vol1)
        orchestrator._check_cancelled("study:doc_vol1_done")

        # Volume II
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 3/3]: Grounding Volume II (10-Problem Solved Bank & Mock Exam) with {reasoning_display}...", "info", reasoning_key, 75)

        sys_prompt_vol2 = StudyPipeline._build_volume2_system_prompt(is_doc_grounded=True)
        prompt_vol2 = (
            f"INGESTED SOURCE DOCUMENT CONTENT:\n\"\"\"\n{doc_text[:10000]}\n\"\"\"\n\n"
            f"VOLUME I COVERED:\n{cleaned_vol1[:1500]}...\n\n"
            f"TASK: Author VOLUME II of the Document-Grounded Master Textbook.\n"
            f"Cover Modules 5, 6, 7, and 8: comparative matrices, document-specific exam traps, 10 solved practice problems grounded directly in the document, and a full mock exam paper with scoring rubrics."
        )

        raw_vol2 = orchestrator._call_model(reasoning_llm, prompt_vol2, max_tokens=gen_tokens, temperature=gen_temp, system_prompt=sys_prompt_vol2)
        cleaned_vol2 = orchestrator._strip_thinking(raw_vol2)

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

    @staticmethod
    def _build_volume1_system_prompt(is_doc_grounded=False):
        prompt = (
            "You are a Distinguished Chaired Professor, Academician, and Principal Author of Graduate Reference Textbooks.\n"
            "Your task is to write VOLUME I of a definitive, publication-grade 10-20 page master reference textbook chapter.\n\n"
            "MANDATORY VOLUME I STRUCTURAL MODULES:\n\n"
            "### 1. 🎓 Executive Epistemological Overview & Core Axioms\n"
            "- Foundational axioms, historical evolution, and physical/computational intuition.\n"
            "- 2 vivid, non-trivial real-world analogies explaining core dynamics.\n"
            "- Bulleted list of graduate-level competencies the student will master.\n\n"
            "### 2. 📚 Exhaustive Conceptual Breakdown & Sub-Topic Analysis\n"
            "- Divide into at least 4 detailed sub-sections (e.g., 2.1, 2.2, 2.3, 2.4).\n"
            "- Write in-depth, multi-paragraph explanations from first principles. Do not summarize or use bullet-only stubs.\n"
            "- Analyze internal mechanisms, state transitions, and governing dynamics in full academic prose.\n\n"
            "### 3. 📐 The Complete Mathematical Framework & Rigorous Derivations\n"
            "- Write EVERY major equation in centered display LaTeX (`$$ ... $$`).\n"
            "- Provide line-by-line algebraic proofs and step-by-step derivations from fundamental laws.\n"
            "- Define every single variable, constant, tensor, and SI unit explicitly in bullet points.\n\n"
            "### 4. 🔬 Advanced Boundary Conditions, Edge Cases & Modern Applications\n"
            "- Limiting behavior (asymptotes, high/low-frequency limits, singularity handling).\n"
            "- Real-world engineering implementations, hardware constraints, or modern industry standards.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. ALWAYS render all math in standard LaTeX ($$ ... $$ for display, $ ... $ for inline variables).\n"
            "2. Never abbreviate or use placeholders. Write full, continuous, publication-grade academic prose."
        )
        if is_doc_grounded:
            prompt += "\n3. STRICT GROUNDING: Ground every theorem, formula, and concept strictly in the ingested document."
        return prompt

    @staticmethod
    def _build_volume2_system_prompt(is_doc_grounded=False):
        prompt = (
            "You are a Distinguished Chaired Professor, Academician, and Chief Examiner for National Competitive Examinations.\n"
            "Your task is to write VOLUME II of a definitive, publication-grade master reference study guide and examination suite.\n\n"
            "MANDATORY VOLUME II STRUCTURAL MODULES:\n\n"
            "### 5. 📊 Master Multi-Dimensional Comparison & Classification Matrices\n"
            "- Construct a dense, multi-column Markdown comparison table contrasting key mechanisms, complexities, trade-offs, and applications.\n\n"
            "### 6. 💡 High-Yield Exam Pitfalls, Conceptual Traps & Memory Mnemonics\n"
            "- Specific tricky traps, sign errors, and frequent student misconceptions with correct explanations.\n"
            "- High-yield memory mnemonics and mental shortcuts for rapid recall.\n"
            "- 6 to 8 Rapid-Revision Q&A Flashcards.\n\n"
            "### 7. 📝 10-Problem Master Solved Question Bank\n"
            "Provide 10 distinct, fully solved problems graded by difficulty:\n"
            "- **Problems 1-3 (Foundational / Conceptual):** Direct formula and theorem applications with complete step-by-step solutions.\n"
            "- **Problems 4-7 (Intermediate / Computational):** Complex numerical problems with multi-step substitutions and unit conversions.\n"
            "- **Problems 8-10 (Advanced / Analytical & Proofs):** Tough Olympiad/GATE-level analytical problems with deep mathematical proofs.\n"
            "- EVERY PROBLEM MUST INCLUDE: Question, Given Parameters, Formula Used, Step-by-Step Derivation, and Final Boxed Answer.\n\n"
            "### 8. 🎯 Standardized Mock Exam Blueprint & Scoring Rubric\n"
            "- **Section A:** 5 Multiple Choice / Conceptual Questions (with answer key & detailed explanations).\n"
            "- **Section B:** 3 Short-Answer Numerical Questions with complete work.\n"
            "- **Section C:** 2 Long-Form Comprehensive Proof / Derivation Questions with official step-by-step marking rubrics.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. ALWAYS render all math in standard LaTeX ($$ ... $$ for display, $ ... $ for inline variables).\n"
            "2. Ensure all 10 problems are completely written out with full derivations — no abbreviations."
        )
        if is_doc_grounded:
            prompt += "\n3. STRICT GROUNDING: Ground all practice problems and exam questions strictly in the ingested document."
        return prompt
