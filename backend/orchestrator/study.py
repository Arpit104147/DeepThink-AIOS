import os
import re
import json
import base64
import io
from backend.sandbox import Sandbox

class StudyPipeline:
    """
    Hierarchical Multi-Chapter Master Curriculum & Pedagogical Synthesis Engine.
    Strictly powered by the Reasoning LLM (DeepSeek-R1) to generate massive,
    exhaustive, 10-20 page level university-grade textbook chapters, complete
    with rigorous step-by-step LaTeX derivations, comparison tables, a 10-problem
    solved question bank, and a full mock exam blueprint.
    """

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None, attached_doc=None, attached_image=None):
        if status_callback:
            status_callback("🎓 Study Engine: Initializing deep pedagogical reasoning core...", "info", "deepseek_r1", 5)

        # 1. Strictly bind to the Reasoning LLM (DeepSeek-R1)
        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        reasoning_key = "deepseek_r1" if orchestrator._is_model_valid(orchestrator._get_model("deepseek_r1", required_ctx=4096)) else "vibethinker"
        reasoning_llm = orchestrator._get_model(reasoning_key, required_ctx=ds_ctx)
        reasoning_display = orchestrator._get_display_model_name(reasoning_key)

        # Boost generation token headroom for exhaustive depth
        study_gen_tokens = min(8192, max(4096, ds_ctx - 1500))
        study_temp = 0.2  # Low temperature for strict mathematical precision

        # 2. Check for uploaded Document payload (Sub-Mode B: Document-Grounded Notes)
        doc_payload = attached_doc or attached_image
        extracted_doc_text = ""
        if doc_payload:
            extracted_doc_text = StudyPipeline._extract_document_text(doc_payload)

        if extracted_doc_text and len(extracted_doc_text) > 100:
            return StudyPipeline._execute_doc_study(
                orchestrator, prompt, extracted_doc_text, reasoning_llm, reasoning_key,
                reasoning_display, study_gen_tokens, study_temp, status_callback
            )
        else:
            return StudyPipeline._execute_web_study(
                orchestrator, prompt, reasoning_llm, reasoning_key,
                reasoning_display, study_gen_tokens, study_temp, status_callback
            )

    @staticmethod
    def _execute_web_study(orchestrator, prompt, reasoning_llm, reasoning_key, reasoning_display, gen_tokens, gen_temp, status_callback):
        """Sub-Mode A: Omni-Web Multi-Query Deep Scraper & Multi-Volume Synthesis."""
        if status_callback:
            status_callback("🎓 Study Engine [Stage 1/4]: Ingesting multi-source curriculum from 15+ academic feeds...", "info", "system", 15)

        raw_contexts = []
        topic_clean = re.sub(r"(teach me|explain|create study notes for|notes on|give me notes for|study guide for|a complete guide on)", "", prompt, flags=re.I).strip()
        if not topic_clean:
            topic_clean = prompt

        # 5 targeted academic sub-queries to capture all theoretical & practical dimensions
        sub_queries = [
            f"{topic_clean} foundational principles axioms theoretical framework overview",
            f"{topic_clean} mathematical derivations proofs equations step by step",
            f"{topic_clean} mechanism dynamics parameter interactions comparison",
            f"{topic_clean} advanced boundary conditions edge cases applications",
            f"{topic_clean} practice problems numerical worked solutions exam questions"
        ]

        if hasattr(orchestrator, "web_search") and orchestrator.web_search:
            try:
                primary_res = orchestrator.web_search.search_and_scrape(topic_clean, max_results=8, max_scrapes=5)
                if isinstance(primary_res, dict) and not primary_res.get("empty", True):
                    raw_contexts.append(primary_res.get("context", ""))

                for sq in sub_queries:
                    try:
                        sq_res = orchestrator.web_search.search(sq, max_results=4)
                        if isinstance(sq_res, list) and sq_res:
                            formatted = [f"[{item.get('title', 'Ref')}] ({item.get('link', '')}):\n{item.get('snippet', '')}" for item in sq_res[:3] if item.get('snippet')]
                            if formatted:
                                raw_contexts.append("\n\n".join(formatted))
                    except Exception:
                        pass
            except Exception as e:
                print(f"Study search notice: {e}")

        aggregated_context = "\n\n---\n\n".join(raw_contexts)

        # ── VOLUME 1: Theory, Mechanisms, Mathematical Proofs & Tables ───────
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 2/4]: Synthesizing Volume 1 (Theory & LaTeX Derivations) with {reasoning_display}...", "info", reasoning_key, 35)

        sys_prompt_vol1 = StudyPipeline._build_volume1_system_prompt()
        prompt_vol1 = (
            f"REFERENCE MATERIALS & ACADEMIC SOURCES:\n{aggregated_context[:14000] if aggregated_context else 'Comprehensive Academic Curriculum Base'}\n\n"
            f"STUDY TOPIC: {topic_clean}\n\n"
            f"TASK: Synthesize VOLUME 1 (Foundations, Deep Theoretical Breakdown, Mathematical Proofs in display LaTeX, Dynamics, and Comparison Tables) "
            f"with exhaustive textbook-level depth. Write full, continuous paragraphs without skipping derivations."
        )
        vol1_raw = orchestrator._call_model(reasoning_llm, prompt_vol1, gen_tokens, gen_temp, system_prompt=sys_prompt_vol1)
        vol1_clean = orchestrator._strip_thinking(vol1_raw)

        # ── VOLUME 2: Problem Book, Exam Traps, Flashcards & Mock Exam ─────────
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 3/4]: Synthesizing Volume 2 (10-Problem Solved Question Bank & Mock Exam) with {reasoning_display}...", "info", reasoning_key, 70)

        sys_prompt_vol2 = StudyPipeline._build_volume2_system_prompt()
        prompt_vol2 = (
            f"STUDY TOPIC: {topic_clean}\n"
            f"CONTEXT SUMMARY FROM VOLUME 1: Theoretical framework established for {topic_clean}.\n\n"
            f"TASK: Synthesize VOLUME 2 (Exam Traps & Mnemonics, 8 Flashcards, a 10-Problem Fully Solved Question Bank with step-by-step math, and a Complete Mock Exam Blueprint) "
            f"at University Graduate / Olympiad rigor. Solve every single problem completely."
        )
        vol2_raw = orchestrator._call_model(reasoning_llm, prompt_vol2, gen_tokens, gen_temp, system_prompt=sys_prompt_vol2)
        vol2_clean = orchestrator._strip_thinking(vol2_raw)

        if status_callback:
            status_callback("🎓 Study Engine [Stage 4/4]: Compiling 10-20 page Master Reference Book...", "info", "system", 95)

        return (
            f"# 🎓 Master Reference Textbook & Comprehensive Study Guide\n"
            f"## 📖 Subject: {topic_clean.title()}\n\n"
            f"---\n\n"
            f"{vol1_clean}\n\n"
            f"---\n\n"
            f"{vol2_clean}"
        )

    @staticmethod
    def _execute_doc_study(orchestrator, prompt, doc_text, reasoning_llm, reasoning_key, reasoning_display, gen_tokens, gen_temp, status_callback):
        """Sub-Mode B: Document-Grounded Multi-Volume Master Study Notes."""
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 1/3]: Parsing uploaded document ({len(doc_text)} chars)...", "info", "system", 20)

        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 2/3]: Synthesizing Volume 1 (Grounded Theory & Mathematical Framework) with {reasoning_display}...", "info", reasoning_key, 50)

        sys_prompt_doc1 = StudyPipeline._build_volume1_system_prompt(is_doc_grounded=True)
        prompt_doc1 = (
            f"INGESTED DOCUMENT CONTENT:\n\"\"\"\n{doc_text[:14000]}\n\"\"\"\n\n"
            f"STUDY FOCUS: {prompt if prompt else 'Create an exhaustive master study guide grounded in this document.'}\n\n"
            f"TASK: Transform the document into Volume 1 of an exhaustive study guide (Modules 1 to 5: Executive Overview, Detailed Topic Breakdown, Formula Proofs in LaTeX, Edge Cases, and Comparison Tables). Strictly ground all concepts in the document."
        )
        vol1_raw = orchestrator._call_model(reasoning_llm, prompt_doc1, gen_tokens, gen_temp, system_prompt=sys_prompt_doc1)
        vol1_clean = orchestrator._strip_thinking(vol1_raw)

        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 3/3]: Synthesizing Volume 2 (Document Practice Problems & Mock Exam) with {reasoning_display}...", "info", reasoning_key, 80)

        sys_prompt_doc2 = StudyPipeline._build_volume2_system_prompt(is_doc_grounded=True)
        prompt_doc2 = (
            f"INGESTED DOCUMENT CONTENT:\n\"\"\"\n{doc_text[:14000]}\n\"\"\"\n\n"
            f"TASK: Construct Volume 2 based on the document (Exam Pitfalls, Flashcards, 10 Solved Practice Problems directly derived from the document's theorems, and a Predicted Mock Exam Paper)."
        )
        vol2_raw = orchestrator._call_model(reasoning_llm, prompt_doc2, gen_tokens, gen_temp, system_prompt=sys_prompt_doc2)
        vol2_clean = orchestrator._strip_thinking(vol2_raw)

        return (
            f"# 🎓 Document-Grounded Master Reference Guide & Exam Blueprint\n\n"
            f"{vol1_clean}\n\n"
            f"---\n\n"
            f"{vol2_clean}"
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
        base = (
            "You are a Distinguished Chaired Professor, Academician, and Principal Author of University Graduate Reference Textbooks.\n"
            "Your task is to write VOLUME 1 (THEORETICAL FOUNDATIONS & MATHEMATICAL APPARATUS) of a definitive, 10-20 page level master reference book.\n\n"
            "MANDATORY STRUCTURAL FORMAT FOR VOLUME 1:\n\n"
            "### 1. 🎓 Executive Epistemological Overview & Core Axioms\n"
            "- Historical development, foundational axioms, and physical/computational intuition.\n"
            "- 2 intuitive, real-world analogies explaining non-obvious dynamics.\n"
            "- Exhaustive list of core competencies & learning objectives.\n\n"
            "### 2. 📚 Exhaustive Conceptual Breakdown & Sub-Topic Analysis\n"
            "- Divide into at least 4 to 6 detailed sub-sections (e.g., 2.1, 2.2, 2.3, 2.4, 2.5).\n"
            "- Write in-depth, multi-paragraph explanations from first principles. Do not summarize.\n"
            "- Analyze internal mechanisms, state transitions, and governing dynamics.\n\n"
            "### 3. 📐 The Complete Mathematical Framework & Rigorous Derivations\n"
            "- Write EVERY equation in centered display LaTeX (`$$ ... $$`).\n"
            "- Provide line-by-line algebraic proofs and derivations from fundamental laws.\n"
            "- Define every single variable, constant, tensor, and SI unit explicitly in bullet points.\n\n"
            "### 4. 🔬 Advanced Boundary Conditions, Edge Cases & Modern Applications\n"
            "- Limiting behavior (e.g., asymptotes, high/low-frequency limits, singularity handling).\n"
            "- Real-world engineering implementations, hardware constraints, or modern industry standards.\n\n"
            "### 5. 📊 Master Multi-Dimensional Comparison & Classification Matrices\n"
            "- Construct a dense, multi-column Markdown comparison table summarizing mechanisms, complexities, trade-offs, and applications.\n\n"
            "STRICT RULES: Use publication-grade academic prose. Render all math in LaTeX (`$$ ... $$` for display, `$ ... $` for inline). Never use placeholders."
        )
        if is_doc_grounded:
            base += "\nSTRICT GROUNDING: Every theorem, formula, and explanation must be strictly grounded in the ingested document."
        return base

    @staticmethod
    def _build_volume2_system_prompt(is_doc_grounded=False):
        base = (
            "You are a Senior Faculty Examiner, Olympiad Chief Judge, and Master Pedagogy Specialist.\n"
            "Your task is to write VOLUME 2 (EXAM MASTERY, SOLVED PROBLEM BOOK & MOCK BLUEPRINT) of the master reference curriculum.\n\n"
            "MANDATORY STRUCTURAL FORMAT FOR VOLUME 2:\n\n"
            "### 6. 💡 High-Yield Exam Pitfalls, Conceptual Traps & Memory Mnemonics\n"
            "- Specific tricky traps, sign errors, and frequent student misconceptions with correct explanations.\n"
            "- High-yield memory mnemonics and mental shortcuts for rapid recall.\n\n"
            "### 7. 🧠 Rapid-Revision Flashcard Deck\n"
            "- 8 to 10 comprehensive Q&A flashcards covering all core definitions, theorems, and edge cases.\n\n"
            "### 8. 📝 10-Problem Master Solved Question Bank\n"
            "Provide 10 distinct, fully solved problems graded by difficulty:\n"
            "- **Problems 1-3 (Foundational / Conceptual):** Direct formula and theorem applications with complete step-by-step solutions.\n"
            "- **Problems 4-7 (Intermediate / Computational):** Complex numerical problems with multi-step substitutions and unit conversions.\n"
            "- **Problems 8-10 (Advanced / Analytical & Proofs):** Tough Olympiad/GATE-level analytical problems with deep mathematical proofs.\n"
            "- EVERY PROBLEM MUST INCLUDE the full Question, Given Parameters, Formula Used, Step-by-Step Derivation, and Final Boxed Answer.\n\n"
            "### 9. 🎯 Standardized Mock Exam Blueprint & Scoring Rubric\n"
            "- **Section A:** 5 Multiple Choice / Conceptual Questions (with answer key & explanations).\n"
            "- **Section B:** 3 Short-Answer Numerical Questions.\n"
            "- **Section C:** 2 Long-Form Comprehensive Proof / Derivation Questions with official step-by-step marking rubrics.\n\n"
            "STRICT RULES: Solve all 10 problems completely line-by-line. Never skip calculation steps. Render all math in LaTeX."
        )
        if is_doc_grounded:
            base += "\nSTRICT GROUNDING: All practice problems and mock exam questions must reflect the uploaded document's concepts."
        return base
