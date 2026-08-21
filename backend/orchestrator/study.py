import os
import re
import json
import base64
import io
from backend.sandbox import Sandbox

class StudyPipeline:
    """
    Hierarchical Master Curriculum & Deep Pedagogical Synthesis Engine.
    Strictly powered by the Reasoning LLM (DeepSeek-R1) to generate massive,
    exhaustive, university-grade master reference textbook chapters, complete
    with rigorous step-by-step LaTeX derivations ($$ ... $$), comparison tables,
    a 10-problem solved question bank, and a full mock exam blueprint.
    """

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None, attached_doc=None, attached_image=None):
        if status_callback:
            status_callback("🎓 Study Engine: Initializing deep pedagogical reasoning core...", "info", "deepseek_r1", 10)

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
        """Sub-Mode A: Fast Multi-Source Academic Web Harvest & Master Textbook Synthesis."""
        if status_callback:
            status_callback("🎓 Study Engine [Stage 1/3]: Ingesting academic curriculum feeds...", "info", "system", 20)

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

        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 2/3]: Structuring topic hierarchy & formulas with {reasoning_display}...", "info", reasoning_key, 50)

        sys_prompt = StudyPipeline._build_master_study_system_prompt()
        prompt_master = (
            f"ACADEMIC KNOWLEDGE BASE & CURRICULUM CONTEXT:\n{aggregated_context[:12000] if aggregated_context else 'Comprehensive Academic Curriculum Base'}\n\n"
            f"TARGET STUDY TOPIC: {topic_clean}\n\n"
            f"TASK:\n"
            f"Write a DEFINITIVE, EXHAUSTIVE MASTER REFERENCE TEXTBOOK CHAPTER for this topic.\n"
            f"Follow all 8 modules rigorously. Provide full continuous paragraph explanations, line-by-line mathematical derivations in display LaTeX ($$ ... $$), dense comparison tables, 10 fully solved practice problems, and a complete mock exam blueprint."
        )

        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 3/3]: Synthesizing exhaustive textbook chapter & 10-problem solved exam with {reasoning_display}...", "info", reasoning_key, 75)

        raw_res = orchestrator._call_model(reasoning_llm, prompt_master, gen_tokens, gen_temp, system_prompt=sys_prompt)
        cleaned_res = orchestrator._strip_thinking(raw_res)

        return (
            f"# 🎓 Master Reference Textbook & Comprehensive Study Guide\n"
            f"## 📖 Subject: {topic_clean.title()}\n\n"
            f"---\n\n"
            f"{cleaned_res}"
        )

    @staticmethod
    def _execute_doc_study(orchestrator, prompt, doc_text, reasoning_llm, reasoning_key, reasoning_display, gen_tokens, gen_temp, status_callback):
        """Sub-Mode B: Document-Grounded Master Study Notes & Solved Exam Blueprint."""
        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 1/2]: Ingesting uploaded document ({len(doc_text)} chars)...", "info", "system", 25)

        if status_callback:
            status_callback(f"🎓 Study Engine [Stage 2/2]: Synthesizing grounded master study guide & mock exam with {reasoning_display}...", "info", reasoning_key, 60)

        sys_prompt_doc = StudyPipeline._build_master_study_system_prompt(is_doc_grounded=True)
        prompt_doc = (
            f"INGESTED DOCUMENT CONTENT:\n\"\"\"\n{doc_text[:14000]}\n\"\"\"\n\n"
            f"STUDY FOCUS / STUDENT QUERY: {prompt if prompt else 'Create an exhaustive master study guide grounded in this document.'}\n\n"
            f"TASK:\n"
            f"Transform the document into an EXHAUSTIVE, HIGH-YIELD MASTER REFERENCE GUIDE & SOLVED EXAM BLUEPRINT.\n"
            f"Strictly ground all concepts, theorems, and formulas in the document. Follow all 8 modules with complete LaTeX equations, 10 solved problems based directly on the document, and a predicted mock exam paper."
        )

        raw_res = orchestrator._call_model(reasoning_llm, prompt_doc, gen_tokens, gen_temp, system_prompt=sys_prompt_doc)
        cleaned_res = orchestrator._strip_thinking(raw_res)

        return (
            f"# 🎓 Document-Grounded Master Reference Guide & Exam Blueprint\n\n"
            f"---\n\n"
            f"{cleaned_res}"
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
    def _build_master_study_system_prompt(is_doc_grounded=False):
        base = (
            "You are a Distinguished Chaired Professor, Academician, and Principal Author of University Graduate Reference Textbooks.\n"
            "Your task is to write a DEFINITIVE, EXHAUSTIVE, 10-20 PAGE LEVEL MASTER REFERENCE TEXTBOOK CHAPTER for students.\n\n"
            "MANDATORY 8-MODULE STRUCTURAL FORMAT:\n\n"
            "### 1. 🎓 Executive Epistemological Overview & Core Axioms\n"
            "- Foundational axioms, historical context, and physical/computational intuition.\n"
            "- 2 intuitive real-world analogies explaining non-obvious dynamics.\n"
            "- Bulleted list of core competencies the student will master.\n\n"
            "### 2. 📚 Exhaustive Conceptual Breakdown & Sub-Topic Analysis\n"
            "- Divide into at least 4 detailed sub-sections (e.g., 2.1, 2.2, 2.3, 2.4).\n"
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
            "- **Section A:** 5 Multiple Choice / Conceptual Questions (with answer key & explanations).\n"
            "- **Section B:** 3 Short-Answer Numerical Questions.\n"
            "- **Section C:** 2 Long-Form Comprehensive Proof / Derivation Questions with official step-by-step marking rubrics.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. ALWAYS render all math in standard LaTeX ($$ ... $$ for display, $ ... $ for inline variables).\n"
            "2. Never abbreviate or use placeholders. Write full, continuous, publication-grade academic prose."
        )
        if is_doc_grounded:
            base += "\n3. STRICT GROUNDING: Every theorem, formula, and explanation must be strictly grounded in the ingested document."
        return base
