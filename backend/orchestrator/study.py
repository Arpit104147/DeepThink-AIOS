import os
import re
import json
import base64
import io
from backend.sandbox import Sandbox

class StudyPipeline:
    """
    Exhaustive Pedagogical Master Study Guide & Educational Synthesis Engine.
    Strictly powered by the Reasoning LLM (DeepSeek-R1) for mathematical proofs,
    formula derivations, structured pedagogical depth, and student mastery.
    """

    @staticmethod
    def execute(orchestrator, prompt, mode="auto", selected_models=None, status_callback=None, attached_doc=None, attached_image=None):
        if status_callback:
            status_callback("🎓 Study Mode [Stage 1/3]: Initializing pedagogical reasoning engine...", "info", "deepseek_r1", 10)

        # 1. Strictly bind to the Reasoning LLM (DeepSeek-R1)
        ds_ctx, oc_ctx, router_ctx, gen_tokens, gen_temp = orchestrator._compute_headroom()
        reasoning_key = "deepseek_r1" if orchestrator._is_model_valid(orchestrator._get_model("deepseek_r1", required_ctx=4096)) else "vibethinker"
        reasoning_llm = orchestrator._get_model(reasoning_key, required_ctx=ds_ctx)
        reasoning_display = orchestrator._get_display_model_name(reasoning_key)

        # Boost generation token headroom for exhaustive, deeply detailed textbook notes
        study_gen_tokens = min(8192, max(4096, ds_ctx - 2000))
        study_temp = 0.2  # Low temperature for strict mathematical & factual precision

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
        """Sub-Mode A: Omni-Web Multi-Query Deep Scraper & Synthesis."""
        if status_callback:
            status_callback("🎓 Study Mode [Stage 1/3]: Ingesting multi-source curriculum & academic textbooks...", "info", "system", 20)

        raw_contexts = []
        topic_clean = re.sub(r"(teach me|explain|create study notes for|notes on|give me notes for|study guide for|a complete guide on)", "", prompt, flags=re.I).strip()
        if not topic_clean:
            topic_clean = prompt

        # Generate 4-5 targeted academic sub-queries to capture all dimensions
        sub_queries = [
            f"{topic_clean} foundational principles theoretical concepts overview",
            f"{topic_clean} mathematical formulas equations derivations proofs",
            f"{topic_clean} key definitions theorems summary table comparison",
            f"{topic_clean} practice problems worked solutions exam review questions"
        ]

        if hasattr(orchestrator, "web_search") and orchestrator.web_search:
            try:
                # Primary exhaustive scrape
                primary_res = orchestrator.web_search.search_and_scrape(topic_clean, max_results=8, max_scrapes=5)
                if isinstance(primary_res, dict) and not primary_res.get("empty", True):
                    raw_contexts.append(primary_res.get("context", ""))

                # Secondary parallel targeted sub-query scrape
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

        if status_callback:
            status_callback(f"🎓 Study Mode [Stage 2/3]: Structuring topic hierarchy & formulas with {reasoning_display}...", "info", reasoning_key, 50)

        sys_prompt = StudyPipeline._build_study_system_prompt()

        synthesis_prompt = (
            f"SOURCE KNOWLEDGE FEEDS & REFERENCE MATERIALS:\n{aggregated_context[:12000] if aggregated_context else 'Comprehensive Academic Curriculum Base'}\n\n"
            f"TARGET STUDY TOPIC / STUDENT REQUEST: {prompt}\n\n"
            f"INSTRUCTION:\n"
            f"Synthesize an EXHAUSTIVE, MASTER-CLASS STUDY GUIDE for this topic.\n"
            f"Do not write a brief summary. Write a complete, deeply educational, textbook-length master reference chapter "
            f"with highlighted display formulas, step-by-step proofs, structured summary tables, key takeaways, and practice problems with full solutions."
        )

        if status_callback:
            status_callback(f"🎓 Study Mode [Stage 3/3]: Synthesizing exhaustive textbook chapter & practice quiz with {reasoning_display}...", "info", reasoning_key, 75)

        raw_res = orchestrator._call_model(reasoning_llm, synthesis_prompt, gen_tokens, gen_temp, system_prompt=sys_prompt)
        cleaned_res = orchestrator._strip_thinking(raw_res)

        return f"# 🎓 Comprehensive Master Study Guide: {topic_clean.title()}\n\n{cleaned_res}"

    @staticmethod
    def _execute_doc_study(orchestrator, prompt, doc_text, reasoning_llm, reasoning_key, reasoning_display, gen_tokens, gen_temp, status_callback):
        """Sub-Mode B: Document-Grounded PDF / Slide Master Notes."""
        if status_callback:
            status_callback(f"🎓 Study Mode [Stage 1/2]: Ingesting uploaded document ({len(doc_text)} chars)...", "info", "system", 25)

        if status_callback:
            status_callback(f"🎓 Study Mode [Stage 2/2]: Synthesizing grounded master study notes with {reasoning_display}...", "info", reasoning_key, 60)

        sys_prompt = StudyPipeline._build_study_system_prompt(is_doc_grounded=True)

        synthesis_prompt = (
            f"INGESTED DOCUMENT CONTENT:\n\"\"\"\n{doc_text[:14000]}\n\"\"\"\n\n"
            f"STUDENT QUERY / STUDY FOCUS: {prompt if prompt else 'Create complete structured study notes for this document.'}\n\n"
            f"INSTRUCTION:\n"
            f"Transform the ingested document into an EXHAUSTIVE, HIGH-YIELD STUDY GUIDE.\n"
            f"Strictly ground all notes in the document content. Highlight all core theorems, extract exact formulas in LaTeX, "
            f"create comparison tables, and construct practice questions based directly on the document concepts."
        )

        raw_res = orchestrator._call_model(reasoning_llm, synthesis_prompt, gen_tokens, gen_temp, system_prompt=sys_prompt)
        cleaned_res = orchestrator._strip_thinking(raw_res)

        return f"# 🎓 Document-Grounded Master Study Notes\n\n{cleaned_res}"

    @staticmethod
    def _extract_document_text(payload):
        """Extract text from base64 PDF or raw text data URL."""
        if not payload or not isinstance(payload, str):
            return ""

        try:
            # Handle Base64 PDF
            if "application/pdf" in payload or payload.startswith("data:application/pdf;base64,"):
                b64_data = payload.split(",", 1)[-1] if "," in payload else payload
                pdf_bytes = base64.b64decode(b64_data)
                
                # 1. Try pypdf
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                    text_pages = [page.extract_text() or "" for page in reader.pages]
                    full_text = "\n\n".join([f"--- Page {i+1} ---\n{t}" for i, t in enumerate(text_pages) if t.strip()])
                    if full_text.strip():
                        return full_text
                except Exception:
                    pass

                # 2. Try fitz / PyMuPDF
                try:
                    import fitz
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    text_pages = [page.get_text() or "" for page in doc]
                    full_text = "\n\n".join([f"--- Page {i+1} ---\n{t}" for i, t in enumerate(text_pages) if t.strip()])
                    if full_text.strip():
                        return full_text
                except Exception:
                    pass

            # Handle plain text data URL
            if payload.startswith("data:text/"):
                b64_data = payload.split(",", 1)[-1] if "," in payload else payload
                return base64.b64decode(b64_data).decode("utf-8", errors="ignore")

            # Fallback if raw text string
            if len(payload) > 50 and not payload.startswith("data:image/"):
                return payload

        except Exception as e:
            print(f"Document text extraction error: {e}")

        return ""

    @staticmethod
    def _build_study_system_prompt(is_doc_grounded=False):
        """Constructs the high-yield pedagogical system prompt for DeepSeek-R1."""
        base_rules = (
            "You are a Distinguished Professor, Principal Scientist, and World-Class Educator.\n"
            "Your mission is to produce an EXHAUSTIVE, RIGOROUS, and PEDAGOGICALLY BRILLIANT MASTER STUDY GUIDE for students.\n\n"
            "MANDATORY STRUCTURAL FORMAT:\n\n"
            "### 1. 🎓 Executive Overview & Core Learning Objectives\n"
            "- High-level conceptual overview and intuitive real-world analogy.\n"
            "- Bulleted list of core competencies the student will master.\n\n"
            "### 2. 📚 Comprehensive Topic-by-Topic Master Notes\n"
            "- Divide into clear, numbered sub-topics (e.g. 2.1, 2.2, 2.3).\n"
            "- Provide deep, thorough, and clear explanations from first principles.\n"
            "- Define every technical term, mechanism, and physical/mathematical interpretation.\n\n"
            "### 3. 📐 Mathematical / Scientific Formulas & Rigorous Derivations\n"
            "- Present every important equation in centered LaTeX display blocks (`$$ ... $$`).\n"
            "- Provide step-by-step mathematical proofs or physical derivations for key formulas.\n"
            "- Define every variable, constant, and SI unit explicitly in bullet points.\n\n"
            "### 4. 💡 Critical Exam Key Takeaways & Common Pitfalls\n"
            "- **Exam Watchouts:** Specific tricky concepts, edge cases, and common student mistakes to avoid.\n"
            "- **Memory Mnemonics / Shortcuts:** High-yield memory aids for rapid recall.\n\n"
            "### 5. 📊 Comprehensive Summary & Comparison Table\n"
            "- Construct a dense, multi-column Markdown comparison table summarizing parameters, mechanisms, pros/cons, or classifications.\n\n"
            "### 6. 🧠 Rapid-Revision Flashcards\n"
            "- 4 to 6 Q&A flashcards for quick revision.\n\n"
            "### 7. 📝 Practice Problems & Conceptual Quiz (with Worked Solutions)\n"
            "- **Problem 1 (Foundational / Easy):** Question followed by step-by-step full solution.\n"
            "- **Problem 2 (Intermediate / Computational):** Question with numerical formula application and step-by-step solution.\n"
            "- **Problem 3 (Advanced / Challenging):** Conceptual/analytical problem with in-depth reasoning.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. ALWAYS render all math formulas in standard LaTeX using `$$ ... $$` for display equations and `$ ... $` for inline variables.\n"
            "2. Make the notes EXTENSIVE, IN-DEPTH, and EXHAUSTIVE. Never truncate, abbreviate, or use placeholders.\n"
            "3. Ensure the tone is inspiring, crystal-clear, and academic."
        )

        if is_doc_grounded:
            base_rules += "\n4. STRICT DOCUMENT GROUNDING: All concepts, formulas, and facts MUST be derived directly from the uploaded document without external hallucinations."

        return base_rules
