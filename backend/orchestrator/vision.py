import os
import io
import base64

class VisionEngine:
    """Multimodal Vision Engine & Native Page-by-Page PDF Extractor."""

    @staticmethod
    def transcribe_image(orchestrator, image_input, user_prompt=None, status_callback=None):
        if status_callback:
            status_callback("👁️ Processing image...", "info", "qwen_vl", 5)

        data_url = None
        raw_bytes = None
        is_pdf = False

        # ── Step 1: Decode & Normalize input data (Image or PDF) ──
        if isinstance(image_input, str):
            if image_input.startswith("data:"):
                data_url = image_input
                if "application/pdf" in image_input:
                    is_pdf = True
                try:
                    header, encoded = image_input.split(",", 1)
                    raw_bytes = base64.b64decode(encoded)
                except Exception:
                    pass
            elif os.path.exists(image_input):
                if image_input.lower().endswith(".pdf"):
                    is_pdf = True
                try:
                    with open(image_input, "rb") as f:
                        raw_bytes = f.read()
                except Exception:
                    pass
        elif isinstance(image_input, bytes):
            raw_bytes = image_input
            if raw_bytes.startswith(b"%PDF"):
                is_pdf = True

        # ── Step 2: Native PDF Document Extraction ──
        if is_pdf and raw_bytes:
            if status_callback:
                status_callback("📄 Parsing PDF Document pages...", "info", "qwen_vl", 15)
            pdf_text = VisionEngine._extract_pdf_text(raw_bytes)
            if pdf_text:
                return f"📄 Extracted PDF Document Content:\n{pdf_text}"

        # ── Step 3: PIL RGB Image Pre-Processing & Downsampling (1024x1024) ──
        if raw_bytes:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(raw_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize if exceeds max dimensions (1024x1024)
                max_dim = 1024
                if img.width > max_dim or img.height > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                jpeg_bytes = buf.getvalue()
                b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
                data_url = f"data:image/jpeg;base64,{b64_str}"
            except Exception as e:
                print(f"⚠️ PIL Image pre-processing warning: {e}")

        if not data_url and isinstance(image_input, str) and image_input.startswith("data:"):
            data_url = image_input

        if not data_url:
            return "Error: Could not decode image payload."

        # ── Step 4: Vision Model Execution ──
        try:
            vl_model = orchestrator._get_model("qwen_vl", required_ctx=4096)
        except Exception as ex:
            return f"⚠️ Vision model loading error: {str(ex)}"

        prompt_text = user_prompt if user_prompt else "Describe the text and contents of this image in detail."

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            response = vl_model.create_chat_completion(messages=messages, max_tokens=1024)
            text_out = response['choices'][0]['message']['content']
            return orchestrator._strip_thinking(text_out)
        except Exception as e:
            return f"👁️ Image transcription notice: {str(e)}"

    @staticmethod
    def _extract_pdf_text(pdf_bytes):
        """Native multi-engine PDF page-by-page text extractor."""
        # Try PyPDF2 / pypdf first
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages_text = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages_text.append(f"--- Page {idx+1} ---\n{txt.strip()}")
            if pages_text:
                return "\n\n".join(pages_text)
        except Exception:
            pass

        # Try PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for idx, page in enumerate(doc):
                txt = page.get_text() or ""
                if txt.strip():
                    pages_text.append(f"--- Page {idx+1} ---\n{txt.strip()}")
            if pages_text:
                return "\n\n".join(pages_text)
        except Exception:
            pass

        return None
