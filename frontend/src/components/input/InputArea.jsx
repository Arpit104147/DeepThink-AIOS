import React, { useRef } from "react";
import { SEARCH_MODES, SEARCH_MODE_LABELS, SEARCH_MODE_COLORS } from "../../utils/constants";

/**
 * @component InputArea
 * Chat input area with textarea, file upload, search mode selector,
 * and send/stop controls.
 */
const InputArea = ({
  prompt,
  setPrompt,
  isGenerating,
  attachedImage,
  setAttachedImage,
  menuOpen,
  setMenuOpen,
  searchMode,
  setSearchMode,
  handleSend,
  handleStop,
  setSettingsOpen,
  textareaRef,
}) => {
  const fileInputRef = useRef(null);

  const handleTextareaInput = (e) => {
    setPrompt(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = e.target.scrollHeight + "px";
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      setAttachedImage(reader.result);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image") !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          const reader = new FileReader();
          reader.onloadend = () => setAttachedImage(reader.result);
          reader.readAsDataURL(file);
          e.preventDefault();
          break;
        }
      }
    }
  };

  return (
    <div className="input-area">
      {/* Hidden file input permanently mounted in DOM so onChange never gets lost */}
      <input
        type="file"
        accept="image/*,.pdf,application/pdf"
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={handleFileUpload}
      />

      <div className="input-wrapper">
        {/* Centered Quick-Access Mode Selector Bar */}
        <div className="quick-mode-bar">
          {SEARCH_MODES.map((m) => {
            const isActive = searchMode === m;
            const color = SEARCH_MODE_COLORS[m];
            return (
              <button
                key={m}
                type="button"
                className={`quick-mode-pill ${isActive ? "active" : ""}`}
                onClick={() => setSearchMode(m)}
                style={{
                  borderColor: isActive ? (color || "rgba(255,255,255,0.4)") : "transparent",
                  background: isActive
                    ? (color ? `${color}22` : "rgba(255,255,255,0.12)")
                    : "transparent",
                  color: isActive ? (color || "#ffffff") : "#8e8e93",
                  boxShadow: isActive && color ? `0 0 12px ${color}33, inset 0 0 8px ${color}15` : "none",
                }}
              >
                {SEARCH_MODE_LABELS[m]}
              </button>
            );
          })}
        </div>

        {/* Text Input Container */}
        <div className="input-box-container">
          {attachedImage && (
            <span className="image-badge">
              {attachedImage.includes("application/pdf") ? (
                <span style={{ fontSize: "1.1rem" }}>📄</span>
              ) : (
                <img src={attachedImage} alt="Attached" className="image-badge-preview" />
              )}
              <span className="image-badge-text">
                {attachedImage.includes("application/pdf") ? "PDF Document attached" : "Image attached"}
              </span>
              <button
                onClick={() => setAttachedImage(null)}
                className="image-badge-remove"
                title="Remove attachment"
              >
                ✕
              </button>
            </span>
          )}

          {/* Popup menu for upload & settings */}
          {menuOpen && (
            <div className="popup-menu">
              <button className="popup-item" onClick={() => { setMenuOpen(false); setTimeout(() => fileInputRef.current?.click(), 50); }}>
                <span className="popup-icon">📷</span> Upload photo or file
              </button>
              <div className="popup-divider" />
              <button className="popup-item" onClick={() => { setSettingsOpen(true); setMenuOpen(false); }}>
                <span className="popup-icon">⚙️</span> Settings
              </button>
            </div>
          )}

          <button
            className={`input-plus-btn ${menuOpen ? "active" : ""}`}
            onClick={() => setMenuOpen(!menuOpen)}
          >
            ＋
          </button>

          <textarea
            ref={textareaRef}
            className="input-box"
            rows={1}
            placeholder={
              searchMode === "study"
                ? "Enter study topic or attach PDF notes for comprehensive revision guide..."
                : searchMode === "prediction"
                ? "Enter topic or market symbol for predictive ML modeling..."
                : searchMode === "extreme"
                ? "Enter deep research query for multi-source academic survey..."
                : "Ask anything"
            }
            value={prompt}
            onChange={handleTextareaInput}
            onPaste={handlePaste}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(e); } }}
            disabled={isGenerating}
          />

          {isGenerating ? (
            <button className="send-btn stop" onClick={handleStop} title="Stop">■</button>
          ) : (
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={!prompt.trim() && !attachedImage}
              title="Send"
            >
              ↑
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default InputArea;
