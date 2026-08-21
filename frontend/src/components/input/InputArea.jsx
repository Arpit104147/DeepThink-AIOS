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

        {/* Popup menu */}
        {menuOpen && (
          <div className="popup-menu">
            <button className="popup-item" onClick={() => { setMenuOpen(false); setTimeout(() => fileInputRef.current?.click(), 50); }}>
              <span className="popup-icon">📷</span> Upload photo or file
            </button>
            <div className="popup-divider" />
            <div className="popup-item popup-search-section">
              <div className="popup-search-header">
                <span><span className="popup-icon">🌐</span> Web Search</span>
                {searchMode !== "off" && (
                  <span
                    className="popup-search-badge"
                    style={{
                      background: `${SEARCH_MODE_COLORS[searchMode]}22`,
                      color: SEARCH_MODE_COLORS[searchMode],
                    }}
                  >
                    {searchMode}
                  </span>
                )}
              </div>
              <div className="segmented-control-container">
                <div
                  className="segmented-slider-backdrop"
                  style={{
                    width: `calc(${100 / SEARCH_MODES.length}% - 3px)`,
                    transform: `translateX(calc(${SEARCH_MODES.indexOf(searchMode)} * (100% + 3px)))`,
                    background: searchMode !== "off"
                      ? `linear-gradient(135deg, ${SEARCH_MODE_COLORS[searchMode]}cc, ${SEARCH_MODE_COLORS[searchMode]}88)`
                      : "rgba(255,255,255,0.06)",
                    boxShadow: searchMode !== "off"
                      ? `0 0 12px ${SEARCH_MODE_COLORS[searchMode]}44, inset 0 1px 0 rgba(255,255,255,0.1)`
                      : "inset 0 1px 0 rgba(255,255,255,0.05)",
                  }}
                />
                {SEARCH_MODES.map((m) => (
                  <button
                    key={m}
                    className={`segmented-button ${searchMode === m ? "active" : ""}`}
                    onClick={(e) => { e.stopPropagation(); setSearchMode(m); }}
                    style={{
                      color: searchMode === m ? "#ffffff" : "#777",
                      fontSize: m === "off" ? "0.68rem" : "0.65rem",
                    }}
                  >
                    {SEARCH_MODE_LABELS[m]}
                  </button>
                ))}
              </div>
            </div>
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
  );
};

export default InputArea;
