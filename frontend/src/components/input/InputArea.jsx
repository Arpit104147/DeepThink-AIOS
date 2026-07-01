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
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => setAttachedImage(reader.result);
    reader.readAsDataURL(file);
  };

  return (
    <div className="input-area">
      <div className="input-wrapper">
        {attachedImage && (
          <span className="image-badge">
            📎 Image attached
            <button
              onClick={() => setAttachedImage(null)}
              className="image-badge-remove"
              title="Remove image"
            >
              ✕
            </button>
          </span>
        )}

        {/* Popup menu */}
        {menuOpen && (
          <div className="popup-menu">
            <input type="file" accept="image/*" ref={fileInputRef} style={{ display: "none" }} onChange={handleFileUpload} />
            <button className="popup-item" onClick={() => { fileInputRef.current?.click(); setMenuOpen(false); }}>
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
                    width: "calc(25% - 3px)",
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
                      fontSize: m === "off" ? "0.7rem" : "0.68rem",
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
          placeholder="Ask anything"
          value={prompt}
          onChange={handleTextareaInput}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) handleSend(e); }}
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
