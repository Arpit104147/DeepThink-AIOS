import React, { useState, useEffect, useRef } from "react";
import MessageRenderer from "./MessageRenderer";

/**
 * @component UserMessage
 * Renders a user's chat message with collapse/expand support
 * (truncates after 4 lines) and a copy-to-clipboard button.
 */
const UserMessage = ({ text }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isTruncated, setIsTruncated] = useState(false);
  const [copied, setCopied] = useState(false);
  const textRef = useRef(null);

  useEffect(() => {
    const checkTruncation = () => {
      if (textRef.current) {
        setIsTruncated(textRef.current.scrollHeight > 108);
      }
    };
    checkTruncation();
    const timer = setTimeout(checkTruncation, 50);
    window.addEventListener("resize", checkTruncation);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", checkTruncation);
    };
  }, [text]);

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="user-message-container">
      <div
        ref={textRef}
        className={`user-message-content ${isExpanded ? "expanded" : "collapsed"}`}
      >
        <MessageRenderer text={text} />
      </div>
      <div className="user-message-actions">
        {isTruncated && (
          <button
            className="user-message-toggle"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? "Show Less ▲" : "Show More ▼"}
          </button>
        )}
        <button className="user-copy-btn" onClick={handleCopy} title="Copy prompt">
          {copied ? (
            <>Copied! ✓</>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default UserMessage;
