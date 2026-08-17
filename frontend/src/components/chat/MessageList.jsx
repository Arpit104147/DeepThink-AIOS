import React, { useEffect, useRef, useState } from "react";
import ThinkingBlock from "./ThinkingBlock";
import UserMessage from "./UserMessage";
import MessageRenderer from "./MessageRenderer";

const MessageList = ({ history, isGenerating, currentLogs, currentStream, displayText }) => {
  const bottomRef = useRef(null);
  const containerRef = useRef(null);
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, currentStream, currentLogs]);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isUp = scrollHeight - scrollTop - clientHeight > 150;
    setShowScrollBottom(isUp);
  };

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    setShowScrollBottom(false);
  };

  if (history.length === 0) {
    return (
      <div className="empty-state">
        <h1>
          {displayText}
          <span className="cursor-blink">|</span>
        </h1>
      </div>
    );
  }

  return (
    <div className="chat-messages" ref={containerRef} onScroll={handleScroll}>
      {history.map((msg, i) => (
        <div key={i} className={`msg-row ${msg.type}`}>
          <div className={`msg-avatar ${msg.type}`}>
            {msg.type === "user" ? "A" : "✦"}
          </div>
          <div className="msg-body">
            {msg.type === "ai" && msg.logs && msg.logs.length > 0 && (
              <ThinkingBlock logs={msg.logs} isActive={false} />
            )}
            {msg.type === "user" ? (
              <UserMessage text={msg.text} image={msg.image} />
            ) : (
              <MessageRenderer text={msg.text} animate={false} />
            )}
          </div>
        </div>
      ))}

      {isGenerating && (
        <div className="msg-row ai">
          <div className="msg-avatar ai">✦</div>
          <div className="msg-body">
            <ThinkingBlock logs={currentLogs} isActive={true} />
            {currentStream && <MessageRenderer text={currentStream} />}
          </div>
        </div>
      )}

      {showScrollBottom && (
        <button className="scroll-bottom-btn" onClick={scrollToBottom} title="Scroll to bottom">
          ↓
        </button>
      )}

      <div ref={bottomRef} />
    </div>
  );
};

export default MessageList;
