import React, { useEffect, useRef } from "react";
import ThinkingBlock from "./ThinkingBlock";
import UserMessage from "./UserMessage";
import MessageRenderer from "./MessageRenderer";

const MessageList = ({ history, isGenerating, currentLogs, currentStream, displayText }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, currentStream, currentLogs]);

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
    <div className="chat-messages">
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
              <MessageRenderer text={msg.text} animate={i === history.length - 1 && !isGenerating} />
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

      <div ref={bottomRef} />
    </div>
  );
};

export default MessageList;
