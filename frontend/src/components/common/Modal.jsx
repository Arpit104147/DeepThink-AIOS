import React, { useEffect } from "react";
import { X } from "lucide-react";

/**
 * @component Modal
 * Unified modal primitive with consistent overlay, escape key handling,
 * click-outside-to-close, and header with title + close button.
 */
const Modal = ({ open, onClose, title, maxWidth = "800px", children }) => {
  useEffect(() => {
    if (!open) return;
    const handleEsc = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleEsc);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleEsc);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-primitive-overlay" onClick={onClose}>
      <div
        className="modal-primitive-container"
        style={{ "--modal-max-width": maxWidth }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-primitive-header">
          <h2 className="modal-primitive-title">{title}</h2>
          <button className="modal-primitive-close" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <div className="modal-primitive-body">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Modal;
