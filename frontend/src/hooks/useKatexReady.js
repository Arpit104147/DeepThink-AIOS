import { useState, useEffect } from "react";

/**
 * Custom hook that triggers a re-render when KaTeX finishes loading.
 * KaTeX loads with defer for faster page startup, so math formulas
 * auto-render only after the library is available.
 */
const useKatexReady = () => {
  const [ready, setReady] = useState(!!window.katex);

  useEffect(() => {
    if (window.katex) {
      setReady(true);
      return;
    }
    const onReady = () => setReady(true);
    window.addEventListener("katex-ready", onReady);
    return () => window.removeEventListener("katex-ready", onReady);
  }, []);

  return ready;
};

export default useKatexReady;
