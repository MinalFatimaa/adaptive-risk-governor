import React, { useEffect, useMemo, useRef, useState } from "react";
import "./LetterGlitch.css";

const CHARACTERS =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<>[]{}//\\\\_+-=:.#";

export default function LetterGlitch({
  glitchSpeed = 75,
  centerVignette = true,
  outerVignette = true,
  smooth = true,
  glitchColors = ["#17384f", "#1c9fca", "#d6fbff"],
}) {
  const [text, setText] = useState("");
  const ref = useRef(null);

  const seed = useMemo(
    () =>
      Array.from({ length: 90 }, () =>
        CHARACTERS[
          Math.floor(Math.random() * CHARACTERS.length)
        ]
      ).join(""),
    []
  );

  useEffect(() => {
    let timer = 0;

    const tick = () => {
      const next = Array.from(seed)
        .map((char, index) => {
          const probability =
            index % 7 === 0 ? 0.65 : 0.08;

          if (Math.random() < probability) {
            return CHARACTERS[
              Math.floor(Math.random() * CHARACTERS.length)
            ];
          }

          return char;
        })
        .join("");

      setText(next);
      timer = window.setTimeout(tick, glitchSpeed);
    };

    tick();

    return () => window.clearTimeout(timer);
  }, [glitchSpeed, seed]);

  return (
    <div
      ref={ref}
      className={`letter-glitch ${
        smooth ? "smooth" : ""
      } ${centerVignette ? "center-vignette" : ""} ${
        outerVignette ? "outer-vignette" : ""
      }`}
      style={{
        "--glitch-a": glitchColors[0],
        "--glitch-b": glitchColors[1],
        "--glitch-c": glitchColors[2],
      }}
      aria-hidden="true"
    >
      <span>{text}</span>
    </div>
  );
}