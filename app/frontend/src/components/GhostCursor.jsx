import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import "./GhostCursor.css";

export default function GhostCursor({
  color = "#35D7FF",
  trailLength = 24,
  inertia = 0.18,
  grainIntensity = 0.018,
  bloomStrength = 0.7,
  bloomRadius = 0.55,
  bloomThreshold = 0.05,
  brightness = 0.7,
  edgeIntensity = 0.25,
  maxDevicePixelRatio = 1.5,
  mixBlendMode = "screen",
}) {
  const ref = useRef(null);

  useEffect(() => {
    const container = ref.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();

    const camera = new THREE.OrthographicCamera(
      -1,
      1,
      1,
      -1,
      0,
      10
    );

    camera.position.z = 1;

    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });

    renderer.setPixelRatio(
      Math.min(window.devicePixelRatio || 1, maxDevicePixelRatio)
    );

    renderer.setSize(
      container.clientWidth,
      container.clientHeight
    );

    renderer.setClearColor(0x000000, 0);

    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.mixBlendMode = mixBlendMode;

    container.appendChild(renderer.domElement);

    const geometry = new THREE.BufferGeometry();

    const positions = new Float32Array(trailLength * 3);
    const sizes = new Float32Array(trailLength);

    for (let i = 0; i < trailLength; i += 1) {
      positions[i * 3] = -10;
      positions[i * 3 + 1] = -10;
      positions[i * 3 + 2] = 0;
      sizes[i] = 1 - i / trailLength;
    }

    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3)
    );

    geometry.setAttribute(
      "aSize",
      new THREE.BufferAttribute(sizes, 1)
    );

    const material = new THREE.PointsMaterial({
      color,
      size: 0.035,
      transparent: true,
      opacity: brightness,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    const cursor = new THREE.Vector3(-10, -10, 0);
    const target = new THREE.Vector3(-10, -10, 0);

    const history = Array.from(
      { length: trailLength },
      () => new THREE.Vector3(-10, -10, 0)
    );

    const onPointerMove = (event) => {
      const rect = container.getBoundingClientRect();

      const x =
        ((event.clientX - rect.left) / rect.width) * 2 - 1;

      const y =
        -((event.clientY - rect.top) / rect.height) * 2 + 1;

      target.set(x, y, 0);
    };

    window.addEventListener("pointermove", onPointerMove);

    const resize = () => {
      renderer.setSize(
        container.clientWidth,
        container.clientHeight
      );
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);

    let raf = 0;
    let active = true;

    const visibilityObserver = new IntersectionObserver(
      ([entry]) => {
        active = entry.isIntersecting;
      },
      { threshold: 0.01 }
    );

    visibilityObserver.observe(container);

    const animate = () => {
      raf = requestAnimationFrame(animate);

      if (!active) return;

      cursor.x += (target.x - cursor.x) * inertia;
      cursor.y += (target.y - cursor.y) * inertia;

      history[0].copy(cursor);

      for (let i = 1; i < history.length; i += 1) {
        history[i].lerp(history[i - 1], 0.32);
      }

      const attribute = geometry.attributes.position;

      history.forEach((point, index) => {
        attribute.setXYZ(
          index,
          point.x,
          point.y,
          0
        );
      });

      attribute.needsUpdate = true;

      material.size =
        0.018 +
        bloomStrength * 0.018 +
        edgeIntensity * 0.006;

      material.opacity =
        brightness * (1 - grainIntensity * 2);

      renderer.render(scene, camera);
    };

    raf = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(raf);
      visibilityObserver.disconnect();
      resizeObserver.disconnect();
      window.removeEventListener("pointermove", onPointerMove);

      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [
    color,
    trailLength,
    inertia,
    grainIntensity,
    bloomStrength,
    bloomRadius,
    bloomThreshold,
    brightness,
    edgeIntensity,
    maxDevicePixelRatio,
    mixBlendMode,
  ]);

  return <div ref={ref} className="ghost-cursor" />;
}