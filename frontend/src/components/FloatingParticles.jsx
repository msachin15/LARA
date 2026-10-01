/**
 * Decorative background: 50 tiny dots that drift using the CSS `float` animation.
 * The dots are plain DOM nodes, created once when the component mounts
 * and removed when it unmounts so they do not leak.
 */
import React, { useRef, useEffect } from 'react';

const FloatingParticles = () => {
  // The div we attach the dots to.
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    const particles = [];
    const numParticles = 50;

    for (let i = 0; i < numParticles; i++) {
      const particle = document.createElement('div');
      particle.className = 'absolute w-1 h-1 bg-gray-500 rounded-full opacity-20';
      // Random start position, as a percentage of the parent.
      particle.style.left = `${Math.random() * 100}%`;
      particle.style.top = `${Math.random() * 100}%`;
      // 5–15 seconds so the dots do not all move in lockstep.
      particle.style.animation = `float ${5 + Math.random() * 10}s linear infinite`;
      particles.push(particle);
      mount.appendChild(particle);
    }

    // Cleanup: React runs this when the component leaves the page.
    return () => {
      particles.forEach(particle => {
        mount.removeChild(particle);
      });
    };
  }, []);

  return <div ref={mountRef} className="absolute inset-0 z-0" />;
};

export default FloatingParticles;