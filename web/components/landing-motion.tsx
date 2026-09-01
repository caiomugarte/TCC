"use client";

import { useEffect } from "react";

export function LandingMotion() {
  useEffect(() => {
    const root = document.querySelector<HTMLElement>("[data-landing-page]");
    let disposed = false;
    let revert: (() => void) | undefined;

    if (!root || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return undefined;
    }

    const animate = async () => {
      const [{ gsap }, { ScrollTrigger }] = await Promise.all([
        import("gsap"),
        import("gsap/ScrollTrigger"),
      ]);

      if (disposed) {
        return;
      }

      gsap.registerPlugin(ScrollTrigger);

      const context = gsap.context(() => {
        const select = <T extends Element>(selector: string) =>
          Array.from(root.querySelectorAll<T>(selector));

        gsap
          .timeline({ defaults: { ease: "power2.out" } })
          .from(select(".hero-copy > *"), {
            duration: 0.5,
            opacity: 0,
            y: 18,
            stagger: 0.07,
          })
          .from(
            select(".hero-visual"),
            { duration: 0.65, opacity: 0, y: 24 },
            "-=0.35",
          )
          .from(
            select(".hero-visual .bar span"),
            {
              duration: 0.5,
              scaleX: 0,
              transformOrigin: "left center",
              stagger: 0.08,
            },
            "-=0.25",
          );

        select<HTMLElement>("[data-reveal]").forEach((element) => {
          gsap.from(element, {
            duration: 0.6,
            opacity: 0,
            y: 24,
            ease: "power2.out",
            scrollTrigger: {
              trigger: element,
              start: "top 86%",
              once: true,
            },
          });
        });
      }, root);

      revert = () => context.revert();
    };

    void animate();

    return () => {
      disposed = true;
      revert?.();
    };
  }, []);

  return null;
}
