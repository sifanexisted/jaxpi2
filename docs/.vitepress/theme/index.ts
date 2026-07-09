import { h } from "vue";
import DefaultTheme from "vitepress/theme";
import "./custom.css";

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      // Looping vorticity animation in the hero image slot
      "home-hero-image": () =>
        h("video", {
          src: "/jaxpi2/gallery/hero.mp4",
          autoplay: true,
          loop: true,
          muted: true,
          playsinline: true,
          width: 320,
          "aria-label": "Kolmogorov flow vorticity simulation",
        }),
    });
  },
};
