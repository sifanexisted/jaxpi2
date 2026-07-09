import { defineConfig } from "vitepress";

const guideSidebar = [
  { text: "Getting Started", link: "/getting-started" },
  {
    text: "Guide",
    items: [
      { text: "Core Concepts", link: "/guide/concepts" },
      { text: "The Trainer", link: "/guide/trainer" },
      { text: "Training Techniques", link: "/guide/training-techniques" },
      { text: "Architectures", link: "/guide/architectures" },
      { text: "Evaluation", link: "/guide/evaluation" },
      { text: "Checkpointing & Resume", link: "/guide/checkpointing" },
      { text: "Write Your Own Example", link: "/guide/new-example" },
    ],
  },
  {
    text: "Methods",
    items: [
      { text: "PirateNets", link: "/methods/piratenet" },
      { text: "Loss Balancing", link: "/methods/loss-balancing" },
      { text: "Causal Training", link: "/methods/causal-training" },
      { text: "SOAP & Gradient Alignment", link: "/methods/soap" },
      { text: "Pseudo-Time Stepping", link: "/methods/pseudo-time" },
    ],
  },
  { text: "Theory", link: "/theory" },
  { text: "About", link: "/about" },
];

export default defineConfig({
  title: "JAXPI",
  description:
    "Physics-informed neural networks at scale in JAX: multi-GPU training, advanced PINN algorithms, and 16 benchmark examples.",
  base: "/jaxpi2/",
  cleanUrls: true,
  lastUpdated: true,

  head: [["link", { rel: "icon", type: "image/svg+xml", href: "/jaxpi2/logo.svg" }]],

  markdown: {
    math: true,
  },

  vue: {
    // Raw-HTML media in markdown references public assets with explicit
    // /jaxpi2/... URLs; disable Vite's asset-import transform for them.
    template: {
      transformAssetUrls: {
        video: [],
        source: [],
        img: [],
        image: [],
        use: [],
      },
    },
  },

  themeConfig: {
    logo: "/logo.svg",

    nav: [
      { text: "Getting Started", link: "/getting-started" },
      {
        text: "Guide",
        items: [
          { text: "Core Concepts", link: "/guide/concepts" },
          { text: "The Trainer", link: "/guide/trainer" },
          { text: "Training Techniques", link: "/guide/training-techniques" },
          { text: "Architectures", link: "/guide/architectures" },
          { text: "Evaluation", link: "/guide/evaluation" },
          { text: "Checkpointing & Resume", link: "/guide/checkpointing" },
          { text: "Write Your Own Example", link: "/guide/new-example" },
        ],
      },
      {
        text: "Methods",
        items: [
          { text: "PirateNets", link: "/methods/piratenet" },
          { text: "Loss Balancing", link: "/methods/loss-balancing" },
          { text: "Causal Training", link: "/methods/causal-training" },
          { text: "SOAP & Gradient Alignment", link: "/methods/soap" },
          { text: "Pseudo-Time Stepping", link: "/methods/pseudo-time" },
        ],
      },
      { text: "Examples", link: "/examples/" },
      { text: "API", link: "/api/models" },
      { text: "Theory", link: "/theory" },
    ],

    sidebar: {
      "/guide/": guideSidebar,
      "/getting-started": guideSidebar,
      "/methods/": guideSidebar,
      "/theory": guideSidebar,
      "/about": guideSidebar,
      "/examples/": [
        { text: "Gallery", link: "/examples/" },
        {
          text: "Time-dependent (1D)",
          items: [
            { text: "Advection", link: "/examples/advection" },
            { text: "Allen–Cahn", link: "/examples/allen_cahn" },
            { text: "Burgers", link: "/examples/burgers" },
            { text: "Inviscid Burgers", link: "/examples/inviscid_burgers" },
            { text: "Korteweg–de Vries", link: "/examples/kdv" },
            { text: "Kuramoto–Sivashinsky", link: "/examples/ks" },
            { text: "Wave", link: "/examples/wave" },
          ],
        },
        {
          text: "Time-dependent (2D/3D)",
          items: [
            { text: "Ginzburg–Landau", link: "/examples/ginzburg_landau" },
            { text: "Gray–Scott", link: "/examples/gray_scott" },
            { text: "Kolmogorov Flow", link: "/examples/kolmogorov_flow" },
            { text: "Rayleigh–Taylor", link: "/examples/rayleigh_taylor" },
            { text: "Taylor–Green Vortex", link: "/examples/taylor_green" },
          ],
        },
        {
          text: "Boundary-value problems",
          items: [
            { text: "Lid-driven Cavity", link: "/examples/lid_driven_cavity" },
            { text: "Backward-facing Step", link: "/examples/bfs_flow" },
          ],
        },
      ],
      "/api/": [
        {
          text: "API Reference",
          items: [
            { text: "jaxpi.models", link: "/api/models" },
            { text: "jaxpi.training", link: "/api/training" },
            { text: "jaxpi.archs", link: "/api/archs" },
            { text: "jaxpi.samplers", link: "/api/samplers" },
            { text: "jaxpi.evaluator", link: "/api/evaluator" },
            { text: "jaxpi.checkpointing", link: "/api/checkpointing" },
            { text: "jaxpi.logging", link: "/api/logging" },
            { text: "jaxpi.utils", link: "/api/utils" },
          ],
        },
      ],
    },

    search: {
      provider: "local",
    },

    socialLinks: [{ icon: "github", link: "https://github.com/sifanexisted/jaxpi2" }],

    footer: {
      message: "Released under the Apache 2.0 License.",
      copyright: "Copyright © 2026 Sifan Wang",
    },

    outline: { level: [2, 3] },
  },
});
