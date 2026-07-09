/**
 * Generate the method illustration SVGs with real LaTeX typesetting.
 *
 * Formulas are typeset by MathJax (TeX fonts) into SVG fragments and composed
 * into hand-designed scenes. Run from the docs/ directory:
 *
 *     node scripts/gen_method_svgs.mjs
 */

import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "methods");
mkdirSync(OUT, { recursive: true });

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const mjDoc = mathjax.document("", {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: "none" }),
});

/** Typeset TeX and return a positioned <svg> fragment. */
function tex(texStr, { x, y, size = 16, color = "#0f172a", anchor = "middle" } = {}) {
  const node = mjDoc.convert(texStr, { display: false });
  let s = adaptor.outerHTML(node);
  s = s.slice(s.indexOf("<svg"), s.lastIndexOf("</svg>") + 6);
  const ex = size * 0.485;
  const w = parseFloat(s.match(/width="([\d.]+)ex"/)[1]) * ex;
  const h = parseFloat(s.match(/height="([\d.]+)ex"/)[1]) * ex;
  s = s.replace(/width="[\d.]+ex"/, `width="${w.toFixed(1)}"`);
  s = s.replace(/height="[\d.]+ex"/, `height="${h.toFixed(1)}"`);
  const left = anchor === "middle" ? x - w / 2 : anchor === "end" ? x - w : x;
  return `<g transform="translate(${left.toFixed(1)},${(y - h / 2).toFixed(1)})" style="color:${color}">${s}</g>`;
}

const FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif";
const INDIGO = "#4f46e5";
const CYAN = "#0891b2";
const ROSE = "#e11d48";
const SLATE = "#64748b";
const INK = "#0f172a";

const panel = (w, h) =>
  `<rect x="8" y="8" width="${w - 16}" height="${h - 16}" rx="18" fill="#f8fafc" stroke="#e2e8f0"/>`;

const title = (t, w, y = 48) =>
  `<text x="${w / 2}" y="${y}" text-anchor="middle" font-size="19" fill="${INK}" font-weight="600">${t}</text>`;

const label = (t, x, y, { size = 13, color = SLATE, anchor = "middle", weight = 400, style = "" } = {}) =>
  `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" fill="${color}" font-weight="${weight}" ${style}>${t}</text>`;

const markers = `
  <marker id="mIndigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${INDIGO}"/></marker>
  <marker id="mCyan" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${CYAN}"/></marker>
  <marker id="mRose" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${ROSE}"/></marker>
  <marker id="mSlate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${SLATE}"/></marker>`;

function write(name, body) {
  writeFileSync(join(OUT, name), body);
  console.log(`  docs/public/methods/${name}`);
}

/* ==================================================================== causal */
function causal() {
  const W = 960, H = 340;
  const chunks = 12, x0 = 120, x1 = 840, y0 = 158, hBar = 60;
  const cw = (x1 - x0) / chunks;
  let seps = "";
  for (let i = 1; i < chunks; i++)
    seps += `<line x1="${x0 + i * cw}" y1="${y0}" x2="${x0 + i * cw}" y2="${y0 + hBar}" stroke="#ffffff" stroke-width="3" opacity="0.85"/>`;

  write("causal_weights.svg", `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="${FONT}">
  <defs>
    <linearGradient id="flow" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="${INDIGO}"/><stop offset="100%" stop-color="#06b6d4"/>
    </linearGradient>
    ${markers}
  </defs>
  <style>
    .cover { animation: sweep 9s ease-in-out infinite; }
    @keyframes sweep { 0%,12% { width: ${x1 - x0}px; } 72%,100% { width: 0px; } }
    .front { animation: front 9s ease-in-out infinite; }
    @keyframes front {
      0%,12% { transform: translateX(0px); opacity: 1; }
      72% { transform: translateX(${x1 - x0}px); opacity: 1; }
      78% { transform: translateX(${x1 - x0}px); opacity: 0; }
      95% { transform: translateX(0px); opacity: 0; }
      100% { transform: translateX(0px); opacity: 1; }
    }
  </style>
  ${panel(W, H)}
  ${title("Causal gates open forward in time", W)}
  ${tex("w_i \\;=\\; \\exp\\Big(-\\varepsilon \\sum_{k<i} \\mathcal{L}_r(t_k,\\theta)\\Big)", { x: W / 2, y: 92, size: 16.5, color: "#334155" })}

  <rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${hBar}" rx="12" fill="url(#flow)"/>
  ${seps}
  <g transform="translate(${x1},${y0}) scale(-1,1)">
    <rect class="cover" x="0" y="0" width="${x1 - x0}" height="${hBar}" rx="12" fill="#e2e8f0"/>
  </g>
  <rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${hBar}" rx="12" fill="none" stroke="#cbd5e1"/>

  <g class="front">
    <line x1="${x0}" y1="${y0 - 14}" x2="${x0}" y2="${y0 + hBar + 14}" stroke="${ROSE}" stroke-width="2.5" stroke-dasharray="4 3"/>
    <path d="M${x0},${y0 - 20} l-6,-10 h12 z" fill="${ROSE}"/>
    <text x="${x0}" y="${y0 - 38}" text-anchor="middle" font-size="12.5" fill="${ROSE}" font-weight="600">convergence front</text>
  </g>

  <line x1="${x0}" y1="${y0 + hBar + 34}" x2="${x1}" y2="${y0 + hBar + 34}" stroke="${SLATE}" stroke-width="1.5" marker-end="url(#mSlate)"/>
  ${tex("t = 0", { x: x0 + 18, y: y0 + hBar + 56, size: 14, color: SLATE, anchor: "middle" })}
  ${tex("t = T", { x: x1 - 18, y: y0 + hBar + 56, size: 14, color: SLATE, anchor: "middle" })}
  ${label("time-sorted collocation chunks", W / 2, y0 + hBar + 60, { size: 13 })}

  <rect x="252" y="${H - 40}" width="16" height="16" rx="5" fill="url(#flow)"/>
  ${tex("\\text{converged: gate open } (w_i \\approx 1)", { x: 370, y: H - 31, size: 13.5, color: "#334155", anchor: "start" }).replace('translate(370', 'translate(278')}
  <rect x="552" y="${H - 40}" width="16" height="16" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/>
  ${tex("\\text{still gated } (w_i \\approx 0)", { x: 670, y: H - 31, size: 13.5, color: "#334155", anchor: "start" }).replace('translate(670', 'translate(578')}
</svg>\n`);
}

/* ============================================================ loss balancing */
function lossBalancing() {
  const W = 960, H = 410;
  const L = { cx: 250, cy: 235 };
  const R = { cx: 714, cy: 235 };

  write("loss_balancing.svg", `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="${FONT}">
  <defs>${markers}</defs>
  ${panel(W, H)}

  ${label("Unweighted: one gradient drowns the rest", L.cx, 54, { size: 17, color: INK, weight: 600 })}
  <g>
    <circle cx="${L.cx}" cy="${L.cy}" r="5" fill="${INK}"/>
    <line x1="${L.cx}" y1="${L.cy}" x2="${L.cx - 30}" y2="${L.cy - 24}" stroke="${INDIGO}" stroke-width="3.5" marker-end="url(#mIndigo)"/>
    ${tex("\\nabla_{\\theta}\\mathcal{L}_{ic}", { x: L.cx - 62, y: L.cy - 42, size: 15, color: INDIGO })}
    <line x1="${L.cx}" y1="${L.cy}" x2="${L.cx - 52}" y2="${L.cy + 48}" stroke="${CYAN}" stroke-width="3.5" marker-end="url(#mCyan)"/>
    ${tex("\\nabla_{\\theta}\\mathcal{L}_{bc}", { x: L.cx - 88, y: L.cy + 70, size: 15, color: CYAN })}
    <line x1="${L.cx}" y1="${L.cy}" x2="${L.cx + 178}" y2="${L.cy - 126}" stroke="${ROSE}" stroke-width="4.5" marker-end="url(#mRose)"/>
    ${tex("\\nabla_{\\theta}\\mathcal{L}_{r}", { x: L.cx + 168, y: L.cy - 148, size: 16, color: ROSE })}
    <line x1="${L.cx}" y1="${L.cy}" x2="${L.cx + 124}" y2="${L.cy - 96}" stroke="${SLATE}" stroke-width="2.5" stroke-dasharray="6 4" marker-end="url(#mSlate)"/>
    ${label("update &#8776; residual only", L.cx + 76, L.cy + 8, { size: 13, anchor: "start" })}
  </g>
  ${label("initial &amp; boundary conditions barely move", L.cx, H - 28, { size: 13.5 })}

  <line x1="480" y1="42" x2="480" y2="${H - 34}" stroke="#e2e8f0" stroke-width="2"/>

  ${label("Balanced: equal gradient norms", R.cx, 54, { size: 17, color: INK, weight: 600 })}
  ${tex("\\hat{\\lambda}_i = \\frac{\\sum_j \\lVert\\nabla_{\\theta}\\mathcal{L}_j\\rVert}{\\lVert\\nabla_{\\theta}\\mathcal{L}_i\\rVert}", { x: R.cx, y: 96, size: 16, color: "#334155" })}
  <g>
    <circle cx="${R.cx}" cy="${R.cy + 20}" r="5" fill="${INK}"/>
    <line x1="${R.cx}" y1="${R.cy + 20}" x2="${R.cx - 104}" y2="${R.cy - 58}" stroke="${INDIGO}" stroke-width="4" marker-end="url(#mIndigo)"/>
    ${tex("\\hat{\\lambda}_{ic}\\nabla_{\\theta}\\mathcal{L}_{ic}", { x: R.cx - 138, y: R.cy - 78, size: 15, color: INDIGO })}
    <line x1="${R.cx}" y1="${R.cy + 20}" x2="${R.cx - 86}" y2="${R.cy + 104}" stroke="${CYAN}" stroke-width="4" marker-end="url(#mCyan)"/>
    ${tex("\\hat{\\lambda}_{bc}\\nabla_{\\theta}\\mathcal{L}_{bc}", { x: R.cx - 146, y: R.cy + 108, size: 15, color: CYAN })}
    <line x1="${R.cx}" y1="${R.cy + 20}" x2="${R.cx + 130}" y2="${R.cy - 12}" stroke="${ROSE}" stroke-width="4" marker-end="url(#mRose)"/>
    ${tex("\\hat{\\lambda}_{r}\\nabla_{\\theta}\\mathcal{L}_{r}", { x: R.cx + 168, y: R.cy - 30, size: 15, color: ROSE })}
    <line x1="${R.cx}" y1="${R.cy + 20}" x2="${R.cx - 26}" y2="${R.cy + 12}" stroke="${SLATE}" stroke-width="2.5" stroke-dasharray="6 4" marker-end="url(#mSlate)"/>
    ${label("update hears every term", R.cx + 10, R.cy + 52, { size: 13, anchor: "start" })}
  </g>
  ${label("smoothed by a running average, refreshed every ~1000 steps", R.cx, H - 28, { size: 13.5 })}
</svg>\n`);
}

/* =============================================================== pseudo-time */
function pseudoTime() {
  const W = 960, H = 380;
  const beat = (dx, tt) => `
    <g transform="translate(${dx},96)">
      ${label(tt, 140, 0, { size: 14.5, color: "#334155", weight: 600 })}
    </g>`;

  write("pseudo_time.svg", `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="${FONT}">
  <defs>${markers}</defs>
  ${panel(W, H)}
  ${title("How pseudo-time stepping exposes spurious solutions", W)}

  ${beat(40, "1 &#183; the loss looks tiny")}
  <g transform="translate(40,96)">
    <line x1="0" y1="112" x2="280" y2="112" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 4"/>
    <path d="M0,112 H150 C160,112 163,42 170,42 C177,42 180,112 190,112 H280" fill="none" stroke="${ROSE}" stroke-width="3"/>
    <g fill="${INDIGO}">
      <circle cx="22" cy="112" r="4.5"/><circle cx="64" cy="112" r="4.5"/><circle cx="106" cy="112" r="4.5"/>
      <circle cx="143" cy="112" r="4.5"/><circle cx="198" cy="112" r="4.5"/><circle cx="240" cy="112" r="4.5"/><circle cx="272" cy="112" r="4.5"/>
    </g>
    ${tex("\\text{defect of width } h", { x: 170, y: 26, size: 13, color: ROSE })}
    ${label("the sharp layer slips between", 140, 164, { size: 12.5 })}
    ${label("the collocation points", 140, 180, { size: 12.5 })}
  </g>

  <path d="M336,200 h22" stroke="#94a3b8" stroke-width="2.5" fill="none"/><path d="M358,200 l-8,-5 v10 z" fill="#94a3b8"/>

  ${beat(378, "2 &#183; one step amplifies it")}
  <g transform="translate(378,96)">
    <line x1="0" y1="112" x2="280" y2="112" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 4"/>
    <path d="M0,112 H150 C160,112 163,42 170,42 C177,42 180,112 190,112 H280" fill="none" stroke="${ROSE}" stroke-width="2" opacity="0.25"/>
    <path d="M0,112 H128 C142,112 146,16 160,16 C170,16 168,182 182,182 C194,182 198,112 212,112 H280" fill="none" stroke="${INDIGO}" stroke-width="3"/>
    ${tex("u - \\tau\\,\\mathcal{R}[u]", { x: 140, y: 208, size: 15, color: INDIGO })}
    ${tex("O(h^{-1}) \\to O(\\tau^2 h^{-3})", { x: 232, y: 24, size: 12.5, color: SLATE })}
  </g>

  <path d="M674,200 h22" stroke="#94a3b8" stroke-width="2.5" fill="none"/><path d="M696,200 l-8,-5 v10 z" fill="#94a3b8"/>

  ${beat(716, "3 &#183; resampling catches it").replace('140, 0', '110, 0').replace('x="140"', 'x="110"')}
  <g transform="translate(716,96)">
    <line x1="0" y1="112" x2="220" y2="112" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 4"/>
    <path d="M0,112 H100 C112,112 115,16 126,16 C134,16 132,182 143,182 C152,182 155,112 166,112 H220" fill="none" stroke="${INDIGO}" stroke-width="3"/>
    <g fill="${CYAN}">
      <circle cx="18" cy="112" r="4.5"/><circle cx="58" cy="112" r="4.5"/><circle cx="92" cy="112" r="4.5"/><circle cx="196" cy="112" r="4.5"/>
    </g>
    <circle cx="125" cy="18" r="6.5" fill="${ROSE}"/><circle cx="125" cy="18" r="11" fill="none" stroke="${ROSE}" opacity="0.45"/>
    <circle cx="142" cy="176" r="6.5" fill="${ROSE}"/><circle cx="142" cy="176" r="11" fill="none" stroke="${ROSE}" opacity="0.45"/>
    ${label("fresh points see a huge residual &#8212;", 110, 208, { size: 12.5 })}
    ${label("training escapes the spurious solution", 110, 224, { size: 12.5 })}
  </g>

  ${tex("\\mathcal{L}_{\\mathrm{pts}}(\\theta) = \\frac{1}{N}\\sum_{i=1}^{N} \\Big| \\tfrac{u_{\\theta}(x_i) - u_{\\theta^{k-1}}(x_i)}{\\tau} + \\mathcal{R}[u_{\\theta}](x_i) \\Big|^2 \\quad \\text{on freshly resampled points}", { x: W / 2, y: H - 34, size: 15, color: "#334155" })}
</svg>\n`);
}

/* ================================================================= piratenet */
function pirateNet() {
  const W = 960, H = 430;
  const yMain = 210;

  const dense = (x, texLabel) => `
    <rect x="${x}" y="${yMain - 24}" width="96" height="48" rx="12" fill="${INDIGO}"/>
    ${tex(texLabel, { x: x + 48, y: yMain, size: 15, color: "#ffffff" })}`;

  const gate = (cx) => `
    <circle cx="${cx}" cy="${yMain}" r="26" fill="#cffafe" stroke="${CYAN}" stroke-width="1.5"/>
    ${tex("\\odot", { x: cx, y: yMain, size: 17, color: CYAN })}`;

  write("piratenet_block.svg", `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="${FONT}">
  <defs>
    <linearGradient id="pn" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="${INDIGO}"/><stop offset="100%" stop-color="#6366f1"/></linearGradient>
    ${markers}
  </defs>
  ${panel(W, H)}
  ${title("One PirateNet residual block", W)}

  <rect x="330" y="82" width="300" height="44" rx="22" fill="#ecfeff" stroke="${CYAN}" stroke-width="1.5"/>
  ${tex("\\text{gates } U, V \\;\\leftarrow\\; \\Phi(x)", { x: 480, y: 104, size: 15.5, color: CYAN })}
  <line x1="408" y1="126" x2="356" y2="${yMain - 32}" stroke="${CYAN}" stroke-width="2" marker-end="url(#mCyan)"/>
  <line x1="552" y1="126" x2="604" y2="${yMain - 32}" stroke="${CYAN}" stroke-width="2" marker-end="url(#mCyan)"/>

  <rect x="40" y="${yMain - 24}" width="86" height="48" rx="24" fill="#eef2ff" stroke="#c7d2fe"/>
  ${tex("x^{(l)}", { x: 83, y: yMain, size: 16, color: "#334155" })}

  ${dense(160, "\\sigma(W_1\\,\\cdot)")}
  ${gate(348)}
  ${dense(412, "\\sigma(W_2\\,\\cdot)")}
  ${gate(612)}
  ${dense(676, "\\sigma(W_3\\,\\cdot)")}

  <circle cx="832" cy="${yMain}" r="22" fill="#ffffff" stroke="${ROSE}" stroke-width="2.5"/>
  ${tex("+", { x: 832, y: yMain, size: 19, color: ROSE })}
  ${tex("x^{(l+1)}", { x: 908, y: yMain, size: 16, color: "#334155" })}

  <g stroke="${SLATE}" stroke-width="2">
    <line x1="126" y1="${yMain}" x2="152" y2="${yMain}" marker-end="url(#mSlate)"/>
    <line x1="256" y1="${yMain}" x2="314" y2="${yMain}" marker-end="url(#mSlate)"/>
    <line x1="374" y1="${yMain}" x2="404" y2="${yMain}" marker-end="url(#mSlate)"/>
    <line x1="508" y1="${yMain}" x2="578" y2="${yMain}" marker-end="url(#mSlate)"/>
    <line x1="638" y1="${yMain}" x2="668" y2="${yMain}" marker-end="url(#mSlate)"/>
    <line x1="854" y1="${yMain}" x2="876" y2="${yMain}" marker-end="url(#mSlate)"/>
  </g>
  <line x1="772" y1="${yMain}" x2="802" y2="${yMain}" stroke="${ROSE}" stroke-width="2.5" marker-end="url(#mRose)"/>
  ${tex("\\alpha\\, h^{(l)}", { x: 790, y: yMain - 40, size: 14, color: ROSE })}

  ${tex("f \\odot U + (1{-}f) \\odot V", { x: 348, y: yMain + 52, size: 13, color: CYAN })}
  ${tex("g \\odot U + (1{-}g) \\odot V", { x: 612, y: yMain + 52, size: 13, color: CYAN })}

  <path d="M83,${yMain + 24} C83,342 700,342 826,${yMain + 21}" fill="none" stroke="${ROSE}" stroke-width="3.5" marker-end="url(#mRose)"/>
  ${tex("\\text{adaptive skip: } (1-\\alpha)\\,x^{(l)}", { x: 452, y: 356, size: 15.5, color: ROSE })}

  ${tex("\\alpha \\text{ trainable, initialized to } 0 \\;\\Rightarrow\\; \\text{the block starts as the identity and the network deepens itself}", { x: W / 2, y: H - 32, size: 14.5, color: SLATE })}
</svg>\n`);
}

/* ====================================================================== soap */
function soap() {
  const W = 960, H = 440;
  const zig = "65,58.5 85.1,353.2 103.7,110.1 120.9,310.7 136.9,145.2 151.8,281.7 165.5,169.1 178.2,262 190,185.3 201,248.6 211.1,196.4 220.5,239.5 229.3,203.9 237.3,233.2 244.8,209.1 251.8,229 258.2,212.6";
  const newton = "65,58.5 216.2,147.3 284.3,187.3 314.9,205.3 328.7,213.4 334.9,217 337.7,218.7";
  const zigDots = zig.split(" ").filter((_, i) => i % 2 === 0);
  const newtonDots = newton.split(" ");

  write("soap_alignment.svg", `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="${FONT}">
  <defs>
    ${markers}
    <clipPath id="leftPane"><rect x="16" y="60" width="620" height="${H - 100}" rx="12"/></clipPath>
  </defs>
  ${panel(W, H)}
  ${label("Ill-conditioned loss: first-order zigzag vs preconditioned descent", 336, 50, { size: 17, color: INK, weight: 600 })}

  <g fill="none" stroke="#94a3b8" clip-path="url(#leftPane)">
    <ellipse cx="340" cy="220" rx="60" ry="19" opacity="0.55"/>
    <ellipse cx="340" cy="220" rx="110" ry="34" opacity="0.45"/>
    <ellipse cx="340" cy="220" rx="170" ry="53" opacity="0.35"/>
    <ellipse cx="340" cy="220" rx="241" ry="74" opacity="0.27"/>
    <ellipse cx="340" cy="220" rx="319" ry="99" opacity="0.2"/>
    <ellipse cx="340" cy="220" rx="400" ry="124" opacity="0.14"/>
  </g>

  <polyline points="${zig}" fill="none" stroke="${ROSE}" stroke-width="2.5" stroke-linejoin="round"/>
  <g fill="${ROSE}">${zigDots.map((p) => { const [a, b] = p.split(","); return `<circle cx="${a}" cy="${b}" r="3"/>`; }).join("")}</g>
  <polyline points="${newton}" fill="none" stroke="${INDIGO}" stroke-width="3" stroke-linejoin="round"/>
  <g fill="${INDIGO}">${newtonDots.map((p) => { const [a, b] = p.split(","); return `<circle cx="${a}" cy="${b}" r="3.5"/>`; }).join("")}</g>
  <path d="M340,209 l3.5,7.5 8,1 -5.8,5.6 1.4,8 -7.1,-3.9 -7.1,3.9 1.4,-8 -5.8,-5.6 8,-1 z" fill="#f59e0b"/>

  <line x1="90" y1="396" x2="122" y2="396" stroke="${ROSE}" stroke-width="2.5"/>
  ${label("Adam / gradient descent", 130, 401, { size: 13.5, color: "#334155", anchor: "start" })}
  <line x1="320" y1="396" x2="352" y2="396" stroke="${INDIGO}" stroke-width="3"/>
  ${label("SOAP (curvature-preconditioned)", 360, 401, { size: 13.5, color: "#334155", anchor: "start" })}

  <line x1="656" y1="42" x2="656" y2="${H - 32}" stroke="#e2e8f0" stroke-width="2"/>

  ${label("Gradient alignment", 806, 50, { size: 16, color: INK, weight: 600 })}
  <g transform="translate(806,150)">
    <circle r="4.5" fill="${INK}"/>
    <line x1="0" y1="0" x2="-78" y2="-42" stroke="${INDIGO}" stroke-width="3.5" marker-end="url(#mIndigo)"/>
    <line x1="0" y1="0" x2="82" y2="-34" stroke="${CYAN}" stroke-width="3.5" marker-end="url(#mCyan)"/>
    <line x1="0" y1="0" x2="2" y2="-40" stroke="${SLATE}" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#mSlate)"/>
  </g>
  ${tex("g_1", { x: 806 - 96, y: 150 - 56, size: 15, color: INDIGO })}
  ${tex("g_2", { x: 806 + 100, y: 150 - 48, size: 15, color: CYAN })}
  ${label("conflicting &#8212; small, wasteful step", 806, 186, { size: 12.5 })}

  <path d="M806,206 v22" stroke="#94a3b8" stroke-width="2.5" fill="none"/><path d="M806,232 l-5,-8 h10 z" fill="#94a3b8"/>
  ${tex("\\text{precondition with } H^{-1}", { x: 806, y: 252, size: 14, color: "#334155" })}

  <g transform="translate(806,364)">
    <circle r="4.5" fill="${INK}"/>
    <line x1="0" y1="0" x2="-34" y2="-72" stroke="${INDIGO}" stroke-width="3.5" marker-end="url(#mIndigo)"/>
    <line x1="0" y1="0" x2="36" y2="-70" stroke="${CYAN}" stroke-width="3.5" marker-end="url(#mCyan)"/>
    <line x1="0" y1="0" x2="1" y2="-84" stroke="${SLATE}" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#mSlate)"/>
  </g>
  ${tex("H^{-1}g_1", { x: 806 - 74, y: 364 - 84, size: 15, color: INDIGO })}
  ${tex("H^{-1}g_2", { x: 806 + 80, y: 364 - 82, size: 15, color: CYAN })}
  ${label("aligned &#8212; both terms improve", 806, 398, { size: 12.5 })}
</svg>\n`);
}

causal();
lossBalancing();
pseudoTime();
pirateNet();
soap();
