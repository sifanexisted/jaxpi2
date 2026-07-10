/**
 * Problem-setup diagrams (geometry + boundary conditions) for example pages,
 * in the same visual language as the method illustrations. Run from docs/:
 *
 *     node scripts/gen_setup_svgs.mjs
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

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "setup");
mkdirSync(OUT, { recursive: true });

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const mjDoc = mathjax.document("", {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: "none" }),
});

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

const label = (t, x, y, { size = 13, color = SLATE, anchor = "middle", weight = 400 } = {}) =>
  `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" fill="${color}" font-weight="${weight}">${t}</text>`;

const markers = `
  <marker id="mIndigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${INDIGO}"/></marker>
  <marker id="mCyan" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${CYAN}"/></marker>
  <marker id="mRose" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${ROSE}"/></marker>
  <marker id="mSlate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${SLATE}"/></marker>`;

const hatch = (id, color = SLATE) => `
  <pattern id="${id}" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
    <line x1="0" y1="0" x2="0" y2="7" stroke="${color}" stroke-width="1.6" opacity="0.55"/>
  </pattern>`;

function write(name, body) {
  writeFileSync(join(OUT, name), body);
  console.log(`  docs/public/setup/${name}`);
}

/* ============================================== backward-facing step */
function bfs() {
  const W = 980, H = 400;
  // channel box (x: 0..15 compressed, y: -0.5..0.5)
  const X0 = 120, X1 = 880, Ytop = 110, Ybot = 300, Ymid = (Ytop + Ybot) / 2;

  // parabolic inflow arrows on the upper half (u = 24 y (0.5 - y), y in (0, 0.5))
  let inflow = "";
  for (const f of [0.15, 0.3, 0.5, 0.7, 0.85]) {
    const y = Ymid - f * (Ymid - Ytop);
    const u = 24 * (f * 0.5) * (0.5 - f * 0.5); // physical u at this height
    const len = 26 + u * 44;
    inflow += `<line x1="${X0 - 64}" y1="${y}" x2="${X0 - 64 + len}" y2="${y}" stroke="${INDIGO}" stroke-width="2.4" marker-end="url(#mIndigo)"/>`;
  }
  // inflow envelope (half parabola)
  const env = [];
  for (let i = 0; i <= 24; i++) {
    const f = i / 24;
    const y = Ymid - f * (Ymid - Ytop);
    const u = 24 * (f * 0.5) * (0.5 - f * 0.5);
    env.push(`${(X0 - 64 + 26 + u * 44).toFixed(1)},${y.toFixed(1)}`);
  }
  const envelope = `<polyline points="${env.join(" ")}" fill="none" stroke="${INDIGO}" stroke-width="1.6" opacity="0.6" stroke-dasharray="4 3"/>`;

  // outflow arrows
  let outflow = "";
  for (const f of [0.22, 0.5, 0.78]) {
    const y = Ytop + f * (Ybot - Ytop);
    outflow += `<line x1="${X1 + 4}" y1="${y}" x2="${X1 + 52}" y2="${y}" stroke="${CYAN}" stroke-width="2.4" marker-end="url(#mCyan)"/>`;
  }

  // recirculation bubble behind the step
  const bubble = `<path d="M ${X0 + 24} ${Ybot - 26} C ${X0 + 60} ${Ybot - 66}, ${X0 + 128} ${Ybot - 62}, ${X0 + 150} ${Ybot - 22}"
      fill="none" stroke="${ROSE}" stroke-width="2.2" marker-end="url(#mRose)" opacity="0.9"/>`;

  write("bfs_setup.svg", `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="${FONT}">
  <defs>${markers}${hatch("hBfs")}</defs>
  <rect x="8" y="8" width="${W - 16}" height="${H - 16}" rx="18" fill="#f8fafc" stroke="#e2e8f0"/>

  <!-- solid walls (hatched bands outside the channel) -->
  <rect x="${X0}" y="${Ytop - 12}" width="${X1 - X0}" height="12" fill="url(#hBfs)"/>
  <rect x="${X0}" y="${Ybot}" width="${X1 - X0}" height="12" fill="url(#hBfs)"/>
  <rect x="${X0 - 12}" y="${Ymid}" width="12" height="${Ybot - Ymid + 12}" fill="url(#hBfs)"/>

  <!-- channel outline -->
  <line x1="${X0}" y1="${Ytop}" x2="${X1}" y2="${Ytop}" stroke="${INK}" stroke-width="2"/>
  <line x1="${X0}" y1="${Ybot}" x2="${X1}" y2="${Ybot}" stroke="${INK}" stroke-width="2"/>
  <line x1="${X0}" y1="${Ymid}" x2="${X0}" y2="${Ybot}" stroke="${INK}" stroke-width="2"/>
  <line x1="${X1}" y1="${Ytop}" x2="${X1}" y2="${Ybot}" stroke="${CYAN}" stroke-width="2" stroke-dasharray="6 4"/>

  <!-- axis break marks on both walls -->
  <g stroke="${SLATE}" stroke-width="1.6">
    <line x1="512" y1="${Ytop - 16}" x2="524" y2="${Ytop + 6}"/>
    <line x1="522" y1="${Ytop - 16}" x2="534" y2="${Ytop + 6}"/>
    <line x1="512" y1="${Ybot - 6}" x2="524" y2="${Ybot + 16}"/>
    <line x1="522" y1="${Ybot - 6}" x2="534" y2="${Ybot + 16}"/>
  </g>

  ${inflow}${envelope}${outflow}${bubble}

  <!-- labels -->
  ${label("parabolic inflow", X0 - 62, Ytop - 26, { color: INDIGO, weight: 600, anchor: "start" })}
  ${tex("u(y) = 24\\,y\\,(\\tfrac{1}{2} - y),\\; v = 0", { x: X0 + 120, y: Ytop - 26, size: 14, color: INDIGO, anchor: "start" })}
  ${label("no-slip walls", (X0 + X1) / 2 + 60, Ytop - 26, { color: SLATE, weight: 600 })}
  ${tex("u = v = 0", { x: (X0 + X1) / 2 + 165, y: Ytop - 26, size: 14, color: SLATE, anchor: "start" })}
  ${label("step face (no-slip)", X0 - 20, Ybot + 34, { anchor: "start", color: SLATE, weight: 600 })}
  ${label("recirculation", X0 + 88, Ybot - 76, { color: ROSE, weight: 600 })}
  ${label("reattachment", X0 + 210, Ybot + 34, { color: ROSE })}
  <circle cx="${X0 + 210}" cy="${Ybot}" r="4" fill="${ROSE}"/>
  ${label("outflow", X1 + 28, Ytop - 26, { color: CYAN, weight: 600 })}
  ${tex("p = 0", { x: X1 + 28, y: Ytop - 6, size: 14, color: CYAN })}

  <!-- dimensions -->
  <g stroke="${SLATE}" stroke-width="1.3">
    <line x1="${X1 + 62}" y1="${Ytop}" x2="${X1 + 62}" y2="${Ybot}" marker-start="url(#mSlate)" marker-end="url(#mSlate)"/>
    <line x1="${X0 - 34}" y1="${Ymid}" x2="${X0 - 34}" y2="${Ybot}" marker-start="url(#mSlate)" marker-end="url(#mSlate)"/>
    <line x1="${X0}" y1="${Ybot + 52}" x2="${X1}" y2="${Ybot + 52}" marker-start="url(#mSlate)" marker-end="url(#mSlate)"/>
  </g>
  ${tex("y = \\tfrac{1}{2}", { x: X0 - 46, y: Ytop - 18, size: 13, color: SLATE, anchor: "end" })}
  ${tex("y = -\\tfrac{1}{2}", { x: X0 - 46, y: Ybot + 18, size: 13, color: SLATE, anchor: "end" })}
  ${tex("1", { x: X1 + 80, y: Ymid, size: 14, color: SLATE, anchor: "start" })}
  ${tex("\\tfrac{1}{2}", { x: X0 - 48, y: (Ymid + Ybot) / 2 + 26, size: 13, color: SLATE, anchor: "end" })}
  ${tex("0 \\le x \\le 15", { x: (X0 + X1) / 2, y: Ybot + 72, size: 14, color: SLATE })}
  ${tex("\\mathrm{Re} = 800,\\;\\; \\nu = 1/800", { x: (X0 + X1) / 2, y: 52, size: 16, color: INK })}
</svg>
`);
}

/* ============================================== lid-driven cavity */
function ldc() {
  const W = 620, H = 560;
  const X0 = 150, X1 = 470, Ytop = 130, Ybot = 450;
  const cx = (X0 + X1) / 2, cy = (Ytop + Ybot) / 2;

  // moving-lid arrows
  let lid = "";
  for (const x of [X0 + 40, X0 + 130, X0 + 220]) {
    lid += `<line x1="${x}" y1="${Ytop - 22}" x2="${x + 58}" y2="${Ytop - 22}" stroke="${INDIGO}" stroke-width="2.6" marker-end="url(#mIndigo)"/>`;
  }

  // primary vortex (clockwise, since lid moves right)
  const vortex = `<path d="M ${cx - 92} ${cy + 6} A 95 88 0 1 1 ${cx + 60} ${cy + 76}"
      fill="none" stroke="${INDIGO}" stroke-width="2.6" marker-end="url(#mIndigo)" opacity="0.85"/>`;

  // bottom corner eddies (counter-rotating)
  const eddy = (ex, ey, sweep) => `<path d="M ${ex - 26 * sweep} ${ey} A 26 22 0 1 ${sweep === 1 ? 0 : 1} ${ex + 10 * sweep} ${ey + 16}"
      fill="none" stroke="${ROSE}" stroke-width="2" marker-end="url(#mRose)" opacity="0.9"/>`;

  write("ldc_setup.svg", `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="${FONT}">
  <defs>${markers}${hatch("hLdc")}</defs>
  <rect x="8" y="8" width="${W - 16}" height="${H - 16}" rx="18" fill="#f8fafc" stroke="#e2e8f0"/>

  <!-- no-slip walls (hatched outside) -->
  <rect x="${X0 - 12}" y="${Ytop}" width="12" height="${Ybot - Ytop}" fill="url(#hLdc)"/>
  <rect x="${X1}" y="${Ytop}" width="12" height="${Ybot - Ytop}" fill="url(#hLdc)"/>
  <rect x="${X0 - 12}" y="${Ybot}" width="${X1 - X0 + 24}" height="12" fill="url(#hLdc)"/>

  <!-- cavity outline; lid drawn in indigo -->
  <rect x="${X0}" y="${Ytop}" width="${X1 - X0}" height="${Ybot - Ytop}" fill="#ffffff" stroke="${INK}" stroke-width="2"/>
  <line x1="${X0}" y1="${Ytop}" x2="${X1}" y2="${Ytop}" stroke="${INDIGO}" stroke-width="4"/>

  ${lid}${vortex}${eddy(X0 + 42, Ybot - 34, 1)}${eddy(X1 - 42, Ybot - 34, -1)}

  <!-- corner singularities -->
  <circle cx="${X0}" cy="${Ytop}" r="5" fill="none" stroke="${ROSE}" stroke-width="1.8"/>
  <circle cx="${X1}" cy="${Ytop}" r="5" fill="none" stroke="${ROSE}" stroke-width="1.8"/>

  <!-- labels -->
  ${label("moving lid", X0 - 4, Ytop - 22, { color: INDIGO, weight: 600, size: 14, anchor: "end" })}
  ${tex("u = 1,\\; v = 0", { x: X1 + 8, y: Ytop - 48, size: 15, color: INDIGO, anchor: "end" })}
  ${label("no-slip walls", X0 - 36, cy, { color: SLATE, weight: 600, size: 13, anchor: "middle" })}
  ${tex("u = v = 0", { x: X0 - 36, y: cy + 20, size: 13, color: SLATE })}
  ${label("primary vortex", cx, cy - 4, { color: INDIGO, weight: 600 })}
  ${label("corner eddies", cx, Ybot - 58, { color: ROSE, weight: 600, size: 12.5 })}
  ${label("velocity", X1 + 16, Ytop - 20, { color: ROSE, size: 12, anchor: "start" })}
  ${label("discontinuities", X1 + 16, Ytop - 6, { color: ROSE, size: 12, anchor: "start" })}

  <!-- dimensions -->
  <g stroke="${SLATE}" stroke-width="1.3">
    <line x1="${X1 + 40}" y1="${Ytop}" x2="${X1 + 40}" y2="${Ybot}" marker-start="url(#mSlate)" marker-end="url(#mSlate)"/>
    <line x1="${X0}" y1="${Ybot + 42}" x2="${X1}" y2="${Ybot + 42}" marker-start="url(#mSlate)" marker-end="url(#mSlate)"/>
  </g>
  ${tex("1", { x: X1 + 56, y: cy, size: 14, color: SLATE, anchor: "start" })}
  ${tex("1", { x: cx, y: Ybot + 60, size: 14, color: SLATE })}
  ${tex("\\mathrm{Re} = U L / \\nu = 5000", { x: cx, y: 44, size: 16, color: INK })}
</svg>
`);
}

console.log("setup diagrams");
bfs();
ldc();
