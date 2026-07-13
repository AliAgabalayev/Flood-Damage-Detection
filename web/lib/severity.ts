// Sequential color ramp for encoding flood severity (% of area flooded).
// Kept separate from the brand/active-state accent (--accent in globals.css)
// so a 5% reading and a 70% reading are visually distinguishable.

const STOPS: [number, string][] = [
  [0, "#d8cfa0"],
  [10, "#e0b25a"],
  [25, "#dc8a3c"],
  [45, "#cf6a2c"],
  [70, "#b8431f"],
  [100, "#8c1f14"],
];

const NEUTRAL = "#b0a090";

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex([r, g, b]: [number, number, number]): string {
  return "#" + [r, g, b].map((c) => Math.round(c).toString(16).padStart(2, "0")).join("");
}

/** Maps a 0-100 flood percentage to a color on the sequential severity ramp. */
export function severityColor(pct: number | null): string {
  if (pct === null || Number.isNaN(pct)) return NEUTRAL;
  const p = Math.max(0, Math.min(100, pct));

  for (let i = 0; i < STOPS.length - 1; i++) {
    const [p0, c0] = STOPS[i];
    const [p1, c1] = STOPS[i + 1];
    if (p >= p0 && p <= p1) {
      const t = p1 === p0 ? 0 : (p - p0) / (p1 - p0);
      const rgb0 = hexToRgb(c0);
      const rgb1 = hexToRgb(c1);
      const mixed: [number, number, number] = [
        rgb0[0] + (rgb1[0] - rgb0[0]) * t,
        rgb0[1] + (rgb1[1] - rgb0[1]) * t,
        rgb0[2] + (rgb1[2] - rgb0[2]) * t,
      ];
      return rgbToHex(mixed);
    }
  }
  return STOPS[STOPS.length - 1][1];
}

/** Human-readable severity tier, used alongside color so the encoding isn't color-only. */
export function severityLabel(pct: number | null): string {
  if (pct === null || Number.isNaN(pct)) return "No data";
  if (pct < 10) return "Low";
  if (pct < 25) return "Moderate";
  if (pct < 45) return "High";
  if (pct < 70) return "Severe";
  return "Critical";
}
