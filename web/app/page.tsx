"use client";

import Link from "next/link";
import Sidebar from "@/components/Sidebar";
import { Location } from "@/types/location";
import locationsData from "@/public/data/locations.json";
import { severityColor, severityLabel } from "@/lib/severity";

const locations = locationsData as Location[];

function latestSceneOf(loc: Location) {
  return [...loc.scenes].sort((a, b) => (a.date < b.date ? 1 : -1))[0];
}

const withLatest = locations.map((loc) => ({ loc, latest: latestSceneOf(loc) }));

const mostRecentUpdate = withLatest.map(({ latest }) => latest.date).sort().at(-1);

const totalFloodedArea = withLatest.reduce((sum, { latest }) => sum + (latest.flooded_area_km2 ?? 0), 0);

const pctValues = withLatest.map(({ latest }) => latest.flooded_pct).filter((v): v is number => v !== null);
const avgFloodedPct = pctValues.length ? pctValues.reduce((a, b) => a + b, 0) / pctValues.length : null;

const ranked = [...withLatest].sort((a, b) => (b.latest.flooded_pct ?? -1) - (a.latest.flooded_pct ?? -1));

export default function Home() {
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg)" }}>
      <Sidebar locations={locations} />

      {/* Center: map gallery */}
      <main className="flex-1 overflow-y-auto min-w-0">
        <div className="px-8 pt-7 pb-5 flex items-end justify-between flex-wrap gap-3">
          <div>
            <p
              className="text-[10px] font-medium tracking-widest mb-1.5"
              style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}
            >
              AVAILABLE MAPS
            </p>
            <h1
              className="text-2xl font-semibold tracking-tight"
              style={{ fontFamily: "var(--font-display)", color: "var(--text-900)" }}
            >
              Select a flood zone to inspect
            </h1>
          </div>
          <span
            className="text-xs font-medium px-2.5 py-1 rounded-md"
            style={{ background: "var(--signal-soft)", color: "var(--signal-strong)", border: "1px solid var(--signal-line)" }}
          >
            Updated {mostRecentUpdate}
          </span>
        </div>

        <div className="px-8 pb-10 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {withLatest.map(({ loc, latest }) => {
            const scenes = [...loc.scenes].sort((a, b) => (a.date < b.date ? 1 : -1));
            const color = severityColor(latest.flooded_pct);

            return (
              <Link key={loc.id} href={`/location/${loc.id}`} className="block group">
                <div
                  className="rounded-2xl overflow-hidden transition-all group-hover:-translate-y-0.5"
                  style={{
                    background: "var(--panel)",
                    border: "1px solid var(--line)",
                    boxShadow: "0 1px 2px rgba(13,31,51,0.04)",
                  }}
                >
                  <div className="relative w-full h-40" style={{ background: "var(--panel-sunken)" }}>
                    {latest.sar_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={latest.sar_url}
                        alt=""
                        className="absolute inset-0 w-full h-full object-cover"
                        style={{ filter: "grayscale(1) contrast(1.05) brightness(0.95)" }}
                      />
                    ) : null}
                    {latest.mask_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={latest.mask_url}
                        alt=""
                        className="absolute inset-0 w-full h-full object-cover"
                        style={{ opacity: 0.8, mixBlendMode: "multiply" }}
                      />
                    ) : null}
                    <div
                      className="absolute top-2.5 right-2.5 text-[10px] font-semibold px-2 py-1 rounded-md"
                      style={{ background: color, color: "#fff" }}
                    >
                      {severityLabel(latest.flooded_pct)}
                    </div>
                  </div>

                  <div className="p-4">
                    <div className="text-sm font-semibold mb-0.5 truncate" style={{ color: "var(--text-900)" }}>
                      {loc.name}
                    </div>
                    <div
                      className="text-[11px] mb-3"
                      style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}
                    >
                      {loc.center[0].toFixed(2)}N {loc.center[1].toFixed(2)}E
                    </div>

                    <div className="flex items-baseline">
                      <span className="text-lg font-semibold tracking-tight" style={{ color }}>
                        {latest.flooded_area_km2 !== null ? `${latest.flooded_area_km2} km²` : "Pending"}
                      </span>
                    </div>

                    <div className="mt-2.5 pt-2.5 flex items-center gap-1.5" style={{ borderTop: "1px solid var(--line-soft)" }}>
                      <div className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--signal)" }} />
                      <span className="text-[11px]" style={{ color: "var(--text-500)" }}>
                        {latest.date} · {scenes.length} scene{scenes.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </main>

      {/* Right: overview dashboard */}
      <aside
        className="w-80 flex-shrink-0 h-full overflow-y-auto px-6 py-7"
        style={{ background: "var(--panel)", borderLeft: "1px solid var(--line)" }}
      >
        <p
          className="text-[10px] font-medium tracking-widest mb-4"
          style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}
        >
          OVERVIEW
        </p>

        <div className="grid grid-cols-2 gap-2.5 mb-6">
          <StatTile label="Zones monitored" value={String(locations.length)} />
          <StatTile label="Avg. flooded" value={avgFloodedPct !== null ? `${avgFloodedPct.toFixed(1)}%` : "—"} />
          <StatTile label="Total flooded area" value={`${totalFloodedArea.toFixed(1)} km²`} wide />
          <StatTile label="Latest update" value={mostRecentUpdate ?? "—"} wide mono />
        </div>

        <p
          className="text-[10px] font-medium tracking-widest mb-3"
          style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}
        >
          HIGHEST SEVERITY
        </p>
        <div className="flex flex-col gap-3 mb-6">
          {ranked.slice(0, 5).map(({ loc, latest }, i) => {
            const color = severityColor(latest.flooded_pct);
            const width = Math.max(4, Math.min(100, latest.flooded_pct ?? 0));
            return (
              <Link key={loc.id} href={`/location/${loc.id}`} className="block group">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-medium truncate" style={{ color: "var(--text-900)" }}>
                    {i + 1}. {loc.name}
                  </span>
                  <span
                    className="text-[11px] font-semibold flex-shrink-0 ml-2"
                    style={{ fontFamily: "var(--font-mono)", color }}
                  >
                    {latest.flooded_pct !== null ? `${latest.flooded_pct}%` : "—"}
                  </span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--panel-sunken)" }}>
                  <div
                    className="h-full rounded-full transition-all group-hover:opacity-80"
                    style={{ width: `${width}%`, background: color }}
                  />
                </div>
              </Link>
            );
          })}
        </div>

        <div className="pt-5" style={{ borderTop: "1px solid var(--line)" }}>
          <p
            className="text-[10px] font-medium tracking-widest mb-2.5"
            style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}
          >
            PIPELINE
          </p>
          <div className="flex flex-col gap-1.5 text-[11px]" style={{ fontFamily: "var(--font-mono)", color: "var(--text-500)" }}>
            <div>Sensor · Sentinel-1 SAR (VV/VH)</div>
            <div>Model · DeepLabV3+ / SegFormer-B4</div>
            <div>Dataset · Sen1Floods11</div>
            <div>Tiling · 512×512 patches</div>
          </div>
        </div>
      </aside>
    </div>
  );
}

function StatTile({ label, value, wide, mono }: { label: string; value: string; wide?: boolean; mono?: boolean }) {
  return (
    <div
      className={`rounded-xl px-3.5 py-3 ${wide ? "col-span-2" : ""}`}
      style={{ background: "var(--panel-sunken)", border: "1px solid var(--line-soft)" }}
    >
      <div
        className="text-lg font-semibold tracking-tight"
        style={{ fontFamily: mono ? "var(--font-mono)" : "var(--font-display)", color: "var(--text-900)", fontSize: mono ? 15 : undefined }}
      >
        {value}
      </div>
      <div className="text-[11px] mt-0.5" style={{ color: "var(--text-500)" }}>
        {label}
      </div>
    </div>
  );
}
