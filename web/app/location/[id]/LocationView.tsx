"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { Location } from "@/types/location";
import MapView from "./MapView";
import Sidebar from "@/components/Sidebar";
import { severityColor, severityLabel } from "@/lib/severity";
import allLocationsData from "@/public/data/locations.json";

interface Props {
  location: Location;
}

const ALL_LOCATIONS = allLocationsData as unknown as Location[];

export default function LocationView({ location }: Props) {
  const sortedScenes = useMemo(
    () => [...location.scenes].sort((a, b) => (a.date < b.date ? 1 : -1)),
    [location.scenes]
  );

  const [selectedIndex, setSelectedIndex] = useState(0);
  const activeScene = sortedScenes[selectedIndex];
  const color = severityColor(activeScene.flooded_pct);

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg)" }}>
      <Sidebar locations={ALL_LOCATIONS} activeId={location.id} />

      {/* Center: map stage */}
      <div className="flex flex-col flex-1 overflow-hidden min-w-0">
        <div
          className="h-16 flex-shrink-0 flex items-center justify-between px-6"
          style={{ background: "var(--panel)", borderBottom: "1px solid var(--line)" }}
        >
          <div className="flex items-center gap-3 min-w-0">
            <Link
              href="/"
              className="text-xs font-medium flex-shrink-0 transition-colors"
              style={{ color: "var(--text-500)" }}
            >
              ← Overview
            </Link>
            <div style={{ width: "1px", height: "16px", background: "var(--line)" }} />
            <div className="min-w-0">
              <div
                className="text-sm font-semibold tracking-tight truncate"
                style={{ fontFamily: "var(--font-display)", color: "var(--text-900)" }}
              >
                {location.name}
              </div>
            </div>
          </div>

          {sortedScenes.length > 1 && (
            <div className="flex items-center gap-1.5 flex-shrink-0">
              {sortedScenes.map((scene, i) => (
                <button
                  key={scene.scene_id}
                  onClick={() => setSelectedIndex(i)}
                  className="text-xs font-medium px-2.5 py-1.5 rounded-lg transition-all"
                  style={{
                    fontFamily: "var(--font-mono)",
                    background: i === selectedIndex ? "var(--navy-900)" : "transparent",
                    color: i === selectedIndex ? "var(--text-on-navy)" : "var(--text-500)",
                    border: i === selectedIndex ? "1px solid var(--navy-900)" : "1px solid var(--line)",
                  }}
                >
                  {scene.date}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex-1 relative overflow-hidden">
          <MapView location={location} scene={activeScene} />
        </div>
      </div>

      {/* Right: dashboard */}
      <aside
        className="w-80 flex-shrink-0 h-full overflow-y-auto px-6 py-7"
        style={{ background: "var(--panel)", borderLeft: "1px solid var(--line)" }}
      >
        <p
          className="text-[10px] font-medium tracking-widest mb-1.5"
          style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}
        >
          ZONE DASHBOARD
        </p>
        <h1
          className="text-lg font-semibold tracking-tight mb-1"
          style={{ fontFamily: "var(--font-display)", color: "var(--text-900)" }}
        >
          {location.name}
        </h1>
        <div className="text-[11px] mb-5" style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}>
          {location.center[0].toFixed(4)}N {location.center[1].toFixed(4)}E
        </div>

        {/* Headline stat */}
        <div
          className="rounded-xl px-4 py-4 mb-6"
          style={{ background: "var(--panel-sunken)", border: "1px solid var(--line-soft)" }}
        >
          <div className="flex items-center justify-between mb-3">
            <span
              className="text-[10px] font-semibold px-2 py-0.5 rounded-md"
              style={{ background: color, color: "#fff" }}
            >
              {severityLabel(activeScene.flooded_pct)}
            </span>
            <span className="text-[11px]" style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}>
              {activeScene.date}
            </span>
          </div>
          <div className="flex items-baseline gap-2 mb-0.5">
            <span className="text-2xl font-semibold tracking-tight" style={{ fontFamily: "var(--font-display)", color }}>
              {activeScene.flooded_pct !== null ? `${activeScene.flooded_pct}%` : "Pending"}
            </span>
            <span className="text-xs" style={{ color: "var(--text-500)" }}>of area flooded</span>
          </div>
          <div className="text-xs" style={{ color: "var(--text-500)" }}>
            {activeScene.flooded_area_km2 !== null ? `${activeScene.flooded_area_km2} km²` : "—"} flooded surface
          </div>
        </div>

        {/* Scene timeline */}
        <SectionLabel>Scene timeline</SectionLabel>
        <div className="flex flex-col gap-1 mb-6">
          {sortedScenes.map((scene, i) => {
            const scColor = severityColor(scene.flooded_pct);
            const active = i === selectedIndex;
            return (
              <button
                key={scene.scene_id}
                onClick={() => setSelectedIndex(i)}
                className="flex items-center justify-between px-3 py-2 rounded-lg text-left transition-colors"
                style={{ background: active ? "var(--signal-soft)" : "transparent", border: active ? "1px solid var(--signal-line)" : "1px solid transparent" }}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: scColor }} />
                  <span className="text-xs" style={{ fontFamily: "var(--font-mono)", color: "var(--text-900)" }}>
                    {scene.date}
                  </span>
                </div>
                <span className="text-xs font-medium flex-shrink-0" style={{ color: scColor }}>
                  {scene.flooded_pct !== null ? `${scene.flooded_pct}%` : "—"}
                </span>
              </button>
            );
          })}
        </div>

        {/* Layers */}
        <SectionLabel>Map layers</SectionLabel>
        <div className="flex flex-col gap-2 mb-6">
          <LayerRow color="var(--layer-water)" label="Flood mask" available={!!activeScene.mask_url} />
          <LayerRow color="var(--layer-permanent)" label="Permanent water (JRC)" available={!!activeScene.permanent_water_url} />
          <LayerRow color="var(--layer-confidence)" label="Model confidence" available={!!activeScene.probability_url} />
          <LayerRow color="var(--text-500)" label="Raw SAR scene (verify layers)" available={!!activeScene.sar_url} />
        </div>
        <p className="text-[11px] mb-6 -mt-3" style={{ color: "var(--text-300)" }}>
          Toggle layers from the buttons on the map.
        </p>

        {/* Downloads */}
        <SectionLabel>Downloads</SectionLabel>
        <div className="flex flex-col gap-2 mb-6">
          {activeScene.mask_url ? (
            <>
              <a
                href={activeScene.mask_url}
                download={`${location.id}_${activeScene.date}_mask.png`}
                className="text-xs font-medium text-center px-3 py-2 rounded-lg transition-opacity hover:opacity-90"
                style={{ background: "var(--navy-900)", color: "var(--text-on-navy)" }}
              >
                Download mask (PNG)
              </a>
              {activeScene.geotiff_url && (
                <a
                  href={activeScene.geotiff_url}
                  download={`${location.id}_${activeScene.date}_mask.tif`}
                  className="text-xs font-medium text-center px-3 py-2 rounded-lg transition-colors"
                  style={{ background: "var(--panel)", color: "var(--text-900)", border: "1px solid var(--line)" }}
                >
                  Download GeoTIFF
                </a>
              )}
            </>
          ) : (
            <span className="text-xs" style={{ color: "var(--text-300)" }}>
              Downloads available after model output
            </span>
          )}
        </div>

      </aside>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p
      className="text-[10px] font-medium tracking-widest mb-2.5"
      style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}
    >
      {typeof children === "string" ? children.toUpperCase() : children}
    </p>
  );
}

function LayerRow({ color, label, available }: { color: string; label: string; available: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <span
        className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
        style={{ background: available ? color : "var(--panel-sunken)", border: available ? "none" : "1px solid var(--line)" }}
      />
      <span className="text-xs flex-1" style={{ color: available ? "var(--text-900)" : "var(--text-300)" }}>
        {label}
      </span>
      {!available && (
        <span className="text-[10px]" style={{ color: "var(--text-300)" }}>
          pending
        </span>
      )}
    </div>
  );
}
