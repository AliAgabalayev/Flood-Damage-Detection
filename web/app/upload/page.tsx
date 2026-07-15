"use client";

import { useState } from "react";
import Link from "next/link";
import Sidebar from "@/components/Sidebar";
import MapView from "@/app/location/[id]/MapView";
import { Location, SceneArchive } from "@/types/location";
import { severityColor, severityLabel } from "@/lib/severity";
import { buildInterpretation } from "@/lib/interpretation";
import locationsData from "@/public/data/locations.json";

const locations = locationsData as unknown as Location[];

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

interface PredictResponse {
  flooded_area_km2: number;
  flooded_pct: number;
  bounds: [[number, number], [number, number]];
  center: [number, number];
  mask_png_base64: string;
  sar_png_base64: string;
}

type Status = "idle" | "loading" | "error";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ location: Location; scene: SceneArchive } | null>(null);

  async function runPrediction() {
    if (!file) return;
    setStatus("loading");
    setError(null);

    try {
      const body = new FormData();
      body.append("file", file);

      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body });

      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? `Prediction failed (HTTP ${res.status}).`);
      }

      const data: PredictResponse = await res.json();

      const scene: SceneArchive = {
        scene_id: "custom_upload",
        date: new Date().toISOString().slice(0, 10),
        flooded_area_km2: data.flooded_area_km2,
        flooded_pct: data.flooded_pct,
        mask_url: `data:image/png;base64,${data.mask_png_base64}`,
        sar_url: `data:image/png;base64,${data.sar_png_base64}`,
        geotiff_url: null,
        permanent_water_url: null,
        probability_url: null,
        layover_shadow_url: null,
      };
      const location: Location = {
        id: "custom",
        name: file.name,
        center: data.center,
        bounds: data.bounds,
        model: "segformer_b4",
        scenes: [scene],
      };

      setResult({ location, scene });
      setStatus("idle");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed.");
      setStatus("error");
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setStatus("idle");
    setError(null);
  }

  const color = result ? severityColor(result.scene.flooded_pct) : "var(--text-300)";
  const interpretation = result ? buildInterpretation(result.scene) : null;

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg)" }}>
      <Sidebar locations={locations} />

      <div className="flex flex-col flex-1 overflow-hidden min-w-0">
        <div
          className="h-16 flex-shrink-0 flex items-center gap-3 px-6"
          style={{ background: "var(--panel)", borderBottom: "1px solid var(--line)" }}
        >
          <Link href="/" className="text-xs font-medium flex-shrink-0" style={{ color: "var(--text-500)" }}>
            ← Overview
          </Link>
          <div style={{ width: "1px", height: "16px", background: "var(--line)" }} />
          <div
            className="text-sm font-semibold tracking-tight"
            style={{ fontFamily: "var(--font-display)", color: "var(--text-900)" }}
          >
            Upload your own SAR scene
          </div>
        </div>

        <div className="flex-1 relative overflow-hidden">
          {result ? (
            <MapView location={result.location} scene={result.scene} />
          ) : (
            <div className="w-full h-full flex items-center justify-center px-6">
              <div
                className="w-full max-w-md rounded-2xl p-6"
                style={{ background: "var(--panel)", border: "1px solid var(--line)" }}
              >
                <p
                  className="text-[10px] font-medium tracking-widest mb-3"
                  style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}
                >
                  CUSTOM PREDICTION
                </p>

                <label
                  className="flex flex-col items-center justify-center gap-2 rounded-xl px-4 py-8 mb-3 cursor-pointer transition-colors"
                  style={{ border: "1.5px dashed var(--line)", background: "var(--panel-sunken)" }}
                >
                  <input
                    type="file"
                    accept=".tif,.tiff"
                    className="hidden"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  />
                  <span className="text-sm font-medium" style={{ color: "var(--text-900)" }}>
                    {file ? file.name : "Choose a GeoTIFF"}
                  </span>
                  <span className="text-[11px]" style={{ color: "var(--text-300)" }}>
                    .tif or .tiff, Sentinel-1 VV+VH bands
                  </span>
                </label>

                <p
                  className="text-xs mb-4 px-3 py-2 rounded-lg"
                  style={{ background: "var(--signal-soft)", color: "var(--signal-strong)", border: "1px solid var(--signal-line)" }}
                >
                  You should upload a real SAR (Sentinel-1) image.
                </p>

                {status === "error" && error && (
                  <p
                    className="text-xs mb-4 px-3 py-2 rounded-lg"
                    style={{ background: "var(--danger-soft)", color: "var(--danger)", border: "1px solid #f0c4c4" }}
                  >
                    {error}
                  </p>
                )}

                <button
                  onClick={runPrediction}
                  disabled={!file || status === "loading"}
                  className="w-full text-sm font-medium text-center px-3 py-2.5 rounded-lg transition-opacity"
                  style={{
                    background: !file || status === "loading" ? "var(--panel-sunken)" : "var(--navy-900)",
                    color: !file || status === "loading" ? "var(--text-300)" : "var(--text-on-navy)",
                    cursor: !file || status === "loading" ? "not-allowed" : "pointer",
                  }}
                >
                  {status === "loading" ? (
                    <span className="flex items-center justify-center gap-2">
                      <span className="radar-mark" style={{ width: 16, height: 16, borderRadius: 5 }}>
                        <span className="radar-mark__sweep" />
                        <span className="radar-mark__dot" />
                      </span>
                      Running model on your SAR scene…
                    </span>
                  ) : (
                    "Run prediction"
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {result && (
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
            className="text-lg font-semibold tracking-tight mb-1 truncate"
            style={{ fontFamily: "var(--font-display)", color: "var(--text-900)" }}
          >
            {result.location.name}
          </h1>
          <div className="text-[11px] mb-5" style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}>
            {result.location.center[0].toFixed(4)}N {result.location.center[1].toFixed(4)}E
          </div>

          <div
            className="rounded-xl px-4 py-4 mb-6"
            style={{ background: "var(--panel-sunken)", border: "1px solid var(--line-soft)" }}
          >
            <div className="flex items-center justify-between mb-3">
              <span
                className="text-[10px] font-semibold px-2 py-0.5 rounded-md"
                style={{ background: color, color: "#fff" }}
              >
                {severityLabel(result.scene.flooded_pct)}
              </span>
            </div>
            <div className="flex items-baseline gap-2 mb-0.5">
              <span className="text-2xl font-semibold tracking-tight" style={{ fontFamily: "var(--font-display)", color }}>
                {result.scene.flooded_pct}%
              </span>
              <span className="text-xs" style={{ color: "var(--text-500)" }}>of area flooded</span>
            </div>
            <div className="text-xs" style={{ color: "var(--text-500)" }}>
              {result.scene.flooded_area_km2} km² flooded surface
            </div>
          </div>

          {interpretation && (
            <>
              <p
                className="text-[10px] font-medium tracking-widest mb-2.5"
                style={{ fontFamily: "var(--font-mono)", color: "var(--text-300)" }}
              >
                INTERPRETATION
              </p>
              <p
                className="text-xs leading-relaxed mb-6 px-4 py-3.5 rounded-xl"
                style={{ background: "var(--signal-soft)", border: "1px solid var(--signal-line)", color: "var(--text-900)" }}
              >
                {interpretation}
              </p>
            </>
          )}

          <p className="text-[11px] mb-6" style={{ color: "var(--text-300)" }}>
            Permanent-water and layover/shadow layers aren&apos;t available for custom
            uploads — those require pre-fetched reference data for the specific area.
          </p>

          <button
            onClick={reset}
            className="w-full text-xs font-medium text-center px-3 py-2 rounded-lg transition-colors"
            style={{ background: "var(--panel)", color: "var(--text-900)", border: "1px solid var(--line)" }}
          >
            Upload another scene
          </button>
        </aside>
      )}
    </div>
  );
}
