"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Location, SceneArchive } from "@/types/location";

interface Props {
  location: Location;
  scene: SceneArchive;
}

export default function MapView({ location, scene }: Props) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<unknown>(null);
  const overlayRef = useRef<unknown>(null);
  const permanentWaterOverlayRef = useRef<unknown>(null);
  const confidenceOverlayRef = useRef<unknown>(null);
  const sarOverlayRef = useRef<unknown>(null);
  const [maskVisible, setMaskVisible] = useState(true);
  const [permanentWaterVisible, setPermanentWaterVisible] = useState(false);
  const [confidenceVisible, setConfidenceVisible] = useState(false);
  const [sarVisible, setSarVisible] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [zoomedCell, setZoomedCell] = useState<number | null>(null);

  const originalBounds = location.bounds;

  const subBoundsGrid = (() => {
    const [[south, west], [north, east]] = originalBounds;
    const latStep = (north - south) / 3;
    const lngStep = (east - west) / 3;
    const cells: [[number, number], [number, number]][] = [];
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const cellNorth = north - row * latStep;
        const cellSouth = north - (row + 1) * latStep;
        const cellWest = west + col * lngStep;
        const cellEast = west + (col + 1) * lngStep;
        cells.push([[cellSouth, cellWest], [cellNorth, cellEast]]);
      }
    }
    return cells;
  })();

  useEffect(() => {
    if (!mapRef.current) return;

    let map: import("leaflet").Map | null = null;

    async function initMap() {
      try {
        const L = (await import("leaflet")).default;
        await import("leaflet/dist/leaflet.css");

        if (!mapRef.current) return;
        if ((mapRef.current as HTMLElement & { _leaflet_id?: number })._leaflet_id) return;

        map = L.map(mapRef.current, {
          zoomControl: true,
          zoomSnap: 0.1,
        });
        map.fitBounds(location.bounds, { padding: [0, 0] });

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: "© OpenStreetMap contributors",
          maxZoom: 19,
        }).addTo(map as never);

        const bounds: [[number, number], [number, number]] = location.bounds;

        if (scene.sar_url) {
          const sarOverlay = L.imageOverlay(scene.sar_url, bounds, { opacity: 0 });
          sarOverlay.addTo(map as never);
          sarOverlayRef.current = sarOverlay;
        }

        if (scene.mask_url) {
          const overlay = L.imageOverlay(scene.mask_url, bounds, { opacity: 0.6 });
          overlay.addTo(map as never);
          overlayRef.current = overlay;
        }

        if (scene.permanent_water_url) {
          const pwOverlay = L.imageOverlay(scene.permanent_water_url, bounds, { opacity: 0 });
          pwOverlay.addTo(map as never);
          permanentWaterOverlayRef.current = pwOverlay;
        }

        if (scene.probability_url) {
          const confOverlay = L.imageOverlay(scene.probability_url, bounds, { opacity: 0 });
          confOverlay.addTo(map as never);
          confidenceOverlayRef.current = confOverlay;
        }

        mapInstanceRef.current = map;
        setLoading(false);
      } catch (e) {
        setError("Failed to load map. Please refresh the page.");
        setLoading(false);
        console.error(e);
      }
    }

    initMap();

    return () => {
      if (map) {
        map.remove();
        mapInstanceRef.current = null;
        overlayRef.current = null;
        permanentWaterOverlayRef.current = null;
        confidenceOverlayRef.current = null;
        sarOverlayRef.current = null;
      }
    };
  }, [location, scene]);

  useEffect(() => {
    if (!overlayRef.current) return;
    const overlay = overlayRef.current as { setOpacity: (o: number) => void };
    overlay.setOpacity(maskVisible ? 0.6 : 0);
  }, [maskVisible]);

  useEffect(() => {
    if (!permanentWaterOverlayRef.current) return;
    const overlay = permanentWaterOverlayRef.current as { setOpacity: (o: number) => void };
    overlay.setOpacity(permanentWaterVisible ? 0.55 : 0);
  }, [permanentWaterVisible]);

  useEffect(() => {
    if (!confidenceOverlayRef.current) return;
    const overlay = confidenceOverlayRef.current as { setOpacity: (o: number) => void };
    overlay.setOpacity(confidenceVisible ? 0.7 : 0);
  }, [confidenceVisible]);

  useEffect(() => {
    if (!sarOverlayRef.current) return;
    const overlay = sarOverlayRef.current as { setOpacity: (o: number) => void };
    overlay.setOpacity(sarVisible ? 1 : 0);
  }, [sarVisible]);

  const zoomToCell = useCallback((index: number) => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current as { fitBounds: (b: unknown, opts: unknown) => void };
    const cellBounds = subBoundsGrid[index];
    map.fitBounds(cellBounds, { animate: true, padding: [0, 0] });
    setZoomedCell(index);
  }, [subBoundsGrid]);

  const resetZoom = useCallback(() => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current as { fitBounds: (b: unknown, opts: unknown) => void };
    map.fitBounds(originalBounds, { animate: true, padding: [0, 0] });
    setZoomedCell(null);
  }, [originalBounds]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>

      <div ref={mapRef} style={{ width: "100%", height: "100%" }} />

      {loading && (
        <div
          style={{
            position: "absolute", inset: 0, zIndex: 1000,
            background: "var(--navy-900)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          <div className="text-center">
            <div className="radar-mark" style={{ width: 44, height: 44, borderRadius: 12, margin: "0 auto 12px" }}>
              <div className="radar-mark__sweep" />
              <div className="radar-mark__ring" />
              <div className="radar-mark__dot" />
            </div>
            <div className="text-xs" style={{ fontFamily: "var(--font-mono)", color: "var(--text-on-navy-soft)" }}>
              Loading map…
            </div>
          </div>
        </div>
      )}

      {error && (
        <div
          style={{
            position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)",
            zIndex: 1000, background: "var(--danger-soft)", border: "1px solid #f0c4c4",
            borderRadius: 8, padding: "8px 14px", fontSize: 11, color: "var(--danger)",
            whiteSpace: "nowrap", boxShadow: "0 1px 4px rgba(13,31,51,0.08)",
          }}
        >
          {error}
        </div>
      )}

      {!loading && !error && (
        <div
          style={{
            position: "absolute",
            top: 0, left: 0, right: 0, bottom: 0,
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gridTemplateRows: "repeat(3, 1fr)",
            zIndex: 400,
            pointerEvents: "none",
          }}
        >
          {subBoundsGrid.map((_, i) => (
            <div
              key={i}
              onClick={() => zoomToCell(i)}
              style={{
                border: "1px dashed rgba(28,111,214,0.18)",
                cursor: "pointer",
                pointerEvents: "auto",
                transition: "background 0.15s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(28,111,214,0.08)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            />
          ))}
        </div>
      )}

      {!loading && !error && zoomedCell !== null && (
        <div style={{ position: "absolute", bottom: 12, right: 12, zIndex: 1000 }}>
          <button
            onClick={resetZoom}
            className="flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg transition-all"
            style={{
              background: "var(--panel)",
              border: "1px solid var(--line)",
              color: "var(--text-900)",
              boxShadow: "0 1px 4px rgba(13,31,51,0.08)",
            }}
          >
            Full view
          </button>
        </div>
      )}

      {!loading && !error && (
        <div style={{ position: "absolute", top: 12, right: 12, zIndex: 1000, display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
          <button
            onClick={() => setMaskVisible((v) => !v)}
            className="flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg transition-all"
            style={{
              background: maskVisible ? "var(--signal-soft)" : "var(--panel)",
              border: maskVisible ? "1px solid var(--signal-line)" : "1px solid var(--line)",
              color: maskVisible ? "var(--signal-strong)" : "var(--text-500)",
              boxShadow: "0 1px 4px rgba(13,31,51,0.08)",
            }}
          >
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: maskVisible ? "var(--layer-water)" : "#c9d3dd" }} />
            {maskVisible ? "Flood mask on" : "Flood mask off"}
          </button>

          {scene.permanent_water_url ? (
            <button
              onClick={() => setPermanentWaterVisible((v) => !v)}
              className="flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg transition-all"
              style={{
                background: permanentWaterVisible ? "#e4f6f4" : "var(--panel)",
                border: permanentWaterVisible ? "1px solid #a6ded7" : "1px solid var(--line)",
                color: permanentWaterVisible ? "var(--layer-permanent)" : "var(--text-500)",
                boxShadow: "0 1px 4px rgba(13,31,51,0.08)",
              }}
            >
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: permanentWaterVisible ? "var(--layer-permanent)" : "#c9d3dd" }} />
              {permanentWaterVisible ? "Permanent water on" : "Permanent water off"}
            </button>
          ) : (
            <div
              className="text-xs px-3 py-2 rounded-lg"
              style={{ background: "var(--panel)", border: "1px solid var(--line)", color: "var(--text-300)", boxShadow: "0 1px 4px rgba(13,31,51,0.08)" }}
            >
              JRC layer pending
            </div>
          )}

          {scene.sar_url ? (
            <button
              onClick={() => setSarVisible((v) => !v)}
              className="flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg transition-all"
              style={{
                background: sarVisible ? "var(--panel-sunken)" : "var(--panel)",
                border: sarVisible ? "1px solid var(--text-300)" : "1px solid var(--line)",
                color: sarVisible ? "var(--text-900)" : "var(--text-500)",
                boxShadow: "0 1px 4px rgba(13,31,51,0.08)",
              }}
              title="Show the raw Sentinel-1 SAR scene in place of the basemap, to verify layers against the real satellite image"
            >
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: sarVisible ? "var(--text-500)" : "#c9d3dd" }} />
              {sarVisible ? "SAR image on" : "SAR image off"}
            </button>
          ) : null}

          {scene.probability_url ? (
            <button
              onClick={() => setConfidenceVisible((v) => !v)}
              className="flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg transition-all"
              style={{
                background: confidenceVisible ? "#efecfe" : "var(--panel)",
                border: confidenceVisible ? "1px solid #c9bdf6" : "1px solid var(--line)",
                color: confidenceVisible ? "var(--layer-confidence)" : "var(--text-500)",
                boxShadow: "0 1px 4px rgba(13,31,51,0.08)",
              }}
            >
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: confidenceVisible ? "var(--layer-confidence)" : "#c9d3dd" }} />
              {confidenceVisible ? "Confidence on" : "Confidence off"}
            </button>
          ) : (
            <div
              className="text-xs px-3 py-2 rounded-lg"
              style={{ background: "var(--panel)", border: "1px solid var(--line)", color: "var(--text-300)", boxShadow: "0 1px 4px rgba(13,31,51,0.08)" }}
            >
              Confidence layer pending
            </div>
          )}
        </div>
      )}

      {!loading && !error && !scene.mask_url && (
        <div
          style={{
            position: "absolute", bottom: 32, left: "50%", transform: "translateX(-50%)",
            zIndex: 1000, background: "var(--panel)", border: "1px solid var(--line)",
            borderRadius: 8, padding: "8px 14px", fontSize: 11, color: "var(--text-500)",
            whiteSpace: "nowrap", boxShadow: "0 1px 4px rgba(13,31,51,0.08)",
          }}
        >
          No mask yet — waiting for model output
        </div>
      )}
    </div>
  );
}
