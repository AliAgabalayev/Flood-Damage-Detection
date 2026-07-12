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
  const [maskVisible, setMaskVisible] = useState(true);
  const [permanentWaterVisible, setPermanentWaterVisible] = useState(false);
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
        });
        map.fitBounds(location.bounds);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: "© OpenStreetMap contributors",
          maxZoom: 19,
        }).addTo(map as never);

        const bounds: [[number, number], [number, number]] = location.bounds;

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

  const zoomToCell = useCallback((index: number) => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current as { fitBounds: (b: unknown, opts: unknown) => void };
    const cellBounds = subBoundsGrid[index];
    map.fitBounds(cellBounds, { animate: true });
    setZoomedCell(index);
  }, [subBoundsGrid]);

  const resetZoom = useCallback(() => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current as { fitBounds: (b: unknown, opts: unknown) => void };
    map.fitBounds(originalBounds, { animate: true });
    setZoomedCell(null);
  }, [originalBounds]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>

      <div ref={mapRef} style={{ width: "100%", height: "100%" }} />

      {loading && (
        <div
          style={{
            position: "absolute", inset: 0, zIndex: 1000,
            background: "#e8e0d4",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          <div className="text-center">
            <div
              style={{
                width: 32, height: 32, border: "2px solid #e8e2d8",
                borderTop: "2px solid #c8622a", borderRadius: "50%",
                animation: "spin 0.8s linear infinite", margin: "0 auto 10px",
              }}
            />
            <div className="text-xs" style={{ color: "#9a8f7e" }}>Loading map…</div>
          </div>
        </div>
      )}

      {error && (
        <div
          style={{
            position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)",
            zIndex: 1000, background: "#fdf0ec", border: "1px solid #e8c0a0",
            borderRadius: 8, padding: "8px 14px", fontSize: 11, color: "#c8622a",
            whiteSpace: "nowrap", boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
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
                border: "1px dashed rgba(44,36,22,0.15)",
                cursor: "pointer",
                pointerEvents: "auto",
                transition: "background 0.15s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(200,98,42,0.08)")}
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
              background: "#faf8f4",
              border: "1px solid #e8e2d8",
              color: "#3c3020",
              boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
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
              background: maskVisible ? "#fdf5ec" : "#faf8f4",
              border: maskVisible ? "1px solid #e8c0a0" : "1px solid #e8e2d8",
              color: maskVisible ? "#c8622a" : "#9a8f7e",
              boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
            }}
          >
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: maskVisible ? "#c8622a" : "#d0c8be" }} />
            {maskVisible ? "Flood mask on" : "Flood mask off"}
          </button>

          {scene.permanent_water_url ? (
            <button
              onClick={() => setPermanentWaterVisible((v) => !v)}
              className="flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg transition-all"
              style={{
                background: permanentWaterVisible ? "#eaf1f7" : "#faf8f4",
                border: permanentWaterVisible ? "1px solid #a8c8dc" : "1px solid #e8e2d8",
                color: permanentWaterVisible ? "#2c6a8c" : "#9a8f7e",
                boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
              }}
            >
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: permanentWaterVisible ? "#2c6a8c" : "#d0c8be" }} />
              {permanentWaterVisible ? "Permanent water on" : "Permanent water off"}
            </button>
          ) : (
            <div
              className="text-xs px-3 py-2 rounded-lg"
              style={{ background: "#faf8f4", border: "1px solid #e8e2d8", color: "#b0a090", boxShadow: "0 1px 4px rgba(0,0,0,0.08)" }}
            >
              JRC layer pending
            </div>
          )}
        </div>
      )}

      {!loading && !error && !scene.mask_url && (
        <div
          style={{
            position: "absolute", bottom: 32, left: "50%", transform: "translateX(-50%)",
            zIndex: 1000, background: "#faf8f4", border: "1px solid #e8e2d8",
            borderRadius: 8, padding: "8px 14px", fontSize: 11, color: "#9a8f7e",
            whiteSpace: "nowrap", boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
          }}
        >
          No mask yet — waiting for model output
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}