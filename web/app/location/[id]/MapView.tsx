"use client";

import { useEffect, useRef, useState } from "react";
import { Location } from "@/types/location";

interface Props {
  location: Location;
}

export default function MapView({ location }: Props) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<unknown>(null);
  const overlayRef = useRef<unknown>(null);
  const [maskVisible, setMaskVisible] = useState(true);

  useEffect(() => {
    if (!mapRef.current) return;

    let map: { remove: () => void } | null = null;

    async function initMap() {
      const L = (await import("leaflet")).default;
      await import("leaflet/dist/leaflet.css");

      if (!mapRef.current) return;
      if ((mapRef.current as HTMLElement & { _leaflet_id?: number })._leaflet_id) return;

      map = L.map(mapRef.current, {
        center: location.center,
        zoom: 11,
        zoomControl: true,
      });

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
        maxZoom: 19,
      }).addTo(map as never);

      // Add flood mask overlay if available
      if (location.mask_url) {
        const bounds: [[number, number], [number, number]] = location.bounds;
        const overlay = L.imageOverlay(location.mask_url, bounds, {
          opacity: 0.6,
        });
        overlay.addTo(map as never);
        overlayRef.current = overlay;
      }

      mapInstanceRef.current = map;
    }

    initMap();

    return () => {
      if (map) {
        map.remove();
        mapInstanceRef.current = null;
        overlayRef.current = null;
      }
    };
  }, [location]);

  // Toggle overlay visibility
  useEffect(() => {
    if (!overlayRef.current) return;
    const overlay = overlayRef.current as { setOpacity: (o: number) => void };
    overlay.setOpacity(maskVisible ? 0.6 : 0);
  }, [maskVisible]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div ref={mapRef} style={{ width: "100%", height: "100%" }} />

      {/* Toggle button */}
      <div style={{ position: "absolute", top: 12, right: 12, zIndex: 1000 }}>
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
          <div
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: maskVisible ? "#c8622a" : "#d0c8be",
            }}
          />
          {maskVisible ? "Mask on" : "Mask off"}
        </button>
      </div>

      {/* No mask notice */}
      {!location.mask_url && (
        <div
          style={{
            position: "absolute",
            bottom: 32,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 1000,
            background: "#faf8f4",
            border: "1px solid #e8e2d8",
            borderRadius: 8,
            padding: "8px 14px",
            fontSize: 11,
            color: "#9a8f7e",
            whiteSpace: "nowrap",
            boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
          }}
        >
          No mask yet — waiting for model output
        </div>
      )}
    </div>
  );
}