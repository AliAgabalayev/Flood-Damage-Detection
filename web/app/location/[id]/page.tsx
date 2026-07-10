import Link from "next/link";
import { Location } from "@/types/location";
import locationsData from "@/public/data/locations.json";
import MapView from "./MapView";

const locations = locationsData as Location[];

export function generateStaticParams() {
  return locations.map((loc) => ({ id: loc.id }));
}

export default async function LocationPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const location = locations.find((l) => l.id === id);

  if (!location) {
    return (
      <div className="flex h-screen items-center justify-center" style={{ background: "#f0ece4" }}>
        <div className="text-center">
          <div className="text-sm font-medium mb-2" style={{ color: "#c8622a" }}>Location not found</div>
          <Link href="/" className="text-xs underline" style={{ color: "#9a8f7e" }}>Back to locations</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "#f0ece4" }}>

      {/* Sidebar */}
      <aside className="w-14 flex flex-col items-center py-4 gap-1 flex-shrink-0" style={{ background: "#2c2416" }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center mb-5" style={{ background: "#c8622a" }}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="5" stroke="white" strokeWidth="1.2" />
            <path d="M7 4v3l2 1.5" stroke="white" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
        </div>
        {[
          { label: "Map", active: true, symbol: "M" },
          { label: "Layers", active: false, symbol: "L" },
          { label: "Download", active: false, symbol: "D" },
        ].map((item) => (
          <button
            key={item.label}
            aria-label={item.label}
            className="w-9 h-9 rounded-lg flex items-center justify-center transition-colors text-xs font-mono"
            style={{
              background: item.active ? "#3d3020" : "transparent",
              color: item.active ? "#e8c89a" : "#7a6a55",
            }}
          >
            {item.symbol}
          </button>
        ))}
      </aside>

      {/* Main */}
      <div className="flex flex-col flex-1 overflow-hidden">

        {/* Topbar */}
        <header
          className="h-12 flex items-center justify-between px-5 flex-shrink-0"
          style={{ background: "#faf8f4", borderBottom: "1px solid #e8e2d8" }}
        >
          <div className="flex items-center gap-3">
            <Link href="/" className="text-xs transition-colors hover:opacity-70" style={{ color: "#9a8f7e" }}>
              Back
            </Link>
            <div style={{ width: "1px", height: "16px", background: "#e8e2d8" }} />
            <div>
              <div className="text-sm font-semibold tracking-tight" style={{ color: "#1c1710" }}>
                {location.name}
              </div>
              <div className="text-xs font-mono" style={{ color: "#9a8f7e" }}>
                {location.center[0].toFixed(4)}N {location.center[1].toFixed(4)}E
              </div>
            </div>
          </div>
          <div className="text-xs font-mono" style={{ color: "#b0a090" }}>
            {location.scene_date} · {location.model}
          </div>
        </header>

        {/* Map */}
        <div className="flex-1 relative overflow-hidden">
          <MapView location={location} />
        </div>

        {/* Stats strip */}
        <div
          className="flex-shrink-0 grid grid-cols-4"
          style={{ background: "#faf8f4", borderTop: "1px solid #e8e2d8" }}
        >
          {[
            { value: location.flooded_area_km2 !== null ? `${location.flooded_area_km2} km2` : "Pending", label: "Flooded area", accent: true },
            { value: location.flooded_pct !== null ? `${location.flooded_pct}%` : "Pending", label: "Of location", accent: false },
            { value: location.scene_date, label: "Scene date", accent: false },
            { value: location.model, label: "Model", accent: false },
          ].map((stat, i) => (
            <div
              key={i}
              className="px-4 py-3"
              style={{ borderRight: i < 3 ? "1px solid #ede8e0" : "none" }}
            >
              <div
                className="text-base font-semibold mb-0.5 tracking-tight"
                style={{ color: stat.accent ? "#c8622a" : "#1c1710", fontSize: stat.value.length > 10 ? 13 : undefined }}
              >
                {stat.value}
              </div>
              <div className="text-xs" style={{ color: "#9a8f7e" }}>
                {stat.label}
              </div>
            </div>
          ))}
        </div>

        {/* Legend + download */}
        <div
          className="flex-shrink-0 flex items-center justify-between px-4 py-2"
          style={{ background: "#faf8f4", borderTop: "1px solid #ede8e0" }}
        >
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                <div style={{ width: 10, height: 10, borderRadius: 2, background: "#c8622a", opacity: 0.5 }} />
                <span className="text-xs" style={{ color: "#7a7060" }}>Water</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div style={{ width: 10, height: 10, borderRadius: 2, background: "#c8bfb0" }} />
                <span className="text-xs" style={{ color: "#7a7060" }}>Land</span>
              </div>
            </div>
            <div className="text-xs font-mono" style={{ color: "#b0a090" }}>
              S1-SAR · VV/VH · 512x512 patches
            </div>
          </div>

          {/* Download buttons */}
          <div className="flex items-center gap-2">
            {location.mask_url ? (
              <div className="flex items-center gap-2">
                <a
                  href={location.mask_url}
                  download={`${location.id}_mask.png`}
                  className="text-xs font-medium px-3 py-1.5 rounded-lg"
                  style={{ background: "#2c2416", color: "#e8c89a", border: "1px solid #2c2416" }}
                >
                  Download PNG
                </a>
                {location.geotiff_url && (
                  <a
                    href={location.geotiff_url}
                    download={`${location.id}_mask.tif`}
                    className="text-xs font-medium px-3 py-1.5 rounded-lg"
                    style={{ background: "#faf8f4", color: "#3c3020", border: "1px solid #e8e2d8" }}
                  >
                    Download GeoTIFF
                  </a>
                )}
              </div>
            ) : (
              <span className="text-xs" style={{ color: "#b0a090" }}>
                Downloads available after model output
              </span>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}