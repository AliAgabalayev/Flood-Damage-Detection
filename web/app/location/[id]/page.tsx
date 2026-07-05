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
          <Link href="/" className="text-xs underline" style={{ color: "#9a8f7e" }}>← Back to locations</Link>
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
          { label: "Map", active: true, symbol: "⊞" },
          { label: "Layers", active: false, symbol: "◫" },
          { label: "Download", active: false, symbol: "↓" },
        ].map((item) => (
          <button
            key={item.label}
            aria-label={item.label}
            className="w-9 h-9 rounded-lg flex items-center justify-center transition-colors text-base"
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
          className="h-12 flex items-center justify-between px-5 flex-shrink-0 border-b"
          style={{ background: "#faf8f4", borderColor: "#e8e2d8" }}
        >
          <div className="flex items-center gap-3">
            <Link href="/" className="text-xs transition-colors hover:opacity-70" style={{ color: "#9a8f7e" }}>
              ← Back
            </Link>
            <div style={{ width: "1px", height: "16px", background: "#e8e2d8" }} />
            <div>
              <div className="text-sm font-semibold tracking-tight" style={{ color: "#1c1710" }}>
                {location.name}
              </div>
              <div className="text-xs font-mono" style={{ color: "#9a8f7e" }}>
                {location.center[0].toFixed(4)}°N {location.center[1].toFixed(4)}°E
              </div>
            </div>
          </div>
          <div className="text-xs font-mono" style={{ color: "#b0a090" }}>
            {location.scene_date} · {location.model}
          </div>
        </header>

        {/* Map */}
        <div className="flex-1 relative">
          <MapView location={location} />
        </div>

      </div>
    </div>
  );
}