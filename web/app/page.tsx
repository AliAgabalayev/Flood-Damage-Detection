import Link from "next/link";
import { Location } from "@/types/location";
import locationsData from "@/public/data/locations.json";

const locations = locationsData as Location[];

export default function Home() {
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "#f0ece4" }}>

      {/* Sidebar */}
      <aside className="w-14 flex flex-col items-center py-4 gap-1 flex-shrink-0" style={{ background: "var(--sidebar)" }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center mb-5" style={{ background: "var(--accent)" }}>
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
              color: item.active ? "var(--gold)" : "#7a6a55",
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
          style={{ background: "var(--surface)", borderColor: "var(--border)" }}
        >
          <div>
            <div className="text-sm font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
              Flood Damage Detection
            </div>
            <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
              Sen1Floods11 · Sentinel-1 SAR · DeepLabV3+
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium px-2 py-1 rounded" style={{ background: "#f0e8de", color: "#7a4a20", border: "1px solid #e0d0c0" }}>
              5 locations ready
            </span>
            <span className="text-xs font-medium px-2 py-1 rounded" style={{ background: "#e8f0e4", color: "#3a6020", border: "1px solid #c8d8c0" }}>
              Model active
            </span>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-6">
          <div className="mb-6">
            <p className="text-xs font-medium uppercase tracking-widest mb-1" style={{ color: "var(--text-muted)" }}>
              Monitored locations
            </p>
            <h1 className="text-xl font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
              Select a flood zone to inspect
            </h1>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
            {locations.map((loc) => (
              <Link key={loc.id} href={`/location/${loc.id}`} className="block">
                <div
                  className="rounded-xl p-4 cursor-pointer transition-all hover:shadow-md hover:-translate-y-0.5"
                  style={{ background: "#faf8f4", border: "1px solid #e8e2d8" }}
                >
                  {/* Thumbnail */}
                  <div className="w-full h-20 rounded-lg mb-3 overflow-hidden" style={{ background: "var(--sand)" }}>
                    <svg width="100%" height="100%" viewBox="0 0 120 80" preserveAspectRatio="xMidYMid slice">
                      <rect width="120" height="80" fill="#ddd4c4" />
                      <ellipse cx="60" cy="40" rx="38" ry="22" fill="#c8622a" fillOpacity="0.15" stroke="#c8622a" strokeWidth="1" strokeOpacity="0.45" />
                    </svg>
                  </div>

                  <div className="text-xs font-semibold mb-0.5 truncate" style={{ color: "var(--text-primary)" }}>
                    {loc.name}
                  </div>
                  <div className="text-xs font-mono mb-3" style={{ color: "var(--text-muted)" }}>
                    {loc.center[0].toFixed(2)}°N {loc.center[1].toFixed(2)}°E
                  </div>

                  <div className="flex items-baseline justify-between">
                    <span className="text-base font-bold" style={{ color: "var(--accent)" }}>
                      {loc.flooded_area_km2 !== null ? `${loc.flooded_area_km2} km²` : "Pending"}
                    </span>
                    <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                      {loc.flooded_pct !== null ? `${loc.flooded_pct}% flooded` : ""}
                    </span>
                  </div>

                  <div className="mt-2 flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--green)" }} />
                    <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                      {loc.scene_date}
                    </span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}