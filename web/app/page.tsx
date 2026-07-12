import Link from "next/link";
import { Location } from "@/types/location";
import locationsData from "@/public/data/locations.json";

const locations = locationsData as Location[];

export default function Home() {
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "#f0ece4" }}>
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

      <div className="flex flex-col flex-1 overflow-hidden">
        <header
          className="h-12 flex items-center justify-between px-5 flex-shrink-0 border-b"
          style={{ background: "#faf8f4", borderColor: "#e8e2d8" }}
        >
          <div>
            <div className="text-sm font-semibold tracking-tight" style={{ color: "#1c1710" }}>
              Flood Damage Detection
            </div>
            <div className="text-xs" style={{ color: "#9a8f7e" }}>
              Sen1Floods11 · Sentinel-1 SAR · DeepLabV3+
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium px-2 py-1 rounded" style={{ background: "#f0e8de", color: "#7a4a20", border: "1px solid #e0d0c0" }}>
              {locations.length} locations ready
            </span>
            <span className="text-xs font-medium px-2 py-1 rounded" style={{ background: "#e8f0e4", color: "#3a6020", border: "1px solid #c8d8c0" }}>
              Model active
            </span>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-6">
          <div className="mb-6">
            <p className="text-xs font-medium uppercase tracking-widest mb-1" style={{ color: "#b0a090" }}>
              Monitored locations
            </p>
            <h1 className="text-xl font-semibold tracking-tight" style={{ color: "#1c1710" }}>
              Select a flood zone to inspect
            </h1>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
            {locations.map((loc) => {
              const scenes = [...loc.scenes].sort((a, b) => (a.date < b.date ? 1 : -1));
              const latest = scenes[0];

              return (
                <Link key={loc.id} href={`/location/${loc.id}`} className="block">
                  <div
                    className="rounded-xl p-4 cursor-pointer transition-all hover:shadow-md hover:-translate-y-0.5"
                    style={{ background: "#faf8f4", border: "1px solid #e8e2d8" }}
                  >
                    <div className="w-full h-20 rounded-lg mb-3 overflow-hidden" style={{ background: "#ddd4c4" }}>
                      <svg width="100%" height="100%" viewBox="0 0 120 80" preserveAspectRatio="xMidYMid slice">
                        <rect width="120" height="80" fill="#ddd4c4" />
                        <ellipse cx="60" cy="40" rx="38" ry="22" fill="#c8622a" fillOpacity="0.15" stroke="#c8622a" strokeWidth="1" strokeOpacity="0.45" />
                      </svg>
                    </div>

                    <div className="text-xs font-semibold mb-0.5 truncate" style={{ color: "#1c1710" }}>
                      {loc.name}
                    </div>
                    <div className="text-xs font-mono mb-3" style={{ color: "#b0a090" }}>
                      {loc.center[0].toFixed(2)}N {loc.center[1].toFixed(2)}E
                    </div>

                    <div className="flex items-baseline justify-between">
                      <span className="text-base font-bold" style={{ color: "#c8622a" }}>
                        {latest.flooded_area_km2 !== null ? `${latest.flooded_area_km2} km2` : "Pending"}
                      </span>
                      <span className="text-xs" style={{ color: "#b0a090" }}>
                        {latest.flooded_pct !== null ? `${latest.flooded_pct}% flooded` : ""}
                      </span>
                    </div>

                    <div className="mt-2 flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 rounded-full" style={{ background: "#7ab060" }} />
                      <span className="text-xs" style={{ color: "#b0a090" }}>
                        {latest.date} · {scenes.length} scene{scenes.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </main>
      </div>
    </div>
  );
}
