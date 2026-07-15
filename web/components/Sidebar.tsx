"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Location } from "@/types/location";
import { severityColor, severityLabel } from "@/lib/severity";

interface Props {
  locations: Location[];
  activeId?: string;
}

function latestSceneOf(loc: Location) {
  return [...loc.scenes].sort((a, b) => (a.date < b.date ? 1 : -1))[0];
}

const LEGEND_STOPS = [5, 15, 35, 60, 90];

export default function Sidebar({ locations, activeId }: Props) {
  const pathname = usePathname();
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    const withLatest = locations.map((loc) => ({ loc, latest: latestSceneOf(loc) }));
    const filtered = query.trim()
      ? withLatest.filter(({ loc }) => loc.name.toLowerCase().includes(query.trim().toLowerCase()))
      : withLatest;
    return filtered.sort((a, b) => (b.latest.flooded_pct ?? -1) - (a.latest.flooded_pct ?? -1));
  }, [locations, query]);

  return (
    <aside
      className="w-72 flex flex-col flex-shrink-0 h-full"
      style={{ background: "var(--navy-900)", borderRight: "1px solid var(--navy-700)" }}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-5 pt-6 pb-5">
        <div className="min-w-0">
          <div
            className="text-[15px] font-semibold tracking-tight leading-none"
            style={{ fontFamily: "var(--font-display)", color: "var(--text-on-navy)" }}
          >
            Sea the Flood
          </div>
          <div
            className="text-[10px] mt-1 tracking-wide"
            style={{ fontFamily: "var(--font-mono)", color: "var(--text-on-navy-dim)" }}
          >
            SAR FLOOD MONITORING
          </div>
        </div>
      </div>

      {/* Nav */}
      <div className="px-3">
        <Link
          href="/"
          className="flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg transition-colors"
          style={{
            background: pathname === "/" ? "var(--navy-700)" : "transparent",
            color: pathname === "/" ? "var(--text-on-navy)" : "var(--text-on-navy-soft)",
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full flex-shrink-0"
            style={{ background: pathname === "/" ? "var(--glow)" : "var(--text-on-navy-dim)" }}
          />
          Overview
        </Link>
      </div>

      {/* Search */}
      <div className="px-5 pt-4 pb-2">
        <div
          className="text-[10px] font-medium tracking-widest mb-2"
          style={{ fontFamily: "var(--font-mono)", color: "var(--text-on-navy-dim)" }}
        >
          MONITORED ZONES · {locations.length}
        </div>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search zones…"
          className="w-full text-xs px-3 py-2 rounded-lg outline-none transition-colors"
          style={{
            background: "var(--navy-800)",
            border: "1px solid var(--navy-700)",
            color: "var(--text-on-navy)",
            fontFamily: "var(--font-mono)",
          }}
        />
      </div>

      {/* Zone list */}
      <nav className="flex-1 overflow-y-auto px-3 pb-3 min-h-0">
        {rows.length === 0 && (
          <div className="text-xs px-3 py-4" style={{ color: "var(--text-on-navy-dim)" }}>
            No zones match “{query}”
          </div>
        )}
        {rows.map(({ loc, latest }) => {
          const active = loc.id === activeId;
          const color = severityColor(latest.flooded_pct);
          return (
            <Link
              key={loc.id}
              href={`/location/${loc.id}`}
              className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg mb-0.5 transition-colors group"
              style={{
                background: active ? "var(--navy-700)" : "transparent",
                borderLeft: active ? "2px solid var(--glow)" : "2px solid transparent",
              }}
            >
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: color, boxShadow: active ? `0 0 6px 1px ${color}88` : "none" }}
              />
              <span
                className="text-xs flex-1 min-w-0 truncate"
                style={{ color: active ? "var(--text-on-navy)" : "var(--text-on-navy-soft)" }}
              >
                {loc.name}
              </span>
              <span
                className="text-[10px] flex-shrink-0"
                style={{ fontFamily: "var(--font-mono)", color: active ? "var(--glow)" : "var(--text-on-navy-dim)" }}
              >
                {latest.flooded_pct !== null ? `${latest.flooded_pct}%` : "—"}
              </span>
            </Link>
          );
        })}
      </nav>

      {/* Legend */}
      <div className="px-5 py-4" style={{ borderTop: "1px solid var(--navy-700)" }}>
        <div
          className="text-[10px] font-medium tracking-widest mb-2.5"
          style={{ fontFamily: "var(--font-mono)", color: "var(--text-on-navy-dim)" }}
        >
          SEVERITY
        </div>
        <div className="flex flex-col gap-1.5">
          {LEGEND_STOPS.map((pct) => (
            <div key={pct} className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: severityColor(pct) }} />
              <span className="text-[11px]" style={{ color: "var(--text-on-navy-soft)" }}>
                {severityLabel(pct)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Status */}
      <div className="px-5 py-3.5 flex items-center gap-2" style={{ borderTop: "1px solid var(--navy-700)" }}>
        <span className="status-dot w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: "var(--ok)" }} />
        <span className="text-[11px]" style={{ color: "var(--text-on-navy-soft)" }}>
          System operational
        </span>
      </div>
    </aside>
  );
}
