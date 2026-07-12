import Link from "next/link";
import { Location } from "@/types/location";
import locationsData from "@/public/data/locations.json";
import LocationView from "./LocationView";

const locations = locationsData as Location[];

export function generateStaticParams() {
  return locations.map((loc) => ({ id: loc.id }));
}

export default async function LocationPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const location = locations.find((l) => l.id === id);

  if (!location || !location.scenes || location.scenes.length === 0) {
    return (
      <div className="flex h-screen items-center justify-center" style={{ background: "#f0ece4" }}>
        <div className="text-center">
          <div className="text-sm font-medium mb-2" style={{ color: "#c8622a" }}>Location not found</div>
          <Link href="/" className="text-xs underline" style={{ color: "#9a8f7e" }}>Back to locations</Link>
        </div>
      </div>
    );
  }

  return <LocationView location={location} />;
}