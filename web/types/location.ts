export interface SceneArchive {
  scene_id: string;
  timestamp: number;
  date: string;
  flooded_area_km2: number | null;
  flooded_pct: number | null;
  mask_url: string | null;
  sar_url: string | null;
  geotiff_url: string | null;
  permanent_water_url: string | null;
  probability_url: string | null;
}

export interface Location {
  id: string;
  name: string;
  center: [number, number];
  bounds: [[number, number], [number, number]];
  model: string;
  scenes: SceneArchive[];
}