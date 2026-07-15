// Plain-English reading of a prediction, generated from its own stats
// (flooded_pct, whether permanent water was excluded) — never hardcoded per scene.

import { SceneArchive } from "@/types/location";

const NONE_TEXT = "No significant flooding was detected in the analyzed satellite image. The area appears to be largely dry";
const LIMITED_TEXT = "Only a small amount of flooding has been detected in this location. Most of the area appears unaffected, with flooding limited to a few isolated regions.";
const MODERATE_TEXT = "Flooding has been detected in several areas of the region. The highlighted regions indicate land that appears to be temporarily covered by floodwater.";
const EXTENSIVE_TEXT = "Extensive flooding has been detected across most of the analyzed area. Large portions of the land appear to be covered by floodwater, consistent with a significant flood event.";

const WATER_CLAUSE = ", with permanent water bodies excluded from the result.";
const WATER_SENTENCE = " Permanent water bodies have been excluded, so the map focuses on newly flooded areas.";

/** Builds a 2-4 sentence, jargon-free interpretation from prediction stats, or null if no stats are available yet. */
export function buildInterpretation(scene: SceneArchive): string | null {
  if (scene.flooded_pct === null) return null;

  const pct = scene.flooded_pct;
  const excludesPermanentWater = !!scene.permanent_water_url;

  if (pct <= 0) {
    return excludesPermanentWater ? `${NONE_TEXT}${WATER_CLAUSE}` : `${NONE_TEXT}.`;
  }

  const waterSentence = excludesPermanentWater ? WATER_SENTENCE : "";

  if (pct <= 10) return LIMITED_TEXT + waterSentence;
  if (pct <= 35) return MODERATE_TEXT + waterSentence;
  return EXTENSIVE_TEXT + waterSentence;
}
