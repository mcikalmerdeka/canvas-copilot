"""Curated catalog for the guided prompt builder.

The catalog is intentionally plain data: every option pairs a short, human label
with the richer descriptive phrase the model receives. Keeping the vocabulary in
one module means the UI, presets, and brief renderer can never drift apart, and
adding an option is a one-line change.

`build_brief` is the only piece of logic here. It turns the current selections
into a labeled brief — an intermediate representation that `gpt-6-luna` rewrites
into the chosen output format. It never calls Streamlit or the network, so the
whole catalog is cheap to unit-test.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DEFAULT_LABEL = "Not specified"

SUBJECT_TYPES: tuple[str, ...] = ("Person", "Group", "Product", "Place", "Creature")

# Display order for the brief and the builder's section list.
GROUPS: tuple[tuple[str, str], ...] = (
    ("subject", "Subject"),
    ("character", "Character"),
    ("scene", "Scene & light"),
    ("camera", "Camera & composition"),
    ("style", "Style & mood"),
    ("finishing", "Finishing touches"),
)

CHARACTER_SUBJECT_TYPES: tuple[str, ...] = ("Person", "Group")


@dataclass(frozen=True)
class FacetOption:
    """One selectable value: a plain label plus the prose sent to the model."""

    label: str
    text: str


@dataclass(frozen=True)
class Facet:
    """One builder control.

    `kind` is "select", "multiselect", or "text". Text facets have no options;
    their value is forwarded verbatim. `applies_to` empty means every subject
    type, otherwise the facet only appears for the listed types.
    """

    id: str
    label: str
    group: str
    help: str
    options: tuple[FacetOption, ...] = ()
    kind: str = "select"
    applies_to: tuple[str, ...] = ()


def _options(*pairs: tuple[str, str]) -> tuple[FacetOption, ...]:
    return tuple(FacetOption(label, text) for label, text in pairs)


FACETS: tuple[Facet, ...] = (
    # --- Character (people only) -------------------------------------------
    Facet(
        id="character_age",
        label="Age",
        group="character",
        help="Every option describes a clearly adult subject.",
        applies_to=CHARACTER_SUBJECT_TYPES,
        options=_options(
            ("20s", "Adult in their early-to-mid 20s"),
            ("30s", "Adult in their early-to-mid 30s"),
            ("40s", "Adult in their early-to-mid 40s"),
            ("50s", "Adult in their early-to-mid 50s"),
            ("60s+", "Adult aged 60 or older, with natural age character"),
        ),
    ),
    Facet(
        id="character_gender",
        label="Gender presentation",
        group="character",
        help="A visual description, not an identity label.",
        applies_to=CHARACTER_SUBJECT_TYPES,
        options=_options(
            ("Woman", "Adult woman"),
            ("Man", "Adult man"),
            ("Androgynous", "Adult with an androgynous presentation"),
        ),
    ),
    Facet(
        id="character_hair_style",
        label="Hair style",
        group="character",
        help="Shape and length of the hair.",
        applies_to=CHARACTER_SUBJECT_TYPES,
        options=_options(
            ("Short bob", "A chin-length short bob"),
            ("Short textured", "Short, textured hair with a natural finish"),
            ("Long straight", "Long, straight hair falling past the shoulders"),
            ("Wavy shoulder-length", "Wavy, shoulder-length hair"),
            ("Natural curls", "Natural, voluminous curls"),
            ("High ponytail", "Hair pulled into a high ponytail"),
            ("Sleek low bun", "A sleek low bun"),
            ("Messy bun", "A loosely gathered messy bun with flyaway strands"),
            ("Pixie cut", "A short, textured pixie cut"),
            ("Buzz cut", "A close-cropped buzz cut"),
            ("Braid", "Hair gathered into a loose side braid"),
            ("Bald", "A shaved or bald head"),
        ),
    ),
    Facet(
        id="character_hair_color",
        label="Hair colour",
        group="character",
        help="Natural or styled colour.",
        applies_to=CHARACTER_SUBJECT_TYPES,
        options=_options(
            ("Black", "Black hair with a natural sheen"),
            ("Dark brown", "Dark brown hair"),
            ("Chestnut", "Chestnut-brown hair with warm auburn undertones"),
            ("Blonde", "Blonde hair, natural and slightly lived-in"),
            ("Copper red", "Copper-red hair"),
            ("Silver grey", "Silver-grey hair"),
            ("Salt-and-pepper", "Salt-and-pepper hair"),
            ("Pastel", "Hair dyed a soft pastel tone"),
        ),
    ),
    Facet(
        id="character_outfit",
        label="Outfit",
        group="character",
        help="Wardrobe and styling. Keep the scene and weather in mind.",
        applies_to=CHARACTER_SUBJECT_TYPES,
        options=_options(
            (
                "Tailored editorial",
                "A sharply tailored suit with a silk shirt, editorial styling",
            ),
            (
                "Casual denim",
                "A worn denim jacket over a plain white T-shirt and jeans",
            ),
            (
                "Streetwear",
                "Layered streetwear — an oversized hoodie, cargo trousers, and chunky sneakers",
            ),
            (
                "Summer linen",
                "A lightweight linen shirt and relaxed trousers, summer-weight",
            ),
            ("Athletic", "Fitted athletic wear with a light windbreaker"),
            (
                "Evening formal",
                "An elegant evening outfit with a floor-length dress or tuxedo and minimal jewellery",
            ),
            (
                "Work uniform",
                "A clean, practical work uniform with rolled sleeves",
            ),
            (
                "Vintage",
                "Vintage-inspired clothing with visible texture and lived-in details",
            ),
            (
                "Minimal monochrome",
                "A minimal monochrome outfit in a single restrained palette",
            ),
            (
                "Structured outerwear",
                "A structured overcoat layered over knitwear",
            ),
            (
                "Traditional-inspired",
                "Contemporary clothing with traditional tailoring details",
            ),
            (
                "Creative layering",
                "Creative layering with mixed textures, scarves, and statement accessories",
            ),
        ),
    ),
    Facet(
        id="character_pose",
        label="Pose",
        group="character",
        help="What the body is doing at the moment of capture.",
        applies_to=CHARACTER_SUBJECT_TYPES,
        options=_options(
            (
                "Standing relaxed",
                "Standing naturally with relaxed shoulders, weight settled on one leg",
            ),
            ("Candid mid-action", "Captured mid-action, unaware of the camera"),
            ("Seated", "Seated with an easy, unforced posture"),
            (
                "Leaning",
                "Leaning against a wall or doorway with one shoulder",
            ),
            ("Walking", "Walking toward the camera in mid-stride"),
            ("Looking back", "Looking back over one shoulder, caught in motion"),
            (
                "Hands in pockets",
                "Standing with both hands in pockets, shoulders loose",
            ),
            ("Arms crossed", "Standing with arms loosely crossed"),
            ("Crouching", "Crouching low with forearms resting on the knees"),
            ("Stretching", "Stretching after exercise, caught mid-movement"),
            (
                "Dancing",
                "Dancing mid-motion, clothes and hair following the movement",
            ),
            ("Reclining", "Reclining comfortably on a sofa or low wall"),
            ("Working", "Focused on a practical task with both hands busy"),
        ),
    ),
    Facet(
        id="character_expression",
        label="Expression",
        group="character",
        help="Keep it unposed; candid expressions read as more real.",
        applies_to=CHARACTER_SUBJECT_TYPES,
        options=_options(
            (
                "Natural",
                "A natural, unposed expression, neither smiling nor serious",
            ),
            ("Soft smile", "A soft, genuine half-smile"),
            ("Laughing", "A genuine laugh caught mid-breath"),
            ("Calm and serious", "A calm, serious expression with a steady gaze"),
            ("Thoughtful", "A thoughtful expression, eyes slightly narrowed"),
            ("Surprised", "A brief expression of pleasant surprise"),
            (
                "Quiet confidence",
                "A quietly confident expression with a direct gaze",
            ),
            (
                "Averted gaze",
                "Eyes averted from the camera, absorbed in the moment",
            ),
        ),
    ),
    Facet(
        id="character_details",
        label="Character details",
        group="character",
        help="Glasses, freckles, tattoos, makeup, jewellery, distinguishing features.",
        kind="text",
        applies_to=CHARACTER_SUBJECT_TYPES,
    ),
    # --- Scene -------------------------------------------------------------
    Facet(
        id="setting",
        label="Setting",
        group="scene",
        help="Where the scene takes place.",
        options=_options(
            (
                "City street",
                "A busy city street with layered signage, shopfronts, and passing pedestrians",
            ),
            (
                "Café interior",
                "A lived-in café interior with warm wood, ceramic cups, and window light",
            ),
            (
                "Photo studio",
                "A seamless studio backdrop with controlled light and clean negative space",
            ),
            (
                "Beach",
                "An open beach with wet sand, low dunes, and a wide horizon",
            ),
            (
                "Forest",
                "A dense forest with layered undergrowth and filtered light",
            ),
            (
                "Mountain landscape",
                "A mountain landscape with a strong ridge line and open sky",
            ),
            (
                "Modern apartment",
                "A modern apartment interior with clean lines and soft textures",
            ),
            ("Office", "A working office with desks, monitors, and practical clutter"),
            (
                "Train station",
                "A train station platform with signage, benches, and passing travellers",
            ),
            ("Rooftop", "A rooftop overlooking a city skyline"),
            (
                "Desert",
                "A desert landscape with sculpted dunes and a vast sky",
            ),
            ("Garden", "A lush garden with layered planting and dappled shade"),
            (
                "Art gallery",
                "A quiet art gallery with pale walls and polished floors",
            ),
            (
                "Workshop",
                "A working workshop with tools, wood shavings, and practical light",
            ),
            (
                "Night market",
                "A neon-lit night market with stalls, steam, and reflected colour",
            ),
            (
                "Minimal backdrop",
                "A simple seamless backdrop with a single tonal colour",
            ),
            (
                "Old town alley",
                "A narrow old-town alley with worn plaster and laundry lines",
            ),
            ("Lakeside", "A calm lakeside with a mirrored surface and distant treeline"),
        ),
    ),
    Facet(
        id="time_of_day",
        label="Time of day",
        group="scene",
        help="Sets the quality and direction of the available light.",
        options=_options(
            ("Dawn", "Dawn, with cool low light and quiet streets"),
            ("Morning", "Morning, with clear directional light"),
            ("Midday", "Midday, with bright overhead light and short shadows"),
            ("Golden hour", "Golden hour, with warm low sun and long shadows"),
            (
                "Blue hour",
                "Blue hour, with cool ambient light and early artificial lights",
            ),
            ("Night", "Night, with artificial light and deep shadows"),
            ("Overcast", "An overcast day with soft, even light"),
            (
                "Indoors",
                "Indoors, where the time of day only shows through windows",
            ),
        ),
    ),
    Facet(
        id="lighting",
        label="Lighting",
        group="scene",
        help="The single highest-leverage choice for mood and realism.",
        options=_options(
            (
                "Soft window light",
                "Soft window light from one side, with gentle falloff",
            ),
            (
                "Overcast diffused",
                "Even, diffused overcast light with no hard shadows",
            ),
            (
                "Golden backlight",
                "Warm golden-hour backlight with a glowing rim and soft lens flare",
            ),
            (
                "Hard midday sun",
                "Hard midday sun with crisp shadows and high contrast",
            ),
            (
                "Single softbox",
                "A single large softbox from the left at roughly 45 degrees, with a subtle rim light",
            ),
            (
                "Rembrandt",
                "Rembrandt lighting with a small triangle of light on the shadowed cheek",
            ),
            (
                "Rim light",
                "A strong rim light separating the subject from a darker background",
            ),
            ("Neon practicals", "Colourful neon practicals with mixed reflections"),
            ("Candlelight", "Warm candlelight with a soft, flickering falloff"),
            (
                "Chiaroscuro",
                "Dramatic chiaroscuro with deep shadow and selective highlights",
            ),
            (
                "Fluorescent office",
                "Cool fluorescent office light with a slight green cast",
            ),
            (
                "Mixed ambient",
                "Mixed ambient light — window light blended with warm interior lamps",
            ),
            (
                "Studio beauty",
                "Clean beauty-dish lighting with soft shadows and a catchlight in the eyes",
            ),
        ),
    ),
    Facet(
        id="weather_atmosphere",
        label="Atmosphere",
        group="scene",
        help="Weather and air, even indoors.",
        options=_options(
            ("Clear", "Clear air and crisp visibility"),
            ("Light rain", "Light rain with wet surfaces and soft reflections"),
            ("Fog", "Fog softening the background and reducing contrast"),
            ("Snow", "Falling snow and cold, muffled light"),
            ("Wind", "Visible wind moving hair, fabric, and loose debris"),
            ("Haze", "Warm atmospheric haze compressing the distance"),
            ("Dust", "Fine dust suspended in the light"),
            ("Humid", "Humid air with a slight bloom around highlights"),
            ("Storm", "Approaching storm light with bruised clouds"),
        ),
    ),
    Facet(
        id="props_details",
        label="Props & set details",
        group="scene",
        help="Specific objects that make the place feel lived in.",
        kind="text",
    ),
    # --- Camera ------------------------------------------------------------
    Facet(
        id="framing",
        label="Framing",
        group="camera",
        help="How much of the subject and scene is in frame.",
        options=_options(
            ("Extreme close-up", "An extreme close-up isolating one feature"),
            ("Close-up", "A close-up of the face and shoulders"),
            ("Medium close-up", "A medium close-up from the chest up"),
            ("Medium shot", "A medium shot from the waist up"),
            ("Cowboy shot", "A three-quarter shot from mid-thigh up"),
            ("Full body", "A full-body shot with the subject entirely in frame"),
            (
                "Wide environmental",
                "A wide environmental shot with the subject small in a large scene",
            ),
            ("Macro detail", "A macro detail shot of texture and surface"),
            (
                "Product hero",
                "A hero product shot with the object filling most of the frame",
            ),
            ("Overhead flat lay", "A top-down flat lay from directly above"),
            ("Group shot", "A group shot with every subject readable"),
        ),
    ),
    Facet(
        id="camera_angle",
        label="Camera angle",
        group="camera",
        help="Where the camera sits relative to the subject.",
        options=_options(
            ("Eye level", "Shot at eye level"),
            ("Low angle", "A low angle looking slightly up at the subject"),
            ("High angle", "A high angle looking slightly down"),
            ("Dutch tilt", "A subtle Dutch tilt, four to seven degrees off level"),
            ("Over the shoulder", "An over-the-shoulder view"),
            ("Overhead", "A bird's-eye overhead view"),
            ("Worm's eye", "A worm's-eye view from near the ground"),
            (
                "From behind",
                "Shot from behind the subject, looking past them into the scene",
            ),
            (
                "Three-quarter",
                "A three-quarter view of the subject's face and body",
            ),
        ),
    ),
    Facet(
        id="lens",
        label="Lens",
        group="camera",
        help="Focal length changes compression, perspective, and intimacy.",
        options=_options(
            ("24mm wide", "A 24mm wide-angle lens with visible perspective"),
            ("28mm street", "A 28mm street-photography lens"),
            ("35mm documentary", "A 35mm documentary lens"),
            ("50mm natural", "A 50mm lens with a natural field of view"),
            ("85mm portrait", "An 85mm portrait lens with gentle compression"),
            ("100mm macro", "A 100mm macro lens"),
            ("135mm telephoto", "A 135mm telephoto lens with strong compression"),
            ("200mm telephoto", "A 200mm telephoto lens"),
            (
                "Phone main",
                "A smartphone main camera, roughly 24–26mm equivalent",
            ),
            (
                "Phone 3x",
                "A smartphone 3× zoom, with slight digital softness",
            ),
        ),
    ),
    Facet(
        id="depth_of_field",
        label="Depth of field",
        group="camera",
        help="How much of the scene stays sharp.",
        options=_options(
            (
                "Deep focus",
                "Deep focus, with foreground and background both readable",
            ),
            ("Moderate", "Moderate depth of field"),
            (
                "Shallow",
                "Shallow depth of field, f/2.8, with a softly blurred background",
            ),
            (
                "Very shallow",
                "Very shallow depth of field, f/1.4, with pronounced background blur",
            ),
            (
                "Bokeh",
                "A creamy bokeh background with distinct out-of-focus highlights",
            ),
            ("No blur", "No intentional background blur"),
        ),
    ),
    Facet(
        id="composition",
        label="Composition",
        group="camera",
        help="Where the eye travels inside the frame.",
        options=_options(
            (
                "Rule of thirds",
                "Rule-of-thirds composition, subject offset from centre",
            ),
            ("Centered", "Centred, symmetrical composition"),
            ("Negative space", "Generous negative space around the subject"),
            (
                "Leading lines",
                "Leading lines drawing the eye toward the subject",
            ),
            (
                "Frame within frame",
                "A frame-within-a-frame composition using a doorway or window",
            ),
            (
                "Off-center candid",
                "An off-centre candid composition that feels accidental",
            ),
            ("Diagonal", "A strong diagonal composition"),
            (
                "Layered depth",
                "Layered foreground, midground, and background",
            ),
        ),
    ),
    # --- Style -------------------------------------------------------------
    Facet(
        id="medium_style",
        label="Medium & style",
        group="style",
        help="The visual language: photograph, painting, illustration, or render.",
        options=_options(
            (
                "Editorial photo",
                "Editorial fashion photography with polished art direction",
            ),
            (
                "Candid smartphone",
                "A candid smartphone snapshot with natural imperfection",
            ),
            (
                "Cinematic still",
                "A cinematic film still with controlled colour and atmosphere",
            ),
            (
                "Documentary",
                "Documentary photojournalism, observational and unstaged",
            ),
            (
                "Analog film",
                "Analog 35mm film photography, Kodak Portra 400 palette, gentle grain",
            ),
            (
                "Studio product",
                "Studio product photography with clean light and controlled reflections",
            ),
            (
                "Flat vector",
                "Flat vector illustration with clean shapes and a limited palette",
            ),
            (
                "Watercolour",
                "Watercolour illustration with visible paper texture and soft edges",
            ),
            (
                "Oil painting",
                "Oil painting with visible brushwork and layered glazes",
            ),
            (
                "3D render",
                "A polished 3D render with physically based materials",
            ),
            (
                "Anime background",
                "Anime-style background painting with painterly detail",
            ),
            (
                "Pencil sketch",
                "Graphite pencil sketch with visible linework and shading",
            ),
            (
                "Editorial illustration",
                "Editorial illustration with a bold, graphic visual language",
            ),
            (
                "Mixed-media collage",
                "Mixed-media collage with torn paper and layered texture",
            ),
        ),
    ),
    Facet(
        id="palette",
        label="Colour palette",
        group="style",
        help="Two to five named colours beat 'colourful' every time.",
        options=_options(
            ("Natural muted", "A natural, muted palette with soft earth tones"),
            (
                "Warm neutrals",
                "A warm neutral palette of ivory, camel, and soft grey",
            ),
            (
                "Warm amber",
                "A warm amber palette with honey highlights and brown shadows",
            ),
            ("Cool teal", "A cool teal-and-cyan palette with clean highlights"),
            ("Black and white", "High-contrast black and white"),
            ("Pastel", "A soft pastel palette with low saturation"),
            ("Earth tones", "An earthy palette of clay, olive, sand, and charcoal"),
            ("Neon contrast", "Neon magenta and cyan against deep shadows"),
            (
                "Monochrome accent",
                "A monochrome scene with a single saturated accent colour",
            ),
            (
                "Desaturated cinematic",
                "A desaturated cinematic palette with teal shadows and warm skin tones",
            ),
            ("Jewel tones", "Rich jewel tones — emerald, sapphire, and ruby"),
        ),
    ),
    Facet(
        id="mood",
        label="Mood",
        group="style",
        help="The emotional target, stated visually.",
        options=_options(
            (
                "Calm",
                "Calm and quiet, with a slow, contemplative atmosphere",
            ),
            ("Joyful", "Joyful and playful, with light energy"),
            (
                "Nostalgic",
                "Nostalgic and bittersweet, like a remembered moment",
            ),
            (
                "Tense",
                "Tense and dramatic, with a sense of anticipation",
            ),
            (
                "Luxurious",
                "Luxurious and composed, with restrained elegance",
            ),
            ("Melancholic", "Melancholic and introspective"),
            ("Energetic", "Energetic and kinetic"),
            ("Serene", "Serene and open, with breathing room"),
            (
                "Mysterious",
                "Mysterious and ambiguous, with information withheld",
            ),
            ("Intimate", "Intimate and close, as if the viewer is nearby"),
        ),
    ),
    # --- Finishing ---------------------------------------------------------
    Facet(
        id="extra_notes",
        label="Extra notes",
        group="finishing",
        help="Anything else: specific props, on-image text, a reference vibe.",
        kind="text",
    ),
    Facet(
        id="negative_constraints",
        label="Avoid",
        group="finishing",
        help="Common failure modes to keep out of the final image.",
        kind="multiselect",
        options=_options(
            (
                "No text or watermarks",
                "No text, watermarks, logos, or signatures",
            ),
            (
                "No logos or branding",
                "No visible brand logos or trademarked designs",
            ),
            (
                "No distorted anatomy",
                "No extra fingers, duplicated limbs, or distorted anatomy",
            ),
            (
                "No plastic skin",
                "No plastic skin, heavy beauty filters, or doll-like faces",
            ),
            (
                "No harsh flash",
                "No harsh on-camera flash or deer-in-headlights lighting",
            ),
            (
                "No busy background",
                "No busy or distracting background clutter",
            ),
            (
                "No artificial blur",
                "No artificial portrait-mode blur or fake bokeh",
            ),
            (
                "No oversaturation",
                "No oversaturated colours or heavy HDR processing",
            ),
            (
                "No posed smiles",
                "No posed camera smiles or staged Instagram posing",
            ),
        ),
    ),
    Facet(
        id="custom_negatives",
        label="Additional avoid notes",
        group="finishing",
        help="Anything else that must not appear.",
        kind="text",
    ),
)

FACET_BY_ID: dict[str, Facet] = {facet.id: facet for facet in FACETS}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, object]] = {
    "fresh": {
        "label": "Start fresh",
        "subject_type": "Person",
        "subject": "",
        "values": {},
    },
    "editorial_portrait": {
        "label": "Editorial portrait",
        "subject_type": "Person",
        "subject": "A woman in her early 30s with a calm, self-assured presence and a direct gaze.",
        "values": {
            "character_age": "30s",
            "character_gender": "Woman",
            "character_hair_style": "Sleek low bun",
            "character_hair_color": "Dark brown",
            "character_outfit": "Tailored editorial",
            "character_pose": "Standing relaxed",
            "character_expression": "Quiet confidence",
            "setting": "Photo studio",
            "time_of_day": "Indoors",
            "lighting": "Single softbox",
            "framing": "Medium close-up",
            "camera_angle": "Eye level",
            "lens": "85mm portrait",
            "depth_of_field": "Shallow",
            "composition": "Centered",
            "medium_style": "Editorial photo",
            "palette": "Warm neutrals",
            "mood": "Luxurious",
            "negative_constraints": [
                "No text or watermarks",
                "No plastic skin",
                "No posed smiles",
            ],
        },
    },
    "candid_street": {
        "label": "Candid street",
        "subject_type": "Person",
        "subject": "A man in his late 20s walking through a busy city, caught mid-stride.",
        "values": {
            "character_age": "20s",
            "character_gender": "Man",
            "character_hair_style": "Short textured",
            "character_outfit": "Streetwear",
            "character_pose": "Walking",
            "character_expression": "Averted gaze",
            "setting": "City street",
            "time_of_day": "Golden hour",
            "lighting": "Golden backlight",
            "weather_atmosphere": "Clear",
            "framing": "Medium shot",
            "camera_angle": "Eye level",
            "lens": "28mm street",
            "depth_of_field": "Moderate",
            "composition": "Off-center candid",
            "medium_style": "Documentary",
            "palette": "Natural muted",
            "mood": "Energetic",
            "negative_constraints": [
                "No text or watermarks",
                "No posed smiles",
                "No oversaturation",
            ],
        },
    },
    "golden_hour_lifestyle": {
        "label": "Golden-hour lifestyle",
        "subject_type": "Person",
        "subject": "A woman in her mid 20s standing in a sunlit meadow, laughing at something off-camera.",
        "values": {
            "character_age": "20s",
            "character_gender": "Woman",
            "character_hair_style": "Wavy shoulder-length",
            "character_hair_color": "Blonde",
            "character_outfit": "Summer linen",
            "character_pose": "Candid mid-action",
            "character_expression": "Laughing",
            "setting": "Garden",
            "time_of_day": "Golden hour",
            "lighting": "Golden backlight",
            "weather_atmosphere": "Clear",
            "framing": "Medium shot",
            "camera_angle": "Eye level",
            "lens": "50mm natural",
            "depth_of_field": "Shallow",
            "composition": "Negative space",
            "medium_style": "Analog film",
            "palette": "Warm amber",
            "mood": "Joyful",
            "negative_constraints": [
                "No plastic skin",
                "No posed smiles",
                "No oversaturation",
            ],
        },
    },
    "studio_product": {
        "label": "Studio product",
        "subject_type": "Product",
        "subject": "A matte ceramic coffee cup on a low stone plinth.",
        "values": {
            "setting": "Photo studio",
            "lighting": "Single softbox",
            "props_details": "A single dried branch casting a soft shadow across the plinth.",
            "framing": "Product hero",
            "camera_angle": "Three-quarter",
            "lens": "100mm macro",
            "depth_of_field": "Shallow",
            "composition": "Centered",
            "medium_style": "Studio product",
            "palette": "Warm neutrals",
            "mood": "Luxurious",
            "negative_constraints": [
                "No text or watermarks",
                "No logos or branding",
                "No busy background",
            ],
        },
    },
    "moody_cinematic": {
        "label": "Moody cinematic",
        "subject_type": "Person",
        "subject": "A man in his 40s standing in a rain-slicked alley at night, looking past the camera.",
        "values": {
            "character_age": "40s",
            "character_gender": "Man",
            "character_hair_style": "Buzz cut",
            "character_hair_color": "Salt-and-pepper",
            "character_outfit": "Structured outerwear",
            "character_pose": "Hands in pockets",
            "character_expression": "Calm and serious",
            "setting": "Old town alley",
            "time_of_day": "Night",
            "lighting": "Neon practicals",
            "weather_atmosphere": "Light rain",
            "framing": "Medium close-up",
            "camera_angle": "Low angle",
            "lens": "35mm documentary",
            "depth_of_field": "Shallow",
            "composition": "Layered depth",
            "medium_style": "Cinematic still",
            "palette": "Desaturated cinematic",
            "mood": "Tense",
            "negative_constraints": [
                "No text or watermarks",
                "No plastic skin",
                "No oversaturation",
            ],
        },
    },
    "travel_landscape": {
        "label": "Travel landscape",
        "subject_type": "Place",
        "subject": "A lone stone chapel on a ridge above a sea of low clouds at sunrise.",
        "values": {
            "setting": "Mountain landscape",
            "time_of_day": "Dawn",
            "lighting": "Golden backlight",
            "weather_atmosphere": "Haze",
            "framing": "Wide environmental",
            "camera_angle": "High angle",
            "lens": "24mm wide",
            "depth_of_field": "Deep focus",
            "composition": "Leading lines",
            "medium_style": "Cinematic still",
            "palette": "Warm amber",
            "mood": "Serene",
            "custom_negatives": "No crowds or tourist clutter.",
        },
    },
    "illustration": {
        "label": "Illustration",
        "subject_type": "Creature",
        "subject": "A small fox-like forest spirit curled asleep inside a giant acorn.",
        "values": {
            "setting": "Forest",
            "time_of_day": "Blue hour",
            "lighting": "Rim light",
            "weather_atmosphere": "Fog",
            "framing": "Medium shot",
            "camera_angle": "Eye level",
            "lens": "50mm natural",
            "depth_of_field": "Moderate",
            "composition": "Frame within frame",
            "medium_style": "Watercolour",
            "palette": "Pastel",
            "mood": "Mysterious",
            "custom_negatives": "No photorealism, no harsh outlines.",
        },
    },
}


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def facets_for(subject_type: str) -> tuple[Facet, ...]:
    """Return the facets that apply to `subject_type`, in catalog order."""
    return tuple(
        facet
        for facet in FACETS
        if not facet.applies_to or subject_type in facet.applies_to
    )


def option_text(facet_id: str, label: str) -> str:
    """Map a select label to its prose; empty for the default or unknown labels."""
    if not label or label == DEFAULT_LABEL:
        return ""
    facet = FACET_BY_ID.get(facet_id)
    if facet is None:
        return ""
    for option in facet.options:
        if option.label == label:
            return option.text
    return ""


def preset_values(preset_id: str) -> dict[str, str | list[str]]:
    """Return a deep-enough copy of a preset's facet values."""
    preset = PRESETS[preset_id]
    values = preset["values"]
    assert isinstance(values, dict)
    return {
        facet_id: list(value) if isinstance(value, list) else value
        for facet_id, value in values.items()
    }


def _facet_line(facet: Facet, value: object) -> str:
    """Render one facet as a labeled line, or empty when it carries no value."""
    if facet.kind == "text":
        text = str(value).strip() if value is not None else ""
        return f"{facet.label}: {text}" if text else ""
    if facet.kind == "multiselect":
        labels = [label for label in (value or []) if label and label != DEFAULT_LABEL]
        texts = [option_text(facet.id, label) for label in labels]
        joined = "; ".join(text for text in texts if text)
        return f"{facet.label}: {joined}" if joined else ""
    text = option_text(facet.id, str(value or ""))
    return f"{facet.label}: {text}" if text else ""


def build_brief(
    subject: str,
    subject_type: str,
    selections: Mapping[str, str | Sequence[str]],
    notes: str = "",
) -> str:
    """Compose the labeled creative brief sent to the prompt-writing model.

    Only populated facets are emitted, so the brief stays proportional to what
    the user actually chose. The subject and subject type lead, matching how the
    image models weight prompt order.
    """
    blocks: list[str] = []
    subject_text = subject.strip()
    if subject_text:
        blocks.append(f"SUBJECT\n{subject_text}")
    if subject_type:
        blocks.append(f"SUBJECT TYPE\n{subject_type}")

    for group_id, group_label in GROUPS:
        if group_id in ("subject", "finishing"):
            continue
        lines = []
        for facet in facets_for(subject_type):
            if facet.group != group_id:
                continue
            # A bald head has no hair colour; skip a contradictory selection.
            if (
                facet.id == "character_hair_color"
                and selections.get("character_hair_style") == "Bald"
            ):
                continue
            line = _facet_line(facet, selections.get(facet.id, ""))
            if line:
                lines.append(line)
        if lines:
            blocks.append(f"{group_label.upper()}\n" + "\n".join(lines))

    finishing_lines = []
    for facet in facets_for(subject_type):
        if facet.group != "finishing":
            continue
        line = _facet_line(facet, selections.get(facet.id, ""))
        if line:
            finishing_lines.append(line)
    if notes.strip():
        finishing_lines.append(f"Notes: {notes.strip()}")
    if finishing_lines:
        blocks.append("FINISHING TOUCHES\n" + "\n".join(finishing_lines))

    return "\n\n".join(blocks)


def randomize_selections(
    subject_type: str,
    rng: random.Random | None = None,
) -> dict[str, str | list[str]]:
    """Pick a valid, varied option for every non-text facet.

    Used by the builder's "Surprise me" action. Text facets are left untouched
    because random prose would be noise rather than control.
    """
    picker = rng or random.Random()
    values: dict[str, str | list[str]] = {}
    for facet in facets_for(subject_type):
        if facet.kind == "text" or not facet.options:
            continue
        labels = [option.label for option in facet.options]
        if facet.kind == "multiselect":
            count = min(2, len(labels))
            values[facet.id] = picker.sample(labels, count)
        else:
            values[facet.id] = picker.choice(labels)
    return values
