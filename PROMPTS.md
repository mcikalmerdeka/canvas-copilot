# Prompt Library for gpt-image-2.5-sunburst

A working library of 60 ready-to-run prompts for the gpt-image-2.5-sunburst model behind this app. The app passes your prompt to the model verbatim, with no rewriting, so what you paste is exactly what the model reads. Size and Quality are set in the app controls, never in the prompt. Every prompt below is self-contained: copy the whole block, paste it into the app, pick the suggested Size and Quality from the note beneath it, and generate.

## The Prompt Formula That Works

GPT-image models of this generation are strong instruction followers. The prompts that produce consistent, controllable results are built as a stack of explicit layers, roughly in this order:

1. **Subject.** Concrete nouns. "A glacial alpine lake with a jagged granite peak" beats "beautiful nature".
2. **Style.** Name the tradition: "editorial fashion photography", "flat vector infographic", "Studio Ghibli-esque background painting".
3. **Composition.** Where things sit and where the eye travels: framing, angle, thirds, leading lines.
4. **Lighting.** Type and direction: "single softbox from the left at 45 degrees", "flat overcast light", "backlit at golden hour".
5. **Color palette.** Explicit colors, hex codes welcome. Two to five named colors beat "colorful" every time.
6. **Camera or medium.** Lens, aperture, film stock, or rendering approach: "85mm at f/1.8", "100mm macro", "loose watercolor with ink linework".
7. **Mood.** A short emotional target: "serene solitude", "bittersweet longing".
8. **Detail level.** How much rendering effort: "high detail", "extremely high detail", or "medium detail, soft edges" when you want simplicity.
9. **Negative instructions.** End with "DO NOT include ..." and list the failure modes common to your genre.

Assembled in miniature: "An elderly lighthouse keeper (subject), oil portrait style (style), waist-up against the tower window (composition), one shaft of morning light (lighting), slate blue and warm ivory (palette), visible brushwork (medium), quiet dignity (mood), richly detailed (detail). DO NOT include storms or crashing waves (negatives)." Every prompt in this file follows that skeleton.

## Tips Specific to This Model and the App

- **Write long, do not ration tokens.** The model accepts up to 32,000 characters and follows detailed instructions well. Precise description beats magic keywords. If a detail matters to you, say it.
- **Put rendered text in double quotes.** Any word you want to appear inside the image should be quoted and spelled exactly, like "Train 70%" or "vector database". Keep on-image text short; a dozen small labels render far more reliably than paragraphs.
- **Aspect ratio lives in the Size control, not the prompt.** The app offers auto, 1024x1024 (square), 1024x1536 (portrait), and 1536x1024 (landscape). Do not write "16:9" or "vertical format" into the prompt. Instead describe a composition that matches the size you picked: "wide panoramic composition" pairs with 1536x1024, "tall vertical composition" pairs with 1024x1536.
- **Quality trades cost for detail.** Use low or medium while iterating on composition, high for most finished images, and xhigh or max for hero pieces and anything with small text. Infographics with labels should run at high or above, or the lettering degrades.
- **Negatives genuinely work.** A closing "DO NOT include ..." clause is one of the most reliable controls you have. Each prompt below carries the negatives that matter for its genre.
- **Iterate one layer at a time.** When a result is close, change a single layer (lighting, palette, lens) rather than rewriting everything, so you can tell what actually moved the result.

## Table of Contents

1. [Infographic: Data Science, Analytics & AI Concepts](#infographic-data-science-analytics--ai-concepts)
2. [Scenery & Landscapes](#scenery--landscapes)
3. [People & Portraits](#people--portraits)
4. [Food & Product Photography](#food--product-photography)
5. [Sci-Fi & Fantasy Concept Art](#sci-fi--fantasy-concept-art)
6. [Architecture & Interiors](#architecture--interiors)
7. [Anime & Illustration Styles](#anime--illustration-styles)
8. [Logos & Brand Marks](#logos--brand-marks)
9. [Business & Marketing Visuals](#business--marketing-visuals)

---

## Infographic: Data Science, Analytics & AI Concepts

The flagship category. Clean, mobile-first educational diagrams for data science and AI carousels: flat vector style, white or very light background, one or two accent colors, labeled boxes and arrows, short legible text, no dark backgrounds, no decorative clutter.

### 1. How to Split Your Dataset: Holdout and K-Fold

```text
Flat vector educational infographic on a pure white background, titled "How to Split Your Dataset", designed as a single social media slide that stays legible on a phone screen.
Top half: one long horizontal rounded rectangle representing a dataset, divided into three labeled segments, "Train 70%" filled solid indigo, "Validation 15%" filled light indigo, "Test 15%" filled coral, with a thin gray arrow flow underneath reading "fit on train, tune on validation, report on test".
Bottom half: a section titled "5-Fold Cross-Validation" showing five compact rows of five blocks each, where each row has exactly one coral block labeled "validation fold" and four indigo blocks labeled "train", with a small curved arrow suggesting the highlighted fold rotates row by row.
Style: clean flat 2D diagram, rounded corners, thin gray connector lines, generous white space, geometric sans-serif labels, palette limited to indigo #4F46E5, light indigo #C7D2FE, coral #F97316, and gray #9CA3AF on white.
Keep every label short and fully readable at thumbnail size.
DO NOT include dark backgrounds, gradients, drop shadows, 3D perspective, mascots, or decorative icons that do not carry meaning.
```

*Suggested Size: 1024x1024 / Quality: high*

### 2. Overfitting vs Underfitting, and the Bias-Variance Tradeoff

```text
Clean educational infographic on a white background, titled "Overfitting vs Underfitting", built as three side-by-side panel cards.
Each panel shows a small scatter plot of gray dots with a fitted curve drawn over them: left panel labeled "Underfit, high bias" with a straight line that misses the curve of the data; middle panel labeled "Good fit" with a smooth curve following the trend; right panel labeled "Overfit, high variance" with a wiggly curve threading through every single point.
Below the three panels, one wide chart labeled "The bias-variance tradeoff" plots two curves against model complexity on the x-axis: a falling indigo line labeled "training error" and a U-shaped coral line labeled "validation error", with a dashed vertical marker at the minimum labeled "sweet spot".
Style: flat 2D diagram, thin axis lines, rounded panel cards with light gray borders, generous white space, geometric sans-serif labels, palette of indigo and coral on white with gray support tones.
All text short and legible at phone size.
DO NOT include dark mode colors, 3D effects, photorealistic textures, or extra decorative elements.
```

*Suggested Size: 1024x1024 / Quality: high*

### 3. The Confusion Matrix, Decoded

```text
Flat educational infographic on a white background, titled "The Confusion Matrix, Decoded", using a spam filter as the running example.
Center: a 2x2 grid of rounded cells. Column headers in small gray text read "Predicted: Spam" and "Predicted: Not spam"; row labels on the left read "Actual: Spam" and "Actual: Not spam".
Cells: top-left "True Positive, 42, spam caught" in solid coral; top-right "False Positive, 3, good email flagged" in light coral with a dashed warning outline; bottom-left "False Negative, 5, spam missed" in light indigo with a dashed warning outline; bottom-right "True Negative, 50, correctly kept" in solid indigo.
Right side: a slim column of four small formula chips reading "Accuracy = (TP + TN) / all", "Precision = TP / (TP + FP)", "Recall = TP / (TP + FN)", "F1 = balance of precision and recall".
Style: flat 2D vector, rounded rectangles, generous spacing, geometric sans-serif, palette limited to indigo, coral, light gray, and white.
Every label must be spelled exactly as quoted and stay legible on mobile.
DO NOT include dark backgrounds, gradients, 3D, or decorative icons.
```

*Suggested Size: 1024x1024 / Quality: xhigh*

### 4. Attention and the LLM Context Window

```text
Clean diagram infographic on a white background, titled "Attention and the Context Window", explaining how a large language model reads text.
Main visual: a horizontal row of small rounded token tiles spelling "the cat sat on the mat and", where the tile "cat" is highlighted coral as the query, and thin curved arcs connect it to the tiles "the", "sat", and "mat", which are highlighted light indigo as strongly attended tokens, with arc thickness showing attention strength.
Above the token row, a long thin bracket labeled "context window" spans all visible tokens, while a few faded gray tiles sit to the right outside the bracket, labeled "out of window, forgotten".
One short caption underneath reads "the model weighs which earlier words matter most".
Style: flat 2D vector diagram, thin lines, rounded shapes, generous white space, geometric sans-serif labels, palette of indigo, coral, and gray on white.
Keep all quoted text exact and legible at phone size.
DO NOT include dark backgrounds, tangled neural network illustrations, 3D, or glow effects.
```

*Suggested Size: 1024x1024 / Quality: high*

### 5. The Feature Engineering Pipeline, and What Mattered

```text
Flat vector infographic on a white background, titled "Feature Engineering Pipeline", flowing left to right in four stages connected by thin gray arrows.
Stage 1: a small table card labeled "raw data" showing messy rows with one missing value and one categorical column.
Stage 2: three stacked process chips labeled "clean", "encode", "scale".
Stage 3: a tidy table card labeled "feature matrix" with neat columns headed "age", "income", "tenure".
Stage 4: a compact card labeled "model" feeding into a final panel titled "What mattered?", showing a SHAP-style summary: four horizontal rows of small dots, one row per feature, red dots pushing right labeled "raises prediction" and blue dots pushing left labeled "lowers prediction", with feature names "income", "tenure", "age", "region" listed on the left of the rows.
Style: flat 2D diagram, rounded rectangles, thin connectors, generous white space, geometric sans-serif, palette limited to indigo, coral, light red, light blue, and gray on white.
All quoted labels exact and legible on mobile.
DO NOT include dark backgrounds, 3D, gradients, or decorative icons without meaning.
```

*Suggested Size: 1536x1024 / Quality: xhigh*

### 6. How RAG Works

```text
Clean educational infographic on a white background, titled "How RAG Works", split into two horizontal lanes connected by a shared vector database in the middle.
Top lane labeled "Indexing, done once": three rounded boxes in a row, "documents" shown as a small stack icon, "chunk and embed" shown as a small gear icon, then an arrow pointing down into a cylinder shape labeled "vector database".
Bottom lane labeled "Answering, per question": a box labeled "user question" with the quoted example "what is our refund policy?", an arrow to a box "embed the question", an arrow up to the "vector database" cylinder with a small magnifier labeled "similarity search, top 3 chunks", then an arrow to a box "question + retrieved chunks" feeding into a rounded box labeled "LLM", ending in a speech bubble reading "answer with sources".
Style: flat 2D vector diagram, thin gray arrows, rounded rectangles, generous white space, geometric sans-serif labels, palette of indigo and coral on white with light gray support.
Keep every quoted string exact and legible at phone size.
DO NOT include dark backgrounds, 3D, glow effects, or dense unreadable text.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 7. Gradient Descent on the Loss Landscape

```text
Minimal educational infographic on a white background, titled "Gradient Descent", showing a top-down contour map of a loss landscape.
Draw nested elliptical contour lines in light indigo, spaced tighter near the center, marking a bowl-shaped valley.
Overlay a descending path of coral dots connected by small arrows, starting high on the outermost contour and stepping downhill toward the center, with the final dot resting at the minimum marked by a small star labeled "minimum loss".
Add three short annotations along the path: "start, random weights", "each step follows the slope", "learning rate sets the step size".
Include one tiny side note card reading "too large a step can overshoot the valley".
Style: flat 2D diagram, thin clean lines, generous white space, geometric sans-serif labels, palette limited to indigo, coral, and gray on white.
All quoted text exact and legible on a phone screen.
DO NOT include 3D surfaces, dark backgrounds, photorealistic textures, or decorative clutter.
```

*Suggested Size: 1024x1024 / Quality: high*

### 8. The MLOps Loop, with A/B Testing

```text
Flat vector infographic on a white background, titled "The MLOps Loop", showing a machine learning lifecycle as a circular flow of six rounded stage boxes connected by thin gray arrows: "collect data", "train", "evaluate", "deploy", "monitor", "retrain", with the arrow from "monitor" looping back to "retrain" labeled "drift detected".
Inside the circle, one inset panel titled "A/B test in production" shows two overlapping bell curves, a solid indigo curve labeled "control" and a coral curve labeled "variant", with the overlap region shaded light gray and a dashed vertical line labeled "significance threshold".
Style: flat 2D diagram, rounded rectangles, thin arrows, generous white space, geometric sans-serif labels, palette of indigo, coral, and gray on white.
Keep every label short, exact, and legible on mobile.
DO NOT include dark backgrounds, 3D, gradients, server racks, or decorative icons that do not carry meaning.
```

*Suggested Size: 1024x1024 / Quality: xhigh*

---

## Scenery & Landscapes

Photographic and fine-art landscapes across times of day, weather, and light. Wide shots suit 1536x1024, forest and aurora scenes suit the tall 1024x1536.

### 1. Alpine Lake at Golden Hour

```text
Photorealistic landscape photograph of a glacial alpine lake at golden hour, with a jagged granite peak reflected in perfectly still water.
A small wooden dock extends from the right foreground into the lake, giving the eye a leading line toward the mountain.
Warm low sun from the left rakes across the rock faces and catches drifting mist in the valleys, while shadowed slopes hold deep cool blues.
Palette of warm amber highlights against slate blue shadows, with a pale peach sky.
Shot as if on a full-frame camera with a 24mm wide lens at f/8 for front-to-back sharpness, polarized water clarity, fine film grain.
Mood of serene solitude.
High detail, crisp texture in the rock and in the rippled reflections near the dock.
DO NOT include people, buildings, boats, or oversaturated HDR halos.
```

*Suggested Size: 1536x1024 / Quality: high*

### 2. Fog-Laden Pine Forest

```text
Moody atmospheric photograph of a dense pine forest in early morning fog.
Tall straight trunks recede into layered mist, creating soft depth planes, the nearest plane dark and detailed, each farther plane fading toward pale gray.
The camera looks straight down a narrow gap between the trees, symmetrical composition, slightly low angle.
Diffuse overcast light, no visible sun, muted palette of deep fir green, charcoal, and silver gray.
Fine art photography aesthetic, like a large-format film plate with subtle grain and a long tonal range.
Quiet, contemplative, slightly mysterious mood.
High detail in the bark texture where the fog thins near the camera.
DO NOT include people, paths, wildlife, or saturated colors.
```

*Suggested Size: 1024x1536 / Quality: high*

### 3. Minimalist Desert Dunes

```text
Fine art minimalist photograph of sand dunes in a desert in late afternoon.
The frame is filled with two or three clean dune ridgelines only: no horizon, no plants, no footprints, just sculpted sand and shadow.
Long raking light from a low sun throws one dune face into warm gold and leaves the lee side in deep violet shadow, forming an almost abstract graphic pattern of curves.
Palette of gold, apricot, and deep purple shadow.
Shot as if with a 200mm telephoto lens compressing the ridgelines, crisp edge detail where sand slips over a crest.
Serene, abstract, gallery-print mood.
Medium detail, clean shapes over busy texture.
DO NOT include camels, people, tire tracks, or a busy sky.
```

*Suggested Size: 1536x1024 / Quality: medium*

### 4. Aurora Over a Norwegian Fjord

```text
Photorealistic night landscape of the aurora borealis over a Norwegian fjord.
Green and teal curtains of aurora ripple across a starry sky, with faint pink fringes at the upper edge, reflected in the dark still water of the fjord.
Foreground: the silhouette of a low rocky shoreline with one small red fishing cabin providing a single warm accent and a sense of scale.
Composition: cabin in the lower left third, the aurora band sweeping from the upper right down toward the horizon.
Long-exposure look with sharp stars, smooth water, and crisp silhouette edges.
Cold palette of emerald, teal, deep blue, and black, warmed only by the tiny red cabin.
Awe-filled, silent mood.
Extremely high detail.
DO NOT include the moon, crowds, boats with lights, or blurry smeared aurora.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 5. Terraced Rice Fields at Dawn

```text
Photograph of terraced rice fields in rural Japan at early morning, shot from an elevated vantage point looking down the valley.
Curving terrace walls step down the hillside in repeating arcs, each flooded paddy holding a mirror of the pale morning sky, some rows planted with young green rice seedlings in neat lines.
Soft mist hangs between the far ridges, diffusing the light.
Composition: the terraces fill the frame in a diagonal flow from lower left to upper right, with one small farmhouse with a tiled roof near the top for scale.
Palette of silver water, fresh green, and soft blue gray.
Gentle, pastoral, timeless mood.
High detail in the seedling rows and terrace edges.
DO NOT include tourists, power lines, or harsh midday contrast.
```

*Suggested Size: 1536x1024 / Quality: high*

### 6. Lighthouse in a Storm

```text
Dramatic seascape photograph of a stone lighthouse on a cliff during a storm.
Massive gray waves crash against black rocks below, spray exploding upward and caught mid-air, while torn clouds race across a dark pewter sky.
The lighthouse stands solid on the right third of the frame, its white tower with a red top lit by a single shaft of sunlight breaking through the storm, the only warm note in the scene.
High shutter speed look, freezing the spray into sharp droplets, deep texture in the churning water.
Palette of slate gray, sea green, white foam, and one red accent.
Powerful, sublime, slightly ominous mood.
Extremely high detail.
DO NOT include people, ships, rainbows, or a fully clear sky.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 7. Lavender Rows at Sunrise

```text
Photograph of lavender fields in Provence at sunrise.
Endless rows of lavender run in strict parallel lines toward a lone line of trees and an old stone farmhouse on the horizon, the rows filling the frame with a strong converging perspective from the bottom corners.
Low golden side light skims along the lavender tops, warming the purple blooms and pulling long soft shadows across the dirt paths between rows.
Fine mist near the horizon catches the light.
Palette of violet, gold, and dusty green.
Shot as if with a 35mm lens at f/8, deep field of view, crisp detail in the nearest blooms softening toward the horizon.
Peaceful, warm, nostalgic mood.
DO NOT include people, tractors, or oversaturated neon purple.
```

*Suggested Size: 1536x1024 / Quality: high*

---

## People & Portraits

Portraits across studio, street, cinematic, and documentary traditions. Most suit the tall 1024x1536; the corporate headshot works well square.

### 1. Editorial Studio Fashion Portrait

```text
Editorial fashion photography portrait of a woman in a sculptural mustard yellow wool coat with a high collar, standing against a seamless warm gray studio backdrop.
Three-quarter body framing, confident relaxed pose, one hand in a pocket, chin slightly lifted, direct calm gaze into the camera.
Lighting: a single large softbox from the left at 45 degrees with a subtle silver reflector filling the right, gentle shadow under the jaw, clean catchlights in the eyes.
Shot as if on a medium-format camera with an 85mm lens at f/2.8, shallow depth of field, skin texture retained, fine fabric weave visible.
Palette of mustard, warm gray, and natural skin tones.
Mood: quiet luxury, modern magazine cover.
High detail.
DO NOT include text, logos, cluttered jewelry, or heavy retouching that erases skin texture.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 2. Candid Street Portrait of a Bookseller

```text
Candid documentary street portrait of an elderly bookseller laughing behind his stall in a narrow old-town street.
He wears round glasses and a cardigan, holding an open paperback mid-gesture as he tells a story.
Natural late afternoon light falls across his face from between the buildings, warm and soft, while the shaded street behind dissolves into blur.
Shallow depth of field, 50mm lens at f/1.8 look, slight grain like classic reportage film.
Background: out-of-focus spines of used books in muted colors.
Composition: subject slightly off-center to the right, looking toward the left of frame.
Warm, human, unposed mood.
High detail in the face and hands.
DO NOT include studio lighting, a stiff posed expression, or a busy in-focus background.
```

*Suggested Size: 1024x1536 / Quality: high*

### 3. Backlit Meadow Portrait at Golden Hour

```text
Backlit portrait of a young woman standing in a tall wildflower meadow at golden hour.
The sun sits low directly behind her shoulder, rim-lighting her hair and the flower edges with warm gold, while her face stays softly lit by the open sky, gentle and even.
She wears a simple white linen dress, holds a few loose daisies, and looks just past the camera with a calm half smile.
Lens flare blooms softly in the upper corner, and light haze warms the whole frame.
Shot as if with an 85mm lens at f/1.4, creamy bokeh, foreground flowers dissolving into soft shapes.
Palette of gold, cream, and soft green.
Dreamy, nostalgic, late-summer mood.
High detail in the rim-lit hair.
DO NOT include harsh shadows on the face, heavy orange skin tones, or a visible sun disk in the center of the frame.
```

*Suggested Size: 1024x1536 / Quality: high*

### 4. Film Noir Portrait in Hard Light

```text
Black and white film noir style portrait of a man in a fedora and trench coat, standing in a dark hallway beside a window with venetian blinds.
A hard single light source outside the window cuts through the blinds, painting sharp horizontal stripes of light and shadow across his face and the wall behind him.
Half his face falls into deep black shadow, one eye catching the light, cigarette smoke drifting through the beam.
Composition: low angle, tight crop from the chest up, strong verticals from the window frame.
High contrast with deep blacks and bright whites, no muddy midtones, fine 1940s film grain.
Tense, secretive, cinematic mood.
High detail in the smoke and fabric.
DO NOT include color, modern clothing, or soft even lighting.
```

*Suggested Size: 1024x1536 / Quality: medium*

### 5. Woodworker in His Workshop

```text
Environmental portrait of a Japanese woodworker in his sixties inside his workshop, standing at a workbench planing a length of cedar.
Wood shavings curl from the hand plane, and sawdust motes hang in a shaft of daylight from a single side window.
He wears a simple indigo apron with sleeves rolled, focused expression looking down at his work rather than at the camera.
Background: hand tools hung neatly on a pegboard and timber stacked against the wall, all softly out of focus.
Warm natural window light from the left, deep honest shadows, no fill light.
Shot as if with a 35mm lens at f/2, documentary realism, visible skin texture and calloused hands.
Palette of warm wood browns, indigo, and soft daylight.
Mood: mastery, patience, quiet dignity.
Extremely high detail.
DO NOT include studio lighting, a posed smile, or a cluttered chaotic workshop.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 6. Dancer Frozen Mid-Leap

```text
Dramatic stage photograph of a contemporary dancer captured at the peak of a jump in a dark studio theater.
A single hard spotlight from the upper left carves her out of the darkness, dust particles glittering in the beam, her flowing charcoal dress trailing the motion.
Her arms extend in one long diagonal line, fingers stretched, face turned up into the light with eyes closed.
A fast shutter freezes every fold of fabric while the background stays pure black.
High contrast chiaroscuro, palette of black, charcoal, warm skin tone, and the white-hot spotlight.
Composition: dancer on the left third, negative space on the right for the trailing fabric.
Mood: explosive yet weightless.
Extremely high detail.
DO NOT include visible stage floor markings, other dancers, or flat front lighting.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 7. Clean Corporate Headshot

```text
Professional corporate headshot of a man in his thirties wearing a navy suit, a light blue shirt, and no tie, against a soft neutral gray studio background.
Chest-up framing, shoulders squared and slightly angled, arms relaxed, warm confident closed-mouth smile with engaged eyes.
Lighting: a classic three-point setup, key light at 45 degrees camera-left, subtle hair light from behind, gentle fill, clean catchlights.
Shot as if with an 85mm lens at f/4, moderate depth of field, natural skin texture with light retouching only.
Palette of navy, light blue, gray, and natural skin tones.
Mood: approachable, competent, trustworthy.
High detail.
DO NOT include heavy vignetting, gimmick poses, visible logos, or plastic-looking skin.
```

*Suggested Size: 1024x1024 / Quality: high*

---

## Food & Product Photography

Commercial-grade food and product shots with explicit lighting recipes. Overhead and macro shots suit square, bottles and tall products suit portrait.

### 1. Rustic Breakfast Flat Lay

```text
Overhead flat lay food photography of a rustic breakfast table on a weathered oak surface.
A ceramic plate of sourdough toast with smashed avocado and a soft poached egg sits slightly off-center, surrounded by a small bowl of strawberries, a stoneware mug of black coffee with faint steam, a jar of honey with a wooden dipper, and a linen napkin folded loosely in oatmeal beige.
Soft natural window light from the top left creates gentle directional shadows and true colors, white balance slightly warm.
Shot as if with a 50mm lens at f/4 from directly above, everything in clean focus with honest food texture, crumbs and all.
Palette of avocado green, berry red, cream, and warm wood.
Cozy, unhurried weekend mood.
High detail.
DO NOT include hands, harsh flash shadows, or artificial-looking garnishes.
```

*Suggested Size: 1024x1024 / Quality: high*

### 2. Citrus Splash Juice Commercial

```text
Commercial beverage photography of a glass bottle of fresh orange juice on a seamless bright yellow studio background, captured at the exact moment orange slices splash into the juice.
Juice droplets and two orange wedges hang mid-air around the bottle in a frozen arc, backlit so every droplet glows.
Lighting: two hard strobes from behind left and right rim-lighting the liquid, one soft frontal fill, high shutter speed look with razor sharp droplet edges.
The bottle label is blank matte white with no text.
Composition: bottle centered low, splash crown above it, generous yellow negative space at the top for a headline.
Palette of vivid orange, yellow, and white.
Energetic, fresh, premium juice brand mood.
Extremely high detail.
DO NOT include readable brand text, blur on the droplets, or a cluttered background.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 3. Molten Chocolate Dessert, Dark and Moody

```text
Dark and moody macro food photography of a chocolate lava cake on a slate board, cut open so glossy molten chocolate flows slowly toward the edge of the board.
A dusting of cocoa powder and three flakes of sea salt sit on the cracked crust, and one raspberry rests against the cake as a single red accent.
Lighting: a single soft light from the back left, deep shadows everywhere else, chiaroscuro in the manner of a Dutch still life.
Shot as if with a 100mm macro lens at f/4, focus locked on the flowing edge of the molten center, background falling into darkness.
Palette of deep brown, black slate, and one red accent.
Indulgent, rich, restaurant-menu mood.
Extremely high detail in the gloss and crumb structure.
DO NOT include bright even lighting, multiple desserts, or a busy tabletop.
```

*Suggested Size: 1024x1024 / Quality: xhigh*

### 4. Perfume Bottle on a Pedestal

```text
Minimal luxury product photography of a rectangular glass perfume bottle with a brushed gold cap, standing on a polished white pedestal against a soft warm gray seamless background.
A single crisp diagonal shadow falls across the pedestal from the upper right.
Refraction concentrates the amber liquid into a glowing core, with a subtle reflection of the bottle in the pedestal surface.
Lighting: a large softbox from the right creating one long elegant highlight down the glass edge, and a black flag on the left keeping one facet dark for depth.
Shot as if with a 90mm tilt-shift lens at f/8, perfectly parallel lines, flawless glass edges.
Palette of amber, white, warm gray, and gold.
Refined, expensive, quiet mood.
Extremely high detail.
DO NOT include text, logos, flowers, hands, or visible dust.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 5. Floating Sneaker Campaign Shot

```text
Dynamic product photography of a white and coral running sneaker floating diagonally in mid-air against a solid deep teal studio background, laces loosened and drifting upward as if in zero gravity.
Fine dust particles and two small coral foam shards orbit the shoe, frozen in motion.
Lighting: a hard key from the upper left sculpting the mesh texture, a cool rim light from behind separating the sole from the background, soft fill from below.
Shot as if with an 85mm lens at f/5.6, sharp product detail, every knit stitch and sole tread visible.
Composition: shoe crossing the frame from lower left to upper right, empty space in the upper left for a headline.
Palette of white, coral, and deep teal.
Energetic, modern sportswear campaign mood.
Extremely high detail.
DO NOT include brand logos, a visible person, or motion blur on the shoe.
```

*Suggested Size: 1024x1024 / Quality: high*

### 6. Mediterranean Market Stall

```text
Documentary food photography of a Mediterranean market stall piled with summer produce, shot in first-light morning conditions.
Crates of tomatoes, figs, peaches, and purple artichokes stack into a slightly chaotic but rhythmic wall of color, with small chalk price signs leaning against the crates.
The stall owner's hands appear at the edge of frame arranging peaches; everything else is produce.
Natural side light from the open market canopy, warm and directional, deep shadows between the crates.
Shot as if with a 35mm lens at f/2.8, focus on the front crate of tomatoes, layers softening behind.
Palette of tomato red, fig purple, peach orange, and leaf green.
Alive, abundant, travel-memory mood.
High detail.
DO NOT include flash lighting, tourists, or plastic packaging.
```

*Suggested Size: 1536x1024 / Quality: medium*

### 7. Luxury Watch Macro

```text
Ultra macro product photography of a luxury mechanical wristwatch, the dial filling the frame against a pure black background.
The dial is deep midnight blue with applied silver indices, a date window at three o'clock, and the second hand frozen mid-tick.
Lighting: one focused light from the upper left grazing across the dial to reveal the brushed metal texture and the domed crystal edge, with a thin silver rim reflection tracing the case.
Shot as if with a 100mm macro lens at f/11 for edge-to-edge sharpness, focus-stacked look.
Palette of midnight blue, silver, and black.
Precision, engineering, quiet wealth mood.
Extremely high detail in the metal finishing and the tiny printed numerals.
DO NOT include a wrist, brand names, dust, or a busy background.
```

*Suggested Size: 1024x1024 / Quality: xhigh*

---

## Sci-Fi & Fantasy Concept Art

Painterly concept art for worlds that do not exist yet. Vistas suit 1536x1024, interiors and vertical scenes suit 1024x1536.

### 1. Solarpunk Street in Late Afternoon Light

```text
Concept art of a solarpunk city street in the near future, where mid-rise buildings are wrapped in vertical gardens and glass solar facades.
A tram overgrown with flowering vines runs down the center, pedestrians walk among planters of bamboo and fruit trees, and awnings and laundry lines add human scale.
Warm late afternoon sun, long soft shadows, clean air with a faint golden haze.
Style: painterly concept art with confident brushwork and slightly idealized forms, in the tradition of optimistic architectural illustration.
Palette of leaf green, terracotta, warm white, and gold.
Composition: one-point perspective down the street, the tram converging toward a sunlit plaza.
Mood: hopeful, livable, post-scarcity.
High detail.
DO NOT include smog, dystopian decay, flying cars, or neon signage.
```

*Suggested Size: 1536x1024 / Quality: high*

### 2. Inside a Generation Ship

```text
Sci-fi concept art of the interior of a generation ship's cylindrical habitat, viewed from inside the curve.
A river of farmland and orchards wraps up the inner hull overhead, where clouds drift along the rotational axis and a tube of artificial sunlight runs the length of the cylinder.
In the foreground a young girl stands on a wooden bridge over the river, tiny against the vast interior, giving scale.
Style: detailed matte painting, soft atmospheric perspective, hard sci-fi realism in the machinery.
Lighting: warm artificial sunlight from the central tube, cool shadows in the hull structure.
Palette of green farmland, cream light, and gunmetal.
Mood: awe, quiet loneliness, a world inside a machine.
Extremely high detail.
DO NOT include aliens, battle damage, or lens flares.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 3. The Dragon's Hoard

```text
Fantasy concept art of an ancient dragon's hoard cavern.
A massive red dragon sleeps curled around a mountain of gold coins, goblets, and crowns, one eye half open, smoke curling from a nostril.
Shafts of cool blue moonlight fall through a crack in the cavern ceiling, catching the gold in scattered glints and rim-lighting the dragon's scales.
In the lower foreground a tiny adventurer holding a lantern stands at the edge of the shadow line, deciding whether to step closer.
Style: painterly fantasy illustration with rich texture and dramatic scale contrast.
Palette of gold, deep red, and moonlit blue.
Composition: the dragon filling the upper two thirds, the adventurer small at the bottom edge.
Mood: tension, greed, held breath.
Extremely high detail.
DO NOT include cartoon proportions, text, or a fully awake attacking dragon.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 4. Neon Night Market in the Rain

```text
Cyberpunk street scene of a night market in the rain, with neon signs in pink and cyan reflecting in every wet surface.
Vendors under tarpaulins sell noodles and synthetic sushi, steam mixing with drizzle, while holographic koi fish swim through the air above the crowd.
A courier in a translucent poncho weaves past on a motorbike, headlight streaking across the frame.
Shot from a low angle at the end of the market alley, deep one-point perspective with pools of light receding into haze.
Style: cinematic concept art, rain-soaked neon atmosphere with volumetric light and heavy reflections.
Palette of magenta, cyan, sodium orange, and black.
Mood: electric, crowded, beautiful decay.
Extremely high detail.
DO NOT include daylight, clean dry streets, or readable real-world brand logos.
```

*Suggested Size: 1536x1024 / Quality: max*

### 5. Floating Islands at Sunrise

```text
Fantasy vista of floating islands above a sea of clouds at sunrise.
The nearest island carries a stone monastery with waterfalls pouring off its edge into mist, connected to a smaller island by a rope bridge.
Distant islands recede into golden haze, and birds circle the highest spire.
The warm sun crests the cloud line, rim-lighting every island edge, while the cloud valleys hold cool shadow.
Style: epic painterly fantasy landscape with romantic-era composition and a sense of vast scale.
Palette of gold, rose, slate blue, and cloud white.
Composition: the main island on the left third, the bridge leading the eye toward the distant archipelago.
Mood: wonder, sacred solitude.
Extremely high detail.
DO NOT include airships, dragons, or a busy foreground.
```

*Suggested Size: 1536x1024 / Quality: xhigh*

### 6. Industrial Mech Design Sheet

```text
Sci-fi mecha design sheet on a light neutral background, presenting one walking cargo mech in three views: front, three-quarter, and side.
The mech is a heavy industrial loader with hydraulic legs, a crane arm, and a worn pilot cabin, utilitarian rather than heroic, with visible panel lines, warning stripes, and scuffed paint.
Small annotation callouts point to the joint actuators and the cargo winch, each ending in a small blank label plate with no readable text.
Style: clean concept design sheet, industrial design rendering, subtle shading, no background scenery.
Palette of olive drab, safety orange, gunmetal, and off-white.
Mood: believable engineering, a working machine.
High detail.
DO NOT include anime faces, weapons, dramatic action poses, or a dark background.
```

*Suggested Size: 1536x1024 / Quality: high*

### 7. Bioluminescent Alien Forest

```text
Sci-fi concept art of a bioluminescent alien forest at night.
Slender translucent trees glow from within in bands of turquoise and violet, their light pooling in a shallow mirror-flat stream running through dark moss.
Floating spores drift like slow embers, lit from below by the glow.
A lone explorer in a white suit stands hip-deep in the stream, her helmet lamp a single small warm dot against the cool glow.
Style: painterly concept art, soft volumetric light, deep atmospheric layers.
Palette of turquoise, violet, deep indigo, and one warm lamp accent.
Composition: the stream leading from the bottom edge to the explorer at the middle third, glowing trees framing both sides.
Mood: hushed, alien reverence.
Extremely high detail.
DO NOT include monsters, daylight, or a busy composition.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

---

## Architecture & Interiors

Exteriors and interiors, from soft Scandinavian daylight to brutalist mass. Interiors and facades suit 1536x1024, vertical subjects like towers and staircases suit 1024x1536.

### 1. Scandinavian Living Room in Daylight

```text
Interior photography of a Scandinavian living room in a mid-century apartment, mid-morning.
A pale oak floor, a low gray three-seat sofa with a rust-colored wool throw, one cognac leather lounge chair, a black arc floor lamp, and a large monstera beside the tall window.
Soft daylight floods from the left through sheer curtains, casting a gentle bright band across the floor.
Styling: minimal and lived-in, two books and a ceramic mug on the small side table, nothing else.
Shot as if with a 24mm tilt-shift lens at f/5.6, vertical lines perfectly straight, one-point perspective from standing height.
Palette of white, pale oak, gray, and rust.
Mood: calm, cozy, uncluttered.
High detail.
DO NOT include people, pets, clutter, or HDR over-processing.
```

*Suggested Size: 1536x1024 / Quality: high*

### 2. Brutalist Library, Overcast

```text
Architectural photograph of a brutalist public library exterior on an overcast day.
Massive board-formed concrete planes stack in a stepped ziggurat composition, deep horizontal window slots casting dark shadow lines, with one tall blank concrete wall anchoring the left side.
A few students with backpacks cross the vast concrete plaza for scale, dressed in muted colors.
Flat diffuse overcast light, honest gray-on-gray tonality, strong geometry.
Shot as if with a 35mm lens at f/8 from a low angle emphasizing the mass, straight verticals.
Palette of raw concrete gray, charcoal, and small accents of clothing color.
Mood: monumental, severe, quietly beautiful.
High detail in the formwork texture.
DO NOT include golden hour warmth, a blue sky, or surrounding modern glass towers.
```

*Suggested Size: 1024x1536 / Quality: high*

### 3. Tea House in a Moss Garden

```text
Architectural photography of a minimalist Japanese tea house in a moss garden, early morning.
A small dark-stained wooden structure with a single glowing paper shoji panel sits at the end of a stepping-stone path, surrounded by raked moss and two maples just turning orange.
Low mist hangs in the garden, softening the background trees.
Composition: the path leading from the bottom edge to the tea house on the right third, one stone lantern mid-path.
Soft dawn light with a warm interior glow contrasting the cool garden.
Palette of moss green, charcoal wood, warm paper glow, and one maple orange.
Mood: stillness, ritual, wabi-sabi.
High detail in the moss and stone textures.
DO NOT include people, bright midday sun, or flowers in saturated colors.
```

*Suggested Size: 1024x1024 / Quality: high*

### 4. Spiral Staircase, Looking Up

```text
Abstract architectural photograph shot looking straight up a spiral staircase from the bottom.
Worn stone steps spiral counterclockwise around a central column, each step edge catching a different band of light from a skylight at the top, creating a hypnotic curve of light and shadow.
An old iron railing traces the spiral as one thin dark line.
Natural light from directly above, soft and even, no artificial sources.
Shot as if with a 16mm lens at f/8, perfectly centered symmetrical composition, straight-up perspective.
Palette of warm limestone, honey light, and dark iron.
Mood: geometry, infinity, quiet vertigo.
Extremely high detail in the stone wear.
DO NOT include people, tilted framing, or a cluttered central column.
```

*Suggested Size: 1024x1536 / Quality: xhigh*

### 5. Greek Island Courtyard at Midday

```text
Travel architectural photography of a Mediterranean courtyard in a small Greek island town at midday.
Whitewashed walls with rounded corners, blue shutters on two windows, a bougainvillea spilling magenta over one wall, stone floor tiles in soft shadow, and a wooden table with two glasses of iced coffee.
Harsh overhead sun is broken by a grapevine pergola, casting a loose dappled pattern across the floor.
Composition: the courtyard in a balanced square frame, the bougainvillea as the color anchor on the left, a strip of deep blue sky visible above the wall.
Palette of white, cobalt, magenta, and stone gray.
Mood: slow afternoon, holiday calm.
High detail.
DO NOT include crowds, parked scooters, or oversaturated HDR.
```

*Suggested Size: 1024x1024 / Quality: medium*

### 6. Glass House at Dusk

```text
Architectural photography of a modernist glass house on a forested hillside at dusk.
A flat-roofed pavilion of floor-to-ceiling glass glows warm from within, with every interior detail visible: a fireplace, a long dining table set for two, a single pendant lamp.
The surrounding pines stand as dark silhouettes against a deep blue twilight sky, the last violet light fading in the west.
Reflections of the interior layer over the faint mirrored forest on the glass.
Shot as if with a 50mm lens at f/8 from across a reflecting pool, the house doubled in still water.
Palette of warm amber interior, deep blue exterior, and black silhouettes.
Mood: serene isolation, a lit jewel in dark woods.
Extremely high detail.
DO NOT include blown-out interior lights, people, or a starry long-exposure sky.
```

*Suggested Size: 1536x1024 / Quality: xhigh*

---

## Anime & Illustration Styles

Named illustration traditions, from Ghibli-esque backgrounds to retro city pop. Wide scenes suit 1536x1024, character and book scenes suit 1024x1536.

### 1. Summer Countryside, Ghibli-esque

```text
Anime background painting in a Studio Ghibli-esque style of a Japanese countryside in high summer.
A narrow road lined with wild grass runs from the bottom edge toward rolling hills, telephone poles marching alongside, and towering cumulus clouds stack in a deep blue sky.
A girl in a school uniform walks her bicycle up the gentle slope, small against the landscape, wind lifting her hair.
Hand-painted look with soft visible brush texture, warm saturated but natural colors, gentle light with soft shadows.
Palette of summer green, sky blue, cloud white, and sun-bleached road gray.
Composition: the road in a gentle S-curve, the girl at the middle third, cloud shapes echoing the hills.
Mood: nostalgia, an endless vacation afternoon.
High detail.
DO NOT include photorealism, dark tones, or urban clutter.
```

*Suggested Size: 1536x1024 / Quality: high*

### 2. City Intersection After Rain, Shinkai-esque

```text
Anime background painting in a Makoto Shinkai-esque style of a city intersection at sunset just after rain.
Wet asphalt mirrors a sky burning in orange, pink, and violet, power lines crossing the frame in thin silhouette, and a train passes on an elevated track with lit windows.
Two students in uniforms stand at the crosswalk on opposite corners, not looking at each other.
Dramatic lens flare from the low sun between buildings, clouds with luminous rim edges, hyper-saturated but clean color grading.
Palette of sunset orange, pink, teal reflections, and deep violet shadow.
Composition: one-point perspective down the street, the sun at the vanishing point.
Mood: bittersweet, cinematic longing.
Extremely high detail.
DO NOT include motion blur, dull colors, or a daytime sky.
```

*Suggested Size: 1536x1024 / Quality: xhigh*

### 3. Flat Vector: Focus at the Desk

```text
Flat vector illustration of a focused young woman working at her desk in a home office, drawn in a modern editorial tech illustration style.
She sits cross-legged in an ergonomic chair with a laptop, a plant, a mug, and a cat asleep on the windowsill beside her.
Simple geometric shapes, no outlines, a subtle grain texture overlay, restrained proportions with slightly oversized hands and laptop.
Composition: the desk scene from a loose three-quarter angle, generous negative space above for a headline.
Palette limited to four colors: cream background, deep teal, peach, and mustard.
Mood: calm productivity, cozy focus.
Clean and minimal, medium detail.
DO NOT include gradients, photorealism, clutter, or text.
```

*Suggested Size: 1024x1024 / Quality: medium*

### 4. Watercolor Fox at the Bus Stop

```text
Children's picture book illustration in loose watercolor and ink of a small fox in a yellow raincoat waiting at a bus stop in the rain.
The fox looks up at a snail perched on the signpost, both waiting.
Soft wet-on-wet watercolor washes bleeding at the edges, fine brown ink linework for the fox's face and paws, visible cold-press paper texture.
Muted rainy-day palette of wash blues, sage green, and warm fox orange, with the yellow raincoat as the brightest note.
Composition: the bus stop on the left third, puddle reflections below, a soft gray sky above.
Mood: gentle, curious, quietly funny.
Medium detail, soft edges.
DO NOT include digital gradients, harsh outlines, or a scary atmosphere.
```

*Suggested Size: 1024x1536 / Quality: high*

### 5. Retro City Pop Night

```text
Retro anime illustration in a late-1980s city pop style of a woman leaning on a white sports car at night, a city skyline of bokeh lights behind her.
She has windblown hair, hoop earrings, and a confident half-smile, one hand on her hip, wearing a tailored blazer over a crop top.
Airbrushed cel shading with visible analog film grain, halation glowing around distant neon signs, a crescent moon and scattered stars in a deep indigo sky.
Palette of hot pink, cyan, chrome white, and indigo.
Composition: the figure on the right third, the car diagonal across the bottom, the skyline low on the left.
Mood: cool, nostalgic, urban romance.
High detail in the airbrushed shading.
DO NOT include modern flat shading, photorealism, or a busy daytime scene.
```

*Suggested Size: 1024x1536 / Quality: high*

### 6. Ink Line Art: The Staircase of Books

```text
Editorial illustration in black ink line art with one selective color accent.
A person climbs an impossibly tall spiral staircase made of stacked books, reaching for a small red kite floating just out of reach.
Dense cross-hatching builds shadow on the book spines, lighter hatching on the figure, clean white paper background everywhere else.
The only color is the red kite and its red string winding down the staircase.
Composition: the staircase as a strong diagonal from bottom left to upper right, the figure three quarters of the way up, the kite near the top edge.
Style: literary magazine illustration, confident and varied line weight, hand-drawn feel.
Mood: ambition, whimsy, the unreachable.
High detail in the hatching.
DO NOT include additional colors, gray fills, or photorealistic textures.
```

*Suggested Size: 1024x1536 / Quality: high*

---

## Logos & Brand Marks

Flat vector marks and wordmarks on clean backgrounds, ready for brand work. Logos are best generated square, then cropped by the app or exported as-is.

### 1. Minimal Mountain Badge

```text
Minimal geometric logo design of a mountain range for an outdoor equipment brand, on a plain white background.
Two overlapping triangular peaks form an abstract mountain with a negative-space path running down the valley between them, the whole mark contained inside a rounded square badge.
Pure flat vector, two colors only: deep forest green and slate gray, no gradients, no outlines.
The mark sits centered with generous clear space around it, presented large and clean.
Style: modern Scandinavian minimalism in logo design, balanced optical weight, sharp vector edges.
Mood: dependable, rugged, calm.
DO NOT include photorealistic mountains, drop shadows, 3D bevels, mockups, text, or a busy background.
```

*Suggested Size: 1024x1024 / Quality: high*

### 2. Serif Wordmark for a Coffee Roastery

```text
Typographic logo design for a specialty coffee roastery: the wordmark "Ember" set in a warm modern serif with slightly rounded terminals, with a small subline "coffee roasters" in spaced-out uppercase sans-serif letters beneath it.
To the left of the wordmark, a small icon of a coffee bean formed by two simple strokes.
Layout: the lockup centered on a plain cream background with generous clear space.
Flat vector, palette of espresso brown and burnt orange on cream, no gradients, no shadows.
Style: contemporary artisan branding, refined kerning, balanced optical spacing.
Mood: warm, crafted, trustworthy.
DO NOT include 3D effects, busy illustrations, more than two typefaces, a dark background, or mockups.
```

*Suggested Size: 1024x1024 / Quality: high*

### 3. Abstract SaaS Tech Mark

```text
Abstract tech company logo mark: a bold geometric symbol built from three rounded parallelogram strokes rotating around a central point, suggesting both a data stream and a forward arrow, forming a clean pinwheel-like glyph.
Flat vector on a plain white background, a single color of deep indigo, thick uniform stroke weight, perfectly balanced negative space inside the mark.
Presented large and centered with generous clear space, no text and no mockups.
Style: modern SaaS brand identity, geometric precision, memorable even at a 16-pixel favicon size.
Mood: intelligent, forward motion, minimal.
DO NOT include gradients, thin fragile lines, 3D effects, mockups, or letters hidden inside the mark.
```

*Suggested Size: 1024x1024 / Quality: medium*

### 4. Vintage Bakery Emblem

```text
Vintage emblem logo for an artisan bakery, designed as a circular badge with a double ring border.
Inside the ring, the text "STONE OVEN BAKERY" arcs along the top curve and "EST. 1987" arcs along the bottom, set in sturdy hand-lettered vintage serif capitals.
Center: a simple line illustration of a wheat sheaf crossed with a rolling pin.
Flat vector, palette of wheat gold and cocoa brown on an off-white background, with a slightly distressed texture kept subtle and clean.
Style: American craft bakery branding, balanced emblem composition, lettering legible at small sizes.
Mood: honest, warm, handmade tradition.
DO NOT include photorealism, gradients, more than two colors, mockups, or a modern minimalist look.
```

*Suggested Size: 1024x1024 / Quality: high*

### 5. Electric Fitness Bolt

```text
Dynamic logo mark for a high-intensity fitness brand: an abstract glyph of a lightning bolt fused with a rising heartbeat pulse line, drawn as one continuous thick angular stroke with a sharp forward lean suggesting speed.
Flat vector on a plain black background, a single color of electric lime, crisp edges, aggressive but balanced geometry, readable as an app icon.
Presented centered with generous clear space, no text and no mockups.
Style: bold sports branding, energetic geometry, an instantly readable silhouette.
Mood: power, urgency, performance.
DO NOT include gradients, thin lines, mascots, dumbbells, mockups, or a light background.
```

*Suggested Size: 1024x1024 / Quality: high*

### 6. Negative-Space Leaf

```text
Minimal negative-space logo for an environmental nonprofit: a solid deep green circle from which a single leaf shape is cleanly cut out in negative space, the leaf tilted as if falling, its stem formed by one thin flowing line that continues just outside the circle edge.
Flat vector on a plain white background, one color only, perfect geometry, generous clear space around the mark, no text.
Style: smart modern identity design of the kind featured in logo design annuals, simple enough to embroider on a shirt.
Mood: growth, care, clarity.
DO NOT include gradients, multiple colors, 3D effects, trees, a globe, mockups, or text.
```

*Suggested Size: 1024x1024 / Quality: high*

---

## Business & Marketing Visuals

Illustration assets for decks, sites, ads, and social. Banners and headers suit 1536x1024, feed posts suit 1024x1024.

### 1. Isometric SaaS Dashboard

```text
Isometric illustration for a SaaS landing page of a floating analytics dashboard interface.
A rounded rectangle dashboard panel hovers in space showing a line chart trending upward, a donut chart, and three small stat cards, each element drawn as a separate slightly raised flat layer with soft long shadows.
Around the main panel float two smaller cards: a notification card and a user avatar card.
Style: modern isometric tech illustration, flat colors, thin line details, abstract placeholder bars instead of readable text.
Palette of white panels, indigo accents, and a soft lavender background.
Composition: the dashboard angled 30 degrees from the top left, empty space at the top for a headline.
Mood: clean, capable, modern software.
High detail.
DO NOT include readable text, people, or a cluttered UI.
```

*Suggested Size: 1536x1024 / Quality: high*

### 2. Consultant Profile Banner

```text
Professional banner illustration for a data analytics consultant's profile page.
Wide horizontal composition on a soft off-white background: the right side features a delicate line-art network of connected nodes, small bar charts, and one rising trend line in deep indigo, drawn with thin uniform strokes like a refined technical drawing.
The left two thirds stays as clean negative space with only a faint grid pattern, reserved for a future headline.
Style: modern corporate illustration, minimal and precise, no mascots.
Palette of deep indigo, slate, and off-white.
Mood: analytical, credible, understated expertise.
Medium detail.
DO NOT include people, readable text, neon colors, or a dark background.
```

*Suggested Size: 1536x1024 / Quality: medium*

### 3. Startup Team Hero Illustration

```text
Flat corporate illustration for a startup website hero section, showing a diverse four-person team collaborating around a large table display, pointing at a project board with sticky notes and a roadmap arrow.
Simple geometric human figures with varied skin tones and modern casual clothes, minimal facial detail beyond simple eyes and smiles, slightly oversized heads in friendly proportions.
Style: contemporary flat vector with subtle grain, no outlines, rounded shapes.
Palette of a white background, navy, coral, and soft yellow.
Composition: the team grouped on the left two thirds, the roadmap arrow pointing toward empty space on the right for a headline.
Mood: optimistic, energetic collaboration.
Medium detail.
DO NOT include photorealism, clutter, readable text on the sticky notes, or more than four figures.
```

*Suggested Size: 1024x1024 / Quality: medium*

### 4. Newsletter Header Illustration

```text
Illustrated email newsletter header for a weekly data science newsletter, in a wide horizontal format.
A cozy desk scene in flat illustration style: an open laptop showing a tiny chart, a coffee mug with curling steam, scattered notebook sheets with small abstract data doodles, and a potted plant, all arranged in a loose horizontal strip.
Style: friendly modern editorial illustration, rounded shapes, thin accents, subtle paper grain.
Palette of a cream background, teal, amber, and ink navy.
Composition: objects distributed evenly across the width with clear space in the center for a newsletter title, nothing touching the frame edges.
Mood: curious, personal, weekly ritual.
Medium detail.
DO NOT include readable text, people, or heavy shadows.
```

*Suggested Size: 1536x1024 / Quality: medium*

### 5. Growth Staircase for a Deck

```text
Isometric business illustration of steady growth for a marketing deck.
A wide ascending staircase of four rounded platform steps rises from left to right, each platform slightly larger than the last, with a small flag planted on the top step and a simple upward arrow tracing the stair profile in the background.
Tiny abstract elements sit on the steps: a seedling, a gear, a chat bubble, and a rocket, suggesting stages of a customer journey.
Style: clean isometric vector illustration, flat colors, soft ambient shadows, no text.
Palette of white, mint green, and deep blue on a very light background.
Composition: the staircase running diagonally across the frame, empty upper right for a headline.
Mood: progress, momentum, clarity.
Medium detail.
DO NOT include people, readable text, or a dark background.
```

*Suggested Size: 1024x1024 / Quality: medium*

### 6. Focus App Social Ad

```text
Social media ad visual for a focus timer app, in a square format for feed placement.
Center: a simplified smartphone mockup showing a large circular countdown timer ring with a play button, no readable text on the screen.
Around the phone, floating flat elements tell the story of deep focus: a crescent moon, a steaming tea cup, a muted bell with a slash, and soft cloud shapes, all gently orbiting the phone.
Style: friendly flat vector illustration, rounded geometry, subtle grain, soft long shadows.
Palette of a deep navy background, soft cream, and warm amber accents.
Composition: the phone centered slightly low, clear space at the top for a headline.
Mood: calm, focused, an evening work session.
Medium detail.
DO NOT include readable text, clutter, or photorealism.
```

*Suggested Size: 1024x1024 / Quality: high*
