# Building a Video Analysis AI Agent for Defense / Intel Use Cases — Deep Research

The high-level pitch is clear; the question now is what to actually build, with what models, on a single Mac Studio. This is what the field looks like in April 2026, what is open enough to use, and what an MVP should cost-budget for. Citations are inline.

## 1. Prior Art — Who Has Actually Built This

### Defense primes & startups

**Palantir / Maven Smart System.** Maven is the reference architecture in this space and the only one with public, recent operational numbers. As of the [Palantir March 2026 blog post](https://blog.palantir.com/maven-smart-system-innovating-for-the-alliance-5ebc31709eea?gi=3360c48b1394) and [Tom's Hardware coverage](https://www.tomshardware.com/tech-industry/artificial-intelligence/pentagon-formalizes-palantirs-maven-ai-as-a-core-military-system-with-multi-year-funding-platforms-investment-grows-to-usd13-billion-from-usd480-million-in-2024), Maven ingests >150 source types (EO/IR drone video, satellite, radar, SIGINT, geolocation), fuses them through a unified pipeline, and produces target nominations at ~1,000/hour per analyst — a 10x jump after CV alone, then another 5x after LLM integration. As of February 2026 Maven runs on AWS with Anthropic Claude in the loop ([Wikipedia: Project Maven](https://en.wikipedia.org/wiki/Project_Maven)). Architecturally, what's public points to: per-feed CV detectors → entity fusion in Palantir Ontology → LLM agents over the ontology for analyst Q&A and report drafting.

**Anduril / Lattice.** [Lattice SDK](https://www.anduril.com/lattice/lattice-sdk) is a sensor-fusion + autonomy platform, not a VLM agent. Per [MIT Tech Review's December 2024 demo](https://www.technologyreview.com/2024/12/10/1108354/we-saw-a-demo-of-the-new-ai-system-powering-andurils-vision-for-war/) and the [March 2026 $20B Army deal](https://ubos.tech/news/anduril-wins-up-to-20-billion-us-army-ai-defense-contract-full-details/), Lattice ingests video/radar/lidar/SIGINT, runs deep-learning detect/classify/track at sub-second latency, and pushes a fused common operating picture. Edge compute is a first-class concept (distributed nodes, mesh networking), reflecting the assumption that comms will be denied.

**Shield AI / Hivemind.** Less relevant for an analyst-style agent — Hivemind is an autonomy stack for unmanned platforms (V-BAT, CCAs). Useful as a mental model for safety wrapping around a model, less so for offline intel analysis.

**Helsing.** Their [Altra Recce-Strike platform](https://helsing.ai/) does the same battlefield-data-fusion play in Europe. [Bloomberg's April 2025 piece](https://www.bloomberg.com/news/articles/2025-04-08/helsing-europe-s-most-valuable-defense-tech-company-is-facing-allegations-from) reported reliability issues — a cautionary tale: shipping a CV stack into a real war exposes failure modes that lab metrics hide.

**Scale AI Donovan.** [Scale's defense LLM page](https://scale.com/donovan/defense-llm) and [Defense Llama](https://defensescoop.com/2024/11/04/scale-ai-unveils-defense-llama-large-language-model-llm-national-security-users/) describe a model-agnostic RAG platform deployed at IL4 / SC2S / SIPR / JWICS levels. Architecturally close to a generic agent platform with classified-network deployment as the differentiator.

**Clarifai ISR.** Won an [AFRL contract for full-motion video analysis](https://www.clarifai.com/press-release/clarifai-awarded-air-force-ai-fmv-contract). Object detection on drone feeds in real time at the edge — managed CV-API play, not agentic.

**Maris-Tech & VisionWave/xClibre.** [VisionWave acquired the xClibre IP for $60M in April 2026](https://www.investing.com/news/company-news/visionwave-acquires-xclibre-ai-video-platform-for-60m-valuation-93CH-4610315), positioned as "video-as-a-sensor" with edge-first architecture and RF-fusion for counter-drone — fusing video VLMs with RF detections is the current frontier.

### Open-source / academic

**[NVIDIA VSS Blueprint](https://docs.nvidia.com/vss/latest/content/architecture.html)** is the most directly copy-able reference. The architecture has: stream handler → VLM pipeline (chunked, GPU-parallel) → CA-RAG (Context-Aware RAG) + GraphRAG → MCP-exposed tools → top-level agent. Captions + transcripts + metadata land in both a vector DB and a graph DB. Source on [GitHub](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization). **This is the architecture to copy.**

**PyImageSearch's [SAM 3 + Qwen agentic loop tutorial (April 6, 2026)](https://pyimagesearch.com/2026/04/06/agentic-ai-vision-system-object-segmentation-with-sam-3-and-qwen/)** is the cleanest open recipe for a VLM-as-orchestrator loop: Qwen proposes noun phrases, SAM 3 returns masks, Qwen evaluates and iterates. Their [SAM 3 for video tutorial (March 2026)](https://pyimagesearch.com/2026/03/02/sam-3-for-video-concept-aware-segmentation-and-object-tracking/) covers concept-based tracking specifically.

## 2. Best Current Models per Stage (April 2026)

### Vision-Language Models

| Model | Long video | Temporal grounding | OCR | On-device | Notes |
|---|---|---|---|---|---|
| **Qwen3-VL** ([tech report](https://arxiv.org/abs/2511.21631), [GH](https://github.com/QwenLM/Qwen3-VL)) | 256K native, 1M tested; 100% NIAH @256K, 99.5% @1M; 2-hour videos | Interleaved MRoPE + text-timestamp alignment for second-level grounding ([the-decoder](https://the-decoder.com/qwen3-vl-can-scan-two-hour-videos-and-pinpoint-nearly-every-detail/)) | Strong | 8B/32B/Thinking variants on [Ollama](https://ollama.com/library/qwen3-vl) | **Best open choice for this use case** |
| **Gemma 4 26B-A4B / 31B** ([Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [HF blog](https://huggingface.co/blog/gemma4)) | Multimodal text+vision+audio | Less mature than Qwen3-VL for video | Strong | Day-0 MLX support; [unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit) | A4B = 26B total, ~4B active — perfect for M3 Ultra |
| **InternVL3** ([arXiv](https://arxiv.org/html/2504.10479v1)) | Competitive with Gemini 2.5 Pro at 78B | SOTA on spatial reasoning (V*Bench 48.9) | Strong | 8B fits comfortably | Best for spatial/geometric reasoning |
| **Llama 4 Vision** | Long context | OK | OK | 109B/400B — too big for prototyping | Skip for solo Mac build |
| **Gemini 2.5 Pro** | Per [TimeScope](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models): only model maintaining accuracy past 1 hour; 84.8% VideoMME | Best in class | Best in class | API only | Use as **reference oracle** for evals |
| **GPT-5V** ([Latent.Space](https://www.latent.space/p/gpt5-vision)) | Strong | Strong | Strong | API only | Frontier vision-reasoning |

**Picks for a Mac Studio build:** Qwen3-VL-32B-Instruct (or Thinking) as primary, Gemma-4-26B-A4B as fast secondary, Gemini 2.5 Pro via API for high-stakes verification.

### Segmentation

[**SAM 3**](https://arxiv.org/abs/2511.16719) (Nov 2025) is the right default: accepts text + exemplar prompts and tracks across video natively, eliminating the previous need to chain Grounding DINO → SAM 2. [SAM 3.1 (March 27, 2026)](https://ai.meta.com/blog/segment-anything-model-3/) added Object Multiplexing — 16 objects per forward pass, ~32 fps on a single H100. Code at [facebookresearch/sam3](https://github.com/facebookresearch/sam3); [Ultralytics integration](https://docs.ultralytics.com/models/sam-3/) is the easy path.

[**Grounded SAM 2**](https://pyimagesearch.com/2026/01/19/grounded-sam-2-from-open-set-detection-to-segmentation-and-tracking/) and **OWL-ViT v2** are still relevant fallbacks per [Roboflow's 2025 zero-shot guide](https://roboflow.com/model-feature/zero-shot-detection).

### Tracking

For a single-camera drone feed, **[BoT-SORT](https://github.com/NirAharon/BoT-SORT)** with a ReID head is the de-facto pick when you care about identity persistence under occlusion (per [Veroke's 2025 review](https://www.veroke.com/insights/how-top-ai-multi-object-trackers-perform-in-real-world-scenarios/) and the [LITE paper](https://arxiv.org/html/2409.04187v1)). **ByteTrack** is faster but weaker on re-association. **OC-SORT** specifically helps with occlusion drift common in surveillance footage. SAM 3 itself does object tracking; only fall back to a dedicated tracker when SAM 3 is too slow.

For cross-camera person/vehicle re-ID, OSNet / TransReID embeddings stored in a vector index over track snapshots is the standard pattern.

### Geolocation

**[PIGEON (CVPR 2024)](https://lukashaas.github.io/PIGEON-CVPR24/)**: 91.96% country accuracy, 40.36% within 25 km on its held-out set — superhuman on GeoGuessr-style outdoor street imagery. **[GeoCLIP](https://github.com/VicenteVivan/geo-clip)** is the cleanest base for adapting to ISR; worldwide GPS retrieval via contrastive alignment on 1.2M image-location pairs. **StreetCLIP** is a smaller fine-tune on 1.1M Street View images. The [MIT 2024 thesis](https://dspace.mit.edu/bitstream/handle/1721.1/162630/Borg_Final_Pixels_to_Places.pdf) and [GeoSeer 2026 guide](https://geoseeer.com/blog/ai-geolocation-models-academic-research-guide-2026) are the best surveys. **Indoor 3.6M** ([OpenReview](https://openreview.net/forum?id=Nw7vkJKHba)) shows fine-tuning these on indoor imagery gets continent-level only — a real gap.

For "single-frame geolocation" by VLMs, Gemini 2.5 Pro and GPT-5V are anecdotally strong but uncalibrated. Treat outputs as hypotheses. Right pattern: VLM proposes a region → GeoCLIP retrieval narrows it → human/agent verifies via Sentinel Hub overlay.

### Audio

Stack: **WhisperX** (whisper-large-v3-turbo) + **pyannote Precision-2** for diarization. Per the [pyannote changelog](https://www.pyannote.ai/changelog), Precision-2 is +14% over Precision-1; orchestrates with Whisper Large V3 Turbo natively. Open-source path: [pyannote.audio 4.0 Community-1](https://github.com/m-bain/whisperX). Add **Silero VAD** in front for cheap activity gating.

Audio event detection for gunshots/vehicles/explosions has no clean SOTA, but you can fine-tune a lightweight CNN on **MFCC + Mel-spectrogram + chromagram** features per the [Military Audio Dataset (MAD), Nature Sci Data 2024](https://www.nature.com/articles/s41597-024-03511-w) — 8,075 samples, 7 classes, ~12 hours. The TensorFlow [Binary-Urban8K work](https://github.com/hasnainnaeem/Gunshot-Detection-in-Audio) hits 97.5% on gunshot/no-gunshot. Use **YAMNet** (pretrained on AudioSet) as a baseline before training anything custom.

## 3. Architecture Patterns

### Frame sampling

Three regimes, in increasing cost: **uniform** (every Nth frame — wasteful on static scenes), **shot-based** (PySceneDetect cuts → 1-2 frames per shot), **event-driven** (motion + change detection → expensive analysis only on triggered windows).

The right hybrid for ISR is the [TransNetV2 + PySceneDetect cascade](https://github.com/soCzech/TransNetV2): fast PySceneDetect first pass → TransNetV2 to recover misses → VLM caption per shot. NVIDIA VSS chunks video into "a few seconds to a few minutes" and parallelizes across GPUs. Maven's specifics aren't public, but the throughput claim (1,000 nominations/hour from one analyst's queue) implies aggressive cheap pre-filtering.

For drone feeds where the whole world is the "shot," substitute optical-flow-based change detection (cv2.calcOpticalFlowFarneback) plus a "novelty" heuristic against a background model.

### Entity graph vs. caption log

Both — that's what NVIDIA VSS does. The captions feed retrieval ("what happened in the parking lot at 14:32"); the entity graph feeds analytic queries ("which vehicle was in three locations today"). Graph nodes = (entity_id, type, attributes), edges = (saw_at, near, exited_with). Use ArangoDB or DuckDB+graph extension for prototyping; both are MLX-friendly.

The pure caption-log approach (what most VLM demos do) breaks the moment you ask cross-clip queries. The pure graph approach is brittle to entity merge errors. Keeping both lets the agent fall back across modes.

### Hierarchical pipelines

The most useful pattern for cost control on long video:

1. **Tier 0 (free):** decode + downscale + scene-cut + motion mask
2. **Tier 1 (cheap):** YOLOv11 or SAM 3 small + ByteTrack on every keyframe → filtered list of "interesting" windows
3. **Tier 2 (expensive):** Qwen3-VL on selected windows → captions + entity proposals
4. **Tier 3 (premium):** Gemini 2.5 Pro / GPT-5V on the 1% of clips flagged for analyst escalation

This is essentially how NVIDIA VSS structures its [ingestion vs. retrieval split](https://docs.nvidia.com/vss/latest/content/architecture.html), and a recurring pattern in Maven coverage.

### Agentic loop structure

Two patterns work well:

- **VLM-as-orchestrator** (per the [PyImageSearch SAM 3 + Qwen tutorial](https://pyimagesearch.com/2026/04/06/agentic-ai-vision-system-object-segmentation-with-sam-3-and-qwen/)): VLM proposes noun phrases → SAM 3 segments → VLM evaluates masks → iterate until satisfied. Good for interactive analyst use.
- **Tool-using agent** (NVIDIA VSS-style): top-level LLM with MCP-exposed tools — `vlm_caption(clip)`, `semantic_search(query)`, `geolocate(image)`, `web_search(query)`, `reverse_image(image)`. Good for batch report generation.

Both should run with explicit tool-call traces (LangGraph checkpoints or Pydantic AI message log) so you can re-run any branch deterministically.

### Vector store

For long-video retrieval: chunk by shot (or 5–10s windows) → embed caption + frame thumbnail (CLIP) + audio transcript snippet → store in **LanceDB** (Apple Silicon native, columnar, supports hybrid search). For the entity graph, **Kùzu** (embedded graph DB, Cypher-compatible) avoids needing a server. NVIDIA's VSS uses Milvus + ArangoDB; for solo prototyping, LanceDB + Kùzu is dramatically lower friction.

## 4. Datasets & Benchmarks

**ISR-relevant video datasets:**
- **[VIRAT](https://www.crcv.ucf.edu/research/data-sets/virat/)** — DARPA-funded, includes EO/IR aerial footage from a military aircraft >1000m altitude. Closest public proxy to drone surveillance.
- **[VisDrone](https://github.com/VisDrone/VisDrone-Dataset)** — 288 video clips, 261,908 frames from drone cameras. Standard drone-detection benchmark.
- **[xView](https://xviewdataset.org/)** — WorldView-3 satellite imagery, 0.3m GSD, >1M instances across 60 classes. Static, but good for satellite-image VLM evals.
- **FMOW (Functional Map of the World)** — temporal satellite imagery for change detection.
- **RarePlanes / SpaceNet** — narrower aircraft / building datasets.

**Video VLM benchmarks:**
- **[VideoMME](https://video-mme.github.io/home_page.html)** (CVPR 2025) — 900 videos, the de facto standard. Gemini 2.5 Pro at 84.8%.
- **[LongVideoBench](https://longvideobench.github.io/)** — 3,763 videos up to 1 hour with subtitles. Better signal for long-form.
- **TimeScope** — single best benchmark for long-video accuracy degradation.
- **EgoSchema**, **Perception Test** — first-person and physical-reasoning evals.

**Geolocation benchmarks:** **Im2GPS3k**, **YFCC4k/26k**, **GWS15k**.

**You should not benchmark your full system on these alone.** Build a held-out set of 20–50 of your own ISR clips with hand-labeled answers (entities, locations, key events) before tuning anything. A custom eval is the only one that matters for a domain-specific build.

## 5. Tooling for the Agent Layer

**Web search / OSINT.** Per [websearchapi.ai's 2026 comparison](https://websearchapi.ai/blog/tavily-alternatives) and [HumAI's review](https://www.humai.blog/ai-search-apis-compared-tavily-vs-exa-vs-perplexity/): **Exa** for semantic / entity-grounded queries (1,200+ domain filters, neural index — best for "find articles about this specific event"); **Tavily** for factual lookup with LLM-friendly extraction; **Perplexity Sonar** for synthesized answers with citations (~358ms — useful when an agent needs a "did X happen near Y on Z date?" answer fast). Use all three from one tool layer; route by intent.

**Reverse image search.** Public APIs are limited. Pragmatic stack: TinEye for exact-match, Yandex (best for geographic/face match), Google Lens via headless browser. Bellingcat's [Search by Image extension](https://bellingcat.gitbook.io/toolkit/more/all-tools/search-by-image) supports 41 services and is open-source — vendor it as a Playwright workflow.

**OSINT-specific.** Bellingcat's [full toolkit](https://bellingcat.gitbook.io/toolkit) is the canonical reference. **Sentinel Hub APIs** ([docs](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview.html)) give programmatic access to Sentinel-2 and commercial archives with built-in change detection (Statistical API). Maltego is GUI-heavy and not script-friendly; Shodan API is genuinely useful for grounding "what device is this" type questions.

**Agent frameworks.** Per [ZenML's PydanticAI vs LangGraph review](https://www.zenml.io/blog/pydantic-ai-vs-langgraph) and the [April 2026 Towards AI 6-framework comparison](https://pub.towardsai.net/i-compared-6-python-ai-agent-frameworks-so-you-dont-have-to-langgraph-vs-crewai-vs-pydanticai-vs-d8a5e6e43262):
- **Pydantic AI** — best for solo build; type-safe tool definitions, structured outputs, thin abstraction layer. Recommended primary.
- **LangGraph** — best for complex stateful workflows with checkpointing/resume. Use when you need explicit state machines or pause-for-human-review.
- **smolagents** — CodeAct pattern (model writes Python, executes); harder to sandbox in a defense context.
- **Letta** — long-term memory focus; not the right fit for batch video analysis.
- **Anthropic's tool use directly** — fewest moving parts; works fine if you don't need a graph.

For this build: **Pydantic AI** for the orchestrator, **LangGraph** only if you find yourself rolling your own state machine.

## 6. Critical Constraints for the Intel Domain

**Provenance / chain of evidence.** [STANAG 4609](https://impleotv.com/2025/03/11/stanag-4609-isr-video/) is the NATO Digital Motion Imagery Standard for ISR FMV: MPEG-2 TS container with MISB-defined KLV metadata embedded per-frame (sensor lat/lon, altitude, pointing angles, camera FOV, timestamps). Any serious build needs to preserve the KLV stream end-to-end and reference frame-time + KLV metadata in every assertion the agent emits. [Fraunhofer IOSB ships a free STANAG 4609 validator](https://www.iosb.fraunhofer.de/en/projects-and-products/stanag-4609-validator.html). On the data-exchange side, NIEM is the cross-agency XML schema; less directly relevant to a video build but shows up in report-output formats.

**Human in the loop / human judgment.** The [DoD AI Strategy (Jan 2026)](https://media.defense.gov/2026/Jan/12/2003855671/-1/-1/0/ARTIFICIAL-INTELLIGENCE-STRATEGY-FOR-THE-DEPARTMENT-OF-WAR.PDF) and [DoDD 3000.09](https://www.congress.gov/crs-product/IF11150) require "appropriate levels of human judgment over the use of force." The [ICRC March 2026 position paper](https://www.icrc.org/sites/default/files/2026-03/4896_002_Autonomous_Weapons_Systems_-_IHL-ICRC.pdf) is more restrictive and calls for binding rules. Translate to engineering: every agent assertion that would inform targeting must be inspectable, attributable to source frames, and reversible. Build the audit log from day one.

**Air-gapped / on-prem.** Palantir and Anduril both ship on-prem and into classified environments (SC2S/SIPR/JWICS). Scale Donovan ships into IL4. For an MLX-based MVP, this is actually a feature: everything running locally on a Mac Studio is air-gap-by-default. Avoid hard dependencies on cloud APIs for any function on the critical path.

**Failure modes to plan for:**
- **Hallucinated detections** — VLMs invent entities. Mitigation: ground every entity assertion to a SAM-3 mask + bbox + frame time; agent must cite or it's discarded.
- **Adversarial camouflage** — small adversarial patches reliably defeat YOLO-family detectors and generalize across architectures ([2008.13671](https://ar5iv.labs.arxiv.org/html/2008.13671)). [MDPI Sensors 2026](https://www.mdpi.com/1424-8220/26/6/1895) shows AI-generated camouflage is now a published technique. Mitigation: ensemble across detector families (YOLO + DETR + SAM 3 prompted) and flag low cross-model agreement.
- **Bias** — training-set bias on Western vehicles/uniforms is recurring. Mitigation: domain-specific evals, explicit "unknown" class.
- **Temporal hallucination** — VLMs confabulate about events they can't see in sampled frames. Mitigation: only let the VLM caption frames it actually receives, and force "I cannot tell from this clip" as a valid output.

## 7. Concrete Recommendations

### Reference architecture (M3 Ultra 192 GB, single-machine MVP)

```
[mp4/ts in] → ffmpeg decode + KLV extract (STANAG 4609 if present)
            → [PySceneDetect + TransNetV2] cascade → shot list
            → per-shot: keyframe(s) + audio segment
                ├─ Tier-1 detector: SAM 3 small with class prompts
                │    ["person","vehicle","weapon","aircraft"]
                ├─ Tracker: BoT-SORT with OSNet ReID embeddings → entity tracks
                ├─ Audio: WhisperX (large-v3-turbo MLX) + pyannote Community-1
                └─ "Interesting?" gate (motion + novel-entity + audio-event)
            → for interesting shots only:
                ├─ Tier-2 VLM: Qwen3-VL-32B-Instruct (MLX 4-bit)
                │    → caption + entity attributes
                └─ Geolocate: GeoCLIP retrieval over frame thumbnail
            → write to:
                ├─ LanceDB (caption+frame-CLIP+transcript, hybrid search)
                └─ Kùzu graph (entities, sightings, co-occurrences)
            → Pydantic AI agent with tools:
                [vlm_recaption, semantic_search, graph_query, geolocate,
                 reverse_image, web_search(Exa/Tavily/Perplexity),
                 sentinel_hub_overlay, escalate_to_gemini25pro]
            → Report writer: markdown with inline frame thumbnails + KLV citations
```

Memory budget on the M3 Ultra 192 GB: Qwen3-VL-32B at 4-bit ≈ ~22 GB; SAM 3 ≈ 4 GB; WhisperX large-v3-turbo ≈ 3 GB; pyannote ≈ 1 GB; GeoCLIP ≈ 1 GB; system + DBs ≈ 16 GB. Comfortable. The [Will-It-Run-AI guide](https://willitrunai.com/blog/qwen-3-5-mlx-apple-silicon-guide) reports M3 Ultra hitting 80+ tok/s on Qwen 3.5 35B-A3B at 8-bit; expect similar from Qwen3-VL-32B at 4-bit, and roughly 2x that from Gemma-4-26B-A4B given only ~4B active params.

### "v0 in a weekend"

One Python script, one MP4, one report:

1. `ffmpeg -i in.mp4 -vf fps=1 frames/%05d.jpg` (1 fps brutally simple)
2. PySceneDetect to get cuts
3. WhisperX-MLX on the audio
4. For each shot, send the middle frame to Qwen3-VL-32B (MLX) with a fixed prompt: "Describe everything visible. List people, vehicles, weapons, signage, locations with bounding boxes."
5. Stuff captions + transcript chunks into LanceDB
6. Pydantic AI agent with three tools: `search_video(query)`, `get_clip(start, end)`, `web_search(query)` (Tavily)
7. Single CLI: `report.py video.mp4 --question "summarize threats and key entities"` → markdown

That's the minimum loop. Doing this end-to-end on one real clip exposes 80% of the integration pain before you spend a week on tracking.

### 6-week build plan

- **Week 1.** v0 above. Commit to LanceDB + Kùzu + Pydantic AI now, don't shop. Stand up an eval set of 10 ISR-style clips (VIRAT + VisDrone subsets). Define "good" output schema.
- **Week 2.** Add TransNetV2, BoT-SORT, OSNet ReID. Track entities across shots. Replace 1-fps with shot-keyframe sampling. Measure recall vs. v0.
- **Week 3.** Add SAM 3 with text prompts as Tier-1 detector. Build the entity graph in Kùzu. Add audio event detection (YAMNet → custom CNN if needed). Hook KLV metadata extraction (klvdata).
- **Week 4.** Promote the agent: Pydantic AI with the full tool set (semantic_search, graph_query, geolocate via GeoCLIP, reverse_image via Playwright + Yandex/TinEye, web_search via Exa+Tavily+Perplexity router). Add Gemini 2.5 Pro escalation tool with a strict budget.
- **Week 5.** Provenance + audit. Every agent assertion cites (clip_id, frame_time, KLV_lat_lon, source_tool, confidence). STANAG 4609 metadata threaded through. Markdown report template with embedded thumbnails. Re-run determinism via LangGraph checkpoints if you've hit graph complexity.
- **Week 6.** Adversarial robustness pass: ensemble Tier-1 detection (SAM 3 + YOLOv11 + DETR), flag disagreement. Bias testing with held-out non-Western imagery. Performance pass — VLM prefix caching and batch inference on the M3 Ultra. Document failure modes.

### Specific model picks with justification

| Stage | Pick | Why |
|---|---|---|
| Primary VLM | **Qwen3-VL-32B-Instruct (MLX 4-bit)** | Best open long-video model; second-level temporal grounding; NIAH 99.5%@1M; runs comfortably on M3 Ultra |
| Fast secondary VLM | **Gemma-4-26B-A4B (MLX)** | Day-0 MLX, MoE means ~4B active = very fast for cheap captions |
| Verifier (paid) | **Gemini 2.5 Pro** | Only model that holds accuracy past 1 hour; use as escalation only |
| Segmentation | **SAM 3 / 3.1** | Native concept prompts + video tracking in one model; replaces Grounded-SAM stack |
| Tracking | **BoT-SORT + OSNet** | Best ReID-aware tracker; SAM 3 fallback for hard cases |
| Geolocation | **GeoCLIP** + Gemini 2.5 Pro verifier | Open + retrievable; LLM hypothesis verification |
| ASR | **WhisperX (large-v3-turbo)** + **pyannote Community-1** | Best open combo; pyannote Precision-2 if budget permits |
| Audio events | **YAMNet** baseline → fine-tune CNN on **MAD** | Lightweight, pretrained on AudioSet |
| Agent | **Pydantic AI** | Type-safe, thin, easy to debug solo |
| Vector store | **LanceDB** | Apple Silicon native, hybrid search, zero ops |
| Graph store | **Kùzu** | Embedded, Cypher-compatible, zero ops |
| Web search | **Exa + Tavily + Perplexity** routed by intent | Each excels at different query type |

### Three highest-risk technical decisions

1. **VLM hallucination on entities you can't verify.** The whole pipeline depends on the VLM not inventing a "white pickup truck" that isn't in the frame. *De-risk:* require every entity assertion to be backed by a SAM 3 mask or it's dropped. Build a "claim → evidence" check as a separate eval.
2. **Geolocation false confidence.** A confident wrong geolocation in an intel report is worse than no geolocation. *De-risk:* GeoCLIP returns top-K with explicit confidence; only surface when (a) GeoCLIP top-1 score > threshold AND (b) a VLM verifier corroborates AND (c) Sentinel Hub overlay is feasible. Otherwise output "indeterminate."
3. **Throughput on a single Mac.** Doing real-time video agentic analysis on one M3 Ultra is at the edge of feasibility. *De-risk:* hierarchical tier gating (only 1–5% of frames hit the VLM); batch VLM calls; use the MoE Gemma A4B for first-pass captions; benchmark end-to-end on a 30-minute clip in week 1, not week 6.

---

The opinionated short version: copy the [NVIDIA VSS architecture](https://docs.nvidia.com/vss/latest/content/architecture.html), substitute MLX-friendly models (Qwen3-VL + Gemma 4 + SAM 3 + WhisperX), wire it together with Pydantic AI and LanceDB+Kùzu, and ground every agent assertion in a STANAG-4609-aware audit trail before adding anything else.
