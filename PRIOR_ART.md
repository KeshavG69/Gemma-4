# Autonomous Long-Video Anomaly Surveillance: Prior Art

The pattern — ingest long video → unsupervised baseline → multi-signal anomaly scoring → VLM analyst pass on candidates → human-readable report — has crystallised in 2025. Reinventors exist in physical security, factory operations, AV development, and academic CVPR submissions.

## 1. Commercial Products

### Security & Surveillance

The most direct analog is **[Ambient.ai's Pulsar](https://www.ambient.ai/blog/introducing-ambient-pulsar-agentic-physical-security)**, GA November 2025: "reasoning Vision-Language Model" running at the edge, trained on >1M hours, processing 500K hours/day. Advertises "open-set detection" and "contextual intent recognition." Marketing leans on 150+ "validated threat signatures" as a safety net ([press release](https://www.prnewswire.com/news-releases/ambientai-unveils-pulsar-the-first-reasoning-vision-language-model-for-agentic-physical-security-302619452.html)). Architecture inferred: reasoning VLM scores frames, signature library disambiguates. That's the "VLM on candidates only" pattern.

**[Avigilon (Motorola)](https://www.motorolasolutions.com/content/dam/msi/images/solutions/safety-reimagined/xu/acc-7-software.pdf)** ships *Unusual Motion Detection (UMD)* and *Unusual Activity Detection (UAD)* — the cleanest deployed example of unsupervised-baseline. UMD continuously learns "what typical activity looks like" with no rules ([UMD docs](https://docs.avigilon.com/bundle/designing-site-for-analytics/page/umd/unusual-motion-detection.htm)). Statistical/density model, not generative.

**[Verkada](https://www.verkada.com)** is publicly evasive about model details; pitch is "AI search" — natural-language queries over indexed footage — rather than autonomous anomaly surfacing.
**[Spot AI](https://www.spot.ai/blog/customers-turn-their-cameras-into-ai-teammates)** markets "Video AI Agents" that "learn normal patterns then flag deviations like loitering after hours." Likely simple statistical baselines + chat layer.
**[Calipsa](https://www.networkoptix.com/blog/2022/05/04/calipsa-detect-object-detection-works-with-nx)** is the inverse — a deep classifier filtering human/vehicle vs. noise to suppress 93% false alarms.
**[Actuate](https://actuate.ai/)** is rule-based threat detection.

### Workplace Safety

**[Voxel AI](https://www.voxelai.com/platform)** runs supervised YOLO/R-CNN for PPE/ergonomics, hazard-specific ([architecture overview](https://www.machinedesign.com/automation-iiot/article/21274068/voxel-ai-how-voxel-combines-video-feeds-and-ai-to-assess-and-mitigate-workplace-safety)). **[Intenseye](https://www.intenseye.com/products/core-ai)** markets "50+ leading indicators" — curated rule library. **[Protex AI](https://www.protex.ai/)** uses "precise rules and business logic." This vertical converged on supervised hazard libraries because false positives are operationally expensive and EHS teams want explainable categories.

### Retail Loss Prevention

**[Veesion](https://veesion.io/en/our-solution/)** uses gesture recognition, supervised on shoplifting examples. Closest published architecture is **[Shopformer](https://arxiv.org/abs/2504.19970)** (2025): transformer encoder-decoder over GCN-derived pose embeddings, two-stage train (GCAE then transformer reconstruction). Reconstruction-based unsupervised on pose tokens not raw pixels.

### Industrial / Manufacturing

**[Drishti](https://www.gatewayhouse.in/drishti-foresight-to-digital-manufacturing/)** (acquired by Apple, Sept 2023) does cycle times and step verification — supervised process analysis. **[Retrocausal](https://www.retrocausal.ai)** does "step verification" copilots — structural anomaly (out-of-order/missing step). Manufacturing converged on **[Anomalib](https://github.com/open-edge-platform/anomalib)** at the still-image level.

### Smart City / Traffic

**[Rekor](https://www.rekor.ai/software/command)** explicitly markets baseline-learning: "Roadway Intelligence Engine" aggregates data "to train their AI model to understand what is expected on the roadways, in order to identify anomalies." Claims 9-min faster incident detection. Architecture not public. **[Hayden AI](https://www.hayden.ai/platform)** runs supervised violation classes (bus-lane intrusion).

### Healthcare

**[Care.ai](https://www.care.ai/patient-monitoring.html)** and **[Ouva](https://www.ouva.co/landing/fall-prevention)** do ambient hospital-room monitoring. Supervised pose+behaviour models. Representative architecture: [Frontiers in Imaging 2025 paper](https://www.frontiersin.org/journals/imaging/articles/10.3389/fimag.2025.1547166/full).

### Wildlife / Conservation

**[Wildlife Insights](https://www.wildlifeinsights.org/about-wildlife-insights-ai)** runs Google's *SpeciesNet* + Microsoft's *MegaDetector* as a first-pass animal/empty filter. Same operational shape as your candidate-filter step. 2024 paper integrates VLMs+RAG ([PMC11679253](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679253/)).

### Autonomous Driving

**[Foretellix](https://www.foretellix.com/foretify-toolchain-overview/)** and **[Applied Intuition](https://www.appliedintuition.com/blog/ai-for-mining-massive-autonomy-datasets)** do *edge-case mining* over driving logs — explicitly unsupervised clustering+anomaly detection over embeddings to surface rare scenarios. Applied Intuition's blog is unusually candid: classical clustering "introduces biases and struggles to retain semantic information," moved to learned embeddings. This is the closest commercial mirror of your "self-supervised baseline → mine outliers → human report" loop.

## 2. Open-Source

**[Anomalib](https://github.com/open-edge-platform/anomalib)** (Intel, v2.2 Oct 2025) — canonical taxonomy of unsupervised image AD: 23+ algorithms (PaDiM, PatchCore, FastFlow, EfficientAD, Reverse Distillation, Dinomaly), unified API, OpenVINO export. October 2025 release adds **[Fuvas (ICASSP 2025)](https://anomalib.readthedocs.io/)** for few-shot unsupervised video AD.

**[GVAED](https://github.com/fudanyliu/GVAED)** maintains the canonical survey ([ACM CSur 2024](https://dl.acm.org/doi/10.1145/3645101)) splitting field into UVAD / WAED / FVAD / SVAD. **[NSVAD](https://github.com/fdjingliu/NSVAD)** covers deployment.

**[LAVAD](https://github.com/lucazanella/lavad)** (CVPR 2024) — first solid training-free pipeline using BLIP-2 for captions + Llama 2 for temporal aggregation + ImageBind for cleaning. Outperformed unsupervised baselines on UCF-Crime/XD-Violence with no training. Architecturally exactly your pattern minus the agent loop.

**[VadCLIP](https://github.com/nwpu-zxr/VadCLIP)** (AAAI 2024) — canonical CLIP-adapter for weakly-supervised VAD: 84.51% AP / 88.02% AUC on XD-Violence/UCF-Crime, dual-branch (visual classifier + language-vision alignment).

**[PANDA](https://github.com/showlab/PANDA)** (NeurIPS 2025) — *the* single closest academic system to your pattern. "Agentic AI engineer" built on MLLMs with: scene-aware strategy planning, goal-driven heuristic reasoning, tool-augmented self-reflection, self-improving chain-of-memory. Self-adaptive scene-aware RAG retrieves anomaly-specific knowledge per scene. SOTA on multi-scenario / open-set / complex without training. **Read this repo first.**

**[Holmes-VAU](https://github.com/pipixin321/HolmesVAU)** (CVPR 2025 Highlight) — HIVAU-70k (70K multi-granular annotations at clip/event/video level) + *Anomaly-focused Temporal Sampler* (ATS) that uses an anomaly scorer to feed the VLM only frames likely to contain anomalies. Exactly the "VLM on candidates only" cost-saving move.

**[Dinomaly](https://github.com/guojiajeremy/Dinomaly)** (CVPR 2025) — first multi-class unsupervised AD model competing with single-class SOTAs. DINOv2-Register encoder + noisy bottleneck + linear attention + loose reconstruction. 99.6% AUROC on MVTec-AD. Being merged into Anomalib ([issue #2782](https://github.com/open-edge-platform/anomalib/issues/2782)).

**[Awesome-Anomaly-Detection-Foundation-Models](https://github.com/mala-lab/Awesome-Anomaly-Detection-Foundation-Models)** is the maintained reading list.

**[NVIDIA VSS](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)** — Blueprint that most closely mirrors your architecture. Anomaly mode uses *RT-VLM* — Cosmos Reason1/2 or Qwen3-VL — to caption and flag *after* a perception layer generates candidate alerts. Agent layer uses MCP for incident records and tool orchestration. [VSS architecture page](https://docs.nvidia.com/vss/latest/content/architecture.html) is the cleanest open document of the pattern. Heavy: needs NIM microservices and an H100.

**[Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io/product/models-overview)** — closest commercial-API analog: Marengo for video embeddings (kNN anomaly scoring) + Pegasus for grounded Q&A over up-to-2-hour clips. The [Qdrant tutorial](https://qdrant.tech/documentation/tutorials-build-essentials/video-anomaly-edge-part-1/) shows a worked anomaly-detection pipeline. Worth borrowing: "embeddings serve dual duty: kNN scoring AND semantic search."

## 3. Academic State-of-Art (2025–26)

### Weakly-supervised on UCF-Crime / XD-Violence

**[GS-MoE](https://openaccess.thecvf.com/content/ICCV2025/papers/Amicantonio_Mixture_of_Experts_Guided_by_Gaussian_Splatters_Matters_A_new_ICCV_2025_paper.pdf)** (ICCV 2025) — 91.58% AUC on UCF-Crime, beating VadCLIP by 3.56 pts. **[π-VAD](https://openaccess.thecvf.com/content/CVPR2025/papers/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.pdf)** (CVPR 2025) — teacher-student with "poly-modal inductor" generating pose/depth/semantic/text/motion at training time only. **[VadCLIP++](https://www.sciencedirect.com/science/article/abs/pii/S1051200425005822)** is the 2025 dynamic VLM extension.

### Memory-augmented & future-frame prediction

**[HF²-VAD](https://arxiv.org/abs/2108.06852)** combines ML-MemAE-SC for optical-flow reconstruction with CVAE for next-frame prediction. **[MemATr](https://dl.acm.org/doi/10.1145/3719203)** (2025) lightweight memory-augmented transformer. **[Patch Diffusion Model](https://arxiv.org/html/2412.09026v1)** (2024) — diffusion-based reconstruction error. Survey: [transformers for VAD (Springer 2025)](https://link.springer.com/article/10.1007/s00521-025-11218-1).

### Training-free / Zero-shot VLM approaches

- **[LAVAD](https://lucazanella.github.io/lavad/)** (CVPR 2024) — captioning + LLM aggregation
- **AnomalyRuler** — induce-then-deduce: build normality rules from few normal references, deductively flag
- **[VERA](https://vera-framework.github.io/)** (CVPR 2025) — verbalised learning of guiding questions on a frozen VLM
- **[Holmes-VAU](https://arxiv.org/abs/2412.06171)** — hierarchical multi-granular reasoning
- **[Fine-Grained Prompting](https://arxiv.org/html/2510.02155v1)** (Oct 2025) — action-centric structured prompts unlock frozen VLMs
- **[Sparse Reasoning is Enough / ReCoVAD](https://arxiv.org/html/2511.17094)** (Nov 2025) — biological-inspired sparse VLM calls only on suspect frames; matches your "VLM on candidates only" exactly

### Multi-modal fusion

**[Multimodal VAD (IEEE 2025)](https://ieeexplore.ieee.org/iel8/19/10764799/11037404.pdf)** combines audio-vision-language. **[Nature Sci Rep 2025](https://www.nature.com/articles/s41598-025-01146-4)** — video+audio fusion gets 10–12% AUC lift on UCSD Ped2 / Avenue.

### Datasets that matter

[UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) (1,900 surveillance vids, 13 classes, weakly labelled), [XD-Violence](https://roc-ng.github.io/XD-Violence/) (4,754 vids w/ audio, 6 categories), [ShanghaiTech & UCSD Ped] (campus, semi-supervised), [MSAD](https://github.com/Tom-roujiang/MSAD) (NeurIPS 2024, 14 scenarios), [NWPU Campus](https://campusvad.github.io/) (43 scenes, scene-dependent, 16 hrs). The 2025 [Rethinking Metrics paper](https://arxiv.org/html/2505.19022v1) re-annotates these and introduces UCF-HN/MSAD-HN "hard normal" benchmarks via diffusion synthesis — important because the original AUC numbers are saturated and arguably leak.

## 4. Adjacent Agentic Patterns

The genuine *agents* (decide what to investigate, call tools, write a report):

- **[PANDA](https://arxiv.org/abs/2509.26386)** (NeurIPS 2025) — agentic AI engineer with tool-augmented self-reflection and chain-of-memory. Clearest "VLM analyst pass" prototype in research.
- **[NVIDIA VSS Agent](https://developer.nvidia.com/blog/advance-video-analytics-ai-agents-using-the-nvidia-ai-blueprint-for-video-search-and-summarization/)** — top-level agent uses MCP for incident records, calls VLM/embedding/clip-retrieval as tools.
- **[Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2)** — NVIDIA's reasoning VLM with chain-of-thought, 256K input tokens, 2B/8B sizes. Open weights.
- **[ReCoVAD](https://arxiv.org/html/2511.17094)** — biological-inspired sparse reasoning: fast reflex pathway flags candidates, slow conscious pathway runs the VLM. Same architecture as your design.
- **[Customizable-VLM](https://github.com/Xiaohao-Xu/Customizable-VLM)** (CSCWD 2025 Best Student Paper) — multimodal anomaly detection with customisable VLM.
- **[Agentic and LLM-Based Multimodal AD survey](https://www.preprints.org/manuscript/202602.1368)** (Feb 2026) — taxonomy of architectures and tradeoffs.

## 5. Production Reports & Case Studies

Public deployment writeups are sparse. Clearest is Ambient.ai's [TikTok USDS data centre case study](https://www.ambient.ai/customer-story/tiktok-usds-ambient-ai). The [VAST Data + NVIDIA Cosmos Reason smart-city blog](https://www.vastdata.com/blog/vast-nvidia-cosmos-reason-smart-cities) describes Hitachi/Milestone/VAST customers using Cosmos Reason. The **[Qdrant tutorial](https://qdrant.tech/documentation/tutorials-build-essentials/video-anomaly-edge-part-1/)** is the closest open recipe — Marengo embeddings + kNN scoring + Pegasus Q&A on edge.

A typical "shift report" output: timestamped event list, severity, thumbnail strips, natural-language description, links back to source clip.

## 6. The Honest Assessment

**Where it works.** Industrial *image* AD on controlled lighting (Anomalib + DINOv2 backbones hit 99%+ AUROC on MVTec). Edge-case *mining* over AV logs. Coarse motion-baseline anomalies in fixed-camera surveillance. Constrained, signature-based detection. False-alarm filtering for alarm centres.

**Where it breaks.** [Beyond Academic Benchmarks (CVPR 2025 workshop)](https://arxiv.org/html/2503.23451v1) is the most useful honest critique: 99.9% AUROC on MVTec degrades significantly on real production data. [2025 survey](https://arxiv.org/html/2508.14203v1) catalogues failure modes: domain shift between cameras, day/night cycles, weather, dynamic backgrounds (trees in wind = false positives), per-camera calibration drift, catastrophic forgetting on incremental updates.

**Specific honest gap for VLM agents:** VLM frame-by-frame is expensive enough that nobody is actually running it on every frame in production. Holmes-VAU's ATS sampler, ReCoVAD's reflex pathway, and PANDA's scene-aware planner all exist *because* you cannot afford to call a VLM on a 24-hour stream. Your "VLM analyst pass on candidates only" is the right move — it's already what the SOTA does.

**Second gap:** hallucination on negative examples. VLMs cheerfully describe anomalies that aren't there with leading prompts. [Benchmarking Compact VLMs (PMC12653427)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12653427/) finds compact VLMs at clip level surprisingly weak under weak supervision.

**Third gap:** nobody has solved *normality drift* (your normal slowly changes over months) without constant retraining or scoring drift. Open problem.

---

## The 5 Closest Analogs (ranked, most → least applicable)

1. **[PANDA (NeurIPS 2025)](https://github.com/showlab/PANDA)** — agentic AI engineer for VAD with scene-aware planning, tool use, self-reflection, chain-of-memory. Same architecture as you, with code. **Start here.**
2. **[NVIDIA VSS Blueprint](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)** — production-grade reference for the exact pipeline. Heavy on NVIDIA hardware.
3. **[Holmes-VAU (CVPR 2025 Highlight)](https://github.com/pipixin321/HolmesVAU)** — Anomaly-focused Temporal Sampler is *the* mechanism for "VLM on candidates only." Borrow directly.
4. **[Ambient.ai Pulsar](https://www.ambient.ai/blog/introducing-ambient-pulsar-agentic-physical-security)** — closest commercial analog in security. Edge VLM + signature library hybrid.
5. **[Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io/product/models-overview)** with the **[Qdrant tutorial](https://qdrant.tech/documentation/tutorials-build-essentials/video-anomaly-edge-part-1/)** — easiest API path to a working prototype.

## The 3 Things Nobody Has Solved Well Yet

1. **Normality drift over time.** Every system bakes in a stationary baseline. In real surveillance, normal slowly shifts. Published incremental-learning approaches all suffer catastrophic forgetting on rare events.
2. **Long-context temporal reasoning at scale.** Pegasus claims 2-hour context, Cosmos Reason claims 256K tokens. Real intel/security feeds are days/weeks. Hierarchical summarisation (Holmes-VAU's clip→event→video) is the current best answer but lossy where rare anomalies hide.
3. **Calibrated anomaly *severity*, not just *score*.** All SOTA produces an AUC-friendly score; few produce calibrated risk that ranks by triage value. Severity is multi-axis (rarity × consequence × confidence) — current systems collapse to one number. Commercial products lean on hand-curated signature libraries; open-source side has no good answer.

## Specific Techniques/Papers Worth Borrowing

- **[Holmes-VAU's Anomaly-focused Temporal Sampler](https://arxiv.org/abs/2412.06171)** — density-aware frame sampling driven by an anomaly scorer. Drop-in for "candidates only" stage.
- **[LAVAD's two-stage pipeline](https://lucazanella.github.io/lavad/)** (BLIP-2 captions → LLM temporal aggregation → ImageBind cleanup) — fully training-free baseline you can build on Apple Silicon in a day.
- **[Dinomaly's DINOv2 + noisy bottleneck + loose reconstruction](https://github.com/guojiajeremy/Dinomaly)** — best-in-class unsupervised baseline scorer. DINOv2 runs cleanly via MLX/coreML.
- **[VadCLIP's dual-branch](https://github.com/nwpu-zxr/VadCLIP)** (vision classifier + language-vision alignment) — cheap explainability via class prompts without training.
- **[PANDA's chain-of-memory + scene-aware RAG](https://arxiv.org/abs/2509.26386)** — borrow the agent loop and per-scene knowledge retrieval pattern wholesale.
- **[ReCoVAD's reflex/conscious dual pathway](https://arxiv.org/html/2511.17094)** — formalises your "fast scorer + slow VLM"; gives the right mental model and citation.
- **[π-VAD's poly-modal inductor (training only)](https://openaccess.thecvf.com/content/CVPR2025/papers/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.pdf)** — distil pose/depth/text into RGB at train time so inference stays cheap.
- **[MSAD dataset](https://github.com/Tom-roujiang/MSAD)** + **[Rethinking Metrics paper](https://arxiv.org/html/2505.19022v1)** — your evaluation set; original UCF-Crime AUC is saturated and arguably leaky.
- **[Anomalib](https://github.com/open-edge-platform/anomalib)** — scaffold for swapping unsupervised image-level scorers; OpenVINO export gets you onto Apple Silicon via ONNX/coreML.
- **[Cosmos Reason 2 (8B)](https://github.com/nvidia-cosmos/cosmos-reason2)** — open-weights reasoning VLM with chain-of-thought tuned for physical understanding; runs on a single GPU.

For Apple Silicon solo build: realistic stack is **DINOv2 (via MLX)** for baseline scoring + **temporal sampler à la Holmes-VAU** + **Qwen2.5-VL-7B or Cosmos Reason 1 7B (via mlx-vlm)** for analyst pass + small agent loop borrowing **PANDA's planner pattern**. Skip NVIDIA VSS as a runtime — read it as a reference architecture only.
