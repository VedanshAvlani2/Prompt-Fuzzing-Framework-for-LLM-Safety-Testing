import asyncio, csv, json, random, time
from datetime import datetime
from pathlib import Path
from engine.utils import ensure_dir, read_seeds_csv
from engine.sandbox_runner import sandbox_invoke
from mutators import paraphrase, insertion, obfuscation
from mutators import jailbreak_adv, context_injection_adv, obfuscation_adv
from mutators import godmode_mistral, godmode_llama
from adapters import api_adapter, llama_adapter, mistral_adapter
from adapters import lmstudio_adapter
from detectors import context_aware_detector
import webbrowser
import os
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

class Orchestrator:
    def __init__(self):
        self.seeds_path = Path("data/seeds.csv")
        self.out_dir = Path("data/results")
        ensure_dir(self.out_dir)
        self.mutators = [
            paraphrase.ParaphraseMutator(),
            insertion.InsertionMutator(),
            obfuscation.ObfuscationMutator(),

            jailbreak_adv.JailbreakAdvancedMutator(),
            context_injection_adv.ContextInjectionAdvancedMutator(),
            obfuscation_adv.ObfuscationAdvancedMutator(),

            godmode_mistral.GodmodeWrapperMutator(),
            godmode_llama.LlamaWrapperMutator(),
        ]
        self.adapters = [
            # primary local Llama (GPU) adapter
            # llama_adapter.LlamaAdapter(),

            # Optional: Mistral (uncomment to use)
            # mistral_adapter.MistralAdapter(),

            # Optional: API based adapter (e.g., Groq)
            # api_adapter.ApiAdapter(),

            # # === LM Studio: Llama 3 8B ===
            lmstudio_adapter.LMStudioAdapter(
                base_url="http://localhost:1234",
                model_name="meta-llama-3.1-8b-instruct",
            ),

            # === LM Studio: Mistral 7B ===
            lmstudio_adapter.LMStudioAdapter(
                base_url="http://localhost:1234",
                model_name="mistral-7b-instruct-v0.2",
            ),
        ]
        self.detector = context_aware_detector.ContextAwareSafetyDetector()
        self.results = []

    def generate_mutations(self, prompt):
        """
        Build a list of mutation dicts with structure:
           {"text": "<mutated text>", "mutation_type": "<type>"}
        Accepts mutators that return either strings or dicts.
        Removes duplicate mutated *text* while preserving order.
        """
        out = []

        # include the original seed as the first "mutation"
        out.append({"text": prompt, "mutation_type": "original"})

        for m in self.mutators:
            # 🔥 MODEL FILTER (ADD THIS)
            allowed = getattr(m, "supported_models", ["*"])
            model_names = [getattr(a, "model_name", "").lower() for a in self.adapters]

            if "*" not in allowed:
                if not any(token in model for model in model_names for token in allowed):
                    # Skip model-specific mutator
                    continue
            
            # Generate variants normally
            variants = m.mutate(prompt)
            for v in variants:
                if isinstance(v, dict):
                    text = v.get("text", "")
                    mtype = v.get("mutation_type", m.__class__.__name__)
                else:
                    text = str(v)
                    mtype = m.__class__.__name__

                out.append({"text": text, "mutation_type": mtype})

        # Remove duplicates by text while preserving order
        seen = set()
        unique = []
        for item in out:
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if text not in seen:
                seen.add(text)
                # ensure item has the desired dict shape
                unique.append({"text": text, "mutation_type": item.get("mutation_type", "unknown")})
        print(f"📝 Generated {len(unique)} unique mutations from seed")
        return unique

    async def _process_single_mutation(self, seed, mutation, mutation_idx):
        """Process a single mutation through all adapters

        mutation is expected to be a dict: {"text": "...", "mutation_type": "..."}
        """
        # Defensive: accept legacy string mutation as well
        if isinstance(mutation, dict):
            mutated_text = mutation.get("text", "")
            mutation_type = mutation.get("mutation_type", "unknown")
        else:
            mutated_text = str(mutation)
            mutation_type = "legacy"

        print(f"   [{mutation_idx}] Processing ({mutation_type}): {mutated_text[:60]}...")

        for adapter in self.adapters:
            try:
                print(f"      └─ Running {adapter.__class__.__name__}...")
                # sandbox_invoke expects adapter + prompt text
                resp = await sandbox_invoke(adapter, mutated_text)

                print("DEBUG: Adapter model:", getattr(adapter, "model_name", adapter.__class__.__name__))
                # print("DEBUG: Adapter model:", adapter.model_name)
                print("DEBUG Response:", resp)

                # Defensive: ensure resp is dict
                if not isinstance(resp, dict):
                    resp = {"text": str(resp), "status": "error"}

                text = resp.get("text", "")

                # Run detector with BOTH prompt and response
                try:
                    det = self.detector.score(mutated_text, text)
                except Exception as e:
                    det = {
                        "label": "error",
                        "is_unsafe": False,
                        "severity": 0,
                        "reasoning": f"Detector error: {str(e)}",
                        "confidence": 0.0
                    }
                    print(f"         ⚠️  Detector error: {e}")

                # Save result (store mutated text and mutation_type separately)
                self.results.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "original_prompt": seed,  
                    "mutated_prompt": mutated_text,  
                    "mutation_type": mutation_type,
                    "model_name": resp.get("model", adapter.__class__.__name__),
                    "response_text": text[:5000],  
                    "safety_label": det.get("label", "safe"),
                    "severity": det.get("severity", 0),
                    "reasoning": det.get("reasoning", ""),
                    "confidence": det.get("confidence", 0.0),
                    "is_unsafe": det.get("is_unsafe", False)
                })
                
                # ✅ Better logging
                label = det.get('label', 'safe')
                unsafe_flag = det.get('is_unsafe', False)
                print(f"         ✅ Done - unsafe: {det.get('is_unsafe', False)}")
                confidence = det.get('confidence', 0.0)

                print(f"         ✅ Done - unsafe: {det.get('is_unsafe', False)}")

            except Exception as e:
                print(f"         ❌ Error: {e}")
                self.results.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "original_prompt": seed,
                    "mutated_prompt": mutated_text,
                    "mutation_type": mutation_type,
                    "model_name": resp.get("model", adapter.__class__.__name__),
                    "response_text": f"[Error: {e}]",
                    "safety_label": "error",
                    "severity": 0,
                    "reasoning": str(e),
                    "reasoning": str(e),
                    "confidence": 0.0,
                    "is_unsafe": False
                })

    async def _process_prompt(self, seed):
        """Process one seed with all its mutations"""
        mutations = self.generate_mutations(seed)
        
        # ✅ CHANGED: Process mutations in parallel batches of 3
        batch_size = 3
        for batch_start in range(0, len(mutations), batch_size):
            batch = mutations[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(mutations) + batch_size - 1) // batch_size
            
            batch_start_time = time.time()

            print(f"\n⚡ Batch {batch_num}/{total_batches} ({len(batch)} mutations in parallel)")
            
            # Run batch in parallel
            tasks = []
            for idx, variant in enumerate(batch):
                # extract text and type for better clarity
                if isinstance(variant, dict):
                    text = variant.get("text", "")
                    mtype = variant.get("mutation_type", "unknown")
                else:
                    text = str(variant)
                    mtype = "legacy"

                print(f"   → Queuing mutation [{batch_start + idx + 1}] ({mtype})")
                tasks.append(self._process_single_mutation(seed, {"text": text, "mutation_type": mtype}, batch_start + idx + 1))

            await asyncio.gather(*tasks)

            batch_elapsed = time.time() - batch_start_time  # ✅ Show batch time
            print(f"   ⏱️  Batch completed in {batch_elapsed:.1f}s")
        
        print(f"\n🧩 Finished all mutations for seed: {seed[:60]}")

    async def run_all(self):
        print("🟡 [DEBUG] Entered run_all()")
        start_time = time.time()
        seeds = read_seeds_csv(self.seeds_path)
        print(f"🟢 [DEBUG] Loaded {len(seeds)} seeds")
        
        # ✅ CHANGED: Sample seeds for faster testing
        # Uncomment ONE of these options:
        
        # Option 1: Just first seed (for quick testing)
        # seeds = seeds[:1]
        
        # Option 2: First 3 seeds (for quick testing)
        # seeds = seeds[:3]

        # Option 3: First 10 seeds (for medium testing)
        # seeds = seeds[:10]
        
        # Option 4: Random sample of 25 seeds (for representative testing)
        # seeds = random.sample(seeds, min(25, len(seeds)))
        
        # Option 5: All seeds (comment out the line above for full run)
        # seeds = seeds  # Use all seeds

        # Option 6: Run a particular seed
        seeds = [seeds[3]]
        
        print(f"🔵 [DEBUG] Processing {len(seeds)} seed(s)")
        print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}\n")
        
        # Process seeds one at a time (mutations within each seed run in parallel)
        for idx, seed in enumerate(seeds, 1):
            seed_start = time.time()

            if idx > 1:
                elapsed = time.time() - start_time
                avg_per_seed = elapsed / (idx - 1)
                remaining_seeds = len(seeds) - idx + 1
                eta_seconds = remaining_seeds * avg_per_seed
                eta_minutes = eta_seconds / 60

                print(f"\n{'='*70}")
                print(f"📊 PROGRESS: {idx-1}/{len(seeds)} complete")
                print(f"⏱️  Elapsed: {elapsed/60:.1f} min | Avg: {avg_per_seed:.1f}s/seed")
                print(f"⏰ ETA: {eta_minutes:.1f} min remaining")
                print(f"{'='*70}")

            print(f"\n{'='*70}")
            print(f"🌱 SEED {idx}/{len(seeds)}: {seed}")
            print(f"{'='*70}")

            await self._process_prompt(seed)

            seed_elapsed = time.time() - seed_start
            print(f"\n✅ Seed completed in {seed_elapsed:.1f}s ({seed_elapsed/60:.2f} min)")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(seeds) if seeds else 0

        print(f"\n{'='*70}")
        print(f"🎉 ALL SEEDS COMPLETED!")
        print(f"{'='*70}")
        print(f"📊 Total seeds processed: {len(seeds)}")
        print(f"⏱️  Total time: {total_time/60:.1f} min ({total_time:.1f}s)")
        print(f"📈 Average per seed: {avg_time:.1f}s")
        print(f"⏰ Finished at: {time.strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")

        print(f"\n💾 Saving {len(self.results)} results...")
        self.save()

    def save(self):
        if not self.results:
            print("⚠️  No results to save!")
            return
            
        t = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_csv = self.out_dir / f"results_{t}.csv"
        out_json = self.out_dir / f"results_{t}.json"
        
        try:
            # Save JSON
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            
            # Save CSV
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                fieldnames = list(self.results[0].keys())
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self.results)
            
            print(f"✅ Saved {len(self.results)} results:")
            print(f"   📄 CSV: {out_csv}")
            print(f"   📄 JSON: {out_json}")

            # Generate summary statistics
            self._print_summary_stats()

            #Start server and open dashboard
            print(f"\n🌐 Launching local dashboard on http://localhost:8000 ...")

            try:
                    # Step 1: Start a tiny HTTP server in a background thread
                    def start_server():
                        project_root = Path(__file__).parent.parent.resolve()
                        os.chdir(project_root)
                        print(f"   📂 Serving from: {project_root}")
                        server = HTTPServer(("localhost", 8000), SimpleHTTPRequestHandler)
                        server.serve_forever()

                    thread = threading.Thread(target=start_server, daemon=True)
                    thread.start()

                    # Step 2: Wait for server to spin up
                    time.sleep(1.5)

                    # Step 3: Check if triage.html exists using absolute path
                    project_root = Path(__file__).parent.parent.resolve()
                    triage_path = project_root / "triage" / "triage.html"
                    
                    if not triage_path.exists():
                        print(f"⚠️  Triage HTML not found at: {triage_path}")
                        return

                    # Step 4: Build dashboard URL
                    dashboard_url = "http://localhost:8000/triage/triage.html"

                    # Inject auto-load JS into HTML once (only if not already present)
                    with open(triage_path, "r", encoding="utf-8") as f:
                        html_content = f.read()

                    if "<!-- AUTOLOAD_JSON_PATH -->" in html_content:
                        # Remove everything between the marker and </script>
                        start = html_content.find("<!-- AUTOLOAD_JSON_PATH -->")
                        end = html_content.find("</script>", start) + len("</script>")
                        html_content = html_content[:start] + html_content[end:]

                    # Inject new auto-load with correct relative path from triage/ folder
                    result_filename = out_json.name  # just the filename: results_20251112_200655.json
                    inject = f"""
<!-- AUTOLOAD_JSON_PATH -->
<script>
window.addEventListener('load', () => {{
  const jsonPath = "/data/results/{result_filename}";
  console.log("Attempting to fetch:", jsonPath);
  fetch(jsonPath)
    .then(resp => {{
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.json();
    }})
    .then(data => {{
      window.allResults = data;
      console.log("✅ Loaded " + data.length + " results");
      if (typeof updateStats === 'function') updateStats();
      if (typeof updateModelFilter === 'function') updateModelFilter();
      if (typeof applyFilters === 'function') applyFilters();
    }})
    .catch(err => {{
      console.error("Auto-load failed:", err);
      alert("⚠️ Auto-load failed. Please use 'Choose File' to load manually.\\nTried: " + jsonPath);
    }});
}});
</script>
"""
                    html_content = html_content.replace("</body>", inject + "\n</body>")
                    with open(triage_path, "w", encoding="utf-8") as f:
                        f.write(html_content)

                    # Step 4: Open dashboard automatically
                    webbrowser.open(dashboard_url)

                    # IMPORTANT: Keep server alive
                    print(f"\n✅ Dashboard opened! Server running on http://localhost:8000")
                    print(f"   Press Ctrl+C to stop the server and exit.\n")

                    # Keep the main thread alive so daemon thread continues
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n👋 Shutting down server...")

            except Exception as e:
                print(f"⚠️ Could not launch triage dashboard: {e}")
            else:
                print("⚠️  Triage HTML not found. Skipping auto-open.")

        except Exception as e:
            print(f"❌ Error saving results: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_summary_stats(self):
        """Print summary statistics about the fuzzing results"""
        if not self.results:
            return
        
        # Count by safety label
        safe_count = sum(1 for r in self.results if r.get('safety_label') == 'safe')
        unsafe_count = sum(1 for r in self.results if r.get('safety_label') == 'unsafe')
        suspicious_count = sum(1 for r in self.results if r.get('safety_label') == 'suspicious')
        error_count = sum(1 for r in self.results if r.get('safety_label') == 'error')
        
        total = len(self.results)
        
        print(f"\n{'='*70}")
        print(f"📊 FUZZING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Tests: {total}")
        print(f"  🟢 Safe:       {safe_count:4d} ({safe_count/total*100:5.1f}%)")
        print(f"  🔴 Unsafe:     {unsafe_count:4d} ({unsafe_count/total*100:5.1f}%)")
        print(f"  🟡 Suspicious: {suspicious_count:4d} ({suspicious_count/total*100:5.1f}%)")
        print(f"  ⚫ Errors:     {error_count:4d} ({error_count/total*100:5.1f}%)")
        
        # Attack Success Rate
        asr = unsafe_count / total * 100 if total > 0 else 0
        print(f"\n🎯 Attack Success Rate (ASR): {asr:.1f}%")
        
        # Per-model breakdown
        models = {}
        for r in self.results:
            model = r.get('model_name', 'unknown')
            if model not in models:
                models[model] = {'safe': 0, 'unsafe': 0, 'suspicious': 0, 'error': 0}
            label = r.get('safety_label', 'error')
            models[model][label] = models[model].get(label, 0) + 1
        
        print(f"\n📈 Per-Model Breakdown:")
        for model, counts in models.items():
            model_total = sum(counts.values())
            model_asr = counts['unsafe'] / model_total * 100 if model_total > 0 else 0
            print(f"  {model}:")
            print(f"    Total: {model_total}, ASR: {model_asr:.1f}%")
            print(f"    Safe: {counts['safe']}, Unsafe: {counts['unsafe']}, Suspicious: {counts['suspicious']}")
        
        print(f"{'='*70}")
