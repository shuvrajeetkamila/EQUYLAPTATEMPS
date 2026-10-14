import json, os, urllib.request
CANDS = [("HuggingFaceTB/SmolLM-135M", "1d461723eec654e65efdc40cf49301c89c0c92f4"),
         ("HuggingFaceTB/SmolLM-360M", "59f7ef243ee09a72cbc14cb054393a3e3b771d41"),
         ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6")]
out = []
for repo, sha in CANDS:
    rec = {"id": repo, "sha": sha}
    for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
        try:
            with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{sha}/{name}", timeout=60) as r:
                rec[name] = json.loads(r.read().decode())
        except Exception as exc:
            rec[name] = f"ERROR {exc}"
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, revision=sha)
        rec["license_tags"] = [t for t in (info.tags or []) if "license" in t]
        rec["card_license"] = (info.card_data or {}).get("license") if info.card_data else None
        rec["files"] = [(s.rfilename, s.size) for s in (info.siblings or []) if s.rfilename.endswith((".safetensors", ".md"))]
    except Exception as exc:
        rec["hub_error"] = str(exc)
    out.append(rec)
body = "<!-- campaign07-probe2 -->\n```json\n" + json.dumps(out, indent=1)[:60000] + "\n```"
req = urllib.request.Request(
    "https://api.github.com/repos/shuvrajeetkamila/EQUYLAPTATEMPS/issues/1/comments",
    data=json.dumps({"body": body}).encode(), method="POST",
    headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"})
print(urllib.request.urlopen(req).status)
