# part1_resume_cache.py
# ---------------------
# Part-1: Extract resumes → JSON using Groq LLM
# Caching + Error handling + Logging (with enhancements)
# Compatible with folder structure:
#
# cache/
# resume/
# logs/
# output/
# jd/
#
# Requires: pip install groq python-dotenv PyPDF2 python-docx tenacity

import os
import json
import hashlib
import logging
import time
import contextlib
import textwrap
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq, APIStatusError  # Groq official SDK

# [GROQ FIX] import SDK-specific connection/timeout errors; keep optional fallback
try:
    from groq import APIConnectionError, APITimeoutError  # type: ignore
except Exception:  # pragma: no cover
    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
)
from PyPDF2 import PdfReader
from docx import Document

# Cross-platform (best-effort) file lock: POSIX gets real lock; others no-op
try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False


@contextlib.contextmanager
def file_lock(lock_path: Path):
    """
    Simple advisory file lock to prevent concurrent writers from corrupting JSON files.
    POSIX only; on non-POSIX it's a no-op but still safe thanks to atomic replace.
    """
    if _HAS_FCNTL:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    else:
        # Fallback: no-op
        yield


# -----------------------------
# Folder Setup
# -----------------------------
BASE = Path(__file__).parent.resolve()

RESUME_DIR = BASE / "resume"
CACHE_DIR = BASE / "cache"
LOGS_DIR = BASE / "logs"

CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Logging
# -----------------------------
LOG_FILE = LOGS_DIR / "part1.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# -----------------------------
# Config (env-overridable)
# -----------------------------
SCHEMA_VERSION = "v1"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # override via .env
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "2048"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "120000"))

# [GROQ FIX] Tunables for network robustness
GROQ_TIMEOUT_SECONDS = int(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))   # per-request timeout
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "8"))           # total retries
GROQ_BACKOFF_MAX = int(os.getenv("GROQ_BACKOFF_MAX", "20"))          # max backoff
SLEEP_BETWEEN_FILES = float(os.getenv("SLEEP_BETWEEN_FILES", "0.8")) # debounce to avoid 429

# [CACHE HEAL] Flags
PURGE_FAILED_CACHE_ON_START = os.getenv("PURGE_FAILED_CACHE_ON_START", "1") == "1"
IGNORE_CACHE_COMPLETELY = os.getenv("IGNORE_CACHE", "0") == "1"


# -----------------------------
# Utility Functions
# -----------------------------
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize(text: str) -> str:
    return " ".join((text or "").split())


def read_pdf(path: Path) -> str:
    """
    More robust PDF extraction:
    - Per-page try/except (so one bad page doesn't kill whole file)
    - Warn if text extracted is very small (likely scanned PDF)
    """
    try:
        reader = PdfReader(str(path))
        pages = []
        for i, p in enumerate(reader.pages[:MAX_PAGES]):
            try:
                t = p.extract_text() or ""
                pages.append(t)
            except Exception as ie:
                logging.warning(f"Page {i} parse failed for {path.name}: {ie}")
        text = "\n".join(pages).strip()
        if len(text) < 100:
            logging.warning(f"Very low text extracted from {path.name} (len={len(text)}). Likely scanned PDF.")
        return text
    except Exception as e:
        logging.warning(f"PDF read failed for {path.name}: {e}")
        return ""


def read_docx(path: Path) -> str:
    try:
        doc = Document(str(path))
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        logging.warning(f"DOCX read failed: {path.name}: {e}")
        return ""


def read_any(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in [".doc", ".docx"]:
        # NOTE: python-docx doesn't read legacy .doc; this will likely fail for .doc
        # We still try and log if it fails.
        return read_docx(path)
    if ext in [".txt", ".md"]:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logging.warning(f"Text read failed: {path.name}: {e}")
            return ""
    logging.info(f"Unsupported type skipped: {path.name}")
    return ""


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"Cache corrupted at {path.name}. Backing up. Err: {e}")
        try:
            path.rename(path.with_suffix(".bak"))
        except Exception:
            pass
        return {}


def dump_json(path: Path, data: Dict[str, Any]):
    """
    Atomic write using temp file + optional file lock to prevent concurrent writers.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    with file_lock(lock_path):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


# -----------------------------
# Schema helpers
# -----------------------------
REQUIRED_KEYS = {"name", "total_experience_years", "skills", "education", "recent_roles", "summary"}


def validate_and_default(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    (1) JSON validity & schema guarantees:
    - Ensure required keys exist
    - Coerce basic types
    - Sanitize role list
    """
    out = {
        "name": obj.get("name") or "",
        "total_experience_years": obj.get("total_experience_years") or 0,
        "skills": obj.get("skills") or [],
        "education": obj.get("education") or [],
        "recent_roles": obj.get("recent_roles") or [],
        "summary": obj.get("summary") or "",
    }

    # type coercions
    try:
        out["total_experience_years"] = float(out["total_experience_years"])
    except Exception:
        out["total_experience_years"] = 0.0

    if not isinstance(out["skills"], list):
        out["skills"] = []
    if not isinstance(out["education"], list):
        out["education"] = []
    if not isinstance(out["recent_roles"], list):
        out["recent_roles"] = []

    roles = []
    for r in out["recent_roles"]:
        if isinstance(r, dict):
            roles.append({
                "title": str(r.get("title") or "").strip(),
                "company": str(r.get("company") or "").strip(),
                "start": str(r.get("start") or "").strip(),
                "end": str(r.get("end") or "").strip(),
            })
    out["recent_roles"] = roles
    return out


def clean_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final sanitizer for the extracted JSON:
    - Trim strings, coerce types
    - Normalize roles
    - Remove roles that look like projects/apps instead of employers
    - Remove roles if total_experience_years == 0
    - Deduplicate roles (title+company+start+end)
    """
    def _s(x):
        return x.strip() if isinstance(x, str) else ""

    def _ls(xs):
        return [_s(s) for s in xs if isinstance(s, str)]

    # --- base fields ---
    name = _s(obj.get("name"))
    try:
        total_exp = float(obj.get("total_experience_years", 0) or 0)
    except Exception:
        total_exp = 0.0

    skills = _ls(obj.get("skills", []))
    education = _ls(obj.get("education", []))

    # Normalize casing/duplicates in skills (optional but helpful)
    # e.g., "JavaScript" and "javascript" collapse to "JavaScript"
    seen_skill = set()
    norm_skills = []
    for sk in skills:
        key = sk.lower()
        if key not in seen_skill and sk:
            seen_skill.add(key)
            norm_skills.append(sk)
    skills = norm_skills

    # --- roles normalize ---
    raw_roles = obj.get("recent_roles", [])
    roles: List[Dict[str, str]] = []
    if isinstance(raw_roles, list):
        for r in raw_roles:
            if not isinstance(r, dict):
                continue
            title = _s(r.get("title"))
            company = _s(r.get("company"))
            start = _s(r.get("start"))
            end = _s(r.get("end"))

            # Skip completely empty entries
            if not any([title, company, start, end]):
                continue

            roles.append({
                "title": title,
                "company": company,
                "start": start,
                "end": end,
            })

    # --- company sanitizer: remove obvious project/app names ---
    # If you find more patterns later, add here
    blocked_keywords = [
        "project", "app", "planner", "website", "web app", "clone",
        "portfolio", "repo", "github", "assignment", "mini project",
        "capstone", "case study", "hackathon"
    ]

    validated_roles: List[Dict[str, str]] = []
    for r in roles:
        company_lc = r["company"].lower()
        # allow empty company (unknown), but if company text exists and looks like project → drop
        if r["company"] and any(k in company_lc for k in blocked_keywords):
            continue
        validated_roles.append(r)

    # --- roles deduplication (by tuple) ---
    deduped: List[Dict[str, str]] = []
    seen = set()
    for r in validated_roles:
        key = (r["title"].lower(), r["company"].lower(), r["start"].lower(), r["end"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # --- EXPERIENCE RULE: if total_exp == 0 → no roles ---
    if total_exp == 0:
        deduped = []

    # --- optional length caps (protect against runaway outputs) ---
    # Adjust these as per your needs
    MAX_SKILLS = 80
    MAX_EDUCATION = 20
    MAX_ROLES = 12
    skills = skills[:MAX_SKILLS]
    education = education[:MAX_EDUCATION]
    deduped = deduped[:MAX_ROLES]

    # --- final object ---
    final = {
        "name": name,
        "total_experience_years": float(total_exp),
        "skills": skills,
        "education": education,
        "recent_roles": deduped,
        "summary": _s(obj.get("summary")),
        "_status": "ok",
    }

    return final


def is_valid_cached_data(data: Any) -> bool:
    """
    A cache entry is valid iff:
    - it's a dict
    - it does NOT have _status == "failed"
    - it has all REQUIRED_KEYS (best-effort)
    """
    if not isinstance(data, dict):
        return False
    if data.get("_status") == "failed":
        return False
    # Must have the keys (values can be empty)
    for k in REQUIRED_KEYS:
        if k not in data:
            return False
    return True


def prioritize_resume_text(full: str, max_chars: int) -> str:
    """
    (3) Smarter truncation: keep head+tail to preserve contact/summary (top)
    and recent roles (often near bottom) rather than naive char slice.
    """
    full = (full or "").strip()
    if len(full) <= max_chars:
        return full
    head = full[: int(max_chars * 0.7)]
    tail = full[-int(max_chars * 0.3):]
    return (head + "\n...\n" + tail)[:max_chars]


# -----------------------------
# Groq LLM Client wrapper
# -----------------------------
RETRIABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retriable(e: Exception) -> bool:
    """
    Retry only on network-ish / 5xx / 429 errors.
    [GROQ FIX] Catch SDK-specific connection/timeout exceptions.
    """
    if isinstance(e, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(e, APIStatusError):
        try:
            return getattr(e, "status_code", 0) in RETRIABLE_STATUS
        except Exception:
            return False
    return False


class GroqResumeExtractor:
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = Groq(api_key=api_key)

    @retry(
        retry=retry_if_exception(_is_retriable),
        wait=wait_exponential_jitter(initial=1, max=GROQ_BACKOFF_MAX),
        stop=stop_after_attempt(GROQ_MAX_RETRIES),
        reraise=True
    )
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract strictly-typed JSON using Groq; robust JSON recovery + schema validation.
        """
        SYSTEM_PROMPT = textwrap.dedent("""
            You are an expert resume information extractor.

            STRICT RULES:

            1. NAME EXTRACTION
            - "name" MUST be the candidate's real human name.
            - Extract ONLY from the FIRST 3 LINES of the resume text.
            - NEVER use project names, app names, team names, product names, website names, college names, or section headers as the name.
            - If the name is not clearly found, return an empty string: "name": "".

            2. COMPANY EXTRACTION
            - "company" MUST be a real employer/organization.
            - DO NOT treat project names (e.g., Bharat Bhraman, AI Trip Planner, Netflix Clone), app names, GitHub repositories, college names, or website names as companies.
            - If no employer/company is mentioned, set "company": "".

            3. EXPERIENCE RULE (IMPORTANT)
            - If "total_experience_years" is 0 or missing:
              → "recent_roles" MUST be an empty list.
            - Projects, internships without a company, and academic coursework MUST NOT be added as job roles.

            4. OUTPUT MUST BE STRICT JSON ONLY with EXACT keys:
            {
              "name": str,
              "total_experience_years": float,
              "skills": [str],
              "education": [str],
              "recent_roles": [{"title": str, "company": str, "start": str, "end": str}],
              "summary": str
            }

            5. ABSOLUTELY DO NOT OUTPUT:
            - Markdown
            - Explanations
            - Extra text
            - Code fences
            - Comments

            Return JSON ONLY.
        """)

        try:
            payload = prioritize_resume_text(text, MAX_TEXT_CHARS)

            # Groq API call with explicit timeout
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract details from resume:\n{payload}"},
                ],
                temperature=0,
                max_tokens=MAX_COMPLETION_TOKENS,
                timeout=GROQ_TIMEOUT_SECONDS,  # crucial fix
            )

            # Guard against empty choices (rare)
            if not getattr(resp, "choices", None):
                raise ValueError("Empty response from Groq model")

            raw = (resp.choices[0].message.content or "").strip()

            # Strict JSON parse with fallback bracket slicing
            try:
                parsed = json.loads(raw)
            except Exception:
                start, end = raw.find("{"), raw.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(raw[start:end + 1])
                else:
                    raise ValueError("Invalid JSON returned by LLM")

            # Validate & coerce shapes/types
            parsed = validate_and_default(parsed)
            return parsed

        except APIStatusError as e:
            # Respect Retry-After for 429
            if getattr(e, "status_code", None) == 429:
                try:
                    retry_after = int(getattr(e, "response", None).headers.get("retry-after", 2))  # type: ignore
                except Exception:
                    retry_after = 2
                logging.warning(f"Rate limit hit, waiting {retry_after}s")
                time.sleep(retry_after)
                raise
            logging.error(f"Groq API error: {e}")
            raise

        except Exception as e:
            logging.error(f"Groq extract failed: {e}")
            raise


# -----------------------------
# Cache healing utilities
# -----------------------------
def heal_cache(cache: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Remove poisoned/invalid entries from resume_cache.json.
    Keeps any non-hash metadata keys (keys starting with '_').
    Returns (new_cache, removed_count).
    """
    if not isinstance(cache, dict):
        return {}, 0

    removed = 0
    new_cache: Dict[str, Any] = {}
    for k, v in cache.items():
        if isinstance(k, str) and k.startswith("_"):
            # carry forward meta keys if any
            new_cache[k] = v
            continue
        if is_valid_cached_data(v):
            new_cache[k] = v
        else:
            removed += 1
    return new_cache, removed


# -----------------------------
# PART‑1: Build Resume Cache
# -----------------------------
def build_cache():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("❌ GROQ_API_KEY missing in .env")

    extractor = GroqResumeExtractor(api_key, DEFAULT_MODEL)

    cache_file = CACHE_DIR / "resume_cache.json"
    master_file = CACHE_DIR / "resume_master.json"

    cache = load_json(cache_file)     # hashed resume → raw LLM data
    master = load_json(master_file)   # list of cleaned objects
    master.setdefault("schema_version", SCHEMA_VERSION)
    master.setdefault("llm_model", DEFAULT_MODEL)
    master.setdefault("resumes", [])

    # [CACHE HEAL] purge invalid/failed cache entries so we don’t get cache_hit=True | status=failed
    if PURGE_FAILED_CACHE_ON_START and cache:
        healed, removed = heal_cache(cache)
        if removed > 0:
            logging.info(f"Cache healing: removed {removed} invalid/failed entries from {cache_file.name}")
            cache = healed
            dump_json(cache_file, cache)

    # Optionally ignore cache completely
    if IGNORE_CACHE_COMPLETELY:
        logging.info("IGNORE_CACHE=1 set. Skipping cache and forcing fresh extraction for all files.")
        cache = {}

    # Index for quick lookups
    known_hashes = {r.get("_hash") for r in master["resumes"]}
    file_index = {r.get("_file"): i for i, r in enumerate(master["resumes"]) if r.get("_file")}

    processed = hits = new_calls = failed = 0

    for file in sorted(RESUME_DIR.iterdir()):
        if file.suffix.lower() not in [".pdf", ".docx", ".doc", ".txt", ".md"]:
            continue

        # Debounce between files to avoid rate limits
        if SLEEP_BETWEEN_FILES > 0:
            time.sleep(SLEEP_BETWEEN_FILES)

        start_ts = time.time()
        text = read_any(file)
        if not text:
            logging.warning(f"No text extracted from {file.name}")
            failed += 1
            continue

        norm = normalize(text)
        h = sha256(norm)

        # --- Cache check (treat invalid/failed entries as miss) ---
        cached_data = cache.get(h)
        is_hit = is_valid_cached_data(cached_data)
        if is_hit:
            data = cached_data  # type: ignore
            logging.info(f"Cache hit: {file.name}")
            hits += 1
        else:
            # If there is a poisoned cache entry, drop it so we can re-extract next time too.
            if h in cache and not is_valid_cached_data(cache[h]):
                cache.pop(h, None)
                dump_json(cache_file, cache)
                logging.info(f"Removed invalid cache entry for {file.name}; re-extracting.")
            logging.info(f"Groq call for: {file.name}")
            try:
                data = extractor.extract(norm)
            except Exception:
                # Do NOT cache failed entries
                failed += 1
                duration = time.time() - start_ts
                logging.info(f"Processed {file.name} in {duration:.2f}s | cache_hit=False | status=failed")
                continue

            # Successful extraction → cache it
            cache[h] = data
            dump_json(cache_file, cache)
            new_calls += 1

        # Safety guard (shouldn’t happen now, but keep it)
        if data.get("_status") == "failed":
            failed += 1
            duration = time.time() - start_ts
            logging.info(f"Processed {file.name} in {duration:.2f}s | cache_hit={is_hit} | status=failed")
            continue

        # Final clean to standardize shape/whitespace
        cleaned = clean_schema(data)
        cleaned["_hash"] = h
        cleaned["_file"] = file.name
        cleaned["_ingested_at"] = datetime.utcnow().isoformat() + "Z"

        # De-duplication/update strategy:
        # - If same filename exists, replace (assumes updated resume with same name)
        # - Else, if new content hash, append
        if file.name in file_index:
            idx = file_index[file.name]
            master["resumes"][idx] = cleaned
        elif h not in known_hashes:
            master["resumes"].append(cleaned)
            file_index[file.name] = len(master["resumes"]) - 1
            known_hashes.add(h)
        else:
            # Already present by hash; skip append
            pass

        processed += 1
        duration = time.time() - start_ts
        logging.info(f"Processed {file.name} in {duration:.2f}s | cache_hit={is_hit} | status=ok")

    dump_json(master_file, master)

    print("=== Part‑1 Done ===")
    print(f"Processed: {processed}")
    print(f"Cache hits: {hits}")
    print(f"New LLM calls: {new_calls}")
    print(f"Failed: {failed}")
    print(f"Cache stored at: {cache_file}")
    print(f"Master stored at: {master_file}")
    print(f"Logs: {LOG_FILE}")


if __name__ == "__main__":
    build_cache()