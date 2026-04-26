"""
Heuristic semantic grouping of the gate-ranked top-15 APIs per family.

Buckets:
    network              — HTTP/URL/socket/wifi/connectivity
    filesystem           — File/IO/SharedPreferences/SQLite
    telephony_sms        — TelephonyManager/SmsManager/SmsMessage
    reflection_obfuscation — REFL: prefix or Method.invoke / Class.forName
    crypto               — javax.crypto / Cipher / MessageDigest / KeyGenerator
    process_runtime      — Runtime.exec / Process / pm install / am start
    other                — anything else

We classify by substring matching on the API string (case-insensitive),
then write `results/api_semantic_groups.json` keyed by family with the
bucket label and a per-family bucket histogram. This is heuristic and
documented as such in the file header.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "results" / "top_apis_per_family.json"
OUTPUT = REPO_ROOT / "results" / "api_semantic_groups.json"

TOP_K = 15


def classify(api: str) -> str:
    s = api.lower()
    # Reflection / obfuscation marker first — set by the resolver
    if api.startswith("REFL:") or "method.invoke" in s or "class.forname" in s:
        return "reflection_obfuscation"
    # Telephony / SMS
    if any(t in s for t in (
        "telephonymanager", "smsmanager", "smsmessage", "android.telephony",
        "getdeviceid", "getsubscriberid", "getsiminfo", "getline1number",
        "sendtextmessage", "sendmultipartmessage",
    )):
        return "telephony_sms"
    # Network
    if any(t in s for t in (
        "java.net.url", "openconnection", "httpurlconnection", "httpclient",
        ".httpget", ".httppost", "connectivitymanager", "wifimanager",
        "socket", "inetaddress", "android.webkit", "websettings",
    )):
        return "network"
    # Crypto
    if any(t in s for t in (
        "javax.crypto", "cipher", "messagedigest", "keygenerator", "secretkey",
        "ivparameterspec", "mac.", "java.security.",
    )):
        return "crypto"
    # Process / Runtime / package install
    if any(t in s for t in (
        "java.lang.runtime.exec", "processbuilder", "system.loadlibrary",
        "packageinstaller", "packagemanager.install", "am.start",
    )):
        return "process_runtime"
    # Filesystem
    if any(t in s for t in (
        "java.io.file", "fileoutputstream", "fileinputstream", "filewriter",
        "filereader", "sharedpreferences", "android.database.sqlite",
        "context.getfilesdir", "context.opendatabase", "openorcreatedatabase",
        "contentresolver", "contentvalues", "cursor.",
    )):
        return "filesystem"
    return "other"


def main() -> None:
    with open(INPUT) as f:
        data = json.load(f)

    out = {
        "_about": (
            "Heuristic semantic grouping of the gate-ranked top-{k} APIs per family. "
            "Buckets: network, filesystem, telephony_sms, reflection_obfuscation, "
            "crypto, process_runtime, other. Classification is by substring matching "
            "and is approximate; review individual labels for fine-grained claims."
        ).format(k=TOP_K),
        "top_k": TOP_K,
        "buckets": [
            "network", "filesystem", "telephony_sms",
            "reflection_obfuscation", "crypto", "process_runtime", "other",
        ],
        "by_family": {},
    }

    for fam, ranked in data.items():
        labelled = []
        bucket_counts: Counter = Counter()
        for entry in ranked[:TOP_K]:
            api, score = entry[0], entry[1]
            bucket = classify(api)
            labelled.append({"api": api, "gate_score": score, "bucket": bucket})
            bucket_counts[bucket] += 1
        out["by_family"][fam] = {
            "apis": labelled,
            "bucket_histogram": dict(bucket_counts),
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {OUTPUT}")
    for fam, body in out["by_family"].items():
        print(f"  {fam:14s}  {body['bucket_histogram']}")


if __name__ == "__main__":
    main()
